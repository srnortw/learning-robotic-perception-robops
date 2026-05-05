"""
Phase C — DETR Training Script (Colab Pro+)

Run this in Google Colab. Copy each section as a separate cell.
Before running, set these Colab Secrets (left panel → key icon):
  - AWS_ACCESS_KEY_ID
  - AWS_SECRET_ACCESS_KEY
  - DAGSHUB_TOKEN

Cell 1 — Install dependencies
==============================
!pip install -q transformers torch torchvision mlflow dagshub \
             optimum[onnxruntime] mosaicml-streaming pyyaml \
             pycocotools boto3 accelerate

Cell 2 — Auth + clone repo
===========================
"""

# ── Imports ────────────────────────────────────────────────────────────────
import json
import os
import time
from pathlib import Path

import boto3
import mlflow
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# ── Config ─────────────────────────────────────────────────────────────────
REPO_OWNER = "srnortw"
REPO_NAME = "learning-robotic-perception-robops"
S3_BUCKET = "my-perception-robops-data-2026-688567275774-eu-central-1-an"
AWS_REGION = "eu-central-1"


def setup_colab():
    """Run in Colab to set credentials from Colab Secrets."""
    try:
        from google.colab import userdata
        os.environ["AWS_ACCESS_KEY_ID"] = userdata.get("AWS_ACCESS_KEY_ID")
        os.environ["AWS_SECRET_ACCESS_KEY"] = userdata.get("AWS_SECRET_ACCESS_KEY")
        os.environ["AWS_DEFAULT_REGION"] = AWS_REGION
        dagshub_token = userdata.get("DAGSHUB_TOKEN")
    except ImportError:
        print("Not running in Colab — using environment variables directly.")
        dagshub_token = os.environ.get("DAGSHUB_TOKEN", "")

    import dagshub
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    mlflow.set_experiment("detr-round1")
    print("Auth complete. MLflow tracking to DagsHub.")


def load_params(params_path: str = "pipeline/phase_c/detr/params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


# ── Dataset ────────────────────────────────────────────────────────────────
def build_transform(params: dict):
    size = params["preprocessing"]["image_size"]
    mean = params["preprocessing"]["normalize_mean"]
    std = params["preprocessing"]["normalize_std"]
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def collate_fn(batch):
    images, annotations = zip(*batch)
    return torch.stack(images), list(annotations)


def build_dataloaders(params: dict):
    from streaming import StreamingDataset

    mds_path = params["dataset"]["mds_path"]
    transform = build_transform(params)

    class DETRStreamingDataset(StreamingDataset):
        def __getitem__(self, idx):
            sample = super().__getitem__(idx)
            from PIL import Image as PILImage
            import io
            img = PILImage.open(io.BytesIO(sample["image"])).convert("RGB")
            return transform(img), sample["annotations"]

    train_ds = DETRStreamingDataset(
        local="/tmp/mds_cache/train",
        remote=mds_path + "train/",
        shuffle=True,
    )
    val_ds = DETRStreamingDataset(
        local="/tmp/mds_cache/val",
        remote=mds_path + "val/",
        shuffle=False,
    )

    train_loader = DataLoader(train_ds, batch_size=4, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=4, num_workers=2, collate_fn=collate_fn)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader


# ── Model ──────────────────────────────────────────────────────────────────
def build_model(num_classes: int, freeze_backbone: bool = True):
    model = AutoModelForObjectDetection.from_pretrained(
        "microsoft/conditional-detr-resnet-50",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    if freeze_backbone:
        for param in model.model.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen (Stage 1 coarse)")
    return model


# ── Eval ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, val_loader, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_conf = 0.0
    steps = 0

    for images, annotations in val_loader:
        outputs = model(pixel_values=images.to(device), labels=annotations)
        total_loss += outputs.loss.item()

        logits = outputs.logits  # [B, num_queries, num_classes+1]
        scores = torch.softmax(logits, dim=-1)[:, :, :-1].max(dim=-1).values
        total_conf += scores.mean().item()
        steps += 1
        if steps >= 20:  # cap validation steps for speed
            break

    return total_loss / max(steps, 1), total_conf / max(steps, 1)


# ── Upload to S3 ───────────────────────────────────────────────────────────
def upload_weights(local_path: str, dataset_version: str):
    s3 = boto3.client("s3", region_name=AWS_REGION)
    key = f"weights/detr/{dataset_version}/model.pt"
    s3.upload_file(local_path, S3_BUCKET, key)
    s3_url = f"s3://{S3_BUCKET}/{key}"
    print(f"Weights uploaded → {s3_url}")
    return s3_url


# ── Register in MLflow ─────────────────────────────────────────────────────
def register_model(run_id: str, dataset_version: str, stage: str = "coarse"):
    client = mlflow.tracking.MlflowClient()
    model_name = "detr-conditional-resnet50"
    model_uri = f"runs:/{run_id}/model"

    try:
        client.create_registered_model(model_name)
    except Exception:
        pass  # already exists

    version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
        tags={"stage": stage, "architecture": "detr", "dataset_version": dataset_version},
    )
    client.transition_model_version_stage(
        name=model_name, version=version.version, stage="Staging"
    )
    print(f"Model registered as Staging (version={version.version})")
    print(f"Run ID: {run_id}  ← copy this for GitHub Actions ci_deploy.yml")
    return version.version


# ── Main training loop ─────────────────────────────────────────────────────
def train(params_path: str = "pipeline/phase_c/detr/params.yaml"):
    params = load_params(params_path)
    dataset_version = params["dataset"]["dataset_version"]
    num_classes = params["dataset"]["num_classes"]
    epochs = params["training"]["epochs"]
    lr = params["training"]["learning_rate"]
    weight_decay = params["training"]["weight_decay"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Dataset: {dataset_version} | Classes: {num_classes}")

    train_loader, val_loader = build_dataloaders(params)
    model = build_model(num_classes, freeze_backbone=True).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=params["training"]["lr_drop_epoch"],
        gamma=0.1,
    )

    with mlflow.start_run(run_name=f"detr-stage1-coarse-{dataset_version}") as run:
        mlflow.log_params({
            "model": "conditional-detr-resnet-50",
            "stage": "coarse",
            "image_size": params["preprocessing"]["image_size"],
            "backbone_frozen": True,
            "epochs": epochs,
            "lr": lr,
            "batch_size": params["training"]["batch_size"],
            "mds_path": params["dataset"]["mds_path"],
            "dataset_version": dataset_version,
            "num_classes": num_classes,
        })

        best_val_loss = float("inf")
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            t0 = time.time()

            for step, (images, annotations) in enumerate(train_loader):
                outputs = model(pixel_values=images.to(device), labels=annotations)
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), params["training"]["gradient_clip"]
                )
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            avg_train_loss = epoch_loss / max(len(train_loader), 1)
            val_loss, val_conf = evaluate(model, val_loader, device)

            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_mean_confidence": val_conf,
                "lr": scheduler.get_last_lr()[0],
            }, step=epoch)

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"conf={val_conf:.3f} | "
                f"{elapsed:.1f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "/tmp/detr_best.pt")

        # Upload best weights to S3
        s3_url = upload_weights("/tmp/detr_best.pt", dataset_version)
        mlflow.log_param("s3_weights_url", s3_url)
        mlflow.pytorch.log_model(model, "model")

        run_id = run.info.run_id

    # Register in MLflow registry
    register_model(run_id, dataset_version)

    # Save drift baseline (used by Phase B from Round 2 onward)
    save_drift_baseline(params, val_loader, model, device)

    return run_id


def save_drift_baseline(params: dict, val_loader, model, device):
    """Save training distribution for Phase B drift gate (Round 2+)."""
    baseline_path = Path("pipeline/phase_b/drift_baseline_detr.json")

    class_counts: dict[int, int] = {}
    total_conf = 0.0
    steps = 0

    model.eval()
    with torch.no_grad():
        for images, annotations in val_loader:
            outputs = model(pixel_values=images.to(device), labels=annotations)
            logits = outputs.logits
            scores = torch.softmax(logits, dim=-1)[:, :, :-1]
            conf = scores.max(dim=-1).values.mean().item()
            total_conf += conf
            for ann_list in annotations:
                for ann in ann_list:
                    cat_id = ann.get("category_id", 0)
                    class_counts[cat_id] = class_counts.get(cat_id, 0) + 1
            steps += 1
            if steps >= 50:
                break

    total = sum(class_counts.values()) or 1
    # Map class ids to names from label_schema
    class_freq = {str(k): v / total for k, v in class_counts.items()}

    baseline = {
        "architecture": "detr",
        "dataset_version": params["dataset"]["dataset_version"],
        "class_frequency": class_freq,
        "mean_confidence": round(total_conf / max(steps, 1), 4),
        "mean_brightness": 128.0,  # placeholder — compute from raw images if needed
    }

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"Drift baseline saved → {baseline_path}")
    print("Commit with: git add pipeline/phase_b/drift_baseline_detr.json")


if __name__ == "__main__":
    setup_colab()
    params = load_params()
    print(f"MDS path: {params['dataset']['mds_path']}")
    run_id = train()
    print(f"\nTraining complete. Run ID: {run_id}")
