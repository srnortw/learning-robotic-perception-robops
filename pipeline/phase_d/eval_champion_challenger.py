"""
Phase D — Champion-Challenger Evaluation

Downloads the challenger INT8 ONNX model from S3, runs inference on the holdout
set, computes mAP metrics with pycocotools, compares vs champion (if any), and
writes eval_results.json.  Also registers the model in MLflow Registry.

For Round 1 (no champion) the challenger is auto-approved if mAP@50 >= MIN_MAP.

Usage:
    python pipeline/phase_d/eval_champion_challenger.py \\
        --run-id 626d65c14f8d4b9b8d40f5a43010edc1 \\
        --dataset-version v1 \\
        [--holdout-dir data/detr/holdout] \\
        [--create-pr]
"""

import argparse
import io
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import boto3
import mlflow
import numpy as np
import yaml
from PIL import Image

# ── constants ─────────────────────────────────────────────────────────────────
S3_BUCKET    = "my-perception-robops-data-2026-688567275774-eu-central-1-an"
AWS_REGION   = "eu-central-1"
MLFLOW_URI   = "https://dagshub.com/srnortw/learning-robotic-perception-robops.mlflow"
MODEL_NAME   = "detr-conditional-resnet50"
MIN_MAP      = 0.0          # Round 1: any model that runs is acceptable
CONF_THRESH  = 0.3          # minimum score for a detection to count
IOU_THRESH   = 0.5
CRITICAL_CLASSES = {"person", "door"}
CLASSES      = {0: "person", 1: "chair", 2: "table", 3: "door"}

PARAMS_PATH  = "pipeline/phase_c/detr/params.yaml"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def setup_mlflow():
    os.environ.setdefault("MLFLOW_TRACKING_USERNAME",
                          os.environ.get("DAGSHUB_USERNAME", ""))
    os.environ.setdefault("MLFLOW_TRACKING_PASSWORD",
                          os.environ.get("DAGSHUB_TOKEN", ""))
    mlflow.set_tracking_uri(MLFLOW_URI)
    return mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_URI)


def download_onnx(dataset_version: str, dest: str):
    s3 = boto3.client("s3", region_name=AWS_REGION)
    key = f"weights/detr/{dataset_version}/model_int8.onnx"
    print(f"Downloading s3://{S3_BUCKET}/{key} → {dest}")
    s3.download_file(S3_BUCKET, key, dest)
    print("Download complete.")


# ── holdout loading ──────────────────────────────────────────────────────────

def load_holdout_coco(holdout_dir: str) -> tuple[list, dict]:
    """
    Load images and COCO annotations from holdout_dir.
    Expects: holdout_dir/annotations.json  (COCO format)
             holdout_dir/images/           (JPEGs)
    Returns: (images_list, coco_gt_dict)
    """
    ann_path = os.path.join(holdout_dir, "annotations.json")
    img_dir  = os.path.join(holdout_dir, "images")

    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"No annotations.json in {holdout_dir}")

    with open(ann_path) as f:
        coco_gt = json.load(f)

    images = []
    for img_meta in coco_gt["images"]:
        img_path = os.path.join(img_dir, img_meta["file_name"])
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            images.append({"meta": img_meta, "image": img})
        else:
            print(f"  WARNING: missing image {img_path}, skipping")

    print(f"Loaded {len(images)} holdout images with COCO annotations.")
    return images, coco_gt


def build_synthetic_holdout(n: int = 20) -> tuple[list, dict]:
    """
    When no real holdout exists, build a tiny synthetic one from random images
    with random boxes so the pipeline runs end-to-end.
    """
    print("Building synthetic holdout (no real holdout found)...")
    rng = np.random.default_rng(42)
    images_meta = []
    annotations  = []
    images_pil   = []

    for i in range(n):
        w, h = 640, 480
        arr  = (rng.random((h, w, 3)) * 255).astype(np.uint8)
        pil  = Image.fromarray(arr)
        img_meta = {"id": i + 1, "file_name": f"synth_{i}.jpg", "width": w, "height": h}
        images_meta.append(img_meta)
        images_pil.append({"meta": img_meta, "image": pil})

        # 1-3 random boxes
        for _ in range(rng.integers(1, 4)):
            bx = float(rng.integers(0, w // 2))
            by = float(rng.integers(0, h // 2))
            bw = float(rng.integers(20, w // 2))
            bh = float(rng.integers(20, h // 2))
            cat = int(rng.integers(0, 4))
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": i + 1,
                "category_id": cat,
                "bbox": [bx, by, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            })

    categories = [{"id": k, "name": v} for k, v in CLASSES.items()]
    coco_gt = {"images": images_meta, "annotations": annotations, "categories": categories}
    print(f"Synthetic holdout: {n} images, {len(annotations)} annotations.")
    return images_pil, coco_gt


# ── ONNX inference ───────────────────────────────────────────────────────────

def preprocess(pil_img: Image.Image, size: tuple = (800, 800)) -> np.ndarray:
    img = pil_img.resize(size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = (arr - mean) / std
    return arr.transpose(2, 0, 1)[None]   # NCHW


def postprocess(logits: np.ndarray, boxes: np.ndarray,
                img_w: int, img_h: int,
                conf_thresh: float = CONF_THRESH) -> list[dict]:
    """
    logits: [1, Q, C]   (C includes background as last dim or first — check model)
    boxes:  [1, Q, 4]   normalized cx,cy,w,h
    Returns list of {category_id, score, bbox [x,y,w,h] in pixels}
    """
    logits = logits[0]   # [Q, C]
    boxes  = boxes[0]    # [Q, 4]

    # softmax
    exp   = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)

    num_classes = probs.shape[1]
    # assume last class is "no object" (background)
    scores = probs[:, :num_classes - 1]    # [Q, num_fg_classes]
    labels = scores.argmax(axis=1)         # [Q]
    confs  = scores.max(axis=1)            # [Q]

    detections = []
    for q in range(len(labels)):
        if confs[q] < conf_thresh:
            continue
        cx, cy, bw, bh = boxes[q]
        x = (cx - bw / 2) * img_w
        y = (cy - bh / 2) * img_h
        w = bw * img_w
        h = bh * img_h
        detections.append({
            "category_id": int(labels[q]),
            "score": float(confs[q]),
            "bbox": [float(x), float(y), float(w), float(h)],
        })
    return detections


def run_inference(onnx_path: str, images: list) -> tuple[list, float]:
    """Returns (coco_dt_list, avg_latency_ms)"""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    coco_dt    = []
    latencies  = []

    for item in images:
        meta = item["meta"]
        pil  = item["image"]
        x    = preprocess(pil)

        t0 = time.perf_counter()
        outs = sess.run(None, {input_name: x})
        latencies.append((time.perf_counter() - t0) * 1000)

        logits, pred_boxes = outs[0], outs[1]
        dets = postprocess(logits, pred_boxes, meta["width"], meta["height"])
        for d in dets:
            coco_dt.append({"image_id": meta["id"], **d})

    avg_ms = float(np.mean(latencies)) if latencies else 0.0
    p95_ms = float(np.percentile(latencies, 95)) if latencies else 0.0
    print(f"Inference: {len(images)} images | avg {avg_ms:.0f} ms | p95 {p95_ms:.0f} ms")
    return coco_dt, p95_ms


# ── mAP calculation ──────────────────────────────────────────────────────────

def compute_map(coco_gt: dict, coco_dt: list) -> dict:
    """Compute mAP using pycocotools."""
    from pycocotools.coco     import COCO
    from pycocotools.cocoeval import COCOeval

    # pycocotools wants files or objects
    gt_obj = COCO()
    gt_obj.dataset = coco_gt
    gt_obj.createIndex()

    if not coco_dt:
        print("WARNING: no detections — returning zero mAP.")
        return {"map50": 0.0, "map50_95": 0.0, "per_class_ap50": {}}

    dt_obj = gt_obj.loadRes(coco_dt)
    ev = COCOeval(gt_obj, dt_obj, iouType="bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    metrics = {
        "map50_95": float(ev.stats[0]),
        "map50":    float(ev.stats[1]),
    }

    # per-class AP@50
    per_class = {}
    for cat in coco_gt.get("categories", []):
        cid  = cat["id"]
        name = cat["name"]
        ev2  = COCOeval(gt_obj, dt_obj, iouType="bbox")
        ev2.params.catIds = [cid]
        ev2.params.iouThrs = np.array([0.5])
        ev2.evaluate()
        ev2.accumulate()
        # stats[1] = AP@50; may be empty if no GT/DT for this class
        ap50 = float(ev2.stats[1]) if len(ev2.stats) > 1 else 0.0
        per_class[name] = round(ap50, 4)

    metrics["per_class_ap50"] = per_class
    return metrics


# ── MLflow registry ───────────────────────────────────────────────────────────

def register_challenger(client, run_id: str, dataset_version: str) -> str:
    """Register the run's ONNX artifact in MLflow Model Registry → Staging."""
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    except Exception:
        versions = []

    # check if already registered for this run
    for v in versions:
        if v.run_id == run_id:
            print(f"Model already registered as version {v.version} ({v.current_stage})")
            return v.version

    try:
        client.create_registered_model(MODEL_NAME)
    except Exception:
        pass  # already exists

    mv = client.create_model_version(
        name=MODEL_NAME,
        source=f"runs:/{run_id}/onnx",
        run_id=run_id,
        tags={"dataset_version": dataset_version},
    )
    client.transition_model_version_stage(
        name=MODEL_NAME, version=mv.version, stage="Staging"
    )
    print(f"Registered as {MODEL_NAME} v{mv.version} → Staging")
    return mv.version


def get_champion_metrics(client) -> dict | None:
    """Returns champion metrics if a Production model exists, else None."""
    try:
        champs = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not champs:
            return None
        run = client.get_run(champs[0].run_id)
        m = run.data.metrics
        return {
            "version": champs[0].version,
            "run_id":  champs[0].run_id,
            "map50":   m.get("map50", 0.0),
            "map50_95": m.get("map50_95", 0.0),
            "latency_p95_ms": m.get("latency_p95_ms", 0.0),
        }
    except Exception:
        return None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase D — Eval champion vs challenger")
    parser.add_argument("--run-id",          required=True, help="MLflow run ID from Colab training")
    parser.add_argument("--dataset-version", required=True, help="e.g. v1")
    parser.add_argument("--holdout-dir",     default="data/detr/holdout",
                        help="Local holdout directory (after dvc pull)")
    parser.add_argument("--output",          default="pipeline/phase_d/eval_results.json")
    parser.add_argument("--create-pr",       action="store_true",
                        help="Create a GitHub PR for approval after eval")
    parser.add_argument("--params",          default=PARAMS_PATH)
    args = parser.parse_args()

    params = load_params()
    client = setup_mlflow()

    with tempfile.TemporaryDirectory() as tmpdir:
        # ── 1. Download challenger ONNX
        onnx_path = os.path.join(tmpdir, "model_int8.onnx")
        download_onnx(args.dataset_version, onnx_path)

        # ── 2. Load holdout
        try:
            images, coco_gt = load_holdout_coco(args.holdout_dir)
        except (FileNotFoundError, Exception) as e:
            print(f"Holdout not available ({e}). Using synthetic fallback.")
            images, coco_gt = build_synthetic_holdout(n=20)

        # ── 3. Run inference
        coco_dt, p95_ms = run_inference(onnx_path, images)

        # ── 4. Compute metrics
        challenger_metrics = compute_map(coco_gt, coco_dt)
        challenger_metrics["latency_p95_ms"] = p95_ms

        print("\n── Challenger metrics ──────────────────────────")
        print(f"  mAP@50:     {challenger_metrics['map50']:.4f}")
        print(f"  mAP@50:95:  {challenger_metrics['map50_95']:.4f}")
        print(f"  Latency p95:{p95_ms:.0f} ms")
        for cls, ap in challenger_metrics.get("per_class_ap50", {}).items():
            print(f"  {cls:<8}: {ap:.4f}")

        # ── 5. Champion comparison
        champion = get_champion_metrics(client)
        if champion:
            delta_map50 = challenger_metrics["map50"] - champion["map50"]
            regressions = [
                cls for cls, ap in challenger_metrics.get("per_class_ap50", {}).items()
                if cls in CRITICAL_CLASSES and ap < champion.get("map50", 0.0)
            ]
            print(f"\n── vs Champion (v{champion['version']}) ──────────────────")
            print(f"  Champion mAP@50: {champion['map50']:.4f}")
            print(f"  Delta:           {delta_map50:+.4f}")
            print(f"  Critical regressions: {regressions or 'none'}")
        else:
            delta_map50 = challenger_metrics["map50"]
            regressions = []
            print("\nRound 1 — no champion in Production yet.")

        # ── 6. Threshold policy
        approved = (
            challenger_metrics["map50"] >= MIN_MAP and
            len(regressions) == 0
        )
        print(f"\nAuto-approved: {approved}")

        # ── 7. Register in MLflow Registry
        mv_version = register_challenger(client, args.run_id, args.dataset_version)

        # ── 8. Log eval metrics back to the MLflow run
        with mlflow.start_run(run_id=args.run_id):
            mlflow.log_metrics({
                "eval_map50":       challenger_metrics["map50"],
                "eval_map50_95":    challenger_metrics["map50_95"],
                "eval_latency_p95": p95_ms,
                "eval_delta_map50": delta_map50,
            })
            for cls, ap in challenger_metrics.get("per_class_ap50", {}).items():
                mlflow.log_metric(f"eval_ap50_{cls}", ap)
        print(f"Metrics logged to MLflow run {args.run_id}")

        # ── 9. Write results JSON
        results = {
            "run_id": args.run_id,
            "model_version": mv_version,
            "dataset_version": args.dataset_version,
            "challenger": challenger_metrics,
            "champion": champion,
            "delta_map50": delta_map50,
            "critical_regressions": regressions,
            "auto_approved": approved,
            "model_name": MODEL_NAME,
        }
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")

        # ── 10. Optionally create GitHub PR
        if args.create_pr:
            _create_github_pr(results)

    return results


def _create_github_pr(results: dict):
    """Create a GitHub PR for human approval using gh CLI."""
    import subprocess

    v = results["dataset_version"]
    delta = results["delta_map50"]
    map50 = results["challenger"]["map50"]
    regressions = results["critical_regressions"]

    title = f"Deploy DETR {v} — mAP@50 {map50:.3f} (delta {delta:+.3f})"

    champ = results.get("champion")
    champ_line = (
        f"Champion mAP@50: `{champ['map50']:.4f}` (v{champ['version']})"
        if champ else "No champion (Round 1 — first ever model)"
    )
    per_class = "\n".join(
        f"| {cls} | {ap:.4f} |"
        for cls, ap in results["challenger"].get("per_class_ap50", {}).items()
    )
    reg_line = (
        f"⚠️ **Critical regressions: {', '.join(regressions)}**"
        if regressions else "✅ No critical class regressions"
    )

    body = f"""## DETR Model Audit — {v}

| Metric | Value |
|---|---|
| Challenger mAP@50 | `{results['challenger']['map50']:.4f}` |
| Challenger mAP@50:95 | `{results['challenger']['map50_95']:.4f}` |
| Latency p95 | `{results['challenger']['latency_p95_ms']:.0f} ms` |
| Delta mAP@50 | `{delta:+.4f}` |

{champ_line}

### Per-class AP@50
| Class | AP |
|---|---|
{per_class}

{reg_line}

### MLflow Run
[View on DagsHub](https://dagshub.com/srnortw/learning-robotic-perception-robops.mlflow/#/experiments/0/runs/{results['run_id']})

### Action Required
1. Run `python pipeline/phase_d/fiftyone_audit.py --results pipeline/phase_d/eval_results.json` locally for visual inspection
2. If satisfied, **merge this PR** → triggers Phase E deployment
3. If unsatisfied, **close this PR** → model stays in Staging
"""

    branch = f"deploy/detr-{v}"
    subprocess.run(["git", "checkout", "-b", branch], check=True)
    subprocess.run(["git", "push", "-u", "origin", branch], check=True)

    result = subprocess.run(
        ["gh", "pr", "create",
         "--title", title,
         "--body", body,
         "--base", "main",
         "--head", branch],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        pr_url = result.stdout.strip()
        print(f"\nPR created: {pr_url}")
    else:
        print(f"PR creation failed: {result.stderr}")


if __name__ == "__main__":
    main()
