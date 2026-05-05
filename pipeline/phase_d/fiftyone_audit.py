"""
Phase D — FiftyOne Visual Audit

Loads holdout images with challenger (and optionally champion) predictions,
sorts by ascending IoU (worst cases first), and launches the FiftyOne browser
for human review.

Run locally after `dvc pull data/detr/holdout.dvc`:

    python pipeline/phase_d/fiftyone_audit.py \\
        --results pipeline/phase_d/eval_results.json \\
        [--holdout-dir data/detr/holdout] \\
        [--onnx-int8 /tmp/model_int8.onnx]
"""

import argparse
import io
import json
import os
import tempfile
import time

import boto3
import numpy as np
from PIL import Image

S3_BUCKET  = "my-perception-robops-data-2026-688567275774-eu-central-1-an"
AWS_REGION = "eu-central-1"
CLASSES    = {0: "person", 1: "chair", 2: "table", 3: "door"}
CONF_THRESH = 0.3


# ── ONNX inference (same helpers as eval script) ─────────────────────────────

def preprocess(pil_img: Image.Image, size: tuple = (800, 800)) -> np.ndarray:
    img = pil_img.resize(size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = (arr - mean) / std
    return arr.transpose(2, 0, 1)[None]


def postprocess(logits, boxes, img_w, img_h):
    logits, boxes = logits[0], boxes[0]
    exp   = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    num_classes = probs.shape[1]
    scores = probs[:, :num_classes - 1]
    labels = scores.argmax(axis=1)
    confs  = scores.max(axis=1)
    dets   = []
    for q in range(len(labels)):
        if confs[q] < CONF_THRESH:
            continue
        cx, cy, bw, bh = boxes[q]
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h
        dets.append({
            "label": CLASSES.get(int(labels[q]), f"class_{labels[q]}"),
            "confidence": float(confs[q]),
            "bounding_box": [x1 / img_w, y1 / img_h,
                             (x2 - x1) / img_w, (y2 - y1) / img_h],
        })
    return dets


def run_model(sess, pil_img: Image.Image) -> list[dict]:
    input_name = sess.get_inputs()[0].name
    x = preprocess(pil_img)
    outs = sess.run(None, {input_name: x})
    return postprocess(outs[0], outs[1], pil_img.width, pil_img.height)


# ── holdout loading ──────────────────────────────────────────────────────────

def load_holdout(holdout_dir: str) -> tuple[list, dict | None]:
    """Returns (list_of_pil_paths, coco_gt_or_None)."""
    ann_path = os.path.join(holdout_dir, "annotations.json")
    img_dir  = os.path.join(holdout_dir, "images")

    if not os.path.isdir(holdout_dir):
        raise FileNotFoundError(f"Holdout dir not found: {holdout_dir}")

    coco_gt = None
    if os.path.exists(ann_path):
        with open(ann_path) as f:
            coco_gt = json.load(f)

    img_paths = sorted([
        os.path.join(img_dir, fn)
        for fn in os.listdir(img_dir)
        if fn.lower().endswith((".jpg", ".jpeg", ".png"))
    ]) if os.path.isdir(img_dir) else []

    return img_paths, coco_gt


def build_synthetic_images(n: int = 10) -> list[tuple[str, Image.Image]]:
    """Creates temp PNG files for FiftyOne when no real holdout exists."""
    import tempfile
    rng   = np.random.default_rng(7)
    items = []
    for i in range(n):
        arr  = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
        pil  = Image.fromarray(arr)
        path = os.path.join(tempfile.gettempdir(), f"synth_audit_{i}.jpg")
        pil.save(path)
        items.append((path, pil))
    return items


# ── FiftyOne dataset builder ─────────────────────────────────────────────────

def build_fo_dataset(img_paths: list[str], pil_imgs: list[Image.Image],
                     challenger_preds: list[list[dict]],
                     coco_gt: dict | None) -> "fo.Dataset":
    import fiftyone as fo

    dataset_name = "detr_phase_d_audit"
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(dataset_name)
    samples = []

    # Build GT lookup by filename if available
    gt_by_file: dict[str, list] = {}
    if coco_gt:
        id_to_file = {img["id"]: img["file_name"] for img in coco_gt["images"]}
        for ann in coco_gt.get("annotations", []):
            fname = id_to_file.get(ann["image_id"], "")
            gt_by_file.setdefault(fname, []).append(ann)

    for path, pil, preds in zip(img_paths, pil_imgs, challenger_preds):
        sample = fo.Sample(filepath=path)

        # Challenger predictions
        det_list = []
        for d in preds:
            x, y, w, h = d["bounding_box"]
            det_list.append(fo.Detection(
                label=d["label"],
                bounding_box=[x, y, w, h],
                confidence=d["confidence"],
            ))
        sample["challenger"] = fo.Detections(detections=det_list)

        # Ground truth (if available)
        fname = os.path.basename(path)
        if fname in gt_by_file:
            gt_dets = []
            for ann in gt_by_file[fname]:
                bx, by, bw, bh = ann["bbox"]
                gt_dets.append(fo.Detection(
                    label=CLASSES.get(ann["category_id"], f"class_{ann['category_id']}"),
                    bounding_box=[bx / pil.width, by / pil.height,
                                  bw / pil.width, bh / pil.height],
                ))
            sample["ground_truth"] = fo.Detections(detections=gt_dets)

        samples.append(sample)

    dataset.add_samples(samples)

    # Evaluate and sort by worst IoU
    if coco_gt and all("ground_truth" in s.field_names for s in dataset):
        try:
            results = dataset.evaluate_detections(
                "challenger",
                gt_field="ground_truth",
                eval_key="eval",
                iou=0.5,
            )
        except Exception as e:
            print(f"WARNING: evaluation failed ({e}), skipping sort.")

    return dataset


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase D — FiftyOne visual audit")
    parser.add_argument("--results",     default="pipeline/phase_d/eval_results.json")
    parser.add_argument("--holdout-dir", default="data/detr/holdout")
    parser.add_argument("--onnx-int8",   default=None,
                        help="Path to model_int8.onnx (downloads from S3 if not set)")
    args = parser.parse_args()

    # Load eval results for context
    results = {}
    if os.path.exists(args.results):
        with open(args.results) as f:
            results = json.load(f)
        print(f"Eval results loaded: mAP@50 = {results.get('challenger', {}).get('map50', 'N/A')}")

    # Resolve ONNX path
    onnx_path = args.onnx_int8
    _tmp = None
    if not onnx_path or not os.path.exists(onnx_path):
        _tmp = tempfile.mkdtemp()
        onnx_path = os.path.join(_tmp, "model_int8.onnx")
        dv = results.get("dataset_version", "v1")
        print(f"Downloading model from S3 (version={dv})...")
        s3 = boto3.client("s3", region_name=AWS_REGION)
        s3.download_file(S3_BUCKET, f"weights/detr/{dv}/model_int8.onnx", onnx_path)

    # Load ONNX session
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print("ONNX model loaded.")

    # Load holdout
    synthetic = False
    try:
        img_paths, coco_gt = load_holdout(args.holdout_dir)
        if not img_paths:
            raise FileNotFoundError("No images found in holdout dir")
        pil_imgs = [Image.open(p).convert("RGB") for p in img_paths]
        print(f"Holdout: {len(img_paths)} images")
    except FileNotFoundError as e:
        print(f"Holdout not found ({e}). Using synthetic images.")
        items    = build_synthetic_images(10)
        img_paths = [i[0] for i in items]
        pil_imgs  = [i[1] for i in items]
        coco_gt   = None
        synthetic = True

    # Run inference
    print("Running challenger inference...")
    challenger_preds = []
    for pil in pil_imgs:
        dets = run_model(sess, pil)
        challenger_preds.append(dets)
    print(f"Total detections: {sum(len(d) for d in challenger_preds)}")

    # Build FiftyOne dataset
    print("Building FiftyOne dataset...")
    dataset = build_fo_dataset(img_paths, pil_imgs, challenger_preds, coco_gt)

    # Sort by worst IoU (if eval was run)
    view = dataset
    try:
        view = dataset.sort_by("eval_tp", reverse=False)
    except Exception:
        pass

    print(f"\nDataset '{dataset.name}' ready with {len(dataset)} samples.")
    if synthetic:
        print("NOTE: Using synthetic images — run `dvc pull data/detr/holdout.dvc` for real data.")

    print("\nLaunching FiftyOne App... (Ctrl+C to exit)")
    print("Look for:")
    print("  - Low-confidence detections")
    print("  - Missed ground-truth objects")
    print("  - Wrong class labels")
    print("  - Critical class failures (person, door)\n")

    import fiftyone as fo
    session = fo.launch_app(view, remote=False)
    session.wait()

    if _tmp:
        import shutil
        shutil.rmtree(_tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
