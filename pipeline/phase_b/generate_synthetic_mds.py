"""
Synthetic MDS data generator — Phase B smoke test utility.

Creates fake images with random bounding box annotations in COCO format,
converts them to MDS shards, and uploads to S3. Use this to validate the
full Phase C training pipeline before real data from Phase A is available.

Usage:
    python generate_synthetic_mds.py --num-images 100 --version v1
"""

import argparse
import json
import os
import random
import tempfile
from pathlib import Path

import boto3
import numpy as np
import yaml
from PIL import Image as PILImage

S3_BUCKET = "my-perception-robops-data-2026-688567275774-eu-central-1-an"
AWS_REGION = "eu-central-1"
PARAMS_PATH = Path(__file__).parents[2] / "pipeline" / "phase_c" / "detr" / "params.yaml"

# Match label_schema.yaml
CLASSES = {0: "person", 1: "chair", 2: "table", 3: "door"}


def load_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def make_synthetic_image(width: int = 640, height: int = 480) -> PILImage.Image:
    """Random noise image — enough to test the data pipeline."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return PILImage.fromarray(arr)


def make_synthetic_annotations(image_id: int, width: int = 640, height: int = 480) -> list[dict]:
    """Random bounding boxes in COCO format."""
    num_boxes = random.randint(1, 4)
    annotations = []
    for ann_id in range(num_boxes):
        x = random.randint(0, width - 50)
        y = random.randint(0, height - 50)
        w = random.randint(30, min(150, width - x))
        h = random.randint(30, min(150, height - y))
        annotations.append({
            "id": image_id * 10 + ann_id,
            "image_id": image_id,
            "category_id": random.choice(list(CLASSES.keys())),
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0,
        })
    return annotations


def build_coco_json(num_images: int) -> tuple[dict, list[PILImage.Image]]:
    images_meta = []
    all_annotations = []
    pil_images = []

    for i in range(num_images):
        img = make_synthetic_image()
        pil_images.append(img)
        images_meta.append({"id": i, "file_name": f"frame_{i:06d}.jpg", "width": 640, "height": 480})
        all_annotations.extend(make_synthetic_annotations(i))

    coco = {
        "images": images_meta,
        "annotations": all_annotations,
        "categories": [{"id": k, "name": v} for k, v in CLASSES.items()],
    }
    return coco, pil_images


def write_mds_shards(coco: dict, pil_images: list, out_path: str, split: str):
    try:
        from streaming import MDSWriter
    except ImportError:
        print("ERROR: mosaicml-streaming not installed. Run: pip install mosaicml-streaming")
        raise

    import io
    split_path = f"{out_path}{split}/"
    columns = {"image": "bytes", "annotations": "json", "image_id": "int"}

    print(f"  Writing {len(pil_images)} samples → {split_path}")
    with MDSWriter(out=split_path, columns=columns) as writer:
        for img_info, pil_img in zip(coco["images"], pil_images):
            img_id = img_info["id"]
            anns = [a for a in coco["annotations"] if a["image_id"] == img_id]

            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            writer.write({
                "image": buf.getvalue(),
                "annotations": anns,
                "image_id": img_id,
            })


def update_params_yaml(version: str):
    with open(PARAMS_PATH) as f:
        params = yaml.safe_load(f)
    params["dataset"]["mds_path"] = f"s3://{S3_BUCKET}/mds/detr/{version}/"
    params["dataset"]["dataset_version"] = version
    with open(PARAMS_PATH, "w") as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)
    print(f"  params.yaml updated: mds_path → s3://{S3_BUCKET}/mds/detr/{version}/")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic MDS data for pipeline smoke test")
    parser.add_argument("--num-images", type=int, default=100, help="Total images (split 85/15 train/val)")
    parser.add_argument("--version", default="v1", help="Dataset version e.g. v1")
    args = parser.parse_args()

    params = load_params()
    train_split = params["dataset"]["train_split"]

    num_train = int(args.num_images * train_split)
    num_val = args.num_images - num_train

    out_path = f"s3://{S3_BUCKET}/mds/detr/{args.version}/"
    print(f"Generating synthetic MDS dataset ({args.version})")
    print(f"  Train: {num_train} | Val: {num_val}")
    print(f"  Output: {out_path}")

    print("\nBuilding train split...")
    train_coco, train_imgs = build_coco_json(num_train)
    write_mds_shards(train_coco, train_imgs, out_path, "train")

    print("Building val split...")
    val_coco, val_imgs = build_coco_json(num_val)
    write_mds_shards(val_coco, val_imgs, out_path, "val")

    print("\nUpdating params.yaml...")
    update_params_yaml(args.version)

    print(f"\nDone! Synthetic MDS dataset ready at {out_path}")
    print("Now commit params.yaml and run the Colab notebook to test the full pipeline.")
    print(f"\n  git add pipeline/phase_c/detr/params.yaml")
    print(f"  git commit -m 'data: synthetic mds {args.version} smoke test'")
    print(f"  git push")


if __name__ == "__main__":
    main()
