"""
Phase B — Sub-pipe 4b: COCO JSON + images → MDS shards → S3.

Reads annotated COCO JSON from data/detr/raw/annotations.json, converts
each image+annotation pair to MDS shards split by train/val, and writes
directly to S3 at:
    s3://bucket/mds/detr/vN/train/
    s3://bucket/mds/detr/vN/val/

Split detection:
  - Roboflow downloads: file_name is prefixed with "train/", "valid/", "test/"
    → written to train/ and val/ automatically.
  - Single-file COCO (CVAT export): all images → train/ (no val split).

Usage:
    python convert_to_mds.py --version v2 --coco-json data/detr/raw/annotations.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parents[2]
PARAMS_PATH = REPO_ROOT / "pipeline" / "phase_c" / "detr" / "params.yaml"
BUCKET = os.environ.get("ROBOPS_S3_BUCKET", "my-perception-robops-data-2026-688567275774-eu-central-1-an")

# "valid" from Roboflow maps to "val" in MDS path (matches train.py expectation)
SPLIT_MAP = {"train": "train", "valid": "val", "test": "val"}

# Allow running as a standalone script without the package installed
sys.path.insert(0, str(REPO_ROOT))
from pipeline.utils.schema import Schema


def load_coco(coco_path: Path) -> dict:
    with open(coco_path) as f:
        return json.load(f)


def build_image_annotation_map(coco: dict) -> dict[int, list]:
    ann_map: dict[int, list] = {img["id"]: [] for img in coco["images"]}
    for ann in coco.get("annotations", []):
        ann_map[ann["image_id"]].append(ann)
    return ann_map


def detect_splits(coco: dict) -> dict[str, list]:
    """
    Group images by split.
    If file_name starts with a known split prefix (train/, valid/, test/),
    use that. Otherwise all images go to 'train'.
    Returns {"train": [...], "val": [...]}
    """
    split_images: dict[str, list] = defaultdict(list)
    for img in coco["images"]:
        fname = img["file_name"]
        prefix = fname.split("/")[0] if "/" in fname else ""
        mds_split = SPLIT_MAP.get(prefix, "train")
        split_images[mds_split].append(img)
    return dict(split_images)


def write_split(images: list, ann_map: dict, image_dir: Path,
                out_path: str, split_name: str, MDSWriter, PILImage):
    """Write one MDS split (train or val) to S3."""
    columns = {"image": "jpeg", "annotations": "json", "image_id": "int"}
    written = skipped = 0

    with MDSWriter(out=out_path, columns=columns) as writer:
        for img_info in images:
            img_path = image_dir / img_info["file_name"]
            if not img_path.exists():
                print(f"[convert_to_mds] WARN: Missing {img_path} — skipping")
                skipped += 1
                continue

            img = PILImage.open(img_path).convert("RGB")
            anns = ann_map.get(img_info["id"], [])
            writer.write({"image": img, "annotations": anns, "image_id": img_info["id"]})
            written += 1

    print(f"[convert_to_mds]   {split_name}: {written} written, {skipped} skipped → {out_path}")
    return written


def convert(coco_path: Path, version: str, dry_run: bool):
    try:
        from streaming import MDSWriter
        from PIL import Image as PILImage
    except ImportError:
        print("[convert_to_mds] ERROR: Install mosaicml-streaming and Pillow first.")
        sys.exit(1)

    coco = load_coco(coco_path)
    ann_map = build_image_annotation_map(coco)
    image_dir = coco_path.parent
    splits = detect_splits(coco)

    base_path = f"s3://{BUCKET}/mds/detr/{version}"
    print(f"[convert_to_mds] Total images: {len(coco['images'])}")
    for split, imgs in splits.items():
        print(f"[convert_to_mds]   {split}: {len(imgs)} images")

    if dry_run:
        print("[convert_to_mds] DRY RUN — no data written.")
        return

    for split, images in splits.items():
        out_path = f"{base_path}/{split}/"
        write_split(images, ann_map, image_dir, out_path, split, MDSWriter, PILImage)

    print(f"[convert_to_mds] MDS shards written to {base_path}/")


def update_params_yaml(version: str):
    with open(PARAMS_PATH) as f:
        params = yaml.safe_load(f)

    params["dataset"]["mds_path"] = f"s3://{BUCKET}/mds/detr/{version}/"
    params["dataset"]["dataset_version"] = version

    with open(PARAMS_PATH, "w") as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)

    print(f"[convert_to_mds] params.yaml updated: mds_path → s3://{BUCKET}/mds/detr/{version}/")

    # Sync num_classes + detr_params.yaml class_names from label_schema.yaml
    Schema.sync()

    print(f"[convert_to_mds] Now run:")
    print(f"    git add pipeline/phase_c/detr/params.yaml ros2_ws/src/detr_node/config/detr_params.yaml")
    print(f"    git commit -m 'data: bump mds_path to detr {version} — triggers training CI'")
    print(f"    git push")


def main():
    parser = argparse.ArgumentParser(description="Phase B MDS conversion")
    parser.add_argument("--version", required=True, help="Dataset version e.g. v1")
    parser.add_argument(
        "--coco-json",
        default=str(REPO_ROOT / "data" / "detr" / "raw" / "annotations.json"),
        help="Path to COCO JSON annotation file",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    coco_path = Path(args.coco_json)
    if not coco_path.exists():
        print(f"[convert_to_mds] ERROR: {coco_path} not found. Export COCO JSON from CVAT first.")
        sys.exit(1)

    convert(coco_path, args.version, args.dry_run)

    if not args.dry_run:
        update_params_yaml(args.version)


if __name__ == "__main__":
    main()
