"""
Phase B — Sub-pipe 4b: COCO JSON + images → MDS shards → S3.

Reads annotated COCO JSON from data/detr/raw/annotations.json, converts
each image+annotation pair to an MDS shard, and writes directly to S3.
After success, updates params.yaml with the new mds_path version.

Usage:
    python convert_to_mds.py --version v1 --coco-json data/detr/raw/annotations.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parents[2]
PARAMS_PATH = REPO_ROOT / "pipeline" / "phase_c" / "detr" / "params.yaml"
BUCKET = os.environ.get("ROBOPS_S3_BUCKET", "my-perception-robops-data-2026-688567275774-eu-central-1-an")

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

    out_path = f"s3://{BUCKET}/mds/detr/{version}/"
    print(f"[convert_to_mds] Writing MDS shards → {out_path}")
    print(f"[convert_to_mds] Total images: {len(coco['images'])}")

    if dry_run:
        print("[convert_to_mds] DRY RUN — no data written.")
        return

    columns = {"image": "jpeg", "annotations": "json", "image_id": "int"}

    with MDSWriter(out=out_path, columns=columns) as writer:
        for img_info in coco["images"]:
            img_path = image_dir / img_info["file_name"]
            if not img_path.exists():
                print(f"[convert_to_mds] WARN: Missing image {img_path} — skipping")
                continue

            img = PILImage.open(img_path).convert("RGB")
            anns = ann_map.get(img_info["id"], [])

            writer.write({
                "image": img,
                "annotations": anns,
                "image_id": img_info["id"],
            })

    print(f"[convert_to_mds] MDS shards written to {out_path}")


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
