"""
Phase B — Export Holdout Set

Extracts the test split from the local Roboflow raw download, remaps
class IDs to match label_schema.yaml, and writes the result to
data/detr/holdout/ in the format eval_champion_challenger.py expects:

    data/detr/holdout/
    ├── annotations.json   (COCO format, schema-aligned IDs)
    └── images/            (flat directory, all test images)

Then tracks the holdout directory with DVC and pushes to S3.

Usage:
    python pipeline/phase_b/export_holdout.py
    python pipeline/phase_b/export_holdout.py --dry-run   # skip DVC
"""

import argparse
import json
import shutil
from pathlib import Path

import yaml

REPO_ROOT      = Path(__file__).parents[2]
SCHEMA_PATH    = REPO_ROOT / "pipeline" / "phase_b" / "label_schema.yaml"
ROBOFLOW_RAW   = REPO_ROOT / "data" / "detr" / "_roboflow_raw"
HOLDOUT_DIR    = REPO_ROOT / "data" / "detr" / "holdout"
IMAGES_DIR     = HOLDOUT_DIR / "images"
ANN_OUT        = HOLDOUT_DIR / "annotations.json"


def load_schema() -> dict[str, int]:
    with open(SCHEMA_PATH) as f:
        schema = yaml.safe_load(f)
    return {c["name"]: c["id"] for c in schema["classes"]}


def find_test_split() -> Path:
    """Find the test/ directory under _roboflow_raw (one level deep)."""
    # Direct: _roboflow_raw/test/
    direct = ROBOFLOW_RAW / "test"
    if (direct / "_annotations.coco.json").exists():
        return direct
    # One sub-directory deep (Roboflow sometimes nests under project name)
    for child in ROBOFLOW_RAW.iterdir():
        if child.is_dir():
            candidate = child / "test"
            if (candidate / "_annotations.coco.json").exists():
                return candidate
    raise FileNotFoundError(
        f"Could not find test/_annotations.coco.json under {ROBOFLOW_RAW}. "
        "Run roboflow_download.py first."
    )


def build_holdout(test_dir: Path, schema: dict[str, int]) -> dict:
    with open(test_dir / "_annotations.coco.json") as f:
        coco = json.load(f)

    # Map old category id → new schema id (drop classes not in schema)
    old_to_new: dict[int, int] = {}
    kept_categories = []
    for cat in coco["categories"]:
        if cat["name"] in schema:
            new_id = schema[cat["name"]]
            old_to_new[cat["id"]] = new_id
            kept_categories.append({"id": new_id, "name": cat["name"], "supercategory": ""})
        else:
            print(f"  SKIP class '{cat['name']}' — not in label_schema.yaml")

    # Filter and remap annotations
    kept_anns: list[dict] = []
    for ann in coco["annotations"]:
        if ann["category_id"] in old_to_new:
            new_ann = dict(ann)
            new_ann["category_id"] = old_to_new[ann["category_id"]]
            kept_anns.append(new_ann)

    # Keep only images that still have annotations
    annotated_ids = {a["image_id"] for a in kept_anns}
    kept_images = [img for img in coco["images"] if img["id"] in annotated_ids]

    # Flatten file_name to just the basename (images go into holdout/images/)
    for img in kept_images:
        img["file_name"] = Path(img["file_name"]).name

    print(f"  Images kept : {len(kept_images)} / {len(coco['images'])}")
    print(f"  Anns kept   : {len(kept_anns)} / {len(coco['annotations'])}")
    print(f"  Classes kept: {[c['name'] for c in kept_categories]}")

    return {
        "images":      kept_images,
        "annotations": kept_anns,
        "categories":  kept_categories,
    }


def copy_images(test_dir: Path, coco: dict):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for img in coco["images"]:
        src = test_dir / img["file_name"]
        dst = IMAGES_DIR / img["file_name"]
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    print(f"  Copied {copied} images → {IMAGES_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Build holdout locally but skip DVC track + push")
    args = parser.parse_args()

    print("[export_holdout] Loading schema...")
    schema = load_schema()

    print("[export_holdout] Locating test split...")
    test_dir = find_test_split()
    print(f"  Found: {test_dir}")

    print("[export_holdout] Building COCO holdout...")
    coco = build_holdout(test_dir, schema)

    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    copy_images(test_dir, coco)

    with open(ANN_OUT, "w") as f:
        json.dump(coco, f)
    print(f"  Saved annotations → {ANN_OUT}")

    if args.dry_run:
        print("[export_holdout] DRY RUN — skipping DVC steps.")
        return

    print("[export_holdout] Tracking with DVC...")
    import subprocess, sys

    def run(cmd: list[str]):
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            print(f"ERROR: {' '.join(cmd)} failed")
            sys.exit(result.returncode)

    run(["dvc", "add", "data/detr/holdout"])
    run(["dvc", "push", "data/detr/holdout.dvc"])

    print("[export_holdout] Committing .dvc pointer file...")
    run(["git", "add", "data/detr/holdout.dvc", ".gitignore"])
    run(["git", "commit", "-m", "data: track holdout set with DVC (test split, v2 construction safety)"])
    run(["git", "push"])

    print("[export_holdout] Done. Holdout is in S3 and tracked by DVC.")


if __name__ == "__main__":
    main()
