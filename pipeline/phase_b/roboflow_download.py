"""
Phase B — Roboflow Dataset Download + Class Remapping

Downloads a dataset from Roboflow Universe in COCO JSON format,
remaps class names to match label_schema.yaml, and saves the result
to data/detr/raw/ ready to be picked up by convert_to_mds.py.

Why class remapping?
  Different Roboflow datasets use different names for the same concept:
  "dining table" vs "table", "Persons" vs "person", etc.
  The --remap flag lets you bridge any dataset to this project's schema.

Usage — minimal (API key only, project must have classes matching schema):
    python pipeline/phase_b/roboflow_download.py \\
        --api-key YOUR_KEY \\
        --workspace YOUR_WORKSPACE \\
        --project  YOUR_PROJECT \\
        --version  1

Usage — with class remapping (e.g. COCO "dining table" → "table"):
    python pipeline/phase_b/roboflow_download.py \\
        --api-key YOUR_KEY \\
        --workspace YOUR_WORKSPACE \\
        --project  YOUR_PROJECT \\
        --version  1 \\
        --remap "dining table:table" "Persons:person"

Usage — dry run (checks connectivity + class mapping, no file writes):
    python pipeline/phase_b/roboflow_download.py ... --dry-run

After this script:
    python pipeline/phase_b/convert_to_mds.py \\
        --version v2 \\
        --coco-json data/detr/raw/annotations.json

Recommended datasets (see docs/DATASETS.md for details):
    - COCO 2017 subset  (workspace=roboflow-universe-projects, project=coco-2017,   version=1)
    - Indoor detection  (workspace=master-thesis-lzxhq,        project=indoor-2024, version=1)
    - Custom fork from Roboflow Universe (fork + add missing classes in Roboflow UI)
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT   = Path(__file__).parents[2]
SCHEMA_PATH = REPO_ROOT / "pipeline" / "phase_b" / "label_schema.yaml"
OUTPUT_DIR  = REPO_ROOT / "data" / "detr" / "raw"


def load_schema() -> dict[str, int]:
    """Return {class_name: id} from label_schema.yaml."""
    with open(SCHEMA_PATH) as f:
        schema = yaml.safe_load(f)
    return {cls["name"]: cls["id"] for cls in schema["classes"]}


def build_remap(remap_args: list[str]) -> dict[str, str]:
    """
    Parse --remap "dining table:table" "Persons:person" into
    {"dining table": "table", "Persons": "person"}.
    """
    result: dict[str, str] = {}
    for entry in remap_args or []:
        parts = entry.split(":", 1)
        if len(parts) != 2:
            print(f"[roboflow_download] WARN: Ignoring malformed --remap '{entry}' (use 'original:mapped')")
            continue
        result[parts[0].strip()] = parts[1].strip()
    return result


def download_dataset(api_key: str, workspace: str, project: str, version: int) -> Path:
    """Download COCO format dataset from Roboflow, return the local dataset path."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[roboflow_download] ERROR: roboflow not installed. Run: pip install roboflow")
        sys.exit(1)

    print(f"[roboflow_download] Connecting to Roboflow...")
    rf  = Roboflow(api_key=api_key)
    prj = rf.workspace(workspace).project(project)
    ver = prj.version(version)

    print(f"[roboflow_download] Downloading '{workspace}/{project}' v{version} in COCO format...")
    dataset = ver.download("coco", location=str(REPO_ROOT / "data" / "detr" / "_roboflow_raw"))
    return Path(dataset.location)


def merge_splits(dataset_path: Path) -> tuple[dict, dict]:
    """
    Roboflow downloads train/valid/test splits as separate COCO JSONs.
    Merge them into one combined dict and keep a per-split map for reporting.
    Returns (merged_coco, split_stats).
    """
    splits      = ["train", "valid", "test"]
    merged      = {"images": [], "annotations": [], "categories": []}
    split_stats = {}
    img_id_offset = 0
    ann_id_offset = 0
    categories_set = False

    for split in splits:
        ann_file = dataset_path / split / "_annotations.coco.json"
        if not ann_file.exists():
            continue

        with open(ann_file) as f:
            coco = json.load(f)

        if not categories_set:
            merged["categories"] = coco.get("categories", [])
            categories_set = True

        old_to_new_img: dict[int, int] = {}
        for img in coco.get("images", []):
            new_id = img["id"] + img_id_offset
            old_to_new_img[img["id"]] = new_id
            new_img = dict(img)
            new_img["id"] = new_id
            # Store the split sub-path so convert_to_mds can find the file
            new_img["file_name"] = f"{split}/{img['file_name']}"
            merged["images"].append(new_img)

        for ann in coco.get("annotations", []):
            new_ann = dict(ann)
            new_ann["id"]       = ann["id"] + ann_id_offset
            new_ann["image_id"] = old_to_new_img[ann["image_id"]]
            merged["annotations"].append(new_ann)

        n_imgs = len(coco.get("images", []))
        n_anns = len(coco.get("annotations", []))
        split_stats[split] = {"images": n_imgs, "annotations": n_anns}
        img_id_offset += max((img["id"] for img in coco.get("images", [])), default=0) + 1
        ann_id_offset += max((ann["id"] for ann in coco.get("annotations", [])), default=0) + 1

    return merged, split_stats


def remap_and_filter(
    coco: dict,
    schema: dict[str, int],
    remap: dict[str, str],
) -> tuple[dict, dict]:
    """
    1. Apply name remapping (e.g. "dining table" → "table").
    2. Remove categories not in label_schema.yaml.
    3. Reassign category IDs to match schema IDs.
    4. Drop annotations whose category was removed.
    Returns (filtered_coco, stats).
    """
    # Build old_cat_id → new_cat_id map
    old_id_to_new: dict[int, int] = {}
    kept_categories = []

    for cat in coco.get("categories", []):
        original_name = cat["name"]
        mapped_name   = remap.get(original_name, original_name)

        if mapped_name in schema:
            new_id = schema[mapped_name]
            old_id_to_new[cat["id"]] = new_id
            kept_categories.append({"id": new_id, "name": mapped_name, "supercategory": cat.get("supercategory", "")})
        else:
            print(f"[roboflow_download] SKIP class '{original_name}' — not in label_schema.yaml")

    # Deduplicate categories (same new_id could come from multiple remaps)
    seen_ids: set[int] = set()
    unique_cats = []
    for cat in kept_categories:
        if cat["id"] not in seen_ids:
            unique_cats.append(cat)
            seen_ids.add(cat["id"])

    # Filter annotations
    kept_anns   = []
    dropped_ann = 0
    for ann in coco.get("annotations", []):
        if ann["category_id"] in old_id_to_new:
            new_ann = dict(ann)
            new_ann["category_id"] = old_id_to_new[ann["category_id"]]
            kept_anns.append(new_ann)
        else:
            dropped_ann += 1

    # Remove images that now have zero annotations (optional but cleaner)
    annotated_img_ids = {ann["image_id"] for ann in kept_anns}
    kept_images = [img for img in coco.get("images", []) if img["id"] in annotated_img_ids]
    dropped_imgs = len(coco.get("images", [])) - len(kept_images)

    stats = {
        "original_categories": len(coco.get("categories", [])),
        "kept_categories":     len(unique_cats),
        "original_annotations": len(coco.get("annotations", [])),
        "kept_annotations":     len(kept_anns),
        "dropped_annotations":  dropped_ann,
        "original_images":      len(coco.get("images", [])),
        "kept_images":          len(kept_images),
        "dropped_images":       dropped_imgs,
    }

    filtered = {
        "images":      kept_images,
        "annotations": kept_anns,
        "categories":  unique_cats,
    }
    return filtered, stats


def save_output(coco: dict, dataset_path: Path):
    """Write merged + filtered COCO JSON to data/detr/raw/annotations.json."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy images into OUTPUT_DIR preserving split sub-dirs
    images_copied = 0
    for img in coco["images"]:
        src = dataset_path / img["file_name"]
        dst = OUTPUT_DIR / img["file_name"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            images_copied += 1

    out_json = OUTPUT_DIR / "annotations.json"
    with open(out_json, "w") as f:
        json.dump(coco, f)

    print(f"[roboflow_download] Saved annotations → {out_json}")
    print(f"[roboflow_download] Copied {images_copied} images  → {OUTPUT_DIR}")


def print_report(split_stats: dict, remap_stats: dict, schema: dict[str, int]):
    print("\n── Download Report ───────────────────────────────────")
    for split, s in split_stats.items():
        print(f"  {split:6s}: {s['images']:4d} images, {s['annotations']:5d} annotations")

    print(f"\n  Categories kept : {remap_stats['kept_categories']} / {remap_stats['original_categories']}")
    print(f"  Annotations kept: {remap_stats['kept_annotations']} / {remap_stats['original_annotations']}"
          f"  (dropped {remap_stats['dropped_annotations']})")
    print(f"  Images kept     : {remap_stats['kept_images']} / {remap_stats['original_images']}"
          f"  (dropped {remap_stats['dropped_images']} — no annotations after filter)")

    print(f"\n  Final class mapping:")
    for name, cid in sorted(schema.items(), key=lambda x: x[1]):
        print(f"    id={cid}  {name}")

    print("\n  Next step:")
    print("    python pipeline/phase_b/convert_to_mds.py \\")
    print("        --version v2 \\")
    print("        --coco-json data/detr/raw/annotations.json")
    print("─────────────────────────────────────────────────────\n")


def main():
    parser = argparse.ArgumentParser(description="Phase B — Roboflow dataset download + class remap")
    parser.add_argument("--api-key",   required=True,  help="Roboflow API key")
    parser.add_argument("--workspace", required=True,  help="Roboflow workspace slug")
    parser.add_argument("--project",   required=True,  help="Roboflow project slug")
    parser.add_argument("--version",   required=True,  type=int, help="Roboflow dataset version number")
    parser.add_argument(
        "--remap",
        nargs="*",
        default=[],
        metavar="ORIG:MAPPED",
        help=(
            "Rename dataset classes to match label_schema.yaml. "
            "Example: --remap 'dining table:table' 'Persons:person'"
        ),
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Check class mapping only — do not write any files")
    args = parser.parse_args()

    schema = load_schema()
    remap  = build_remap(args.remap)

    print(f"[roboflow_download] Schema classes: {list(schema.keys())}")
    if remap:
        print(f"[roboflow_download] Active remaps: {remap}")

    dataset_path = download_dataset(args.api_key, args.workspace, args.project, args.version)
    print(f"[roboflow_download] Downloaded to: {dataset_path}")

    print("[roboflow_download] Merging train/valid/test splits...")
    merged_coco, split_stats = merge_splits(dataset_path)

    print("[roboflow_download] Applying class filter + remap...")
    filtered_coco, remap_stats = remap_and_filter(merged_coco, schema, remap)

    print_report(split_stats, remap_stats, schema)

    if args.dry_run:
        print("[roboflow_download] DRY RUN — no files written.")
        return

    if remap_stats["kept_images"] == 0:
        print("[roboflow_download] ERROR: 0 images remain after filtering. "
              "Check class names and use --remap to bridge mismatches.")
        sys.exit(1)

    save_output(filtered_coco, dataset_path)
    print("[roboflow_download] Done.")


if __name__ == "__main__":
    main()
