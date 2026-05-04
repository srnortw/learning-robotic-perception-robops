"""
Phase B — Sub-pipe 4a: DVC versioning.

After CVAT annotation is done and COCO JSON is placed in data/detr/raw/:
  1. dvc add data/detr/raw/          → hash images, update data.dvc
  2. dvc push                        → sync to S3 DVC cache
  3. git add + git commit data.dvc   → version the pointer

Usage:
    python publish_dataset.py --version v1 --num-images 420 --classes "person,chair,table,door"
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
DATA_DIR = REPO_ROOT / "data" / "detr" / "raw"
DVC_FILE = REPO_ROOT / "data" / "detr" / "data.dvc"


def run(cmd: list[str], cwd: Path = REPO_ROOT) -> int:
    print(f"[publish_dataset] $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        print(f"[publish_dataset] ERROR: command failed with exit code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Phase B DVC versioning")
    parser.add_argument("--version", required=True, help="Dataset version, e.g. v1")
    parser.add_argument("--num-images", type=int, required=True)
    parser.add_argument("--classes", required=True, help="Comma-separated class names")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not DATA_DIR.exists():
        print(f"[publish_dataset] ERROR: {DATA_DIR} not found. Pull images first.")
        sys.exit(1)

    if args.dry_run:
        print("[publish_dataset] DRY RUN — no changes will be made.")

    steps = [
        (["dvc", "add", str(DATA_DIR.relative_to(REPO_ROOT))], "DVC tracking images"),
        (["dvc", "push"], "Pushing to S3 DVC cache"),
        (["git", "add", str(DVC_FILE.relative_to(REPO_ROOT)), "data/detr/.gitignore"], "Staging data.dvc"),
        ([
            "git", "commit", "-m",
            f"data: detr dataset {args.version} — {args.num_images} images, classes: {args.classes}"
        ], "Git commit data.dvc"),
    ]

    for cmd, description in steps:
        print(f"\n[publish_dataset] Step: {description}")
        if not args.dry_run:
            code = run(cmd)
            if code != 0:
                print(f"[publish_dataset] Stopping due to error in: {description}")
                sys.exit(code)

    print(f"\n[publish_dataset] Done. Dataset {args.version} versioned and pushed.")
    print(f"[publish_dataset] Next: run convert_to_mds.py --version {args.version}")


if __name__ == "__main__":
    main()
