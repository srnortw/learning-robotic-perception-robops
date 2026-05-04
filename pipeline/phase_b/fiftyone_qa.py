"""
Phase B — Sub-pipe 2: FiftyOne visual QA + CVAT annotation queue.

Loads locally pulled images with MongoDB confidence metadata into a FiftyOne
dataset. Sorts by ascending confidence (lowest = most informative = label first).
Flags bad frames for exclusion. Exports the CVAT-ready queue as a JSON manifest.

Usage:
    python fiftyone_qa.py --architecture detr
    # then open http://localhost:5151 in browser to inspect
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pymongo

RAW_DATA_DIR = Path(__file__).parents[2] / "data" / "detr" / "raw"
CVAT_QUEUE_PATH = Path(__file__).parents[2] / "data" / "detr" / "cvat_queue.json"
MONGO_URI = os.environ.get("MONGO_URI", "")


def load_metadata_from_mongo(architecture: str) -> dict[str, dict]:
    """Returns dict keyed by filename → metadata doc."""
    if not MONGO_URI:
        print("[fiftyone_qa] WARN: MONGO_URI not set — loading images without metadata.")
        return {}
    client = pymongo.MongoClient(MONGO_URI)
    db = client["robops"]
    docs = list(db.telemetry.find({"architecture": architecture, "s3_url": {"$ne": None}}))
    client.close()
    # Key by the last part of s3_url (filename)
    return {doc["s3_url"].split("/")[-1]: doc for doc in docs if doc.get("s3_url")}


def build_fiftyone_dataset(architecture: str):
    try:
        import fiftyone as fo
    except ImportError:
        print("[fiftyone_qa] ERROR: fiftyone not installed. Run: pip install fiftyone")
        sys.exit(1)

    metadata = load_metadata_from_mongo(architecture)
    image_paths = sorted(RAW_DATA_DIR.glob("*.jpg")) + sorted(RAW_DATA_DIR.glob("*.png"))

    if not image_paths:
        print(f"[fiftyone_qa] No images found in {RAW_DATA_DIR}")
        sys.exit(0)

    dataset_name = f"robops_{architecture}_qa"
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(name=dataset_name)
    samples = []

    for img_path in image_paths:
        sample = fo.Sample(filepath=str(img_path))
        fname = img_path.name
        # Strip s3 key prefix that was encoded into the filename
        meta = metadata.get(fname) or {}
        sample["mean_confidence"] = meta.get("mean_confidence", 0.0)
        sample["num_detections"] = meta.get("num_detections", 0)
        sample["robot_id"] = meta.get("robot_id", "unknown")
        sample["source"] = meta.get("source", "shadow")
        sample["flagged_bad"] = False
        samples.append(sample)

    dataset.add_samples(samples)
    # Sort ascending — lowest confidence = most uncertain = label these first
    dataset = dataset.sort_by("mean_confidence", reverse=False)

    print(f"[fiftyone_qa] Dataset loaded: {len(dataset)} samples")
    print(f"[fiftyone_qa] Opening FiftyOne App at http://localhost:5151")
    print(f"[fiftyone_qa] Flag bad frames with 'flagged_bad = True' in the App.")
    print(f"[fiftyone_qa] When done, press Ctrl+C to export the CVAT queue.")

    session = fo.launch_app(dataset, port=5151)

    try:
        session.wait()
    except KeyboardInterrupt:
        pass

    export_cvat_queue(dataset)
    return dataset


def export_cvat_queue(dataset):
    try:
        import fiftyone as fo
    except ImportError:
        return

    good_samples = dataset.match(fo.ViewField("flagged_bad") == False)
    queue = []
    for sample in good_samples.sort_by("mean_confidence", reverse=False):
        queue.append({
            "filepath": sample.filepath,
            "mean_confidence": sample["mean_confidence"],
            "robot_id": sample["robot_id"],
        })

    CVAT_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CVAT_QUEUE_PATH, "w") as f:
        json.dump(queue, f, indent=2)

    print(f"[fiftyone_qa] CVAT queue exported: {len(queue)} frames → {CVAT_QUEUE_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Phase B FiftyOne QA")
    parser.add_argument("--architecture", default="detr")
    args = parser.parse_args()

    if not RAW_DATA_DIR.exists():
        print(f"[fiftyone_qa] ERROR: {RAW_DATA_DIR} does not exist. Run query_and_pull.py first.")
        sys.exit(1)

    build_fiftyone_dataset(args.architecture)


if __name__ == "__main__":
    main()
