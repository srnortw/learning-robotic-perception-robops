"""
Phase F — Observability Dashboard

Loads low-confidence or high-drift production frames from MongoDB into
FiftyOne for visual inspection. Run locally whenever you want to see what
the live model is struggling with.

Usage:
    python pipeline/phase_f/dashboard.py \
        [--hours 24] \
        [--confidence-threshold 0.55] \
        [--port 5151]
"""

import argparse
import os
from datetime import datetime, timedelta, timezone

from pymongo import MongoClient

MONGO_URI = os.environ.get("MONGO_URI", "")
DB_NAME   = "robops"


def connect_mongo():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    return client[DB_NAME]


def fetch_low_confidence_docs(db, hours: int, conf_threshold: float) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    return list(db.production_metrics.find(
        {
            "source": "production",
            "mean_confidence": {"$lt": conf_threshold},
            "timestamp": {"$gte": cutoff.isoformat()},
        },
        {"_id": 0},
    ).sort("mean_confidence", 1).limit(500))


def build_fiftyone_dataset(docs: list[dict], dataset_name: str):
    import fiftyone as fo

    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(dataset_name)
    samples = []

    for doc in docs:
        # FiftyOne requires a filepath — use a placeholder if no S3 URL
        filepath = doc.get("s3_url") or f"/tmp/robops_frame_{doc.get('frame_count', 0)}.jpg"

        sample = fo.Sample(filepath=filepath)
        sample["robot_id"]        = doc.get("robot_id", "")
        sample["model_version"]   = doc.get("model_version", "")
        sample["mean_confidence"] = doc.get("mean_confidence", 0.0)
        sample["num_detections"]  = doc.get("num_detections", 0)
        sample["cpu_percent"]     = doc.get("cpu_percent", 0.0)
        sample["memory_mb"]       = doc.get("memory_mb", 0)
        sample["timestamp"]       = doc.get("timestamp", "")

        class_dist = doc.get("class_distribution", {})
        if class_dist:
            sample["class_distribution"] = str(class_dist)

        samples.append(sample)

    if samples:
        dataset.add_samples(samples)

    dataset.save()
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Phase F — production monitoring dashboard")
    parser.add_argument("--hours",                type=int,   default=24)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--port",                 type=int,   default=5151)
    args = parser.parse_args()

    print(f"Fetching low-confidence frames (conf < {args.confidence_threshold}, last {args.hours}h)...")
    db   = connect_mongo()
    docs = fetch_low_confidence_docs(db, args.hours, args.confidence_threshold)
    print(f"  {len(docs)} frames found.")

    if not docs:
        print("No low-confidence frames in this window. Try increasing --hours or lowering --confidence-threshold.")
        return

    name    = f"robops-drift-monitor-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
    dataset = build_fiftyone_dataset(docs, name)

    print(f"\nDataset '{name}' ready with {len(dataset)} samples.")
    print(f"Sorted by lowest confidence first.")
    print(f"Launching FiftyOne App on http://localhost:{args.port} ...")

    import fiftyone as fo
    session = fo.launch_app(dataset, port=args.port)
    session.wait()


if __name__ == "__main__":
    main()
