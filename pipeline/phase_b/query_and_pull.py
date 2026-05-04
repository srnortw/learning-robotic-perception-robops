"""
Phase B — Sub-pipe 1: MongoDB query + S3 image download.

Queries telemetry (shadow frames) and retrain_queue (Phase F drift priority frames),
merges them (priority first), downloads images from S3 to data/detr/raw/.

Usage:
    python query_and_pull.py --architecture detr --days 7 --limit 500
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import pymongo

MONGO_URI = os.environ.get("MONGO_URI", "")
BUCKET = os.environ.get("ROBOPS_S3_BUCKET", "my-perception-robops-data-2026-688567275774-eu-central-1-an")
RAW_DATA_DIR = Path(__file__).parents[2] / "data" / "detr" / "raw"


def query_priority_frames(db) -> list[dict]:
    """Phase F drift frames — always processed first."""
    frames = list(db.retrain_queue.find({
        "architecture": "detr",
        "retrain_priority": True,
        "processed": False,
    }))
    print(f"[query_and_pull] Priority frames (retrain_queue): {len(frames)}")
    return frames


def query_shadow_frames(db, architecture: str, days: int, limit: int) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    frames = list(db.telemetry.find(
        {
            "architecture": architecture,
            "source": "shadow",
            "timestamp": {"$gte": cutoff},
            "s3_url": {"$ne": None},
        },
        limit=limit,
    ).sort("mean_confidence", 1))  # ascending = lowest confidence first (active learning)
    print(f"[query_and_pull] Shadow frames (telemetry): {len(frames)}")
    return frames


def mark_priority_processed(db, frame_ids: list):
    if not frame_ids:
        return
    db.retrain_queue.update_many(
        {"_id": {"$in": frame_ids}},
        {"$set": {"processed": True}},
    )


def download_from_s3(s3_urls: list[str], dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    downloaded = []

    for url in s3_urls:
        if not url or not url.startswith("s3://"):
            continue
        parts = url[5:].split("/", 1)
        bucket, key = parts[0], parts[1]
        filename = key.replace("/", "_")
        dest = dest_dir / filename

        if dest.exists():
            downloaded.append(dest)
            continue

        try:
            s3.download_file(bucket, key, str(dest))
            downloaded.append(dest)
            print(f"[query_and_pull] Downloaded: {filename}")
        except Exception as e:
            print(f"[query_and_pull] WARN: Failed to download {url}: {e}")

    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Phase B query and pull")
    parser.add_argument("--architecture", default="detr")
    parser.add_argument("--days", type=int, default=7, help="Lookback window in days")
    parser.add_argument("--limit", type=int, default=500, help="Max shadow frames to pull")
    args = parser.parse_args()

    if not MONGO_URI:
        print("[query_and_pull] ERROR: MONGO_URI not set.")
        sys.exit(1)

    client = pymongo.MongoClient(MONGO_URI)
    db = client["robops"]

    priority = query_priority_frames(db)
    shadow = query_shadow_frames(db, args.architecture, args.days, args.limit)

    # Priority frames first, then shadow frames — deduplicate by s3_url
    seen = set()
    all_frames = []
    for f in priority + shadow:
        url = f.get("s3_url")
        if url and url not in seen:
            seen.add(url)
            all_frames.append(f)

    print(f"[query_and_pull] Total unique frames to download: {len(all_frames)}")

    s3_urls = [f["s3_url"] for f in all_frames if f.get("s3_url")]
    downloaded = download_from_s3(s3_urls, RAW_DATA_DIR)

    priority_ids = [f["_id"] for f in priority]
    mark_priority_processed(db, priority_ids)
    client.close()

    print(f"[query_and_pull] Done. {len(downloaded)} images in {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()
