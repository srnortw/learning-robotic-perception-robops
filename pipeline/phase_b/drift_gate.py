"""
Phase B — Sub-pipe 0: Data drift gate.

Compares a new incoming batch of frames (from MongoDB telemetry) against the
saved training baseline distribution using PSI (Population Stability Index).

PSI > 0.2 on any metric = block the pipeline and send a Discord alert.
Skip this gate on Round 1 — no baseline exists yet.

Usage:
    python drift_gate.py --batch-date 2026-05-04 --architecture detr
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

import pymongo
import requests

MONGO_URI = os.environ.get("MONGO_URI", "")
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
BASELINE_PATH = os.path.join(os.path.dirname(__file__), "drift_baseline_detr.json")
PSI_BLOCK_THRESHOLD = 0.2


def load_baseline() -> dict | None:
    if not os.path.exists(BASELINE_PATH):
        print("[drift_gate] No baseline found — skipping gate (Round 1 mode).")
        return None
    with open(BASELINE_PATH) as f:
        return json.load(f)


def fetch_batch(db, architecture: str, date_str: str) -> list[dict]:
    cutoff = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    window_start = cutoff - timedelta(days=7)
    return list(db.telemetry.find({
        "architecture": architecture,
        "source": "shadow",
        "timestamp": {"$gte": window_start.isoformat(), "$lte": cutoff.isoformat()},
    }))


def compute_psi(expected: float, actual: float) -> float:
    """PSI for a single bucket (proportion comparison)."""
    eps = 1e-6
    e = max(expected, eps)
    a = max(actual, eps)
    return (a - e) * math.log(a / e)


def check_confidence_psi(baseline: dict, batch: list[dict]) -> float:
    baseline_mean = baseline.get("mean_confidence", 0.0)
    if not batch:
        return 0.0
    batch_mean = sum(d.get("mean_confidence", 0.0) for d in batch) / len(batch)
    return abs(compute_psi(baseline_mean, batch_mean))


def check_class_psi(baseline: dict, batch: list[dict]) -> float:
    baseline_freq: dict = baseline.get("class_frequency", {})
    if not batch or not baseline_freq:
        return 0.0

    batch_counts: dict[str, int] = {}
    total = 0
    for doc in batch:
        for cls, count in doc.get("class_distribution", {}).items():
            batch_counts[cls] = batch_counts.get(cls, 0) + count
            total += count

    if total == 0:
        return 0.0

    psi_total = 0.0
    for cls, expected_prop in baseline_freq.items():
        actual_prop = batch_counts.get(cls, 0) / total
        psi_total += compute_psi(expected_prop, actual_prop)

    return psi_total


def send_discord_alert(message: str):
    if not DISCORD_WEBHOOK:
        print(f"[drift_gate] DISCORD_WEBHOOK_URL not set — alert not sent.\n{message}")
        return
    payload = {"content": f"**[RoboOps Drift Gate]** {message}"}
    try:
        requests.post(DISCORD_WEBHOOK, json=payload, timeout=5)
    except Exception as e:
        print(f"[drift_gate] Discord alert failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Phase B drift gate")
    parser.add_argument("--batch-date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--architecture", default="detr")
    args = parser.parse_args()

    baseline = load_baseline()
    if baseline is None:
        print("[drift_gate] PASS — Round 1, no baseline. Proceeding.")
        sys.exit(0)

    if not MONGO_URI:
        print("[drift_gate] ERROR: MONGO_URI not set.")
        sys.exit(1)

    client = pymongo.MongoClient(MONGO_URI)
    db = client["robops"]
    batch = fetch_batch(db, args.architecture, args.batch_date)
    client.close()

    if not batch:
        print(f"[drift_gate] No frames found for {args.architecture} on {args.batch_date}.")
        sys.exit(0)

    confidence_psi = check_confidence_psi(baseline, batch)
    class_psi = check_class_psi(baseline, batch)

    print(f"[drift_gate] PSI confidence={confidence_psi:.4f}  class_distribution={class_psi:.4f}")
    print(f"[drift_gate] Batch size: {len(batch)} frames")

    blocked = False
    if confidence_psi > PSI_BLOCK_THRESHOLD:
        msg = (
            f"DRIFT DETECTED on confidence distribution! "
            f"PSI={confidence_psi:.4f} > threshold={PSI_BLOCK_THRESHOLD}. "
            f"Batch: {args.batch_date} | arch: {args.architecture}. "
            f"Pipeline blocked — do NOT proceed to CVAT."
        )
        send_discord_alert(msg)
        print(f"[drift_gate] BLOCKED: {msg}")
        blocked = True

    if class_psi > PSI_BLOCK_THRESHOLD:
        msg = (
            f"DRIFT DETECTED on class distribution! "
            f"PSI={class_psi:.4f} > threshold={PSI_BLOCK_THRESHOLD}. "
            f"Batch: {args.batch_date} | arch: {args.architecture}. "
            f"Pipeline blocked — do NOT proceed to CVAT."
        )
        send_discord_alert(msg)
        print(f"[drift_gate] BLOCKED: {msg}")
        blocked = True

    if blocked:
        sys.exit(1)

    print("[drift_gate] PASS — no significant drift detected. Proceeding to query_and_pull.")
    sys.exit(0)


if __name__ == "__main__":
    main()
