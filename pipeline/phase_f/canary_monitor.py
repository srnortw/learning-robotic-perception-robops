"""
Phase F — Canary Monitor

Runs after Phase E deploys to group-canary.
Polls MongoDB production_metrics every POLL_INTERVAL_MINUTES for
CANARY_MONITOR_HOURS, comparing against a pre-deploy baseline snapshot.

On success (no regression): calls promote_to_fleet.py.
On failure:                  triggers Greengrass rollback + Discord alert.

Usage:
    python pipeline/phase_f/canary_monitor.py \
        --dataset-version v1 \
        [--hours 4] \
        [--poll-minutes 15]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import boto3
from pymongo import MongoClient

MONGO_URI   = os.environ.get("MONGO_URI", "")
DB_NAME     = "robops"
DISCORD_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

CANARY_DEVICE     = "robops-pi3b-001"
CANARY_GROUP_ARN  = "arn:aws:iot:eu-central-1:688567275774:thinggroup/group-canary"
GG_REGION         = "eu-central-1"

# Regression thresholds for canary window
MAX_CONFIDENCE_DROP = 0.15   # >15% vs pre-deploy baseline → rollback
MAX_LATENCY_RATIO   = 2.0    # >2× pre-deploy p95 → rollback
MIN_DOCS_PER_POLL   = 3      # <3 docs in a poll window → device may be unhealthy


def connect_mongo():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    return client[DB_NAME]


def snapshot_baseline(db) -> dict:
    """Capture metrics from the 30 min before deploy as baseline."""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)
    docs = list(db.production_metrics.find(
        {"robot_id": CANARY_DEVICE, "timestamp": {"$gte": cutoff.isoformat()}},
        {"_id": 0, "mean_confidence": 1, "inference_latency_ms": 1}
    ))
    if not docs:
        # No pre-deploy data; use safe defaults
        return {"mean_confidence": 0.0, "latency_p95_ms": 0.0, "doc_count": 0}
    confs = [d["mean_confidence"] for d in docs if "mean_confidence" in d]
    lats  = [d["inference_latency_ms"] for d in docs if "inference_latency_ms" in d]
    return {
        "mean_confidence": sum(confs) / len(confs) if confs else 0.0,
        "latency_p95_ms": sorted(lats)[int(len(lats) * 0.95)] if lats else 0.0,
        "doc_count": len(docs),
    }


def poll_canary(db, since: datetime) -> dict:
    """Fetch metrics written by the canary device since last poll."""
    docs = list(db.production_metrics.find(
        {"robot_id": CANARY_DEVICE,
         "source": "production",
         "timestamp": {"$gte": since.isoformat()}},
        {"_id": 0, "mean_confidence": 1, "inference_latency_ms": 1}
    ))
    if not docs:
        return {"doc_count": 0, "mean_confidence": None, "latency_p95_ms": None}
    confs = [d["mean_confidence"] for d in docs if "mean_confidence" in d]
    lats  = [d["inference_latency_ms"] for d in docs if "inference_latency_ms" in d]
    return {
        "doc_count": len(docs),
        "mean_confidence": sum(confs) / len(confs) if confs else None,
        "latency_p95_ms": sorted(lats)[int(len(lats) * 0.95)] if lats else None,
    }


def check_regression(baseline: dict, poll: dict) -> list[str]:
    issues = []
    if poll["doc_count"] < MIN_DOCS_PER_POLL:
        issues.append(f"low_data ({poll['doc_count']} docs < {MIN_DOCS_PER_POLL})")

    if poll["mean_confidence"] is not None and baseline["mean_confidence"] > 0:
        drop = baseline["mean_confidence"] - poll["mean_confidence"]
        if drop > MAX_CONFIDENCE_DROP:
            issues.append(f"confidence_drop ({drop:.3f} > {MAX_CONFIDENCE_DROP})")

    if poll["latency_p95_ms"] is not None and baseline["latency_p95_ms"] > 0:
        ratio = poll["latency_p95_ms"] / baseline["latency_p95_ms"]
        if ratio > MAX_LATENCY_RATIO:
            issues.append(f"latency_ratio ({ratio:.2f}x > {MAX_LATENCY_RATIO}x)")

    return issues


def trigger_rollback(dataset_version: str):
    """Cancel the Greengrass deployment to group-canary."""
    print("Triggering Greengrass rollback...")
    try:
        gg = boto3.client("greengrassv2", region_name=GG_REGION)
        paginator = gg.get_paginator("list_deployments")
        for page in paginator.paginate(targetArn=CANARY_GROUP_ARN):
            for dep in page.get("deployments", []):
                if dep.get("deploymentStatus") in ("ACTIVE", "COMPLETED"):
                    gg.cancel_deployment(deploymentId=dep["deploymentId"])
                    print(f"Cancelled deployment {dep['deploymentId']}")
                    return
        print("No active deployment found to cancel.")
    except Exception as e:
        print(f"WARNING: Greengrass rollback failed: {e}")


def send_alert(message: str):
    if not DISCORD_URL:
        return
    try:
        import requests
        requests.post(DISCORD_URL, json={"content": message}, timeout=10)
    except Exception:
        pass


def promote_to_fleet(dataset_version: str):
    result = subprocess.run(
        ["python", "pipeline/phase_e/promote_to_fleet.py",
         "--dataset-version", dataset_version],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"WARNING: promote_to_fleet failed: {result.stderr}")


def main():
    parser = argparse.ArgumentParser(description="Phase F — canary monitor")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--hours",        type=float, default=4.0)
    parser.add_argument("--poll-minutes", type=float, default=15.0)
    args = parser.parse_args()

    db       = connect_mongo()
    baseline = snapshot_baseline(db)
    deadline = datetime.now(timezone.utc) + timedelta(hours=args.hours)
    poll_secs= args.poll_minutes * 60
    poll_num = 0

    print(f"\n── Canary Monitor ─────────────────────────────────────────")
    print(f"  Device:    {CANARY_DEVICE}")
    print(f"  Duration:  {args.hours}h ({args.poll_minutes}-min polls)")
    print(f"  Baseline:  conf={baseline['mean_confidence']:.3f}  lat_p95={baseline['latency_p95_ms']:.0f}ms")
    print(f"  Deadline:  {deadline.strftime('%H:%M UTC')}\n")

    last_poll = datetime.now(timezone.utc)

    while datetime.now(timezone.utc) < deadline:
        time.sleep(poll_secs)
        poll_num += 1
        now       = datetime.now(timezone.utc)
        poll      = poll_canary(db, last_poll)
        last_poll = now

        issues = check_regression(baseline, poll)
        status = "✗ REGRESSION" if issues else "✓ healthy"
        conf_str = f"{poll['mean_confidence']:.3f}" if poll['mean_confidence'] else "n/a"
        lat_str  = f"{poll['latency_p95_ms']:.0f}" if poll['latency_p95_ms'] else "n/a"
        print(f"  Poll {poll_num:02d} [{now.strftime('%H:%M')}] "
              f"docs={poll['doc_count']} "
              f"conf={conf_str} "
              f"lat={lat_str}ms "
              f"→ {status}")

        if issues:
            msg = (
                f"**[Phase F] Canary REGRESSION detected on `{CANARY_DEVICE}`**\n"
                f"Issues: `{', '.join(issues)}`\n"
                f"Rolling back Greengrass deployment for DETR {args.dataset_version}."
            )
            print(f"\n{msg}")
            send_alert(msg)
            trigger_rollback(args.dataset_version)
            sys.exit(1)

    print(f"\n✓ Canary healthy for {args.hours}h — promoting to fleet.")
    send_alert(
        f"**[Phase F]** Canary `{CANARY_DEVICE}` passed {args.hours}h monitoring ✓\n"
        f"Promoting DETR `{args.dataset_version}` to fleet."
    )
    promote_to_fleet(args.dataset_version)


if __name__ == "__main__":
    main()
