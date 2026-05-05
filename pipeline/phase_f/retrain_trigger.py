"""
Phase F — Retraining Trigger

Called automatically by drift_detector.py when drift is confirmed.
1. Flags high-drift frames in MongoDB (retrain_priority: true)
2. Inserts a retrain_queue document
3. Fires GitHub Actions workflow_dispatch to kick off Phase A→B→C
4. Sends Discord alert

Usage:
    python pipeline/phase_f/retrain_trigger.py \
        --report pipeline/phase_f/drift_report.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import requests
import yaml
from pymongo import MongoClient

MONGO_URI   = os.environ.get("MONGO_URI", "")
DB_NAME     = "robops"
GH_TOKEN    = os.environ.get("GITHUB_TOKEN", "")
GH_REPO     = "srnortw/learning-robotic-perception-robops"
DISCORD_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
PARAMS_PATH = "pipeline/phase_c/detr/params.yaml"


def current_dataset_version() -> str:
    try:
        with open(PARAMS_PATH) as f:
            return yaml.safe_load(f)["dataset"]["dataset_version"]
    except Exception:
        return "v1"


def connect_mongo():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    return client[DB_NAME]


def flag_drift_frames(db, report: dict):
    """Mark production frames from the drift window as retrain_priority."""
    try:
        from datetime import datetime, timedelta, timezone
        report_ts  = datetime.fromisoformat(report["timestamp"])
        window_hrs = report.get("window_hours", 24)
        window_start = (report_ts - timedelta(hours=window_hrs)).isoformat()
        result = db.production_metrics.update_many(
            {"source": "production",
             "timestamp": {"$gte": window_start, "$lte": report["timestamp"]}},
            {"$set": {"retrain_priority": True}},
        )
        print(f"Flagged {result.modified_count} frames as retrain_priority in MongoDB")
    except Exception as e:
        print(f"WARNING: could not flag frames: {e}")


def push_retrain_queue(db, report: dict):
    """Insert a retrain queue document for Phase A to pick up."""
    doc = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "architecture": "detr",
        "trigger": "phase_f_drift",
        "drift_triggers": report.get("triggers", []),
        "drift_value": max(
            report.get("confidence_drop", 0.0),
            report.get("class_psi", 0.0),
        ),
        "status": "pending",
    }
    try:
        db.retrain_queue.insert_one(doc)
        print("Retrain queue document inserted")
    except Exception as e:
        print(f"WARNING: could not write to retrain_queue: {e}")


def dispatch_github_actions(report: dict) -> str | None:
    """Fire workflow_dispatch on ci_deploy.yml to start Phase A→C retrain."""
    if not GH_TOKEN:
        print("WARNING: GITHUB_TOKEN not set — skipping workflow dispatch.")
        return None

    triggers = report.get("triggers", [])
    primary_trigger = triggers[0] if triggers else "drift"
    drift_val = str(round(max(
        report.get("confidence_drop", 0.0),
        report.get("class_psi", 0.0),
    ), 4))

    url = f"https://api.github.com/repos/{GH_REPO}/actions/workflows/ci_deploy.yml/dispatches"
    payload = {
        "ref": "main",
        "inputs": {
            "model_run_id": "",
            "dataset_version": current_dataset_version(),
            "retrain": "true",
        }
    }
    resp = requests.post(
        url,
        headers={
            "Authorization": f"token {GH_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        },
        json=payload,
        timeout=15,
    )
    if resp.status_code == 204:
        run_url = f"https://github.com/{GH_REPO}/actions"
        print(f"GitHub Actions dispatched: {run_url}")
        return run_url
    else:
        print(f"WARNING: GitHub dispatch failed {resp.status_code}: {resp.text}")
        return None


def send_discord_alert(report: dict, run_url: str | None):
    if not DISCORD_URL:
        return

    conf_drop = report.get("confidence_drop", 0.0)
    psi       = report.get("class_psi", 0.0)
    lat_ratio = report.get("latency_ratio", 1.0)
    triggers  = report.get("triggers", [])

    conf_emoji = "⚠" if conf_drop > 0.10 else "✓"
    psi_emoji  = "⚠" if psi > 0.20 else "✓"
    lat_emoji  = "⚠" if lat_ratio > 1.5 else "✓"

    lines = [
        f"**[Phase F Alert]** DETR drift detected on `robops-pi3b-001`",
        f"",
        f"{conf_emoji} Confidence drop: `{conf_drop:+.3f}` (threshold -0.10)",
        f"{psi_emoji} Class freq PSI:  `{psi:.3f}` (threshold 0.20)",
        f"{lat_emoji} Latency ratio:   `{lat_ratio:.2f}x` (threshold 1.5x)",
        f"",
        f"Triggers: `{', '.join(triggers)}`",
        f"Action: Retraining cycle dispatched automatically.",
    ]
    if run_url:
        lines.append(f"GitHub Actions: {run_url}")

    payload = {"content": "\n".join(lines)}
    try:
        requests.post(DISCORD_URL, json=payload, timeout=10)
        print("Discord alert sent")
    except Exception as e:
        print(f"WARNING: Discord alert failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Phase F — retrain trigger")
    parser.add_argument("--report", default="pipeline/phase_f/drift_report.json")
    args = parser.parse_args()

    with open(args.report) as f:
        report = json.load(f)

    if not report.get("drift_detected", False):
        print("No drift in report — nothing to trigger.")
        sys.exit(0)

    print(f"Drift confirmed: {report['triggers']}")

    try:
        db = connect_mongo()
        flag_drift_frames(db, report)
        push_retrain_queue(db, report)

        # Update drift event status
        db.drift_events.update_one(
            {"timestamp": report["timestamp"]},
            {"$set": {"action": "retrain_dispatched"}},
        )
    except Exception as e:
        print(f"WARNING: MongoDB ops failed: {e}")

    run_url = dispatch_github_actions(report)
    send_discord_alert(report, run_url)

    print("Retrain trigger complete.")


if __name__ == "__main__":
    main()
