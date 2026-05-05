"""
Phase F — Drift & Degradation Detector

Scheduled every 6 hours via .github/workflows/drift_check.yml.
Fetches the last 24 h of production_metrics from MongoDB, compares against
the training baseline distribution using Evidently AI, and fires an alert
+ retrain trigger if thresholds are breached.

Usage:
    python pipeline/phase_f/drift_detector.py \
        [--hours 24] \
        [--baseline data/drift_baselines/detr_v1_baseline.json] \
        [--output pipeline/phase_f/drift_report.json]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml
from pymongo import MongoClient

MONGO_URI      = os.environ.get("MONGO_URI", "")
DB_NAME        = "robops"
PARAMS_PATH    = "pipeline/phase_c/detr/params.yaml"
DRIFT_EVENT_COL= "drift_events"

# Alert thresholds (from Phase F plan)
CONFIDENCE_DROP_THRESHOLD = 0.10   # >10% drop vs baseline mean
PSI_THRESHOLD             = 0.20   # class-frequency Population Stability Index
LATENCY_MULTIPLIER        = 1.5    # p95 > 1.5× baseline → alert

CLASSES = ["person", "chair", "table", "door"]


def load_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def connect_mongo():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    client.server_info()
    return client[DB_NAME]


def fetch_recent(db, hours: int) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    docs = list(db.production_metrics.find(
        {"source": "production", "timestamp": {"$gte": cutoff.isoformat()}},
        {"_id": 0, "mean_confidence": 1, "class_distribution": 1,
         "inference_latency_ms": 1, "timestamp": 1}
    ))
    if not docs:
        return pd.DataFrame()
    return pd.DataFrame(docs)


def load_baseline(baseline_path: str) -> dict | None:
    p = Path(baseline_path)
    if not p.exists():
        print(f"WARNING: baseline file {baseline_path} not found — skipping drift check.")
        return None
    with open(p) as f:
        return json.load(f)


def compute_psi(baseline_dist: dict, current_dist: dict) -> float:
    """
    Population Stability Index for class frequency distributions.
    PSI < 0.1: no drift, 0.1–0.2: moderate, >0.2: significant drift.
    """
    import math
    eps = 1e-6
    psi = 0.0
    all_classes = set(baseline_dist) | set(current_dist)
    total_base = sum(baseline_dist.values()) or 1
    total_curr = sum(current_dist.values()) or 1
    for cls in all_classes:
        b = (baseline_dist.get(cls, 0) / total_base) + eps
        c = (current_dist.get(cls, 0) / total_curr) + eps
        psi += (c - b) * math.log(c / b)
    return round(psi, 4)


def run_evidently_drift(baseline_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    """Run Evidently DataDriftPreset on confidence + latency columns."""
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError:
        print("WARNING: evidently not installed — skipping Evidently report.")
        return {"dataset_drift": False, "evidently_available": False}

    cols = [c for c in ["mean_confidence", "inference_latency_ms"] if c in baseline_df.columns and c in current_df.columns]
    if not cols:
        return {"dataset_drift": False, "evidently_available": True}

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=baseline_df[cols], current_data=current_df[cols])
    result = report.as_dict()
    return result["metrics"][0]["result"]


def write_drift_event(db, event: dict):
    try:
        db[DRIFT_EVENT_COL].insert_one(event)
        print(f"Drift event written to MongoDB: {event['trigger']}")
    except Exception as e:
        print(f"WARNING: could not write drift event to MongoDB: {e}")


def main():
    parser = argparse.ArgumentParser(description="Phase F — drift detector")
    parser.add_argument("--hours",    type=int, default=24)
    parser.add_argument("--baseline", default="data/drift_baselines/detr_v1_baseline.json")
    parser.add_argument("--output",   default="pipeline/phase_f/drift_report.json")
    args = parser.parse_args()

    # ── 1. Load data
    baseline = load_baseline(args.baseline)
    if baseline is None:
        print("No baseline available — exiting without drift check.")
        sys.exit(0)

    print(f"Connecting to MongoDB...")
    try:
        db = connect_mongo()
    except Exception as e:
        print(f"ERROR: MongoDB connection failed: {e}")
        sys.exit(1)

    print(f"Fetching last {args.hours}h of production_metrics...")
    current_df = fetch_recent(db, args.hours)
    if current_df.empty:
        print("No production data in window — skipping drift check.")
        sys.exit(0)

    print(f"  {len(current_df)} records fetched.")

    # ── 2. Compute metrics
    current_mean_conf = float(current_df["mean_confidence"].mean()) if "mean_confidence" in current_df else 0.0
    baseline_mean_conf= baseline.get("mean_confidence", 0.0)
    confidence_drop   = baseline_mean_conf - current_mean_conf

    # Aggregate class distribution from production window
    current_class_dist: dict[str, int] = {}
    for row in current_df.get("class_distribution", pd.Series(dtype=object)).dropna():
        if isinstance(row, dict):
            for cls, cnt in row.items():
                current_class_dist[cls] = current_class_dist.get(cls, 0) + cnt

    baseline_class_dist = baseline.get("class_distribution", {})
    psi = compute_psi(baseline_class_dist, current_class_dist)

    current_lat_p95 = float(current_df["inference_latency_ms"].quantile(0.95)) \
        if "inference_latency_ms" in current_df and not current_df["inference_latency_ms"].isna().all() \
        else 0.0
    baseline_lat_p95 = baseline.get("latency_p95_ms", 0.0)
    latency_ratio    = (current_lat_p95 / baseline_lat_p95) if baseline_lat_p95 > 0 else 1.0

    # ── 3. Evidently report
    # Evidently needs a real sample distribution for meaningful results.
    # A stored scalar baseline (single mean + p95) cannot produce valid
    # statistical tests — skip Evidently and rely on the custom PSI/threshold
    # checks above. Evidently becomes useful once we store full sample vectors.
    baseline_samples = baseline.get("samples")
    if baseline_samples and len(baseline_samples) >= 30:
        baseline_df = pd.DataFrame(baseline_samples)
        evidently_result = run_evidently_drift(baseline_df, current_df)
    else:
        print("NOTE: Evidently skipped — baseline has no sample distribution (scalar only). "
              "Using custom PSI checks instead.")
        evidently_result = {"dataset_drift": False, "evidently_available": False}

    # ── 4. Evaluate thresholds
    triggers = []
    if confidence_drop > CONFIDENCE_DROP_THRESHOLD:
        triggers.append(f"confidence_drop ({confidence_drop:.3f} > {CONFIDENCE_DROP_THRESHOLD})")
    if psi > PSI_THRESHOLD:
        triggers.append(f"class_psi ({psi:.3f} > {PSI_THRESHOLD})")
    if latency_ratio > LATENCY_MULTIPLIER:
        triggers.append(f"latency_ratio ({latency_ratio:.2f}x > {LATENCY_MULTIPLIER}x)")

    drift_detected = len(triggers) > 0

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "window_hours": args.hours,
        "num_records": len(current_df),
        "baseline_mean_confidence": baseline_mean_conf,
        "current_mean_confidence": round(current_mean_conf, 4),
        "confidence_drop": round(confidence_drop, 4),
        "class_psi": psi,
        "baseline_latency_p95_ms": baseline_lat_p95,
        "current_latency_p95_ms": round(current_lat_p95, 1),
        "latency_ratio": round(latency_ratio, 2),
        "evidently_dataset_drift": evidently_result.get("dataset_drift", False),
        "drift_detected": drift_detected,
        "triggers": triggers,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    # ── 5. Print summary
    print(f"\n── Drift Report ──────────────────────────────────────")
    print(f"  Confidence:  {current_mean_conf:.4f} (baseline {baseline_mean_conf:.4f}, drop {confidence_drop:+.4f})")
    print(f"  Class PSI:   {psi:.4f}")
    print(f"  Latency p95: {current_lat_p95:.0f} ms (baseline {baseline_lat_p95:.0f} ms, ratio {latency_ratio:.2f}x)")
    print(f"  Drift detected: {drift_detected}")
    if triggers:
        for t in triggers:
            print(f"    ⚠ {t}")

    # ── 6. Write drift event + trigger retrain
    if drift_detected:
        event = {
            "timestamp": report["timestamp"],
            "architecture": "detr",
            "model_version": baseline.get("model_version", "v1"),
            "trigger": ", ".join(triggers),
            "drift_value": max(confidence_drop, psi),
            "report": report,
            "action": "pending",
        }
        write_drift_event(db, event)

        print(f"\nDrift detected — calling retrain_trigger.py")
        import subprocess
        result = subprocess.run(
            ["python", "pipeline/phase_f/retrain_trigger.py",
             "--report", args.output],
            capture_output=True, text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"WARNING: retrain_trigger failed: {result.stderr}")
    else:
        print("\nNo drift detected — system healthy.")

    sys.exit(0)


if __name__ == "__main__":
    main()
