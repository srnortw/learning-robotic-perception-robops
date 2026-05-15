# Phase F — Drift detection & retrain hook (MLOps)

**Goal:** Compare recent **production_metrics** in MongoDB (if any) against a saved baseline using Evidently-style checks, write `drift_report.json`, optionally notify Discord, and optionally fire `retrain_trigger.py` → GitHub `workflow_dispatch`.

**Receives from:** Any system that still writes `production_metrics` documents (this repo no longer ships edge nodes). If nothing writes metrics, scheduled drift runs will see an empty window — that is expected.

**Triggers:** `retrain_trigger.py` can call the main CI workflow with `retrain=true` (configure repo name inside the script if you fork).

---

## Components (in this repo)

| Artifact | Role |
|---|---|
| `pipeline/phase_f/drift_detector.py` | Pull recent Mongo docs, compare to `data/drift_baselines/*.json`, emit `drift_report.json` |
| `pipeline/phase_f/retrain_trigger.py` | On drift, queue / notify / `workflow_dispatch` |
| `pipeline/phase_f/dashboard.py` | Optional local visualization helper |
| `.github/workflows/drift_check.yml` | Scheduled + manual drift job (uses **self-hosted** runner if configured) |

**Removed:** `canary_monitor.py` (post–Greengrass deploy window) and all ROS2 / Pi-specific monitoring nodes.

---

## Production metrics shape (reference)

If you attach another data source later, documents should be compatible with what `drift_detector.py` expects (`mean_confidence`, `class_distribution`, optional `inference_latency_ms`, ISO `timestamp`, `source`).

---

## Scheduled drift

Cron in `drift_check.yml` (every 6 hours). Requires secrets: `MONGO_URI`, optional `DISCORD_WEBHOOK_URL`, `GITHUB_TOKEN`.

---

## Learning order

Phase F is optional polish after Phase D is stable. Tune thresholds in `drift_detector.py` to match your baseline files under `data/drift_baselines/`.
