# DETR MLOps Pipeline — Documentation

Personal **MLOps** pipeline for construction-safety DETR (and future model slots). **Edge / ROS2 / Raspberry Pi deployment has been removed** from this repository; focus is datasets → training → ONNX → evaluation → model card, with optional scheduled drift checks.

**Training:** Google Colab Pro+ (or workstation Docker `docker/training`)
**CI/CD:** GitHub Actions (`ubuntu-latest` for build/export; optional **self-hosted** runner for `drift_check.yml` only)
**Registry:** AWS ECR **`robops/training`** (amd64) — reproducible training/export image
**Storage:** AWS S3 (images, MDS shards, weights) + MongoDB Atlas (optional, for drift / retrain hooks)
**Experiment tracking:** DagsHub (MLflow + DVC)

---

## Pipeline overview

```
Phase B → Phase C → Phase D  (+ optional Phase F)
   ↑                              |
   └──── drift / retrain ─────────┘
```

| Phase | Name | What happens |
|---|---|---|
| B | Data engineering | Drift gate → DVC → FiftyOne QA → CVAT → MDS shards on S3 |
| C | Training & export | Colab streams MDS from S3 → MLflow → `export_onnx.py` in CI → INT8 on S3 |
| D | Human audit | Champion–challenger eval → Discord → FiftyOne audit → approval PR → model card on merge |
| F | Monitoring (optional) | `drift_detector.py` on Mongo `production_metrics` (if populated) → report / retrain hook |

**Phase A** (live robot ingest) and **Phase E** (device deploy) are documented as **out of scope** in this repo — see plan files below.

---

## Plan files

- [01_PHASE_A_DATA_INGESTION](pipeline_plans/01_PHASE_A_DATA_INGESTION.md) — archived note (robot path removed)
- [02_PHASE_B_DATA_ENGINEERING](pipeline_plans/02_PHASE_B_DATA_ENGINEERING.md)
- [03_PHASE_C_TRAINING_DETR](pipeline_plans/03_PHASE_C_TRAINING_DETR.md)
- [04_PHASE_D_HUMAN_AUDIT](pipeline_plans/04_PHASE_D_HUMAN_AUDIT.md)
- [05_PHASE_E_DEPLOYMENT](pipeline_plans/05_PHASE_E_DEPLOYMENT.md) — edge deploy deferred / separate project
- [06_PHASE_F_MONITORING](pipeline_plans/06_PHASE_F_MONITORING.md) — drift + retrain (no Pi nodes in repo)
- [ROUND1_COMPLETION](ROUND1_COMPLETION.md) — MLOps “done” checklist
- [OPTIONAL_AWS_CLEANUP](OPTIONAL_AWS_CLEANUP.md) — remove unused ECR / IoT resources after dropping edge
- [SESSION_DONE_2026-05-15](SESSION_DONE_2026-05-15.md) — last session: MLOps cleanup, branches, ECR, CI dispatch
- [NEXT_STEPS](NEXT_STEPS.md) — checklist to continue tomorrow
- [DATASETS](DATASETS.md)
---

## Key data flow: COCO / Roboflow → MDS → training

```
Phase B                        Phase C
CVAT / Roboflow COCO JSON  →  MDS shards in S3  →  Colab StreamingDataset
(dvc add)                       (mds/detr/vN/)        (no full dvc pull for training)
```

- **DVC** tracks raw images and holdout (`data/detr/data.dvc`, `holdout.dvc`)
- **MDS** path lives in `pipeline/phase_c/detr/params.yaml`
- **Class names** — single source: `pipeline/phase_b/label_schema.yaml` → sync with `python pipeline/utils/schema.py --sync` (updates `params.yaml` and `pipeline/phase_c/detr/detr_params.yaml`)

---

## Infrastructure setup order

1. GitHub repo + DVC remote (e.g. S3 or DagsHub)
2. DagsHub project (MLflow + DVC)
3. AWS S3 bucket
4. AWS ECR repository **`robops/training`** (amd64) for the training Docker image
5. MongoDB Atlas (optional — for Phase F drift if you have metrics)
6. GitHub Secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN`, optional `MONGO_URI`, `DISCORD_WEBHOOK_URL`
7. For scheduled drift only: self-hosted runner + `drift_check.yml` secrets

---

## Local training / export image (laptop)

```bash
aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.eu-central-1.amazonaws.com
docker build -f docker/training/Dockerfile -t robops/training:local .
```

---

## CI workflow

`.github/workflows/ci_deploy.yml` — ONNX export + INT8, push **training** image to ECR, optional Phase D eval + PR + model card on merge. **No** arm64 edge build, **no** Greengrass.

---

## Architecture rounds

**Round 1 — DETR** (current). Future rounds (RT-DETR, Mask2Former, …) reuse the same **MLOps** skeleton; edge code belongs in another repo if needed.
