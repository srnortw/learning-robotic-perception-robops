# Round 1 (DETR) — MLOps completion checklist

This repo ends at **data → train → export → eval → model card** (plus optional **Phase F** drift checks against MongoDB if you still ingest production metrics from somewhere else).

## Definition of done

1. **`label_schema.yaml`** — five classes: `person`, `helmet`, `vest`, `no-helmet`, `no-vest`.
2. **Phase B** — MDS shards in S3; `params.yaml` `mds_path` matches.
3. **Phase C** — Colab (or workstation) training logged to DagsHub MLflow; weights in `s3://…/weights/detr/<v>/model.pt`.
4. **CI** — `workflow_dispatch` with MLflow run ID runs `export_onnx.py`, uploads `model_int8.onnx`, pushes **training** image to ECR.
5. **Phase D** — `eval_champion_challenger.py` on holdout; approval PR; merge generates `model_card.md`.

## Optional Phase F

- `drift_check.yml` + `drift_detector.py` expect documents in MongoDB `production_metrics`. Without an edge fleet, that collection may be empty; the job can still run but will report no recent data.

## Secrets (GitHub)

Typical: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN`, `MONGO_URI` (for drift), `DISCORD_WEBHOOK_URL`, `GITHUB_TOKEN` for PRs.

Edge-only secrets (`PI_HOST`, etc.) are **not** used by this repo anymore.
