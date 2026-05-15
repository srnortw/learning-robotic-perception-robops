# Phase A — Data ingestion (archived / robot path)

**Status in this repo:** The live-robot ingest path (ROS2 `camera_node`, `detr_node`, shadow mode, `s3_uploader`, `mongo_writer`) has been **removed**. This repository is **MLOps-only** (data engineering → training → eval → optional drift monitoring).

**Current data entry**

- Annotated images and labels via **Roboflow** / **CVAT** → COCO JSON.
- Phase B: DVC, FiftyOne QA, `convert_to_mds.py`, shards on **S3** (`mds_path` in `params.yaml`).

See `pipeline/phase_a/README.md` for a short summary.

If you reintroduce edge inference later, use a **separate** repository or package for ROS2 / deployment; keep this repo focused on datasets, training artifacts, and evaluation.
