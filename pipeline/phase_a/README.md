# Phase A — Data paths (MLOps repo)

Live robot ingest (ROS2 camera, shadow metrics, `s3_uploader` / `mongo_writer` nodes) was **removed** when this repository was narrowed to **MLOps only**.

**How data enters the pipeline now**

- Annotated datasets via **Roboflow** / **CVAT** → COCO JSON → Phase B (`publish_dataset.py`, DVC, MDS).
- Raw images and shards live in **S3**; training reads MDS via `params.yaml` (`mds_path`).

**Still in this folder**

- `setup_mongo_collections.py` — optional helper to create MongoDB collections used by Phase F drift jobs (if you still point drift checks at Atlas).
