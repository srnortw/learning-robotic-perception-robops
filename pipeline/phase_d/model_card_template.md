# DETR Model Card — {dataset_version}

## Model

| Field | Value |
|---|---|
| Architecture | Conditional DETR ResNet-50 |
| HuggingFace ID | microsoft/conditional-detr-resnet-50 |
| Export format | ONNX INT8 (dynamic quantization) |
| Target hardware | Raspberry Pi 3B+ (arm64 CPU) |
| MLflow run | [{run_id_short}]({mlflow_run_url}) |
| S3 weights | `{s3_onnx_path}` |
| Dataset version | {dataset_version} |
| Training date | {training_date} |

## Training Data

| Field | Value |
|---|---|
| Classes | person, chair, table, door |
| Format | COCO JSON → MDS shards (S3) |
| MDS path | `{mds_path}` |
| Train split | 85% |
| Val split | 15% |
| Holdout split | Separate — never used in training |

## Performance on Hold-out Set

| Metric | Value |
|---|---|
| mAP@50 | {map50} |
| mAP@50:95 | {map50_95} |
| Latency p95 | {latency_p95_ms} ms (CPU, ONNX INT8) |
| Holdout size | {holdout_size} images |

### Per-class AP@50

{per_class_table}

## Comparison vs Champion

{champion_section}

## Known Failure Modes

- Performance degrades in low-light conditions (< 50 lux)
- Door class degrades with partial occlusion (> 50% occluded)
- Small objects (< 32×32 px) may be missed
- No performance guarantee outside the robot's indoor environment

## Approved Use Cases

- Shadow mode logging on indoor wheeled robot
- **Not approved** for actuation or safety-critical decisions (shadow mode only)

## Approval

| Field | Value |
|---|---|
| Reviewer | {reviewer} |
| Approval date | {approval_date} |
| GitHub PR | {pr_link} |
| MLflow Registry | {registry_version} |
