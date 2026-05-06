# DETR Model Card — v2

## Model

| Field | Value |
|---|---|
| Architecture | Conditional DETR ResNet-50 |
| HuggingFace ID | microsoft/conditional-detr-resnet-50 |
| Export format | ONNX INT8 (dynamic quantization) |
| Target hardware | Raspberry Pi 3B+ (arm64 CPU) |
| MLflow run | [15548dfd](https://dagshub.com/srnortw/learning-robotic-perception-robops.mlflow/#/experiments/0/runs/15548dfdffdd4098bed0fad2510d2da6) |
| S3 weights | `s3://my-perception-robops-data-2026-688567275774-eu-central-1-an/weights/detr/v2/model_int8.onnx` |
| Dataset version | v2 |
| Training date | 2026-05-06 |

## Training Data

| Field | Value |
|---|---|
| Classes | person, chair, table, door |
| Format | COCO JSON → MDS shards (S3) |
| MDS path | `s3://my-perception-robops-data-2026-688567275774-eu-central-1-an/mds/detr/v2/` |
| Train split | 85% |
| Val split | 15% |
| Holdout split | Separate — never used in training |

## Performance on Hold-out Set

| Metric | Value |
|---|---|
| mAP@50 | `0.4723` |
| mAP@50:95 | `0.2292` |
| Latency p95 | `2201` ms (CPU, ONNX INT8) |
| Holdout size | 90 images |

### Per-class AP@50

| Class | AP@50 |
|---|---|
| helmet | `0.6154` |
| no-helmet | `0.2682` |
| no-vest | `0.0000` |
| person | `0.8350` |
| vest | `0.6428` |

## Comparison vs Champion

This is Round 1 — no prior champion exists. Model is the first to be considered for Production.

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
| Reviewer | srnortw |
| Approval date | 2026-05-06 |
| GitHub PR | [PR](https://github.com/srnortw/learning-robotic-perception-robops/pull/13) |
| MLflow Registry | `detr-conditional-resnet50` v3 |
