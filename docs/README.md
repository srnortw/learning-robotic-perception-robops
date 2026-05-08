# RoboOps + MLOps Pipeline — Documentation

Personal end-to-end robotic perception pipeline. Architecture-agnostic — each model (DETR, RT-DETR, Mask2Former, SAM2, VLM) is a swappable slot in the same A–F pipeline skeleton.

**Edge target:** Raspberry Pi 3B+ (arm64, 1GB RAM)
**ROS2:** Jazzy (Ubuntu 24.04)
**Training:** Google Colab Pro+
**CI/CD:** GitHub Actions (self-hosted runner on workstation)
**Registry:** AWS ECR (**training** amd64; **ros2-full-stack** arm64 for Pi — legacy `inference` / `ros2-stack` images optional)
**Storage:** AWS S3 (images + weights) + MongoDB Atlas (telemetry)
**Experiment tracking:** DagsHub (MLflow + DVC, free tier)

---

## Pipeline Overview

```
Phase A → Phase B → Phase C → Phase D → Phase E → Phase F
  ↑                                                    |
  └────────────────── drift detected ─────────────────┘
```

| Phase | Name | What happens |
|---|---|---|
| A | Data Ingestion | Camera node + DETR shadow mode on Pi → S3 + MongoDB |
| B | Data Engineering | Drift gate → DVC pull → FiftyOne QA → CVAT → DVC version → MDS convert → S3 shards |
| C | Training | Colab streams MDS shards from S3 via StreamingDataset → MLflow on DagsHub → ONNX INT8 export |
| D | Human Audit | Champion-challenger eval → Discord notify → FiftyOne audit → model card |
| E | Deployment | docker buildx arm64 → ECR → Greengrass **`com.robops.stack`** canary → fleet |
| F | Monitoring | Metrics from Pi → Evidently AI drift → retrain trigger → loop back to A |

---

## Plan Files

- [01_PHASE_A_DATA_INGESTION](pipeline_plans/01_PHASE_A_DATA_INGESTION.md) — camera node, shadow mode, S3, MongoDB
- [02_PHASE_B_DATA_ENGINEERING](pipeline_plans/02_PHASE_B_DATA_ENGINEERING.md) — drift gate, DVC pull, FiftyOne QA, CVAT, MDS conversion, params.yaml
- [03_PHASE_C_TRAINING_DETR](pipeline_plans/03_PHASE_C_TRAINING_DETR.md) — Colab streams MDS from S3, ONNX INT8 export, MLflow, GitHub Actions
- [04_PHASE_D_HUMAN_AUDIT](pipeline_plans/04_PHASE_D_HUMAN_AUDIT.md) — champion-challenger, threshold policy, model card
- [05_PHASE_E_DEPLOYMENT](pipeline_plans/05_PHASE_E_DEPLOYMENT.md) — ECR, docker buildx arm64, Greengrass, canary, rollback
- [ROUND1_COMPLETION](ROUND1_COMPLETION.md) — Pi 3B+ constraints, Mongo, camera bridge, “definition of done”
- [06_PHASE_F_MONITORING](pipeline_plans/06_PHASE_F_MONITORING.md) — monitoring node, Evidently AI drift detection, retrain loop
- [DATASETS](DATASETS.md) — Roboflow dataset guide, class remapping, multi-round dataset strategy

---

## Learning Order (Architecture Rounds)

| Round | Model | Key concept |
|---|---|---|
| 1 (current) | Conditional DETR | Transformer detection, Hungarian matching |
| 2 | RT-DETR | Real-time transformers |
| 3 | Mask2Former | Panoptic segmentation |
| 4 | SAM 2 | Promptable zero-shot segmentation |
| 5 | VLM | Scene-level language grounding (workstation only) |

---

## Key Data Flow: Raw Images → MDS → Training

```
Phase A                  Phase B                        Phase C
S3 raw images  →  CVAT annotated COCO JSON  →  MDS shards in S3  →  Colab StreamingDataset
(dvc-cache/)      (dvc add → data.dvc)          (mds/detr/vN/)        (no full download)
```

- **DVC** tracks raw images and holdout set (pointers in `data/detr/data.dvc`, `holdout.dvc`)
- **MDS** (`mosaicml-streaming`) converts annotated COCO JSON → streaming shards at `s3://bucket/mds/detr/vN/`
- **Phase C never calls `dvc pull`** for training data — it reads `mds_path` from `params.yaml` and streams
- Updating `params.yaml` with a new `mds_path` version is what triggers the GitHub Actions CI workflow

---

## Infrastructure Setup Order

1. Create GitHub repo, initialize DVC
2. Create DagsHub project, connect repo (free MLflow + DVC remote)
3. Create AWS S3 bucket (`your-bucket`)
4. Create AWS ECR repos: `training` (amd64), `ros2-full-stack` (arm64, required for Pi); optional legacy `inference`, `ros2-stack`
5. Create MongoDB Atlas free cluster, collections: `telemetry`, `production_metrics`, `retrain_queue`, `drift_events`
6. Set up GitHub Secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `MONGO_URI`, `DISCORD_WEBHOOK_URL`, `GITHUB_TOKEN`
7. Install Greengrass v2 on Pi 3B+ (64-bit OS required)
8. Set up self-hosted GitHub Actions runner on workstation
9. Enable `docker buildx` with QEMU on workstation runner

---

## Important Hardware Notes

- Pi 3B+: 1GB RAM, no GPU, arm64. Run **one** edge container (`robops-stack` / full-stack image); two large stacks will exhaust RAM/swap and stall SSH.
- DETR ONNX INT8 inference is roughly **multi-second per frame** on 3B+ — expected for a demo; not real-time “video rate” inference.
- Pi OS must be 64-bit: `uname -m` must return `aarch64` before Greengrass install.
- Docker images for Pi must be built with `--platform linux/arm64`.
- Use `arm64v8/ros:jazzy-ros-base` base image (not desktop variant — too large).
- **Camera:** CSI frames enter Docker via **host** `pipeline/phase_e/camera_bridge.py` + systemd (`robops-camera-bridge.service`), not V4L2 inside the Ubuntu container.
