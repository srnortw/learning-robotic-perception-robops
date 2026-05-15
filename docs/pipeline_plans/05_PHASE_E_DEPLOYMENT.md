# Phase E — Edge deployment (out of scope)

**This repository no longer includes** Raspberry Pi / ROS2 / Docker edge stacks, **AWS IoT Greengrass** recipes, or CI jobs that deploy to devices.

Artifacts produced here and suitable for a **future** edge stack:

- `s3://…/weights/detr/<version>/model_int8.onnx` — INT8 ONNX (from `export_onnx.py` + CI).
- `688567275774.dkr.ecr.eu-central-1.amazonaws.com/robops/training:<version>` — **amd64** training / export environment only.

To deploy elsewhere, create a separate project (or fork) that owns Greengrass components, device IAM, and arm64 inference images.

---

## Historical note

Earlier iterations of this monorepo documented Greengrass v2, `com.robops.stack`, ECR `ros2-full-stack`, SSH health checks, and canary promotion. Those paths were removed when the project was narrowed to **MLOps only**.
