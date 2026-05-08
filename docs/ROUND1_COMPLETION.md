# Round 1 (DETR) — Completion Checklist

Use this when you want to call the **A→F loop “done”** for personal RoboOps Round 1.

## Already in place

- **Training / eval:** Phase C + D, holdout mAP, `label_schema.yaml` as class source of truth.
- **Edge image:** Single **`robops/ros2-full-stack`** container (`robops-stack`): `camera_node` + `detr_node` + `monitoring_node`, **CycloneDDS**, `--network host`.
- **Camera path:** Host **`robops-camera-bridge`** (picamera2) → UDP JPEG → `camera_node` in Docker (avoids Pi OS vs Ubuntu Docker camera stack conflicts).
- **CI/CD:** `ci_deploy.yml` builds full-stack arm64, publishes Greengrass **`com.robops.stack`**, Phase E health check, Phase F canary window.

## One-time / optional follow-ups

### 1. Pi 3B+ resources (expect this)

- **1 GB RAM + swap:** Keep **one** stack running; avoid parallel `docker pull` while the stack runs.
- **SD card:** Keep **≥1–2 GB** free on `/` before pulls/upgrades (`df -h /`). Prune unused images when tight.

### 2. MongoDB on the device (optional)

Telemetry is **off** until `MONGO_URI` is a real URI inside the container.

- The **`com.robops.stack`** recipe uses **component configuration** `mongoUri` (default empty).
- In **AWS IoT Core → Greengrass → Components →** your deployment / thing group, set **`com.robops.stack`** configuration merge, e.g.:

  ```json
  {
    "mongoUri": "mongodb+srv://USER:PASS@cluster.mongodb.net/robops?retryWrites=true&w=majority"
  }
  ```

  Prefer **AWS Secrets Manager + Greengrass secret reference** for production instead of pasting the URI in the console.

### 3. CSI camera reliability

If `libcamera` reports **camera frontend timeout**, treat it as **hardware first** (ribbon, connector, module). The bridge and ROS stack can be correct while the sensor path is flaky.

### 4. Declare “done”

You are done with Round 1 when you are satisfied with:

1. One **full-stack** container running on the canary Pi.  
2. **Correct five classes** on `detr_node`.  
3. **Camera bridge + stack** verified when hardware is stable.  
4. (Optional) **Mongo** configured if you care about Phase F dashboards.

Next architecture round (RT-DETR, etc.) reuses the same pipeline skeleton.
