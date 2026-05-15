# Next steps — resume when you are ready

Use this checklist **tomorrow** or whenever you continue. No rush.

---

## 1. Confirm the last GitHub Actions run

1. Open: `https://github.com/srnortw/learning-robotic-perception-robops/actions`
2. Open the latest **“CI — DETR MLOps (export, train image, eval, model card)”** run (manual dispatch with `latest`).
3. Check:
   - **Export ONNX + INT8 Quantize** — success (S3 should have `weights/detr/v2/model_int8.onnx`).
   - **Build Training Docker Image → ECR** — success (image pushed to `…/robops/training:v2` and `:latest`).

If **export** failed: open the job log; common issues are MLflow `latest` resolution, DagsHub secrets, or torch/onnx version drift. Fix in repo or re-run with an explicit Colab **run ID**.

If **Docker push** failed: confirm `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` in GitHub secrets and that repo `robops/training` still exists in ECR.

---

## 2. Optional: full Phase D loop with a real run ID

After a Colab training run:

1. Actions → same workflow → **Run workflow**.
2. Set **`model_run_id`** to the **exact** MLflow run ID from Colab (not only `latest` if you want a pinned eval).
3. That should run **Phase D — Evaluate + Open Approval PR** (when `model_run_id` is non-empty).
4. Review PR → merge → **Generate Model Card** job runs on merge.

---

## 3. Optional: AWS / GitHub hygiene

- **`docs/OPTIONAL_AWS_CLEANUP.md`** — IoT Greengrass console cleanup if old deployments/components remain.
- **GitHub → Settings → Secrets:** remove `PI_HOST` and any SSH/deploy secrets you no longer use.

---

## 4. Optional: Phase B or drift

- **Phase B workflow** — Roboflow → MDS when you ingest a new dataset version.
- **`drift_check.yml`** — needs a **self-hosted** runner and `MONGO_URI` if you still want scheduled drift on `production_metrics` (often empty without edge telemetry).

---

## 5. If you add edge inference again later

Use a **separate repository** (or fork) for ROS2 / device Docker / Greengrass. This repo stays the **MLOps** source of truth: weights on S3, eval, model card, training image on ECR.

---

## Related docs

- `docs/README.md` — pipeline overview  
- `docs/SESSION_DONE_2026-05-15.md` — what was completed in the last session  
- `docs/ROUND1_COMPLETION.md` — MLOps “done” definition  
- `docs/OPTIONAL_AWS_CLEANUP.md` — extra AWS cleanup commands  
