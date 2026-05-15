# Next steps — resume when you are ready

---

## Current status (2026-05-15)

- **CI run succeeded:** [Actions run 25922972322](https://github.com/srnortw/learning-robotic-perception-robops/actions/runs/25922972322)
  - Export ONNX + INT8 → S3 ✓
  - Training image → ECR ✓
  - Phase D eval ✓ (fix: `latest` run ID resolved in `eval_champion_challenger.py`)
- **Open approval PR:** [#26 — Promote DETR v2](https://github.com/srnortw/learning-robotic-perception-robops/pull/26) (mAP@50 **0.472**, delta **+0.472**)

**Your move:** merge PR #26 when satisfied → **Generate Model Card** job runs on `main`.

---

## 1. Confirm the last GitHub Actions run (done)

Latest MLOps workflow completed green. Re-run only if you change weights or want a new Colab run ID pinned in eval.

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
