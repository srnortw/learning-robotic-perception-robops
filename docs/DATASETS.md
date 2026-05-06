# Datasets Guide

This pipeline is dataset-agnostic. Any COCO-format dataset from Roboflow (or annotated in CVAT and exported as COCO) can feed Phase B → C as long as the class names are mapped to `label_schema.yaml`.

---

## Current Schema — Round 1 (DETR)

Defined in `pipeline/phase_b/label_schema.yaml`:

| ID | Class | Notes |
|---|---|---|
| 0 | person | Indoor occupant detection |
| 1 | chair | Seating |
| 2 | table | Surfaces (dining, desk, coffee) |
| 3 | door | Interior + exterior doors |

Keep this schema consistent across all architecture rounds (DETR → RT-DETR → Mask2Former → SAM 2) so model performance is **directly comparable** on the same classes.

---

## Recommended Roboflow Datasets

No single public dataset perfectly matches all 4 classes out of the box. Below are the best fits and how to use them.

### Option 1 — COCO 2017 Subset (Recommended for Round 1)

**What it is:** The classic COCO dataset, filtered and re-hosted on Roboflow. Covers person, chair, and dining table with ~100k images. Best class coverage for this schema.

**Workspace/Project:** `roboflow-universe-projects / coco-2017`

**Class mismatch:** COCO calls it `dining table`, not `table`. Use `--remap` to fix:

```bash
python pipeline/phase_b/roboflow_download.py \
    --api-key YOUR_KEY \
    --workspace roboflow-universe-projects \
    --project coco-2017 \
    --version 1 \
    --remap "dining table:table"
```

> **Note:** COCO does not have a `door` class. You can either skip it for Round 1 (set `num_classes: 3` in `params.yaml`) or supplement with a door-specific dataset.

---

### Option 2 — Indoor Object Detection (Has Door)

**What it is:** ~1,200 indoor images with `chair`, `door`, `table`, `couch`, `cabinet`, `window`. Good for door class coverage.

**Workspace/Project:** `master-thesis-lzxhq / indoor-2024` *(verify slug on Roboflow Universe)*

```bash
python pipeline/phase_b/roboflow_download.py \
    --api-key YOUR_KEY \
    --workspace master-thesis-lzxhq \
    --project indoor-2024 \
    --version 1
```

> **Note:** No `person` class. Best combined with Option 1 or Phase A live data.

---

### Option 3 — Fork + Merge on Roboflow (Best Long-Term)

1. Fork both datasets above into your Roboflow workspace
2. Merge them using Roboflow's **Dataset Merge** feature
3. Add any missing classes (annotate `door` on COCO frames, `person` on indoor frames)
4. Export as **COCO JSON** and download:

```bash
python pipeline/phase_b/roboflow_download.py \
    --api-key YOUR_KEY \
    --workspace YOUR_WORKSPACE \
    --project YOUR_MERGED_PROJECT \
    --version 1
```

This gives you all 4 classes in one clean dataset — the recommended approach for a real deployment.

---

## How to Get Your API Key

1. Go to [https://roboflow.com](https://roboflow.com) → Sign up (free)
2. Settings → Roboflow API → copy your **Private API Key**
3. Pass it via `--api-key` or set `ROBOFLOW_API_KEY` as an environment variable / GitHub Secret

---

## Class Remapping Reference

The `--remap` flag maps dataset class names → your `label_schema.yaml` names. Classes not in the schema are silently dropped (and the images with only those classes are removed too).

```bash
# Example: dataset uses "Dining Table" and "Human"
--remap "Dining Table:table" "Human:person" "dining table:table"
```

Run with `--dry-run` to see what survives the filter before writing any files:

```bash
python pipeline/phase_b/roboflow_download.py \
    --api-key YOUR_KEY \
    --workspace ... \
    --project  ... \
    --version  1 \
    --remap "dining table:table" \
    --dry-run
```

---

## Full Phase B Workflow with Roboflow

```bash
# 1. Download + remap
python pipeline/phase_b/roboflow_download.py \
    --api-key YOUR_KEY \
    --workspace YOUR_WORKSPACE \
    --project   YOUR_PROJECT \
    --version   1 \
    --remap "dining table:table"

# 2. (Optional) Visual QA in FiftyOne before converting
python pipeline/phase_b/fiftyone_qa.py

# 3. Convert to MDS shards → S3 (increments dataset version)
python pipeline/phase_b/convert_to_mds.py \
    --version v2 \
    --coco-json data/detr/raw/annotations.json

# 4. Commit params.yaml to trigger training CI
git add pipeline/phase_c/detr/params.yaml
git commit -m "data: bump mds_path to detr v2 — real Roboflow dataset"
git push
```

---

## Adding a Different Dataset for a Different Model (Round 2+)

The pipeline is designed to be **round-agnostic**. For each new model architecture:

1. **Create a new schema** — copy `label_schema.yaml` to `pipeline/phase_b/label_schema_rt_detr.yaml` (or whatever the round is) with new classes if needed
2. **Create a new params.yaml** — copy `pipeline/phase_c/detr/params.yaml` to `pipeline/phase_c/rt_detr/params.yaml`, update `architecture`, `num_classes`, `model_name`
3. **Download a new dataset** — use `roboflow_download.py` with the new schema path (add a `--schema` arg or just update `label_schema.yaml` for that round)
4. **Convert to MDS** — same `convert_to_mds.py` script, different `--version` (e.g. `v1-rt-detr`)
5. Everything else (Phase D eval, Phase E deploy, Phase F monitoring) **stays identical** — just pointing at the new params.yaml

### Example Dataset Ideas Per Round

| Round | Model | Suggested Dataset Type |
|---|---|---|
| 1 | DETR | COCO 2017 (person, chair, table) + indoor dataset (door) |
| 2 | RT-DETR | Same as Round 1 — compare performance directly |
| 3 | Mask2Former | ADE20K or COCO panoptic (segmentation masks needed) |
| 4 | SAM 2 | Any COCO dataset — SAM 2 is promptable, zero-shot |
| 5 | VLM | VQA datasets (COCO Captions, GQA) — needs text+image pairs |

The key design principle: **the A→F pipeline skeleton never changes between rounds. Only the model slot and its dataset format adapter change.**
