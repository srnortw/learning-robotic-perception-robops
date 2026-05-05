"""
Phase D — Model Card Generator

Reads eval_results.json and params.yaml, fills model_card_template.md,
writes model_card.md, and logs it as an artifact to the MLflow run.

Triggered automatically on PR merge via GitHub Actions (or run manually).

Usage:
    python pipeline/phase_d/generate_model_card.py \\
        --results pipeline/phase_d/eval_results.json \\
        [--pr-url https://github.com/.../pull/42] \\
        [--reviewer "Your Name"] \\
        [--output pipeline/phase_d/model_card.md]
"""

import argparse
import json
import os
from datetime import date, datetime
from pathlib import Path

import mlflow
import yaml

TEMPLATE_PATH = Path(__file__).with_name("model_card_template.md")
PARAMS_PATH   = "pipeline/phase_c/detr/params.yaml"
MLFLOW_URI    = "https://dagshub.com/srnortw/learning-robotic-perception-robops.mlflow"
S3_BUCKET     = "my-perception-robops-data-2026-688567275774-eu-central-1-an"
DAGSHUB_REPO  = "srnortw/learning-robotic-perception-robops"


def load_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def setup_mlflow():
    os.environ.setdefault("MLFLOW_TRACKING_USERNAME",
                          os.environ.get("DAGSHUB_USERNAME", ""))
    os.environ.setdefault("MLFLOW_TRACKING_PASSWORD",
                          os.environ.get("DAGSHUB_TOKEN", ""))
    mlflow.set_tracking_uri(MLFLOW_URI)


def build_per_class_table(per_class: dict) -> str:
    lines = ["| Class | AP@50 |", "|---|---|"]
    for cls, ap in per_class.items():
        lines.append(f"| {cls} | `{ap:.4f}` |")
    return "\n".join(lines)


def build_champion_section(champion: dict | None, delta: float) -> str:
    if not champion:
        return (
            "This is Round 1 — no prior champion exists. "
            "Model is the first to be considered for Production."
        )
    regressions = [
        cls for cls, ap in champion.get("per_class_ap50", {}).items()
        if ap > 0
    ]
    return "\n".join([
        f"| Metric | Champion (v{champion['version']}) | Challenger | Delta |",
        "|---|---|---|---|",
        f"| mAP@50 | `{champion['map50']:.4f}` | `{champion['map50']+ delta:.4f}` | `{delta:+.4f}` |",
        f"| mAP@50:95 | `{champion['map50_95']:.4f}` | — | — |",
        f"| Latency p95 | `{champion['latency_p95_ms']:.0f} ms` | — | — |",
    ])


def fill_template(template: str, results: dict, params: dict,
                  reviewer: str, pr_url: str | None) -> str:
    challenger = results["challenger"]
    champion   = results.get("champion")
    run_id     = results["run_id"]
    dv         = results["dataset_version"]
    delta      = results.get("delta_map50", 0.0)

    mlflow_run_url = (
        f"https://dagshub.com/{DAGSHUB_REPO}.mlflow/#/experiments/0/runs/{run_id}"
    )
    s3_onnx_path = f"s3://{S3_BUCKET}/weights/detr/{dv}/model_int8.onnx"
    mds_path = params.get("dataset", {}).get("mds_path", "—")

    per_class = challenger.get("per_class_ap50", {})
    per_class_table = build_per_class_table(per_class) if per_class else "N/A"
    champion_section = build_champion_section(champion, delta)

    registry_info = f"`{results.get('model_name', 'detr-conditional-resnet50')}` v{results.get('model_version', '?')}"
    pr_link = f"[PR]({pr_url})" if pr_url else "—"

    return template.format(
        dataset_version      = dv,
        run_id_short         = run_id[:8],
        mlflow_run_url       = mlflow_run_url,
        s3_onnx_path         = s3_onnx_path,
        training_date        = datetime.utcnow().strftime("%Y-%m-%d"),
        mds_path             = mds_path,
        map50                = f"`{challenger['map50']:.4f}`",
        map50_95             = f"`{challenger['map50_95']:.4f}`",
        latency_p95_ms       = f"`{challenger['latency_p95_ms']:.0f}`",
        holdout_size         = "synthetic (20)" if not champion else "—",
        per_class_table      = per_class_table,
        champion_section     = champion_section,
        reviewer             = reviewer,
        approval_date        = date.today().isoformat(),
        pr_link              = pr_link,
        registry_version     = registry_info,
    )


def log_to_mlflow(run_id: str, card_path: str):
    setup_mlflow()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(card_path, artifact_path="model_card")
        mlflow.set_tag("model_card_generated", "true")
    print(f"Model card logged to MLflow run {run_id}")


def transition_to_production(results: dict):
    """Transition the MLflow registry entry from Staging → Production."""
    setup_mlflow()
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_URI)
    model_name = results.get("model_name", "detr-conditional-resnet50")
    version    = results.get("model_version")
    if not version:
        print("No model version in results — skipping registry transition.")
        return
    client.transition_model_version_stage(
        name=model_name, version=version, stage="Production",
        archive_existing_versions=True,
    )
    print(f"Model {model_name} v{version} → Production in MLflow Registry.")


def main():
    parser = argparse.ArgumentParser(description="Phase D — generate model card")
    parser.add_argument("--results",    default="pipeline/phase_d/eval_results.json")
    parser.add_argument("--pr-url",     default=None)
    parser.add_argument("--reviewer",   default=os.environ.get("GITHUB_ACTOR", "robops"))
    parser.add_argument("--output",     default="pipeline/phase_d/model_card.md")
    parser.add_argument("--promote",    action="store_true",
                        help="Also transition registry entry → Production")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    params = load_params()
    template = TEMPLATE_PATH.read_text()
    card = fill_template(template, results, params, args.reviewer, args.pr_url)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(card)
    print(f"Model card written to {out}")
    print(card)

    log_to_mlflow(results["run_id"], str(out))

    if args.promote:
        transition_to_production(results)


if __name__ == "__main__":
    main()
