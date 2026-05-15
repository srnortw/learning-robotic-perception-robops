"""
Phase D — Open GitHub Approval PR

Reads eval_results.json and creates a GitHub PR using the gh CLI.
Called by GitHub Actions after champion-challenger evaluation.

Usage:
    python pipeline/phase_d/open_pr.py [--results pipeline/phase_d/eval_results.json]
"""

import argparse
import json
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="pipeline/phase_d/eval_results.json")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    dv     = results["dataset_version"]
    delta  = results["delta_map50"]
    map50  = results["challenger"]["map50"]
    regs   = results["critical_regressions"]
    run_id = results["run_id"]

    per_cls_rows = "\n".join(
        f"| {cls} | `{ap:.4f}` |"
        for cls, ap in results["challenger"].get("per_class_ap50", {}).items()
    )
    champ = results.get("champion")
    champ_line = (
        f"Champion (v{champ['version']}): mAP\\@50 = `{champ['map50']:.4f}`"
        if champ else "No champion — Round 1 (first ever model)"
    )
    reg_line = (
        f"WARNING Critical regressions: {', '.join(regs)}"
        if regs else "OK No critical class regressions"
    )

    title = f"Promote DETR {dv} — mAP@50 {map50:.3f} (delta {delta:+.3f})"
    body = (
        f"## DETR Model Audit - {dv}\n\n"
        f"| Metric | Value |\n"
        f"|---|---|\n"
        f"| Challenger mAP@50 | `{map50:.4f}` |\n"
        f"| Challenger mAP@50:95 | `{results['challenger']['map50_95']:.4f}` |\n"
        f"| Latency p95 | `{results['challenger']['latency_p95_ms']:.0f} ms` |\n"
        f"| Delta mAP@50 | `{delta:+.4f}` |\n\n"
        f"{champ_line}\n\n"
        f"### Per-class AP@50\n"
        f"| Class | AP |\n"
        f"|---|---|\n"
        f"{per_cls_rows}\n\n"
        f"{reg_line}\n\n"
        f"### MLflow Run\n"
        f"https://dagshub.com/srnortw/learning-robotic-perception-robops.mlflow"
        f"/#/experiments/0/runs/{run_id}\n\n"
        f"### Action Required\n"
        f"1. Run `python pipeline/phase_d/fiftyone_audit.py` locally\n"
        f"2. **Merge this PR** to record the audit on `main` and generate the model card (MLOps Round 1 gate)\n"
        f"3. **Close PR** to keep champion unchanged\n"
    )

    result = subprocess.run(
        ["gh", "pr", "create",
         "--title", title,
         "--body", body,
         "--base", "main"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        pr_url = result.stdout.strip()
        print(f"PR created: {pr_url}")
    else:
        print(f"PR creation failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
