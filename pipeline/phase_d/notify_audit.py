"""
Phase D — Human Notification

Reads eval_results.json and sends a formatted Discord webhook message
with the full metric comparison and a link to the GitHub PR.

Usage:
    python pipeline/phase_d/notify_audit.py \\
        --results pipeline/phase_d/eval_results.json \\
        [--pr-url https://github.com/.../pull/42]
"""

import argparse
import json
import os
import sys

import requests

DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")

CLASSES = ["person", "chair", "table", "door"]


def delta_emoji(val: float, cls: str | None = None) -> str:
    if cls and cls in ("person", "door") and val < 0:
        return "⚠️ CRITICAL REGRESSION"
    if val > 0:
        return "✅"
    if val < -0.001:
        return "⚠️"
    return "➖"


def build_message(results: dict, pr_url: str | None = None) -> dict:
    challenger = results["challenger"]
    champion   = results.get("champion")
    delta      = results["delta_map50"]
    dv         = results["dataset_version"]
    run_id     = results["run_id"]
    regressions = results.get("critical_regressions", [])
    approved    = results.get("auto_approved", False)

    header = f"**[DETR Audit Request] {dv} — Model Review Needed**"
    status = "✅ Auto-approved (Round 1 — no prior champion)" if (approved and not champion) \
        else ("✅ Improvement detected — review recommended" if delta > 0 else
              "⚠️ Regression or no improvement — review required")

    champ_line = (
        f"Champion (v{champion['version']}): mAP@50 = `{champion['map50']:.4f}`"
        if champion else "Champion: None (Round 1 — first ever model)"
    )

    per_class_lines = []
    champ_per_class = champion.get("per_class_ap50", {}) if champion else {}
    for cls, ap in challenger.get("per_class_ap50", {}).items():
        champ_ap = champ_per_class.get(cls)
        d = (ap - champ_ap) if champ_ap is not None else None
        delta_str = f" ({d:+.4f} {delta_emoji(d, cls)})" if d is not None else ""
        per_class_lines.append(f"  {cls:<8}: `{ap:.4f}`{delta_str}")

    reg_section = ""
    if regressions:
        reg_section = f"\n⚠️ **Critical class regressions: {', '.join(regressions)}**\n"

    pr_section = f"\n🔗 **PR:** {pr_url}" if pr_url else \
                 "\n📋 Run with `--create-pr` to open an approval PR."

    mlflow_url = (
        f"https://dagshub.com/srnortw/learning-robotic-perception-robops.mlflow"
        f"/#/experiments/0/runs/{run_id}"
    )

    description = "\n".join([
        f"{status}",
        "",
        f"**Challenger ({dv})**: mAP@50 = `{challenger['map50']:.4f}` | "
        f"mAP@50:95 = `{challenger['map50_95']:.4f}` | "
        f"Latency p95 = `{challenger['latency_p95_ms']:.0f} ms`",
        champ_line,
        f"Delta mAP@50: `{delta:+.4f}`",
        "",
        "**Per-class AP@50:**",
        *per_class_lines,
        reg_section,
        f"🔬 [MLflow Run]({mlflow_url})",
        pr_section,
        "",
        "**Action:** Review FiftyOne audit then approve or close the PR.",
    ])

    return {
        "username": "RoboOps MLOps",
        "embeds": [{
            "title": header,
            "description": description,
            "color": 0x2ecc71 if (approved and not regressions) else 0xe74c3c,
            "footer": {"text": f"Run ID: {run_id[:16]}..."},
        }]
    }


def send_discord(payload: dict, webhook_url: str):
    resp = requests.post(webhook_url, json=payload, timeout=10)
    resp.raise_for_status()
    print("Discord notification sent.")


def main():
    parser = argparse.ArgumentParser(description="Phase D — notify audit via Discord")
    parser.add_argument("--results", default="pipeline/phase_d/eval_results.json")
    parser.add_argument("--pr-url",  default=None, help="GitHub PR URL")
    parser.add_argument("--webhook", default=DISCORD_WEBHOOK,
                        help="Discord webhook URL (or set DISCORD_WEBHOOK_URL env var)")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    payload = build_message(results, args.pr_url)

    if not args.webhook:
        print("No Discord webhook configured — printing message instead:\n")
        print(json.dumps(payload, indent=2))
        print("\nSet DISCORD_WEBHOOK_URL env var or pass --webhook to send.")
        return

    send_discord(payload, args.webhook)


if __name__ == "__main__":
    main()
