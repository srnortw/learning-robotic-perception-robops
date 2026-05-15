"""
Single source of truth for class definitions.

All pipeline scripts and training code derive class names and num_classes
from pipeline/phase_b/label_schema.yaml.

To add or change classes:
    1. Edit pipeline/phase_b/label_schema.yaml
    2. Run:  python pipeline/utils/schema.py --sync
    This updates params.yaml (num_classes) and detr_params.yaml (class_names)
    automatically. Then commit both files.

Usage in Python:
    from pipeline.utils.schema import Schema
    names = Schema.class_names()   # ['person', 'chair', 'table', 'door']
    n     = Schema.num_classes()   # 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REPO_ROOT   = Path(__file__).parents[2]
SCHEMA_PATH = REPO_ROOT / "pipeline" / "phase_b" / "label_schema.yaml"
PARAMS_PATH = REPO_ROOT / "pipeline" / "phase_c" / "detr" / "params.yaml"
DETR_PARAMS = REPO_ROOT / "pipeline" / "phase_c" / "detr" / "detr_params.yaml"


class Schema:
    _cache: list[dict] | None = None

    @classmethod
    def _load(cls) -> list[dict]:
        if cls._cache is None:
            with open(SCHEMA_PATH) as f:
                data = yaml.safe_load(f)
            cls._cache = sorted(data["classes"], key=lambda c: c["id"])
        return cls._cache

    @classmethod
    def class_names(cls) -> list[str]:
        """Returns class names ordered by ID: ['person', 'chair', 'table', 'door']"""
        return [c["name"] for c in cls._load()]

    @classmethod
    def num_classes(cls) -> int:
        return len(cls._load())

    @classmethod
    def id_to_name(cls) -> dict[int, str]:
        return {c["id"]: c["name"] for c in cls._load()}

    @classmethod
    def name_to_id(cls) -> dict[str, int]:
        return {c["name"]: c["id"] for c in cls._load()}

    @classmethod
    def sync(cls, verbose: bool = True) -> None:
        """
        Propagate label_schema.yaml → params.yaml + detr_params.yaml.
        Call this after editing label_schema.yaml, or run as a script:
            python pipeline/utils/schema.py --sync
        """
        cls._cache = None  # invalidate cache so fresh load happens
        names = cls.class_names()
        n     = cls.num_classes()

        # ── Update pipeline/phase_c/detr/params.yaml ────────────────────────
        with open(PARAMS_PATH) as f:
            params = yaml.safe_load(f)

        old_n = params.get("dataset", {}).get("num_classes")
        params.setdefault("dataset", {})["num_classes"] = n

        with open(PARAMS_PATH, "w") as f:
            yaml.dump(params, f, default_flow_style=False, sort_keys=False)

        if verbose:
            changed = f"{old_n} → {n}" if old_n != n else f"{n} (unchanged)"
            print(f"[schema.sync] params.yaml          num_classes={changed}")

        # ── Update pipeline/phase_c/detr/detr_params.yaml ───────────────────
        with open(DETR_PARAMS) as f:
            rp = yaml.safe_load(f)

        old_names = rp.get("detr_node", {}).get("ros__parameters", {}).get("class_names")
        rp.setdefault("detr_node", {}).setdefault("ros__parameters", {})["class_names"] = names

        with open(DETR_PARAMS, "w") as f:
            yaml.dump(rp, f, default_flow_style=False, sort_keys=False)

        if verbose:
            changed = f"{old_names} → {names}" if old_names != names else f"{names} (unchanged)"
            print(f"[schema.sync] detr_params.yaml     class_names={changed}")

        if verbose:
            print(f"[schema.sync] Done. {n} classes: {names}")

        # ── Warn if drift baselines have stale class keys ────────────────────
        baseline_dir = REPO_ROOT / "data" / "drift_baselines"
        if baseline_dir.exists():
            import json
            for bf in baseline_dir.glob("*.json"):
                try:
                    with open(bf) as f:
                        bl = json.load(f)
                    bl_keys  = set(bl.get("class_distribution", {}).keys())
                    new_keys = set(names)
                    if bl_keys and bl_keys != new_keys:
                        print(
                            f"[schema.sync] WARNING: {bf.name} class_distribution keys "
                            f"{sorted(bl_keys)} don't match schema {sorted(new_keys)}. "
                            f"Update data/drift_baselines/{bf.name} with real production data."
                        )
                except Exception:
                    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync label_schema.yaml → params.yaml + detr_params.yaml")
    parser.add_argument("--sync", action="store_true", help="Write changes to disk")
    parser.add_argument("--show", action="store_true", help="Print current schema (no writes)")
    args = parser.parse_args()

    if args.sync:
        Schema.sync()
    else:
        print(f"Classes ({Schema.num_classes()}):")
        for cid, name in Schema.id_to_name().items():
            print(f"  {cid}: {name}")
        print("\nRun with --sync to propagate to params.yaml + detr_params.yaml")
