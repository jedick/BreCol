#!/usr/bin/env python3
"""
Build Table 9 (SetBERT ablations) as HTML under manuscript/table9_setbert.html.

Reads ``experiments.yaml`` ``train_setbert.experiments`` and aggregates
``tuning.best_epoch`` plus ``metrics.{test,holdout}_auc`` from per-seed
JSON under ``results/setbert/`` (filenames ``{cd|ct}_{name}_s{seed}.json``,
as written by ``scripts/train_setbert.py``). JSON files in
``results/setbert/`` that are not referenced by ``experiments.yaml`` are
ignored.

Each ablation name appears once per task in ``experiments.yaml``. The table
emits one row per unique ablation name (YAML-first-appearance order),
combining cancer_diagnosis (``cd_*``) and cancer_type (``ct_*``) results.

Each task contributes three columns: best epoch (mean across seeds),
test AUC, and holdout AUC. Test/holdout cells render as ``mean \u00b1 sd``
across the random seeds listed in ``experiments.yaml`` (sample standard
deviation; the ``\u00b1 sd`` part is omitted when only one seed is available).

Run from the repository root: ``python helpers/table9_setbert.py``
"""

from __future__ import annotations

import html
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml


# Plain-English row labels keyed by ``train_setbert.experiments[].name``.
# Row order in the rendered table follows the YAML-first-appearance order
# of each unique name across the cancer_type and cancer_diagnosis sections.
ABLATION_DESCRIPTIONS: Dict[str, str] = {
    "head_linear": "Linear classification head (default)",
    "head_cosine": "Cosine-similarity classification head",
    "cosine_random": "Cosine head with random weight initialization",
}

TASK_ABBRV: Dict[str, str] = {
    "cancer_diagnosis": "cd",
    "cancer_type": "ct",
}

METRIC_KEYS: Tuple[str, ...] = ("test_auc", "holdout_auc")

DECIMALS = 2
EM_DASH = "\u2014"
PLUS_MINUS = "\u00b1"
OUTPUT_REL = Path("manuscript") / "table9_setbert.html"


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _parse_seed_spec(raw: object, default_seed: int) -> List[int]:
    if raw is None:
        return [default_seed]
    if isinstance(raw, int):
        return [raw]
    if isinstance(raw, (list, tuple)):
        try:
            return [int(x) for x in raw]
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"random_seed must be a list of integers; got {raw!r}."
            ) from exc
    raise SystemExit(
        "random_seed must be an integer or a list of integers; "
        f"got {type(raw).__name__}."
    )


def _experiment_rows(repo_root: Path) -> List[Tuple[str, str, int]]:
    """Return [(name, task, seed), ...] expanded from experiments.yaml."""
    base_cfg = _load_yaml(repo_root / "defaults.yaml").get("train_setbert") or {}
    if not isinstance(base_cfg, dict):
        raise SystemExit("defaults.yaml must define train_setbert as a mapping.")
    default_seed = int(base_cfg.get("random_seed", 0))

    exp_cfg = _load_yaml(repo_root / "experiments.yaml").get("train_setbert") or {}
    rows = exp_cfg.get("experiments") or []
    if not isinstance(rows, list) or not rows:
        raise SystemExit(
            "experiments.yaml train_setbert.experiments must be a non-empty list."
        )

    out: List[Tuple[str, str, int]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SystemExit(f"experiments.yaml row {idx + 1} is not a mapping.")
        name = str(row.get("name") or "").strip()
        overrides = row.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise SystemExit(f"{name}: overrides must be a mapping.")
        task = str(overrides.get("task") or base_cfg.get("task") or "").strip()
        if task not in TASK_ABBRV:
            raise SystemExit(
                f"experiments.yaml row {idx + 1} ({name!r}): "
                f"unsupported task {task!r} (expected one of {sorted(TASK_ABBRV)})."
            )
        if name not in ABLATION_DESCRIPTIONS:
            raise SystemExit(
                f"experiments.yaml row {idx + 1}: unknown experiment name {name!r}; "
                "add a description to ABLATION_DESCRIPTIONS in helpers/table9_setbert.py."
            )
        for seed in _parse_seed_spec(overrides.get("random_seed"), default_seed):
            out.append((name, task, seed))
    return out


def _read_metrics(path: Path) -> Dict[str, float]:
    """Return {best_epoch, test_auc, holdout_auc} from one JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a JSON object.")
    tuning = data.get("tuning") or {}
    best_epoch = tuning.get("best_epoch")
    if not isinstance(best_epoch, int):
        raise SystemExit(f"{path}: missing or non-integer tuning.best_epoch.")
    metrics = data.get("metrics") or {}
    out: Dict[str, float] = {"best_epoch": float(best_epoch)}
    for key in METRIC_KEYS:
        v = metrics.get(key)
        if v is None:
            raise SystemExit(f"{path}: missing metrics.{key}.")
        try:
            fv = float(v)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"{path}: metrics.{key} is not numeric ({exc}).") from exc
        if not math.isfinite(fv):
            raise SystemExit(f"{path}: metrics.{key} is non-finite ({fv}).")
        out[key] = fv
    return out


def _collect_results(
    setbert_dir: Path,
    experiment_rows: List[Tuple[str, str, int]],
) -> Tuple[List[str], Dict[Tuple[str, str], Dict[str, List[float]]]]:
    """Return (ordered ablation names, {(name, task) -> {metric -> [per-seed values]}})."""
    ordered_names: List[str] = []
    seen: Set[str] = set()
    buckets: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    missing: List[str] = []
    repo_root = setbert_dir.parent.parent
    for name, task, seed in experiment_rows:
        if name not in seen:
            seen.add(name)
            ordered_names.append(name)
        path = setbert_dir / f"{TASK_ABBRV[task]}_{name}_s{seed}.json"
        if not path.is_file():
            missing.append(str(path.relative_to(repo_root)))
            continue
        m = _read_metrics(path)
        bucket = buckets.setdefault((name, task), {})
        for k, v in m.items():
            bucket.setdefault(k, []).append(v)
    if missing:
        bullet = "\n".join(f"  - {p}" for p in missing)
        raise SystemExit("Error: Missing results files:\n" + bullet)
    return ordered_names, buckets


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _sd(values: List[float]) -> Optional[float]:
    """Sample standard deviation; ``None`` when fewer than two values."""
    if len(values) < 2:
        return None
    return statistics.stdev(values)


def _fmt_mean_sd(values: List[float], *, decimals: int) -> str:
    """Render ``mean ± sd`` (or just ``mean`` for a single seed)."""
    if not values:
        return EM_DASH
    mean = _mean(values)
    sd = _sd(values)
    if sd is None:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} {PLUS_MINUS} {sd:.{decimals}f}"


def _fmt_epoch(values: List[float]) -> str:
    v = _mean(values)
    if v is None:
        return EM_DASH
    return str(int(v)) if float(v).is_integer() else f"{v:.1f}"


def format_table_html(
    ordered_names: List[str],
    buckets: Dict[Tuple[str, str], Dict[str, List[float]]],
    *,
    decimals: int,
) -> str:
    thead = (
        "<thead>\n"
        "<tr>\n"
        '<th rowspan="2">Ablation</th>'
        '<th colspan="3">Cancer diagnosis</th>'
        '<th colspan="3">Cancer type</th>\n'
        "</tr>\n"
        "<tr>\n"
        "<th>Epoch</th><th>Test</th><th>Holdout</th>"
        "<th>Epoch</th><th>Test</th><th>Holdout</th>\n"
        "</tr>\n"
        "</thead>\n"
    )

    def task_cells(name: str, task: str) -> Tuple[str, str, str]:
        m = buckets.get((name, task))
        if m is None:
            return (EM_DASH, EM_DASH, EM_DASH)
        ep = _fmt_epoch(m.get("best_epoch", []))
        test = _fmt_mean_sd(m.get("test_auc", []), decimals=decimals)
        hold = _fmt_mean_sd(m.get("holdout_auc", []), decimals=decimals)
        return ep, test, hold

    body_rows: List[str] = []
    for name in ordered_names:
        # Descriptions may contain HTML markup; render as-is (matching table4_hyenadna).
        label = ABLATION_DESCRIPTIONS[name]
        cd_ep, cd_test, cd_hold = task_cells(name, "cancer_diagnosis")
        ct_ep, ct_test, ct_hold = task_cells(name, "cancer_type")
        body_rows.append(
            f"<tr>\n<td>{label}</td>"
            f"<td>{html.escape(cd_ep)}</td>"
            f"<td>{html.escape(cd_test)}</td>"
            f"<td>{html.escape(cd_hold)}</td>"
            f"<td>{html.escape(ct_ep)}</td>"
            f"<td>{html.escape(ct_test)}</td>"
            f"<td>{html.escape(ct_hold)}</td>\n</tr>"
        )
    tbody = "<tbody>\n" + "\n".join(body_rows) + "\n</tbody>\n"
    return f"<table>\n{thead}{tbody}</table>\n"


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    setbert_dir = repo_root / "results" / "setbert"
    if not setbert_dir.is_dir():
        raise SystemExit(f"Not a directory: {setbert_dir}")

    experiment_rows = _experiment_rows(repo_root)
    ordered_names, buckets = _collect_results(setbert_dir, experiment_rows)
    text = format_table_html(ordered_names, buckets, decimals=DECIMALS)
    out_path = repo_root / OUTPUT_REL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
