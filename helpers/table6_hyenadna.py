#!/usr/bin/env python3
"""
Build Table 6 (HyenaDNA ablations) as HTML under manuscript/table6_hyenadna.html.

Reads ``experiments.yaml`` ``train_hyenadna.experiments`` and aggregates
``metrics.{test,holdout}_auc`` from per-seed JSON under ``results/hyenadna/``
(filenames ``{cd|ct}_{name}_{max_length/1024}k_s{seed}.json``, as written by
``scripts/train_hyenadna.py``). Rows whose ``name`` contains ``max_length``
feed Figure 3 instead and are skipped here.

Each ablation name may declare a single task or a list of tasks (e.g.
``task: [cancer_diagnosis, cancer_type]``); the same ablation contributes one
row per unique name (YAML-first-appearance order), combining
cancer_diagnosis (``cd_*``) and cancer_type (``ct_*``) results.

Each task contributes two columns: test AUC and holdout AUC. Cells render
as ``mean`` followed by a smaller-font ``\u00b1 sd`` across the random seeds
listed in ``experiments.yaml`` (sample standard deviation; the
``\u00b1 sd`` part is omitted when only one seed is available). The mean is
wrapped in ``<strong>`` for the row with the highest mean in its column.

Run from the repository root: ``python helpers/table6_hyenadna.py``
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml

from shared_table_code import (
    ABLATION_COLUMNS,
    ABLATION_DESCRIPTIONS,
    ABLATION_METRIC_KEYS,
    TASK_ABBRV,
    AblationRow,
    format_ablation_table_html,
)


DECIMALS = 2
OUTPUT_REL = Path("manuscript") / "table6_hyenadna.html"


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


def _parse_task_spec(raw: object) -> List[str]:
    """Return the (sorted) list of task tokens this experiment row produces."""
    if isinstance(raw, (list, tuple)):
        tokens = [str(tok).strip() for tok in raw if str(tok).strip()]
    else:
        token = str(raw or "").strip()
        tokens = [token] if token else []

    seen: List[str] = []
    for t in tokens:
        if t not in TASK_ABBRV:
            raise SystemExit(
                f"unsupported task {t!r} (expected one of {sorted(TASK_ABBRV)})."
            )
        if t not in seen:
            seen.append(t)
    return sorted(seen)


def _len_token(max_length: int) -> str:
    if max_length <= 0 or max_length % 1024 != 0:
        raise SystemExit(
            f"max_length {max_length} is not a positive multiple of 1024; "
            "cannot reconstruct results JSON filename."
        )
    return f"{max_length // 1024}k"


def _experiment_rows(repo_root: Path) -> List[Tuple[str, str, int, int]]:
    """Return [(name, task, seed, max_length), ...] expanded from experiments.yaml.

    Rows whose ``name`` contains ``max_length`` feed Figure 3 and are skipped.
    """
    base_cfg = _load_yaml(repo_root / "defaults.yaml").get("train_hyenadna") or {}
    if not isinstance(base_cfg, dict):
        raise SystemExit("defaults.yaml must define train_hyenadna as a mapping.")
    default_seed = int(base_cfg.get("random_seed", 0))
    default_tasks = _parse_task_spec(base_cfg.get("task"))

    exp_cfg = _load_yaml(repo_root / "experiments.yaml").get("train_hyenadna") or {}
    rows = exp_cfg.get("experiments") or []
    if not isinstance(rows, list) or not rows:
        raise SystemExit(
            "experiments.yaml train_hyenadna.experiments must be a non-empty list."
        )

    out: List[Tuple[str, str, int, int]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SystemExit(f"experiments.yaml row {idx + 1} is not a mapping.")
        name = str(row.get("name") or "").strip()
        if "max_length" in name:
            continue  # Figure 3 input, not a Table 6 ablation.
        overrides = row.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise SystemExit(f"{name}: overrides must be a mapping.")
        if "task" in overrides:
            try:
                tasks = _parse_task_spec(overrides.get("task"))
            except SystemExit as exc:
                raise SystemExit(
                    f"experiments.yaml row {idx + 1} ({name!r}): {exc}"
                ) from exc
        else:
            tasks = list(default_tasks)
        if not tasks:
            raise SystemExit(
                f"experiments.yaml row {idx + 1} ({name!r}): no task configured."
            )
        if name not in ABLATION_DESCRIPTIONS:
            raise SystemExit(
                f"experiments.yaml row {idx + 1}: unknown experiment name {name!r}; "
                "add a description to ABLATION_DESCRIPTIONS in helpers/shared_table_code.py."
            )
        merged = {**base_cfg, **overrides}
        try:
            max_length = int(merged["max_length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(
                f"{name}: cannot resolve max_length from defaults+overrides ({exc})."
            )
        for task in tasks:
            for seed in _parse_seed_spec(overrides.get("random_seed"), default_seed):
                out.append((name, task, seed, max_length))
    return out


def _read_metrics(path: Path) -> Dict[str, float]:
    """Return {test_auc, holdout_auc} from one per-seed JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a JSON object.")
    metrics = data.get("metrics") or {}
    out: Dict[str, float] = {}
    for key in ABLATION_METRIC_KEYS:
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


def collect_table_rows(
    hyenadna_dir: Path,
    experiment_rows: List[Tuple[str, str, int, int]],
) -> List[AblationRow]:
    """Build one AblationRow per unique ablation ``name`` (YAML order preserved)."""
    ordered_names: List[str] = []
    seen: Set[str] = set()
    buckets: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    missing: List[str] = []
    repo_root = hyenadna_dir.parent.parent
    for name, task, seed, max_length in experiment_rows:
        if name not in seen:
            seen.add(name)
            ordered_names.append(name)
        path = hyenadna_dir / f"{TASK_ABBRV[task]}_{name}_{_len_token(max_length)}_s{seed}.json"
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

    rows: List[AblationRow] = []
    for name in ordered_names:
        cells: Dict[Tuple[str, str], List[float]] = {}
        for task, metric in ABLATION_COLUMNS:
            per_seed = buckets.get((name, task), {}).get(metric, [])
            cells[(task, metric)] = list(per_seed)
        rows.append(AblationRow(label_html=ABLATION_DESCRIPTIONS[name], cells=cells))
    return rows


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    hyenadna_dir = repo_root / "results" / "hyenadna"
    if not hyenadna_dir.is_dir():
        raise SystemExit(f"Not a directory: {hyenadna_dir}")

    experiment_rows = _experiment_rows(repo_root)
    rows = collect_table_rows(hyenadna_dir, experiment_rows)
    text = format_ablation_table_html(rows, decimals=DECIMALS)
    out_path = repo_root / OUTPUT_REL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
