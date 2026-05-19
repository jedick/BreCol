#!/usr/bin/env python3
"""
Build Table 4 (UC/CAP classifiers, selected feature triple) as HTML under
manuscript/table4_uc_cap.html.

Resolves ``results/tetramer_uc_cap/<feat>/`` using ``experiments.yaml`` ``run_uc_cap_pipeline``
rows merged over ``defaults.yaml`` (same ordering as ``helpers/list_uc_cap_feature_outputs.py``),
then loads KNN, SVM, and random forest for both tasks.

The manuscript triple is fixed: *n*<sub>UC</sub> = 1000, *K* = 2000, *n*<sub>CAP</sub> = 5000.

Run from the repository root: ``python helpers/table4_uc_cap.py``
"""

from __future__ import annotations

import html
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml

from list_uc_cap_feature_outputs import merge_run_uc_cap_baseline

TASKS = ("cancer_diagnosis", "cancer_type")
MODELS = ("knn", "svm", "random_forest")

ROW_LABELS: Dict[str, str] = {
    "knn": "KNN",
    "svm": "SVM",
    "random_forest": "Random Forest",
}

TASK_HEADER = {
    "cancer_diagnosis": "Cancer diagnosis",
    "cancer_type": "Cancer type",
}

N_UC = 1000
N_CLUSTERS = 2000
N_CAP = 5000
DECIMALS = 3
OUTPUT_REL = Path("manuscript") / "table4_uc_cap.html"


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_feat_index(
    repo_root: Path,
    *,
    n_uc: int,
    n_clusters: int,
    n_cap: int,
) -> int:
    """1-based ``FEAT`` index for the merged row matching the triple."""
    defaults_cfg = _load_yaml(repo_root / "defaults.yaml")
    experiments_cfg = _load_yaml(repo_root / "experiments.yaml")
    base = merge_run_uc_cap_baseline(defaults_cfg)
    rows = experiments_cfg.get("run_uc_cap_pipeline") or []
    for i, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise SystemExit("experiments.yaml run_uc_cap_pipeline entries must be mappings")
        merged: Dict[str, Any] = {**base, **row}
        nu = int(merged["n_uc"])
        nk = int(merged["n_clusters"])
        nc = int(merged["n_cap"])
        if nu == n_uc and nk == n_clusters and nc == n_cap:
            return i
    raise SystemExit(
        f"No run_uc_cap_pipeline row matches n_uc={n_uc}, n_clusters={n_clusters}, "
        f"n_cap={n_cap} (after merging defaults.yaml baseline)."
    )


def _load_metrics_uc_cap(
    uc_cap_subdir: Path,
) -> Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]]:
    """Map (task, model) -> (test_auroc, holdout_auroc)."""
    out: Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]] = {}
    for task in TASKS:
        for model in MODELS:
            path = uc_cap_subdir / f"{task}_{model}.json"
            if not path.is_file():
                raise SystemExit(
                    f"Missing expected JSON: {path}\n"
                    f"Required files: {{task}}_{{model}}.json for "
                    f"task in {TASKS}, model in {MODELS}."
                )
            data = json.loads(path.read_text(encoding="utf-8"))
            file_task = data.get("task")
            file_model = data.get("model")
            if file_task != task:
                raise SystemExit(f"{path}: expected task {task!r}, got {file_task!r}")
            if file_model != model:
                raise SystemExit(f"{path}: expected model {model!r}, got {file_model!r}")
            metrics = data.get("metrics") or {}
            for split in ("test", "holdout"):
                blob = metrics.get(split)
                if not isinstance(blob, dict):
                    raise SystemExit(
                        f"{path}: expected metrics['{split}'] to be an object with "
                        f"'auroc' (scripts/fit_classifier.py output layout)."
                    )
            test_v = metrics["test"].get("auroc")
            hold_v = metrics["holdout"].get("auroc")
            out[(task, model)] = (
                float(test_v) if test_v is not None else None,
                float(hold_v) if hold_v is not None else None,
            )
    return out


def _fmt_cell(value: Optional[float], *, decimals: int) -> str:
    if value is None:
        return "nan"
    if isinstance(value, float) and not math.isfinite(value):
        return "nan"
    return f"{value:.{decimals}f}"


def format_table_html(
    metrics: Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]],
    *,
    decimals: int,
) -> str:
    thead = (
        "<thead>\n"
        "<tr>\n"
        '<th rowspan="2">Model</th>\n'
        f'<th colspan="2">{html.escape(TASK_HEADER["cancer_diagnosis"])}</th>\n'
        f'<th colspan="2">{html.escape(TASK_HEADER["cancer_type"])}</th>\n'
        "</tr>\n"
        "<tr>\n"
        "<th>Test</th><th>Holdout</th>"
        "<th>Test</th><th>Holdout</th>\n"
        "</tr>\n"
        "</thead>\n"
    )
    body_rows = []
    for model in MODELS:
        label = html.escape(ROW_LABELS[model])
        cells = []
        for task in TASKS:
            test_v, hold_v = metrics[(task, model)]
            cells.append(_fmt_cell(test_v, decimals=decimals))
            cells.append(_fmt_cell(hold_v, decimals=decimals))
        tds = "".join(f"<td>{html.escape(c)}</td>" for c in cells)
        body_rows.append(f"<tr>\n<td>{label}</td>{tds}\n</tr>")
    tbody = "<tbody>\n" + "\n".join(body_rows) + "\n</tbody>\n"
    return f"<table>\n{thead}{tbody}</table>\n"


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    uc_cap_dir = root / "results" / "tetramer_uc_cap"
    if not uc_cap_dir.is_dir():
        raise SystemExit(f"Not a directory: {uc_cap_dir}")

    feat_idx = resolve_feat_index(
        root,
        n_uc=N_UC,
        n_clusters=N_CLUSTERS,
        n_cap=N_CAP,
    )
    sub = uc_cap_dir / str(feat_idx)
    if not sub.is_dir():
        raise SystemExit(f"Missing UC/CAP results directory: {sub}")

    metrics = _load_metrics_uc_cap(sub)
    text = format_table_html(metrics, decimals=DECIMALS)
    out_path = root / OUTPUT_REL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
