#!/usr/bin/env python3
"""
Build Table 3 (tetramer classifiers) as HTML under manuscript/table3_tetramer.html.

Reads eight JSON files under results/tetramer/ named {task}_{model}.json
(e.g. cancer_diagnosis_knn.json), as written by scripts/fit_classifier.py:
tasks cancer_diagnosis and cancer_type; models baseline, knn, svm, and
random_forest. Each file must have metrics.test.auc and
metrics.holdout.auc.

Run from the repository root: ``python helpers/table3_tetramer.py``
"""

from __future__ import annotations

import html
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

TASKS = ("cancer_diagnosis", "cancer_type")
MODELS = ("baseline", "knn", "svm", "random_forest")

ROW_LABELS: Dict[str, str] = {
    "baseline": "Majority class",
    "knn": "KNN",
    "svm": "SVM",
    "random_forest": "Random Forest",
}

TASK_HEADER = {
    "cancer_diagnosis": "Cancer diagnosis",
    "cancer_type": "Cancer type",
}

DECIMALS = 2
OUTPUT_REL = Path("manuscript") / "table3_tetramer.html"


def _load_metrics(
    tetramer_dir: Path,
) -> Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]]:
    """Map (task, model) -> (test_auc, holdout_auc). None if JSON null."""
    out: Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]] = {}
    for task in TASKS:
        for model in MODELS:
            path = tetramer_dir / f"{task}_{model}.json"
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
                        f"'auc' (scripts/fit_classifier.py output layout)."
                    )
            test_v = metrics["test"].get("auc")
            hold_v = metrics["holdout"].get("auc")
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
    """Format with `decimals`; bump to `decimals + 1` only when display would be 1.0... but value < 1.0."""
    text = f"{value:.{decimals}f}"
    if text == f"{1.0:.{decimals}f}" and value != 1.0:
        return f"{value:.{decimals + 1}f}"
    return text


def _render_cell(value: Optional[float], *, decimals: int, bold: bool) -> str:
    text = html.escape(_fmt_cell(value, decimals=decimals))
    return f"<strong>{text}</strong>" if bold else text


def _models_at_max_auc(
    metrics: Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]],
    task: str,
    *,
    split_index: int,
) -> Set[str]:
    """Models tied for the highest AUC in one (task, split) column."""
    values: Dict[str, float] = {}
    for model in MODELS:
        v = metrics[(task, model)][split_index]
        if v is None or not math.isfinite(v):
            continue
        values[model] = v
    if not values:
        return set()
    best = max(values.values())
    return {m for m, v in values.items() if math.isclose(v, best, rel_tol=0.0, abs_tol=1e-12)}


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
    bold_test = {task: _models_at_max_auc(metrics, task, split_index=0) for task in TASKS}
    bold_hold = {task: _models_at_max_auc(metrics, task, split_index=1) for task in TASKS}

    body_rows = []
    for model in MODELS:
        label = html.escape(ROW_LABELS[model])
        cells = []
        for task in TASKS:
            test_v, hold_v = metrics[(task, model)]
            cells.append(
                _render_cell(
                    test_v, decimals=decimals, bold=model in bold_test[task]
                )
            )
            cells.append(
                _render_cell(
                    hold_v, decimals=decimals, bold=model in bold_hold[task]
                )
            )
        tds = "".join(f"<td>{c}</td>" for c in cells)
        body_rows.append(f"<tr>\n<td>{label}</td>{tds}\n</tr>")
    tbody = "<tbody>\n" + "\n".join(body_rows) + "\n</tbody>\n"
    return f"<table>\n{thead}{tbody}</table>\n"


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    tetramer_dir = root / "results" / "tetramer"
    if not tetramer_dir.is_dir():
        raise SystemExit(f"Not a directory: {tetramer_dir}")

    metrics = _load_metrics(tetramer_dir)
    text = format_table_html(metrics, decimals=DECIMALS)
    out_path = root / OUTPUT_REL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
