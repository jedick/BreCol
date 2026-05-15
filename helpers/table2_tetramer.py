#!/usr/bin/env python3
"""
Build Table 2 (tetramer classifiers) as HTML under manuscript/table2_tetramer.html.

Reads eight JSON files under results/tetramer/ named {task}_{model}.json
(e.g. cancer_diagnosis_knn.json), as written by scripts/fit_classifier.py:
tasks cancer_diagnosis and cancer_type; models baseline, knn, svm, and
random_forest. Each file must have metrics.test.auroc and
metrics.holdout.auroc.

Run from the repository root: ``python helpers/table2_tetramer.py``
"""

from __future__ import annotations

import html
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

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

DECIMALS = 3
OUTPUT_REL = Path("manuscript") / "table2_tetramer.html"


def _load_metrics(
    tetramer_dir: Path,
) -> Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]]:
    """Map (task, model) -> (test_auroc, holdout_auroc). None if JSON null."""
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
