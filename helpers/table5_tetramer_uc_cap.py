#!/usr/bin/env python3
"""
Build Table 5 (tetramer UC/CAP classifiers) as HTML under manuscript/table5_tetramer_uc_cap.html.

For each task, scans the 3×6 model × feature-set test AUC grid, picks the feature set
that contains the single best test value, then reports test and holdout AUC for every
model on that feature set only.

Run from the repository root: ``python helpers/table5_tetramer_uc_cap.py``
"""

from __future__ import annotations

import sys
import html
import math
from dataclasses import dataclass
from pathlib import Path

from shared_table_code import (
    TASKS,
    MODELS,
    TASK_HEADER,
    DECIMALS,
    feature_count,
    fmt_cell,
    load_split_metrics,
    select_best_feat_index,
)

ROW_LABELS: Dict[str, str] = {
    "knn": "KNN",
    "svm": "SVM",
    "random_forest": "Random Forest",
}

OUTPUT_REL = Path("manuscript") / "table5_tetramer_uc_cap.html"
RESULTS_SUBDIR = "tetramer_uc_cap"


@dataclass(frozen=True)
class UcCapTableData:
    best_feat_index: Dict[str, int]
    metrics: Dict[Tuple[str, str], SplitMetrics]


def _metrics_json_path(feat_dir: Path, task: str, model: str) -> Path:
    return feat_dir / f"{task}_{model}.json"


def build_uc_cap_table_data(uc_cap_dir: Path, n_features: int) -> UcCapTableData:
    """Select best feature set per task from the test grid, then load all models at that set."""
    best_feat_index: Dict[str, int] = {}
    metrics: Dict[Tuple[str, str], SplitMetrics] = {}
    for task in TASKS:
        feat_idx = select_best_feat_index(uc_cap_dir, task, n_features)
        best_feat_index[task] = feat_idx
        feat_dir = uc_cap_dir / str(feat_idx)
        for model in MODELS:
            path = _metrics_json_path(feat_dir, task, model)
            metrics[(task, model)] = load_split_metrics(path, task=task, model=model)
    return UcCapTableData(best_feat_index=best_feat_index, metrics=metrics)


def format_uc_cap_table_html(data: UcCapTableData, *, decimals: int) -> str:
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
    body_rows: List[str] = []

    bold_test = {task: _models_at_max_auc(data, task, split="test") for task in TASKS}
    bold_hold = {
        task: _models_at_max_auc(data, task, split="holdout") for task in TASKS
    }

    for model in MODELS:
        label = html.escape(ROW_LABELS[model])
        cells = []
        for task in TASKS:
            m = data.metrics[(task, model)]
            cells.append(
                _render_cell(
                    m.test_auc,
                    decimals=decimals,
                    bold=model in bold_test[task],
                )
            )
            cells.append(
                _render_cell(
                    m.holdout_auc,
                    decimals=decimals,
                    bold=model in bold_hold[task],
                )
            )
        tds = "".join(f"<td>{c}</td>" for c in cells)
        body_rows.append(f"<tr>\n<td>{label}</td>{tds}\n</tr>")

    feat_cells = []
    for task in TASKS:
        idx = data.best_feat_index[task]
        feat_cells.append(f'<td colspan="2">{html.escape(str(idx))}</td>')
    body_rows.append(
        "<tr>\n"
        f"<td>{html.escape('Feature set')}</td>"
        + "".join(feat_cells)
        + "\n</tr>"
    )

    tbody = "<tbody>\n" + "\n".join(body_rows) + "\n</tbody>\n"
    return f"<table>\n{thead}{tbody}</table>\n"


def write_uc_cap_table(
    repo_root: Path,
    *,
    results_subdir: str,
    output_rel: Path,
    decimals: int = DECIMALS,
) -> Path:
    """Write HTML table with per-task best feature set and per-model test/holdout AUC."""
    uc_cap_dir = repo_root / "results" / results_subdir
    if not uc_cap_dir.is_dir():
        raise SystemExit(f"Not a directory: {uc_cap_dir}")

    n_features = feature_count(repo_root)
    data = build_uc_cap_table_data(uc_cap_dir, n_features)
    text = format_uc_cap_table_html(data, decimals=decimals)
    out_path = repo_root / output_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path


def _models_at_max_auc(
    data: UcCapTableData, task: str, *, split: str
) -> Set[str]:
    """Models tied for the highest test or holdout AUC in one task column pair."""
    attr = "test_auc" if split == "test" else "holdout_auc"
    values = {m: getattr(data.metrics[(task, m)], attr) for m in MODELS}
    best = max(values.values())
    return {m for m, v in values.items() if math.isclose(v, best, rel_tol=0.0, abs_tol=1e-12)}


def _render_cell(value: float, *, decimals: int, bold: bool) -> str:
    text = html.escape(fmt_cell(value, decimals=decimals))
    return f"<strong>{text}</strong>" if bold else text


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    out_path = write_uc_cap_table(
        root,
        results_subdir=RESULTS_SUBDIR,
        output_rel=OUTPUT_REL,
    )
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
