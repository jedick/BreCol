"""Shared UC/CAP classifier table logic for tetramer and embedding result trees."""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

TASKS = ("cancer_diagnosis", "cancer_type")
MODELS = ("svm", "knn", "random_forest")

ROW_LABELS: Dict[str, str] = {
    "knn": "KNN",
    "svm": "SVM",
    "random_forest": "Random Forest",
}

TASK_HEADER = {
    "cancer_diagnosis": "Cancer diagnosis",
    "cancer_type": "Cancer type",
}

DECIMALS = 2


@dataclass(frozen=True)
class SplitMetrics:
    test_auc: float
    holdout_auc: float


@dataclass(frozen=True)
class UcCapTableData:
    best_feat_index: Dict[str, int]
    metrics: Dict[Tuple[str, str], SplitMetrics]


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def feature_count(repo_root: Path) -> int:
    cfg = _load_yaml(repo_root / "experiments.yaml")
    rows = cfg.get("run_uc_cap_pipeline") or []
    if not isinstance(rows, list) or not rows:
        raise SystemExit("No run_uc_cap_pipeline rows found in experiments.yaml.")
    return len(rows)


def _metrics_json_path(feat_dir: Path, task: str, model: str) -> Path:
    return feat_dir / f"{task}_{model}.json"


def load_split_metrics(path: Path, *, task: str, model: str) -> SplitMetrics:
    if not path.is_file():
        raise SystemExit(f"Missing expected JSON: {path}")
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
    if test_v is None or hold_v is None:
        raise SystemExit(f"{path}: missing metrics.test.auc or metrics.holdout.auc.")
    return SplitMetrics(test_auc=float(test_v), holdout_auc=float(hold_v))


def select_best_feat_index(uc_cap_dir: Path, task: str, n_features: int) -> int:
    """Pick the 1-based feature set with the highest test AUC in the model × feature grid."""
    best_idx: Optional[int] = None
    best_test = float("-inf")
    for feat_idx in range(1, n_features + 1):
        feat_dir = uc_cap_dir / str(feat_idx)
        if not feat_dir.is_dir():
            raise SystemExit(f"Missing UC/CAP results directory: {feat_dir}")
        for model in MODELS:
            path = _metrics_json_path(feat_dir, task, model)
            test = load_split_metrics(path, task=task, model=model).test_auc
            if test > best_test:
                best_test = test
                best_idx = feat_idx
    if best_idx is None:
        raise SystemExit(f"No feature sets found under {uc_cap_dir} for task {task!r}.")
    return best_idx


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


def fmt_cell(value: float, *, decimals: int) -> str:
    """Format an AUC-like value with `decimals` decimals.

    Selective rounding: if the rounded display would be exactly "1.00" but the
    underlying value is not exactly 1.0, render with one extra decimal (e.g.
    "0.997" instead of "1.00").
    """
    if not math.isfinite(value):
        return "nan"
    text = f"{value:.{decimals}f}"
    if text == f"{1.0:.{decimals}f}" and value != 1.0:
        return f"{value:.{decimals + 1}f}"
    return text


def _render_cell(value: float, *, decimals: int, bold: bool) -> str:
    text = html.escape(fmt_cell(value, decimals=decimals))
    return f"<strong>{text}</strong>" if bold else text


def _models_at_max_auc(
    data: UcCapTableData, task: str, *, split: str
) -> Set[str]:
    """Models tied for the highest test or holdout AUC in one task column pair."""
    attr = "test_auc" if split == "test" else "holdout_auc"
    values = {m: getattr(data.metrics[(task, m)], attr) for m in MODELS}
    best = max(values.values())
    return {m for m, v in values.items() if math.isclose(v, best, rel_tol=0.0, abs_tol=1e-12)}


def format_table_html(data: UcCapTableData, *, decimals: int) -> str:
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
    text = format_table_html(data, decimals=decimals)
    out_path = repo_root / output_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path
