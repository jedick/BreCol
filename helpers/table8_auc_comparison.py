#!/usr/bin/env python3
"""
Build Table 8 (per-study cancer-diagnosis AUC vs literature) as HTML under
manuscript/table8_auc_comparison.html.

For each cancer type, lists 7 development + 6 holdout studies from datasets.csv
(row order preserved). Per-study AUC comes from the test or holdout per_study
block of the best tetramer UC/CAP cancer_diagnosis cell, where "best" reuses
Table 6's logic: best feature set by test AUC across the model × feature grid,
then best model within that feature set by test AUC.

Literature AUC for colorectal studies is read from the datasets.csv ``auc``
column (empty cells render as an em dash).

Run from the repository root: ``python helpers/table8_auc_comparison.py``
"""

from __future__ import annotations

import html
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import pandas as pd

from shared_table_code import (
    MODELS,
    feature_count,
    fmt_cell,
    load_split_metrics,
    select_best_feat_index,
)

TASK = "cancer_diagnosis"
RESULTS_SUBDIR = "tetramer_uc_cap"
OUTPUT_REL = Path("manuscript") / "table8_auc_comparison.html"
DECIMALS = 2
EM_DASH = "\u2014"

CANCER_TYPES = ("breast", "colorectal")
CANCER_HEADER = {
    "breast": "Breast cancer",
    "colorectal": "Colorectal cancer",
}
N_DEV_PER_TYPE = 7
N_HOLDOUT_PER_TYPE = 6
N_ROWS = N_DEV_PER_TYPE + N_HOLDOUT_PER_TYPE


def _best_model_within_feat(uc_cap_dir: Path, feat_idx: int) -> str:
    """Return the model name with the highest test AUC at the given feature set."""
    feat_dir = uc_cap_dir / str(feat_idx)
    best_model: Optional[str] = None
    best_test = float("-inf")
    for model in MODELS:
        path = feat_dir / f"{TASK}_{model}.json"
        test_auc = load_split_metrics(path, task=TASK, model=model).test_auc
        if test_auc > best_test:
            best_test = test_auc
            best_model = model
    if best_model is None:
        raise SystemExit(f"No models found under {feat_dir} for task {TASK!r}.")
    return best_model


def _ordered_studies(datasets_df: pd.DataFrame, cancer_type: str) -> List[Dict[str, object]]:
    """Return [dev studies in CSV order] + [holdout studies in CSV order] for one cancer type."""
    sub = datasets_df[datasets_df["cancer_type"] == cancer_type]
    dev = sub[sub["partition"] == "development"]
    holdout = sub[sub["partition"] == "holdout"]
    if len(dev) != N_DEV_PER_TYPE or len(holdout) != N_HOLDOUT_PER_TYPE:
        raise SystemExit(
            f"datasets.csv: expected {N_DEV_PER_TYPE} development and "
            f"{N_HOLDOUT_PER_TYPE} holdout studies for {cancer_type!r}; "
            f"got dev={len(dev)}, holdout={len(holdout)}."
        )
    return list(dev.to_dict("records")) + list(holdout.to_dict("records"))


def _per_study_auc(per_study: Mapping[str, dict], study_name: str) -> Optional[float]:
    entry = per_study.get(study_name)
    if not isinstance(entry, dict):
        return None
    v = entry.get("auc")
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _fmt_auc(value: Optional[float]) -> str:
    if value is None:
        return EM_DASH
    return fmt_cell(value, decimals=DECIMALS)


def _fmt_lit_auc(value: object) -> str:
    """Format a literature AUC from the datasets.csv 'auc' column (NaN/empty → em dash)."""
    if value is None:
        return EM_DASH
    if isinstance(value, float) and not math.isfinite(value):
        return EM_DASH
    if isinstance(value, str) and not value.strip():
        return EM_DASH
    try:
        f = float(value)
    except (TypeError, ValueError):
        return EM_DASH
    if not math.isfinite(f):
        return EM_DASH
    return fmt_cell(f, decimals=DECIMALS)


def _fmt_ref(study_name: str) -> str:
    """Return literal citeproc token; @ + [ ] are not HTML-special."""
    return f"[@{html.escape(study_name)}]"


def _load_per_study(json_path: Path) -> tuple[dict, dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    test_ps = data.get("metrics", {}).get("test", {}).get("per_study")
    hold_ps = data.get("metrics", {}).get("holdout", {}).get("per_study")
    if not isinstance(test_ps, dict) or not isinstance(hold_ps, dict):
        raise SystemExit(
            f"{json_path}: missing metrics.test.per_study or "
            "metrics.holdout.per_study. Re-fit with the current "
            "scripts/fit_classifier.py to populate per-study metrics."
        )
    return test_ps, hold_ps


def _build_table_html(
    studies_by_type: Mapping[str, List[Dict[str, object]]],
    test_per_study: Mapping[str, dict],
    hold_per_study: Mapping[str, dict],
) -> str:
    thead = (
        "<thead>\n"
        "<tr>\n"
        '<th rowspan="2">Partition</th>\n'
        f'<th colspan="2">{html.escape(CANCER_HEADER["breast"])}</th>\n'
        f'<th colspan="3">{html.escape(CANCER_HEADER["colorectal"])}</th>\n'
        "</tr>\n"
        "<tr>\n"
        "<th>Dataset</th><th>AUC</th>"
        "<th>Dataset</th><th>AUC</th><th>AUC (literature)</th>\n"
        "</tr>\n"
        "</thead>\n"
    )

    body_rows: List[str] = []
    for i in range(N_ROWS):
        is_dev = i < N_DEV_PER_TYPE
        per_study = test_per_study if is_dev else hold_per_study
        partition_label = "Development" if is_dev else "Holdout"

        crc = studies_by_type["colorectal"][i]
        bc = studies_by_type["breast"][i]

        crc_study = str(crc["study_name"]).strip()
        bc_study = str(bc["study_name"]).strip()

        crc_auc = _per_study_auc(per_study, crc_study)
        bc_auc = _per_study_auc(per_study, bc_study)

        cells = [
            f"<td>{html.escape(partition_label)}</td>",
            f"<td>{_fmt_ref(bc_study)}</td>",
            f"<td>{html.escape(_fmt_auc(bc_auc))}</td>",
            f"<td>{_fmt_ref(crc_study)}</td>",
            f"<td>{html.escape(_fmt_auc(crc_auc))}</td>",
            f"<td>{html.escape(_fmt_lit_auc(crc.get('auc')))}</td>",
        ]
        body_rows.append("<tr>\n" + "".join(cells) + "\n</tr>")

    tbody = "<tbody>\n" + "\n".join(body_rows) + "\n</tbody>\n"
    return f"<table>\n{thead}{tbody}</table>\n"


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    uc_cap_dir = root / "results" / RESULTS_SUBDIR
    if not uc_cap_dir.is_dir():
        raise SystemExit(f"Not a directory: {uc_cap_dir}")

    n_features = feature_count(root)
    feat_idx = select_best_feat_index(uc_cap_dir, TASK, n_features)
    model = _best_model_within_feat(uc_cap_dir, feat_idx)
    json_path = uc_cap_dir / str(feat_idx) / f"{TASK}_{model}.json"
    test_per_study, hold_per_study = _load_per_study(json_path)

    datasets_df = pd.read_csv(root / "datasets.csv")
    studies_by_type = {ct: _ordered_studies(datasets_df, ct) for ct in CANCER_TYPES}

    text = _build_table_html(studies_by_type, test_per_study, hold_per_study)
    out_path = root / OUTPUT_REL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(out_path)
    print(
        f"Source: {json_path.relative_to(root)} (feat={feat_idx}, model={model})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
