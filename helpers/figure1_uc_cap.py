#!/usr/bin/env python3
"""
Build Figure 1: UC/CAP feature-set stability across test vs holdout AUROC.

Writes ``manuscript/figure1_uc_cap.png`` and ``manuscript/figure1_uc_cap.svg`` from
JSON metrics under ``results/uc_cap/<feat_index>/``.

Run from the repository root: ``python helpers/figure1_uc_cap.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import yaml


TASKS: List[Tuple[str, str]] = [
    ("cancer_diagnosis", "Cancer diagnosis"),
    ("cancer_type", "Cancer type"),
]
MODELS: List[Tuple[str, str]] = [
    ("knn", "KNN"),
    ("svm", "SVM"),
]

OUTPUT_PNG = Path("manuscript") / "figure1_uc_cap.png"
OUTPUT_SVG = Path("manuscript") / "figure1_uc_cap.svg"


def _load_yaml(path: Path) -> Dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _feature_count(repo_root: Path) -> int:
    cfg = _load_yaml(repo_root / "experiments.yaml")
    rows = cfg.get("run_uc_cap_pipeline") or []
    if not isinstance(rows, list) or not rows:
        raise SystemExit("No run_uc_cap_pipeline rows found in experiments.yaml.")
    return len(rows)


def _load_metrics(json_path: Path) -> Tuple[float, float]:
    if not json_path.is_file():
        raise SystemExit(f"Missing expected JSON: {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    metrics = data.get("metrics") or {}
    test = metrics.get("test") or {}
    holdout = metrics.get("holdout") or {}
    if "auroc" not in test or "auroc" not in holdout:
        raise SystemExit(
            f"{json_path}: expected metrics.test.auroc and metrics.holdout.auroc."
        )
    return float(test["auroc"]), float(holdout["auroc"])


def collect_series(
    uc_cap_dir: Path, n_features: int
) -> Dict[Tuple[str, str], Tuple[List[float], List[float]]]:
    out: Dict[Tuple[str, str], Tuple[List[float], List[float]]] = {}
    for task, _ in TASKS:
        for model, _ in MODELS:
            test_vals: List[float] = []
            holdout_vals: List[float] = []
            for feat_idx in range(1, n_features + 1):
                json_path = uc_cap_dir / str(feat_idx) / f"{task}_{model}.json"
                test_auc, holdout_auc = _load_metrics(json_path)
                test_vals.append(test_auc)
                holdout_vals.append(holdout_auc)
            out[(task, model)] = (test_vals, holdout_vals)
    return out


def build_plot(
    series: Dict[Tuple[str, str], Tuple[List[float], List[float]]],
    n_features: int,
    output_path: Path,
) -> None:
    x = list(range(1, n_features + 1))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)

    for row, (task_key, task_label) in enumerate(TASKS):
        for col, (model_key, model_label) in enumerate(MODELS):
            ax = axes[row, col]
            test_vals, holdout_vals = series[(task_key, model_key)]

            ax.plot(
                x,
                test_vals,
                linestyle="--",
                linewidth=1.2,
                marker="o",
                markersize=4,
                label="Test",
                color="#4C78A8",
            )
            ax.plot(
                x,
                holdout_vals,
                linestyle="-",
                linewidth=2.8,
                marker="o",
                markersize=4.5,
                label="Holdout",
                color="#F58518",
            )

            if row == 0:
                ax.set_title(model_label)
            if col == 0:
                ax.set_ylabel(f"{task_label}\nAUROC")

            ax.set_xticks(x)
            ax.set_ylim(0.5, 1.02)
            ax.grid(alpha=0.25, linewidth=0.7)

    for ax in axes[-1, :]:
        ax.set_xlabel("Feature set index")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    uc_cap_dir = repo_root / "results" / "uc_cap"
    if not uc_cap_dir.is_dir():
        raise SystemExit(f"Not a directory: {uc_cap_dir}")

    n_features = _feature_count(repo_root)
    series = collect_series(uc_cap_dir, n_features=n_features)
    png_path = repo_root / OUTPUT_PNG
    svg_path = repo_root / OUTPUT_SVG
    build_plot(series, n_features=n_features, output_path=png_path)
    build_plot(series, n_features=n_features, output_path=svg_path)
    print(png_path)
    print(svg_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
