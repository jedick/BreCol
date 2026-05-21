"""Shared UC/CAP stability figure logic for tetramer and embedding result trees."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import yaml

TASKS: List[Tuple[str, str]] = [
    ("cancer_diagnosis", "Cancer diagnosis"),
    ("cancer_type", "Cancer type"),
]
MODELS_SVM_KNN: List[Tuple[str, str]] = [
    ("svm", "SVM"),
    ("knn", "KNN"),
]
MODELS_SVM_RF: List[Tuple[str, str]] = [
    ("svm", "SVM"),
    ("random_forest", "Random Forest"),
]


def _load_yaml(path: Path) -> Dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def feature_count(repo_root: Path) -> int:
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
    if "auc" not in test or "auc" not in holdout:
        raise SystemExit(
            f"{json_path}: expected metrics.test.auc and metrics.holdout.auc."
        )
    return float(test["auc"]), float(holdout["auc"])


def collect_series(
    uc_cap_dir: Path,
    n_features: int,
    models: List[Tuple[str, str]],
) -> Dict[Tuple[str, str], Tuple[List[float], List[float]]]:
    out: Dict[Tuple[str, str], Tuple[List[float], List[float]]] = {}
    for task, _ in TASKS:
        for model, _ in models:
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
    models: List[Tuple[str, str]],
    output_path: Path,
) -> None:
    x = list(range(1, n_features + 1))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)

    for row, (task_key, task_label) in enumerate(TASKS):
        for col, (model_key, model_label) in enumerate(models):
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
                ax.set_ylabel(f"{task_label}\nAUC")

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


def write_uc_cap_stability_figure(
    repo_root: Path,
    *,
    results_subdir: str,
    models: List[Tuple[str, str]],
    output_png: Path,
    output_svg: Path,
) -> Tuple[Path, Path]:
    """Build PNG and SVG from ``results/<results_subdir>/<feat_index>/`` JSON metrics."""
    uc_cap_dir = repo_root / "results" / results_subdir
    if not uc_cap_dir.is_dir():
        raise SystemExit(f"Not a directory: {uc_cap_dir}")

    n_features = feature_count(repo_root)
    series = collect_series(uc_cap_dir, n_features=n_features, models=models)
    png_path = repo_root / output_png
    svg_path = repo_root / output_svg
    build_plot(series, n_features=n_features, models=models, output_path=png_path)
    build_plot(series, n_features=n_features, models=models, output_path=svg_path)
    return png_path, svg_path
