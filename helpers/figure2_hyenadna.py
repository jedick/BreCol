#!/usr/bin/env python3
"""
Build Figure 2: HyenaDNA AUC vs sequence length per set (test vs holdout).

Layout:
- 2x2 subplots
- rows = task (cancer diagnosis, cancer type)
- columns = early-stop horizon by validation F1: best epoch in epochs 1–10 vs 1–20
- x-axis = length per set (1k, 2k, 4k, 8k, 16k) from experiment max_length
- y-axis = ROC AUC at the chosen epoch (from training log)

Each subplot draws:
- test split: thin dashed line with markers
- holdout split: solid bold line with markers

For each column, AUC is read from ``results/hyenadna/<name>_training.json`` at the
epoch that maximizes ``val_f1_weighted`` among rows with ``epoch`` ≤ the column
cap (10 or 20). Ties break toward a later epoch.

Experiment names come from ``experiments.yaml`` ``train_hyenadna.experiments``
merged with ``defaults.yaml`` ``train_hyenadna`` (single num_sets grid, typically 5).

Writes ``manuscript/figure2_hyenadna.png`` and ``manuscript/figure2_hyenadna.svg``.

Run from the repository root: ``python helpers/figure2_hyenadna.py``
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import yaml


TASKS: List[Tuple[str, str]] = [
    ("cancer_diagnosis", "Cancer diagnosis"),
    ("cancer_type", "Cancer type"),
]
EPOCH_COLUMNS: List[Tuple[int, str]] = [
    (10, "Best val F1 (epochs 1–10)"),
    (20, "Best val F1 (epochs 1–20)"),
]

LENGTHS_BP: Tuple[int, ...] = (1024, 2048, 4096, 8192, 16384)
X_LABELS: Tuple[str, ...] = ("1k", "2k", "4k", "8k", "16k")

OUTPUT_PNG = Path("manuscript") / "figure2_hyenadna.png"
OUTPUT_SVG = Path("manuscript") / "figure2_hyenadna.svg"


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _train_hyenadna_base(repo_root: Path) -> dict:
    cfg = _load_yaml(repo_root / "defaults.yaml")
    sec = cfg.get("train_hyenadna")
    if not isinstance(sec, dict):
        raise SystemExit("defaults.yaml must define train_hyenadna as a mapping.")
    return dict(sec)


def _experiment_grid(repo_root: Path) -> Dict[Tuple[str, int], str]:
    """Map (task, max_length) -> experiment name (results / training log stem)."""
    base = _train_hyenadna_base(repo_root)
    exp_cfg = _load_yaml(repo_root / "experiments.yaml").get("train_hyenadna") or {}
    rows = exp_cfg.get("experiments") or []
    if not isinstance(rows, list) or not rows:
        raise SystemExit("experiments.yaml must list train_hyenadna.experiments.")

    out: Dict[Tuple[str, int], str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise SystemExit("train_hyenadna experiment entry must be a mapping.")
        name = str(row.get("name") or "").strip()
        if not name:
            raise SystemExit("Each train_hyenadna experiment must have a non-empty name.")
        overrides = row.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise SystemExit("experiment overrides must be a mapping.")
        merged = {**base, **overrides}

        task = merged.get("task")
        if task not in ("cancer_diagnosis", "cancer_type"):
            raise SystemExit(f"{name}: expected task cancer_diagnosis or cancer_type, got {task!r}")

        max_length = int(merged["max_length"])
        key = (str(task), max_length)
        if key in out:
            raise SystemExit(
                f"Duplicate experiment for task={task!r}, max_length={max_length}: "
                f"{out[key]!r} and {name!r}."
            )
        out[key] = name

    return out


def _grid_complete(grid: Dict[Tuple[str, int], str]) -> None:
    for task, _ in TASKS:
        for L in LENGTHS_BP:
            key = (task, L)
            if key not in grid:
                raise SystemExit(
                    f"Missing train_hyenadna experiment for task={task!r}, max_length={L}."
                )


def _val_f1_for_argmax(raw: object) -> float:
    if raw is None:
        return float("-inf")
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return float("-inf")
    if not math.isfinite(v):
        return float("-inf")
    return v


def _load_training_rows(path: Path) -> List[dict]:
    if not path.is_file():
        raise SystemExit(f"Missing training log: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise SystemExit(f"{path}: expected a non-empty JSON list of epoch records.")
    return [r for r in data if isinstance(r, dict)]


def _auc_at_best_val_f1(
    rows: Sequence[dict],
    *,
    max_epoch: int,
) -> Tuple[float, float, int]:
    capped = [r for r in rows if int(r.get("epoch", -1)) <= max_epoch]
    if not capped:
        raise SystemExit(f"No epochs with epoch ≤ {max_epoch} in training log.")

    best = max(
        capped,
        key=lambda r: (_val_f1_for_argmax(r.get("val_f1_weighted")), int(r["epoch"])),
    )
    ep = int(best["epoch"])
    t_raw = best.get("test_roc_auc")
    h_raw = best.get("holdout_roc_auc")
    if t_raw is None or h_raw is None:
        raise SystemExit(
            f"Chosen epoch {ep}: need finite test_roc_auc and holdout_roc_auc "
            f"(got {t_raw!r}, {h_raw!r})."
        )
    t_auc = float(t_raw)
    h_auc = float(h_raw)
    if not math.isfinite(t_auc) or not math.isfinite(h_auc):
        raise SystemExit(
            f"Chosen epoch {ep}: test_roc_auc and holdout_roc_auc must be finite "
            f"(got {t_auc}, {h_auc})."
        )
    return t_auc, h_auc, ep


def collect_series(
    hyenadna_dir: Path,
    grid: Dict[Tuple[str, int], str],
) -> Dict[Tuple[str, int], Tuple[List[float], List[float]]]:
    out: Dict[Tuple[str, int], Tuple[List[float], List[float]]] = {}
    for task, _ in TASKS:
        for max_epoch, _ in EPOCH_COLUMNS:
            test_vals: List[float] = []
            holdout_vals: List[float] = []
            for L in LENGTHS_BP:
                name = grid[(task, L)]
                log_path = hyenadna_dir / f"{name}_training.json"
                rows = _load_training_rows(log_path)
                test_auc, holdout_auc, _ep = _auc_at_best_val_f1(rows, max_epoch=max_epoch)
                test_vals.append(test_auc)
                holdout_vals.append(holdout_auc)
            out[(task, max_epoch)] = (test_vals, holdout_vals)
    return out


def build_plot(
    series: Dict[Tuple[str, int], Tuple[List[float], List[float]]],
    output_path: Path,
) -> None:
    x = list(range(len(LENGTHS_BP)))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)

    for row, (task_key, task_label) in enumerate(TASKS):
        for col, (max_epoch, col_title) in enumerate(EPOCH_COLUMNS):
            ax = axes[row, col]
            test_vals, holdout_vals = series[(task_key, max_epoch)]

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
                ax.set_title(col_title)
            if col == 0:
                ax.set_ylabel(f"{task_label}\nROC AUC")

            ax.set_xticks(x)
            ax.set_xticklabels(X_LABELS)
            ax.set_ylim(0.5, 1.02)
            ax.grid(alpha=0.25, linewidth=0.7)

    for ax in axes[-1, :]:
        ax.set_xlabel("Length per set")

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
    hyenadna_dir = repo_root / "results" / "hyenadna"
    if not hyenadna_dir.is_dir():
        raise SystemExit(f"Not a directory: {hyenadna_dir}")

    grid = _experiment_grid(repo_root)
    _grid_complete(grid)

    series = collect_series(hyenadna_dir, grid)
    png_path = repo_root / OUTPUT_PNG
    svg_path = repo_root / OUTPUT_SVG
    build_plot(series, output_path=png_path)
    build_plot(series, output_path=svg_path)
    print(png_path)
    print(svg_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
