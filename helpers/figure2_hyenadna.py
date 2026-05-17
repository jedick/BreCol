#!/usr/bin/env python3
"""
Build Figure 2: HyenaDNA AUROC vs sequence length per set (test vs holdout).

Layout:
- 1 row, 2 columns (cancer diagnosis, cancer type)
- x-axis = length per set (1k, 2k, 4k, 8k, 16k) from max_length train_hyenadna runs
- y-axis = AUROC from ``results/hyenadna/mt_max_length_<bp>_<token>_s*.json``
  (mean and sample stdev across seeds)

Each subplot draws:
- test split: thin dashed line with markers and ±1 stdev error bars
- holdout split: solid bold line with markers and ±1 stdev error bars

Missing ``max_length_*`` result groups are skipped (partial result trees are OK).

Writes ``manuscript/figure2_hyenadna.png`` and ``manuscript/figure2_hyenadna.svg``.

Run from the repository root: ``python helpers/figure2_hyenadna.py``
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt

TASK_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("cancer_diagnosis", "Cancer diagnosis"),
    ("cancer_type", "Cancer type"),
)
LENGTHS_BP: Tuple[int, ...] = (1024, 2048, 4096, 8192, 16384)
RESULTS_GLOB = "mt_max_length_*_*_s*.json"
_RESULT_RE = re.compile(r"mt_max_length_(\d+)_(\d+k)_s\d+\.json$")

OUTPUT_PNG = Path("manuscript") / "figure2_hyenadna.png"
OUTPUT_SVG = Path("manuscript") / "figure2_hyenadna.svg"


@dataclass(frozen=True)
class LengthPoint:
    bp: int
    x_label: str
    test_mean: float
    test_std: float
    holdout_mean: float
    holdout_std: float


def _len_token(bp: int) -> str:
    return f"{bp // 1024}k"


def _read_task_aurocs(path: Path, task_key: str) -> Tuple[float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a JSON object.")
    metrics = data.get("metrics") or {}
    blob = metrics.get(task_key)
    if not isinstance(blob, dict):
        raise SystemExit(f"{path}: missing metrics[{task_key!r}] (expected multitask layout).")
    for split in ("test", "holdout"):
        part = blob.get(split)
        if not isinstance(part, dict):
            raise SystemExit(f"{path}: metrics[{task_key!r}][{split!r}] must be an object.")
    test_v = blob["test"].get("auroc")
    hold_v = blob["holdout"].get("auroc")
    if test_v is None or hold_v is None:
        raise SystemExit(f"{path}: missing test or holdout auroc for {task_key}.")
    t, h = float(test_v), float(hold_v)
    if not math.isfinite(t) or not math.isfinite(h):
        raise SystemExit(f"{path}: non-finite AUROC for {task_key} ({t}, {h}).")
    return t, h


def _discover_result_files(hyenadna_dir: Path) -> Dict[int, List[Path]]:
    """Map max_length (bp) to multitask result JSON paths."""
    by_bp: Dict[int, List[Path]] = {}
    for path in sorted(hyenadna_dir.glob(RESULTS_GLOB)):
        if path.name.endswith("_training.json"):
            continue
        m = _RESULT_RE.match(path.name)
        if m is None:
            continue
        bp = int(m.group(1))
        if m.group(2) != _len_token(bp):
            continue
        by_bp.setdefault(bp, []).append(path)
    return by_bp


def _aggregate_seed_values(values: Sequence[float]) -> Tuple[float, float]:
    mean = statistics.fmean(values)
    if len(values) >= 2:
        std = statistics.stdev(values)
    else:
        std = 0.0
    return mean, std


def collect_series(hyenadna_dir: Path) -> Dict[str, List[LengthPoint]]:
    """Per task key, return length points that have on-disk seed results."""
    by_bp = _discover_result_files(hyenadna_dir)
    out: Dict[str, List[LengthPoint]] = {task_key: [] for task_key, _ in TASK_COLUMNS}
    for bp in LENGTHS_BP:
        paths = by_bp.get(bp) or []
        if not paths:
            continue
        for task_key, _ in TASK_COLUMNS:
            test_seed: List[float] = []
            hold_seed: List[float] = []
            for path in paths:
                t, h = _read_task_aurocs(path, task_key)
                test_seed.append(t)
                hold_seed.append(h)
            test_mean, test_std = _aggregate_seed_values(test_seed)
            hold_mean, hold_std = _aggregate_seed_values(hold_seed)
            out[task_key].append(
                LengthPoint(
                    bp=bp,
                    x_label=_len_token(bp),
                    test_mean=test_mean,
                    test_std=test_std,
                    holdout_mean=hold_mean,
                    holdout_std=hold_std,
                )
            )
    if not any(out.values()):
        raise SystemExit(
            "No max_length results found under "
            f"{hyenadna_dir} (expected mt_max_length_<bp>_<token>_s*.json)."
        )
    return out


def build_plot(series: Dict[str, List[LengthPoint]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for col, (task_key, task_label) in enumerate(TASK_COLUMNS):
        ax = axes[col]
        points = series[task_key]
        if not points:
            raise SystemExit(f"No max_length results for task {task_key!r}.")

        x = list(range(len(points)))
        test_means = [p.test_mean for p in points]
        test_stds = [p.test_std for p in points]
        hold_means = [p.holdout_mean for p in points]
        hold_stds = [p.holdout_std for p in points]
        x_labels = [p.x_label for p in points]

        ax.errorbar(
            x,
            test_means,
            yerr=test_stds,
            linestyle="--",
            linewidth=1.2,
            marker="o",
            markersize=4,
            capsize=3,
            capthick=1,
            elinewidth=1,
            label="Test",
            color="#4C78A8",
        )
        ax.errorbar(
            x,
            hold_means,
            yerr=hold_stds,
            linestyle="-",
            linewidth=2.8,
            marker="o",
            markersize=4.5,
            capsize=3,
            capthick=1,
            elinewidth=1,
            label="Holdout",
            color="#F58518",
        )

        ax.set_title(task_label)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylim(0.5, 1.02)
        ax.grid(alpha=0.25, linewidth=0.7)

    axes[0].set_ylabel("AUROC")
    for ax in axes:
        ax.set_xlabel("Length per set")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    hyenadna_dir = repo_root / "results" / "hyenadna"
    if not hyenadna_dir.is_dir():
        raise SystemExit(f"Not a directory: {hyenadna_dir}")

    series = collect_series(hyenadna_dir)
    png_path = repo_root / OUTPUT_PNG
    svg_path = repo_root / OUTPUT_SVG
    build_plot(series, output_path=png_path)
    build_plot(series, output_path=svg_path)
    print(png_path)
    print(svg_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
