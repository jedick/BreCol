#!/usr/bin/env python3
"""
Build Figure 4: UC/CAP feature-set stability across test vs holdout AUC (HyenaDNA embeddings).

Writes ``manuscript/figure4_embedding_uc_cap.png`` and ``manuscript/figure4_embedding_uc_cap.svg`` from
JSON metrics under ``results/embedding_uc_cap/<feat_index>/``.

Run from the repository root: ``python helpers/figure4_embedding_uc_cap.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

from shared_figure_code import MODELS_SVM_RF, write_uc_cap_stability_figure

OUTPUT_PNG = Path("manuscript") / "figure4_embedding_uc_cap.png"
OUTPUT_SVG = Path("manuscript") / "figure4_embedding_uc_cap.svg"
RESULTS_SUBDIR = "embedding_uc_cap"


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    png_path, svg_path = write_uc_cap_stability_figure(
        repo_root,
        results_subdir=RESULTS_SUBDIR,
        models=MODELS_SVM_RF,
        output_png=OUTPUT_PNG,
        output_svg=OUTPUT_SVG,
    )
    print(png_path)
    print(svg_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
