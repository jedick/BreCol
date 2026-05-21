#!/usr/bin/env python3
"""
Build Table 6 (tetramer UC/CAP classifiers) as HTML under manuscript/table6_tetramer_uc_cap.html.

For each task, scans the 3×6 model × feature-set test AUC grid, picks the feature set
that contains the single best test value, then reports test and holdout AUC for every
model on that feature set only.

Run from the repository root: ``python helpers/table6_tetramer_uc_cap.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

from shared_table_code import write_uc_cap_table

OUTPUT_REL = Path("manuscript") / "table6_tetramer_uc_cap.html"
RESULTS_SUBDIR = "tetramer_uc_cap"


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
