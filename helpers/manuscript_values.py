#!/usr/bin/env python3
"""Compute manuscript text-substitution values and write them to ``manuscript/values.yaml``.

The YAML file is consumed by Pandoc via ``--metadata-file=values.yaml`` plus the
``manuscript/filters/substitute_values.lua`` filter, which expands ``{key}``
patterns in the manuscript body using these values. Substituted text is
re-parsed as Markdown by the filter, so values may contain inline Markdown such
as citation keys (``[@AAA+99]``) that citeproc resolves downstream.

Values produced:

``hyenadna_sequences_per_sample_text``
    Summary statistics for the per-sample sequence count in the 16k HyenaDNA
    run-tensor cache. For each study, sum the per-study totals across the five
    sets (``set_0`` ... ``set_4``) in
    ``outputs/hyenadna_run_tensors/sequence_counts.csv`` and divide by that
    study's sample count (``n_cancer + n_healthy`` in ``datasets.csv``). The
    formatted string reports the mean ± sample standard deviation across
    studies plus the extreme studies, e.g.::

        323 \u00b1 112 (min 50 for ref [@YTK+26], max 540 for ref [@BVW+21])

    The ``[@...]`` keys match BibTeX keys in ``manuscript/references.bib``.

Run from the repository root: ``python helpers/manuscript_values.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SEQ_COUNTS_CSV = REPO_ROOT / "outputs" / "hyenadna_run_tensors" / "sequence_counts.csv"
DATASETS_CSV = REPO_ROOT / "datasets.csv"
VALUES_YAML = REPO_ROOT / "manuscript" / "values.yaml"


def _per_sample_sequence_counts() -> pd.Series:
    """Return one ratio per study: (sum over 5 sets) / (n_cancer + n_healthy)."""
    if not SEQ_COUNTS_CSV.is_file():
        raise SystemExit(
            f"Missing {SEQ_COUNTS_CSV.relative_to(REPO_ROOT)}. "
            "Run `make hyenadna_run_tensors` first."
        )
    if not DATASETS_CSV.is_file():
        raise SystemExit(f"Missing {DATASETS_CSV.relative_to(REPO_ROOT)}.")

    counts = pd.read_csv(SEQ_COUNTS_CSV)
    set_cols = [c for c in counts.columns if c.startswith("set_")]
    if not set_cols:
        raise SystemExit(
            f"{SEQ_COUNTS_CSV.relative_to(REPO_ROOT)} has no set_* columns."
        )
    seqs_per_study = counts.set_index("study")[set_cols].sum(axis=1)

    datasets = pd.read_csv(DATASETS_CSV)
    samples_per_study = (
        datasets.set_index("study_name")[["n_cancer", "n_healthy"]].sum(axis=1)
    )

    aligned = pd.concat(
        {"seqs": seqs_per_study, "samples": samples_per_study},
        axis=1,
        join="inner",
    )
    missing = set(seqs_per_study.index) ^ set(aligned.index)
    if missing:
        raise SystemExit(
            "Mismatched studies between sequence_counts.csv and datasets.csv: "
            f"{sorted(missing)}"
        )

    return aligned["seqs"] / aligned["samples"]


def _hyenadna_sequences_per_sample_text() -> str:
    per_sample = _per_sample_sequence_counts()
    mean = int(round(float(per_sample.mean())))
    # Sample standard deviation (ddof=1), matching R's sd().
    sd = int(round(float(per_sample.std(ddof=1))))
    min_study = str(per_sample.idxmin())
    max_study = str(per_sample.idxmax())
    min_val = int(round(float(per_sample.min())))
    max_val = int(round(float(per_sample.max())))
    return (
        f"{mean} \u00b1 {sd} "
        f"(min {min_val} for ref [@{min_study}], max {max_val} for ref [@{max_study}])"
    )


def compute_values() -> Dict[str, object]:
    return {
        "hyenadna_sequences_per_sample_text": _hyenadna_sequences_per_sample_text(),
    }


def main() -> int:
    values = compute_values()
    VALUES_YAML.parent.mkdir(parents=True, exist_ok=True)
    with VALUES_YAML.open("w", encoding="utf-8") as f:
        yaml.safe_dump(values, f, sort_keys=True, default_flow_style=False, allow_unicode=True)
    for key, val in sorted(values.items()):
        print(f"{key}: {val}")
    print(f"Wrote {VALUES_YAML.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
