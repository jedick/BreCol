"""Shared per-run sequence row selection (FASTA records or tetramer count file rows)."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


def load_sequence_selection(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Read and validate ``sequence_selection`` from a defaults-style mapping."""
    section = cfg.get("sequence_selection")
    if not isinstance(section, dict):
        raise SystemExit("defaults.yaml must define sequence_selection as a mapping.")
    seq_offset = int(section["seq_offset"])
    min_seqs = int(section["min_seqs"])
    sample_mode = str(section["sample_mode"]).strip().lower()
    sampling_seed = int(section["sampling_seed"])
    if seq_offset < 0:
        raise SystemExit("sequence_selection.seq_offset must be >= 0.")
    if min_seqs < 0:
        raise SystemExit("sequence_selection.min_seqs must be >= 0.")
    if sample_mode not in ("head", "random"):
        raise SystemExit("sequence_selection.sample_mode must be 'head' or 'random'.")
    return {
        "seq_offset": seq_offset,
        "min_seqs": min_seqs,
        "sample_mode": sample_mode,
        "sampling_seed": sampling_seed,
    }


def run_sampling_seed(sampling_seed: int, run: str) -> int:
    run_seed = int(hashlib.sha256(run.encode("utf-8")).hexdigest()[:8], 16)
    return int(sampling_seed) + run_seed


def select_row_indices_0based(
    n_rows: int,
    *,
    seq_offset: int,
    min_seqs: int,
    n_max: int,
    sample_mode: str,
    sampling_seed: int,
    run: str,
) -> Optional[np.ndarray]:
    """
    Choose 0-based row indices into a per-run count file.

    Returns None when the run should be skipped (too few rows after offset).
    """
    if n_rows < 0:
        raise ValueError("n_rows must be >= 0")
    if n_max <= 0:
        raise ValueError("n_max must be positive")
    if seq_offset > 0 and n_rows < seq_offset:
        return None
    pool_size = n_rows - seq_offset
    if min_seqs > 0 and pool_size < min_seqs:
        return None
    n_take = min(n_max, pool_size)
    if n_take <= 0:
        return None
    if sample_mode == "head":
        return np.arange(seq_offset, seq_offset + n_take, dtype=np.int64)
    if sample_mode == "random":
        rng = random.Random(run_sampling_seed(sampling_seed, run))
        rel = sorted(rng.sample(range(pool_size), n_take))
        return np.asarray([seq_offset + i for i in rel], dtype=np.int64)
    raise ValueError(f"Unknown sample_mode {sample_mode!r}")


def source_row_indices_1based(indices_0based: np.ndarray) -> np.ndarray:
    """Map 0-based file row indices to 1-based source-file sequence_index values."""
    return indices_0based.astype(np.int64, copy=False) + 1


def skip_reason(
    n_rows: int,
    *,
    seq_offset: int,
    min_seqs: int,
) -> Optional[str]:
    if seq_offset > 0 and n_rows < seq_offset:
        return f"n_rows={n_rows}<{seq_offset}"
    pool_size = n_rows - seq_offset
    if min_seqs > 0 and pool_size < min_seqs:
        return f"n_rows_after_offset={pool_size}<{min_seqs}"
    return None
