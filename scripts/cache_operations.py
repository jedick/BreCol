"""Shared cache layout, FASTA iteration, row selection, and Parquet read/write for sequence caches."""

from __future__ import annotations

import gzip
import hashlib
import random
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from shared_utilities import TETRAMERS

# --- Sequence row selection (sequence_cache in defaults.yaml) ---

def load_sequence_row_selection(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Read and validate row-selection keys from ``sequence_cache`` in a defaults-style mapping."""
    section = cfg.get("sequence_cache")
    if not isinstance(section, dict):
        raise SystemExit("defaults.yaml must define sequence_cache as a mapping.")
    try:
        seq_offset = int(section["seq_offset"])
        min_seqs = int(section["min_seqs"])
        sample_mode = str(section["sample_mode"]).strip().lower()
        sampling_seed = int(section["sampling_seed"])
    except KeyError as exc:
        raise SystemExit(f"sequence_cache missing required key: {exc.args[0]!r}") from exc
    if seq_offset < 0:
        raise SystemExit("sequence_cache.seq_offset must be >= 0.")
    if min_seqs < 0:
        raise SystemExit("sequence_cache.min_seqs must be >= 0.")
    if sample_mode not in ("head", "random"):
        raise SystemExit("sequence_cache.sample_mode must be 'head' or 'random'.")
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


# --- Hive-partitioned Parquet cache layout ---


def tetramer_cache_dataset_root(cache_dir: Path, n_max: int) -> Path:
    return cache_dir / f"n{n_max}"


def hive_partition_path_segment(key: str, value: str) -> str:
    """One hive directory level; encoding matches PyArrow ``write_to_dataset``."""
    return f"{key}={urllib.parse.quote(str(value), safe='')}"


def run_partition_dir(cache_root: Path, study_name: str, run: str) -> Path:
    return cache_root / hive_partition_path_segment(
        "study_name", study_name
    ) / hive_partition_path_segment("Run", run)


def partition_is_up_to_date(
    cache_root: Path,
    study_name: str,
    run: str,
    fasta_gz: Path,
) -> bool:
    """True when a Parquet partition exists and is not older than the source FASTA."""
    part_dir = run_partition_dir(cache_root, study_name, run)
    if not part_dir.is_dir():
        return False
    parquet_files = list(part_dir.rglob("*.parquet"))
    if not parquet_files:
        return False
    if not fasta_gz.is_file():
        return False
    cache_mtime = max(p.stat().st_mtime for p in parquet_files)
    return fasta_gz.stat().st_mtime <= cache_mtime


def split_jobs_by_cache_state(
    jobs: Sequence[Tuple[str, str, Path]],
    cache_root: Path,
    *,
    force: bool,
) -> Tuple[List[Tuple[str, str, Path]], int]:
    """Return (runs still to build, count already up-to-date vs FASTA mtime).

    Jobs with missing FASTA stay in the pending list so callers can report them.
    When ``force`` is true, every job is pending and the skip count is zero.
    """
    if force:
        return list(jobs), 0
    pending: List[Tuple[str, str, Path]] = []
    n_skip = 0
    for study_name, run, fasta_gz in jobs:
        if (
            fasta_gz.is_file()
            and partition_is_up_to_date(cache_root, study_name, run, fasta_gz)
        ):
            n_skip += 1
        else:
            pending.append((study_name, run, fasta_gz))
    return pending, n_skip


# --- FASTA iteration for cache builders ---


def row_is_sample_used(row: Mapping[str, object]) -> bool:
    return (row.get("sample_used") or "").strip().casefold() == "true"


def count_fasta_records(fasta_gz: Path) -> int:
    """Count FASTA records by header lines only (no sequence assembly)."""
    n = 0
    with gzip.open(fasta_gz, "rt", encoding="ascii", errors="replace") as handle:
        for raw in handle:
            if raw.strip().startswith(">"):
                n += 1
    return n


def iter_selected_fasta_sequences(
    fasta_gz: Path,
    wanted: set[int],
    max_index: int,
) -> Iterable[Tuple[int, str]]:
    """Yield (0-based sequence index, sequence) for selected indices only."""
    seq_index = -1
    collecting = False
    chunks: List[str] = []
    with gzip.open(fasta_gz, "rt", encoding="ascii", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if collecting and chunks:
                    yield seq_index, "".join(chunks)
                seq_index += 1
                if seq_index > max_index:
                    return
                collecting = seq_index in wanted
                chunks = []
                continue
            if collecting:
                chunks.append(line)
        if collecting and chunks:
            yield seq_index, "".join(chunks)


# --- Tetramer count cache ---

TETRAMER_COLUMNS: Tuple[str, ...] = TETRAMERS
TETRAMER_FEATURE_NAMES: List[str] = ["sequence_index", *TETRAMER_COLUMNS]


def write_tetramer_run_partition(
    *,
    cache_root: Path,
    study_name: str,
    run: str,
    sequence_index_1based: np.ndarray,
    counts: np.ndarray,
    compression: Optional[str],
) -> None:
    if counts.shape[0] != sequence_index_1based.shape[0]:
        raise ValueError("sequence_index and counts row counts differ")
    if counts.ndim != 2 or counts.shape[1] != 256:
        raise ValueError("counts must have shape (n, 256)")
    arrays = [pa.array(sequence_index_1based, type=pa.int32())]
    for col in range(256):
        arrays.append(pa.array(counts[:, col], type=pa.int32()))
    table = pa.table(
        dict(zip(TETRAMER_FEATURE_NAMES, arrays)),
    )
    table = table.append_column("study_name", pa.array([study_name] * table.num_rows))
    table = table.append_column("Run", pa.array([run] * table.num_rows))
    pq.write_to_dataset(
        table,
        root_path=str(cache_root),
        partition_cols=["study_name", "Run"],
        compression=compression,
        existing_data_behavior="overwrite_or_ignore",
    )


class TetramerCacheReader:
    """Lazy per-run reader for a hive-partitioned tetramer cache."""

    def __init__(self, cache_root: Path) -> None:
        if not cache_root.is_dir():
            raise FileNotFoundError(f"Tetramer cache not found: {cache_root}")
        self.cache_root = cache_root
        self._dataset = ds.dataset(
            str(cache_root),
            format="parquet",
            partitioning="hive",
        )

    def load_run(self, study_name: str, run: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (sequence_index_1based, counts) with counts float64 (n, 256)."""
        filt = (pc.field("study_name") == pc.scalar(study_name)) & (
            pc.field("Run") == pc.scalar(run)
        )
        table = self._dataset.to_table(columns=list(TETRAMER_FEATURE_NAMES), filter=filt)
        if table.num_rows == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty((0, 256), dtype=np.float64),
            )
        idx = table.column("sequence_index").to_numpy(zero_copy_only=False).astype(
            np.int64, copy=False
        )
        cols = [
            table.column(name).to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
            for name in TETRAMER_COLUMNS
        ]
        counts = np.column_stack(cols)
        return idx, counts
