"""Read/write partitioned UC/CAP tetramer count Parquet caches."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow.parquet as pq

from shared_utilities import TETRAMERS


TETRAMER_COLUMNS: Tuple[str, ...] = TETRAMERS
FEATURE_NAMES: List[str] = ["sequence_index", *TETRAMER_COLUMNS]


def tetramer_cache_dataset_root(cache_dir: Path, n_max: int) -> Path:
    return cache_dir / f"n{n_max}"


def run_partition_dir(cache_root: Path, study_name: str, run: str) -> Path:
    return cache_root / f"study_name={study_name}" / f"Run={run}"


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


def write_run_partition(
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
        dict(zip(FEATURE_NAMES, arrays)),
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
        table = self._dataset.to_table(columns=list(FEATURE_NAMES), filter=filt)
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
