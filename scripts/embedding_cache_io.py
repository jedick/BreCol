"""Read/write hive-partitioned per-sequence HyenaDNA embedding Parquet caches."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow.parquet as pq

from sequence_cache_io import cache_dataset_root, partition_is_up_to_date, run_partition_dir

embedding_cache_dataset_root = cache_dataset_root

EMBED_COLUMN_PREFIX = "embed_"


def embed_column_names(embed_dim: int) -> Tuple[str, ...]:
    return tuple(f"{EMBED_COLUMN_PREFIX}{i}" for i in range(embed_dim))


def write_run_partition(
    *,
    cache_root: Path,
    study_name: str,
    run: str,
    sequence_index_1based: np.ndarray,
    embeddings: np.ndarray,
    compression: Optional[str],
) -> None:
    if embeddings.shape[0] != sequence_index_1based.shape[0]:
        raise ValueError("sequence_index and embeddings row counts differ")
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2-D")
    embed_dim = embeddings.shape[1]
    names = embed_column_names(embed_dim)
    arrays = [pa.array(sequence_index_1based, type=pa.int32())]
    for col in range(embed_dim):
        arrays.append(pa.array(embeddings[:, col], type=pa.float32()))
    table = pa.table(dict(zip(["sequence_index", *names], arrays)))
    table = table.append_column("study_name", pa.array([study_name] * table.num_rows))
    table = table.append_column("Run", pa.array([run] * table.num_rows))
    pq.write_to_dataset(
        table,
        root_path=str(cache_root),
        partition_cols=["study_name", "Run"],
        compression=compression,
        existing_data_behavior="overwrite_or_ignore",
    )


class EmbeddingCacheReader:
    """Lazy per-run reader for a hive-partitioned embedding cache."""

    def __init__(self, cache_root: Path) -> None:
        if not cache_root.is_dir():
            raise FileNotFoundError(f"Embedding cache not found: {cache_root}")
        self.cache_root = cache_root
        self._dataset = ds.dataset(
            str(cache_root),
            format="parquet",
            partitioning="hive",
        )
        schema_names = self._dataset.schema.names
        self._embed_cols = sorted(
            c for c in schema_names if c.startswith(EMBED_COLUMN_PREFIX)
        )
        if not self._embed_cols:
            raise ValueError(f"No {EMBED_COLUMN_PREFIX}* columns in embedding cache schema")
        self.embed_dim = len(self._embed_cols)

    def load_run(self, study_name: str, run: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (sequence_index_1based, embeddings) with embeddings float64 (n, d)."""
        columns = ["sequence_index", *self._embed_cols]
        filt = (pc.field("study_name") == pc.scalar(study_name)) & (
            pc.field("Run") == pc.scalar(run)
        )
        table = self._dataset.to_table(columns=columns, filter=filt)
        if table.num_rows == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty((0, self.embed_dim), dtype=np.float64),
            )
        idx = table.column("sequence_index").to_numpy(zero_copy_only=False).astype(
            np.int64, copy=False
        )
        cols = [
            table.column(name).to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
            for name in self._embed_cols
        ]
        return idx, np.column_stack(cols)
