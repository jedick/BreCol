"""Shared hive-partitioned Parquet cache layout for per-sequence feature caches."""

from __future__ import annotations

import urllib.parse
from pathlib import Path
from typing import List, Sequence, Tuple


def cache_dataset_root(cache_dir: Path, n_max: int) -> Path:
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
