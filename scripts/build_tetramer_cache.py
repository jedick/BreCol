#!/usr/bin/env python3
"""
Build a hive-partitioned per-sequence tetramer count cache from FASTA files.

For each data/ CSV row with sample_used=TRUE, reads fasta/<study_name>/<Run>.fasta.gz,
applies sequence_cache row selection from defaults.yaml, then counts 4-mers (Numba when available)
only for selected sequences and writes Parquet under paths.tetramer_cache_dir.

Downstream: calculate_tetramer_frequencies.py (run-level sums) and run_uc_cap_pipeline.py.

Use --first-run-per-study for a quick sanity check.
Use --no-numba to force pure-Python counting. Use --force to rebuild all partitions.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple, Union

import yaml
from tqdm import tqdm

try:
    import numpy as np
    from numba import njit

    _BASE_LUT_NUMBA = np.full(256, -1, dtype=np.int8)
    _BASE_LUT_NUMBA[ord("A")] = 0
    _BASE_LUT_NUMBA[ord("C")] = 1
    _BASE_LUT_NUMBA[ord("G")] = 2
    _BASE_LUT_NUMBA[ord("T")] = 3

    @njit(cache=True)
    def _count_tetramers_numba(
        buf: np.ndarray, counts: np.ndarray, lut: np.ndarray
    ) -> None:
        n = buf.shape[0]
        for i in range(n - 3):
            a = lut[buf[i]]
            if a < 0:
                continue
            b = lut[buf[i + 1]]
            if b < 0:
                continue
            c = lut[buf[i + 2]]
            if c < 0:
                continue
            d = lut[buf[i + 3]]
            if d < 0:
                continue
            idx = (((a << 2) | b) << 2 | c) << 2 | d
            counts[idx] += 1

    _NUMBA_KERNEL_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    _NUMBA_KERNEL_AVAILABLE = False
    _BASE_LUT_NUMBA = None  # type: ignore[assignment]
    _count_tetramers_numba = None  # type: ignore[assignment]

_USE_NUMBA_COUNTING = False

from cache_operations import (  # noqa: E402
    count_fasta_records,
    iter_selected_fasta_sequences,
    load_sequence_row_selection,
    partition_is_up_to_date,
    row_is_sample_used,
    run_partition_dir,
    select_row_indices_0based,
    skip_reason,
    split_jobs_by_cache_state,
    tetramer_cache_dataset_root,
    write_tetramer_run_partition,
)
from hyenadna_fasta_data import fasta_path_for_run, resolve_repo_path
from shared_utilities import RUN_PATTERN

_BASE_BITS = {"A": 0, "C": 1, "G": 2, "T": 3}


def _resolve_repo_path(repo_root: Path, raw: str) -> Path:
    return resolve_repo_path(repo_root, raw)


def count_tetramers_in_sequence(seq: str, counts: List[int]) -> None:
    if len(seq) < 4:
        return
    s = seq.upper()
    n = len(s)
    for i in range(n - 3):
        a, b, c, d = s[i], s[i + 1], s[i + 2], s[i + 3]
        bits = _BASE_BITS.get(a)
        if bits is None:
            continue
        bbits = _BASE_BITS.get(b)
        if bbits is None:
            continue
        cbits = _BASE_BITS.get(c)
        if cbits is None:
            continue
        dbits = _BASE_BITS.get(d)
        if dbits is None:
            continue
        idx = (((bits << 2) | bbits) << 2 | cbits) << 2 | dbits
        counts[idx] += 1


def _warmup_numba_kernel() -> None:
    if not (_NUMBA_KERNEL_AVAILABLE and np is not None and _count_tetramers_numba is not None):
        return
    buf = np.array((65, 67, 71, 84), dtype=np.uint8)
    tmp = np.zeros(256, dtype=np.int64)
    _count_tetramers_numba(buf, tmp, _BASE_LUT_NUMBA)


def configure_counting_backend(use_numba: bool) -> None:
    global _USE_NUMBA_COUNTING
    if use_numba and _NUMBA_KERNEL_AVAILABLE:
        _USE_NUMBA_COUNTING = True
        _warmup_numba_kernel()
    else:
        _USE_NUMBA_COUNTING = False


def accumulate_tetramers_from_sequence(
    seq: str, counts_buffer: Union[List[int], "np.ndarray"]
) -> None:
    if _USE_NUMBA_COUNTING:
        if np is None or _count_tetramers_numba is None or _BASE_LUT_NUMBA is None:
            raise RuntimeError("Numba counting requested but dependencies are missing")
        if len(seq) < 4:
            return
        buf = np.frombuffer(memoryview(seq.upper().encode("ascii")), dtype=np.uint8)
        _count_tetramers_numba(buf, counts_buffer, _BASE_LUT_NUMBA)
    else:
        count_tetramers_in_sequence(seq, counts_buffer)  # type: ignore[arg-type]


def _count_one_sequence(seq: str) -> np.ndarray:
    if np is None:
        raise RuntimeError("NumPy is required")
    row = np.zeros(256, dtype=np.int64)
    accumulate_tetramers_from_sequence(seq, row)
    return row


def selected_counts_from_fasta(
    fasta_gz: Path,
    *,
    seq_offset: int,
    min_seqs: int,
    n_max: int,
    sample_mode: str,
    sampling_seed: int,
    run: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Select row indices from a header-only record count, then 4-mer count those rows only."""
    if np is None:
        raise RuntimeError("NumPy is required")

    n_raw = count_fasta_records(fasta_gz)
    if skip_reason(n_raw, seq_offset=seq_offset, min_seqs=min_seqs) is not None:
        return None, None

    indices_0 = select_row_indices_0based(
        n_raw,
        seq_offset=seq_offset,
        min_seqs=min_seqs,
        n_max=n_max,
        sample_mode=sample_mode,
        sampling_seed=sampling_seed,
        run=run,
    )
    if indices_0 is None or indices_0.size == 0:
        return None, None

    wanted = {int(i) for i in indices_0.tolist()}
    max_index = max(wanted)
    index_rows: List[int] = []
    count_rows: List[np.ndarray] = []
    for seq_index, seq in iter_selected_fasta_sequences(fasta_gz, wanted, max_index):
        count_rows.append(_count_one_sequence(seq))
        index_rows.append(seq_index + 1)

    if len(count_rows) != int(indices_0.size):
        return None, None

    return (
        np.asarray(index_rows, dtype=np.int32),
        np.vstack(count_rows),
    )


def _build_one_run_partition(
    *,
    fasta_path_s: str,
    study_name: str,
    run: str,
    cache_root_s: str,
    n_max: int,
    seq_offset: int,
    min_seqs: int,
    sample_mode: str,
    sampling_seed: int,
    parquet_compression: Optional[str],
    force: bool,
) -> Tuple[str, int]:
    fasta_gz = Path(fasta_path_s)
    cache_root = Path(cache_root_s)

    if not fasta_gz.is_file():
        return ("missing_fasta", 0)
    if not force and partition_is_up_to_date(cache_root, study_name, run, fasta_gz):
        return ("skipped_existing", 0)

    try:
        seq_index, counts = selected_counts_from_fasta(
            fasta_gz,
            seq_offset=seq_offset,
            min_seqs=min_seqs,
            n_max=n_max,
            sample_mode=sample_mode,
            sampling_seed=sampling_seed,
            run=run,
        )
    except OSError as exc:
        raise OSError(f"{study_name}/{run}: {exc}") from exc

    if seq_index is None or counts is None or counts.shape[0] == 0:
        return ("skipped_selection", 0)

    if force:
        part_dir = run_partition_dir(cache_root, study_name, run)
        if part_dir.is_dir():
            shutil.rmtree(part_dir)

    write_tetramer_run_partition(
        cache_root=cache_root,
        study_name=study_name,
        run=run,
        sequence_index_1based=seq_index,
        counts=counts,
        compression=parquet_compression,
    )
    return ("written", int(counts.shape[0]))


def _collect_jobs(repo_root: Path, data_dir: Path, fasta_dir_key: str) -> List[Tuple[str, str, Path]]:
    jobs: List[Tuple[str, str, Path]] = []
    for csv_path in sorted(data_dir.rglob("*.csv")):
        study_name = csv_path.stem
        with open(csv_path, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                run = (row.get("Run") or "").strip()
                if not RUN_PATTERN.match(run):
                    continue
                if not row_is_sample_used(row):
                    continue
                fasta_gz = fasta_path_for_run(repo_root, fasta_dir_key, study_name, run)
                jobs.append((study_name, run, fasta_gz))
    return jobs


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--first-run-per-study",
        action="store_true",
        help="Process only the first eligible run per study CSV.",
    )
    parser.add_argument(
        "--no-numba",
        action="store_true",
        help="Disable Numba JIT counting even if NumPy/Numba are installed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild Parquet partitions even when FASTA is not newer.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    configure_counting_backend(use_numba=not args.no_numba)
    if not _NUMBA_KERNEL_AVAILABLE and not args.no_numba:
        print("Numba unavailable; using pure-Python tetramer counting.", file=sys.stderr)

    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = repo_root / "defaults.yaml"
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        cache_cfg = cfg["sequence_cache"]
        n_max = int(cache_cfg["n_max_per_run"])
        compression_raw = str(cache_cfg["parquet_compression"]).strip()
        max_workers = int(cache_cfg.get("max_workers", 1))
        paths = cfg["paths"]
        data_dir = _resolve_repo_path(repo_root, str(paths["data_dir"]))
        fasta_dir_key = str(paths["fasta_dir"]).strip()
        cache_dir = _resolve_repo_path(repo_root, str(paths["tetramer_cache_dir"]))
        selection = load_sequence_row_selection(cfg)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Invalid pipeline config in {cfg_path}: {exc}", file=sys.stderr)
        return 1

    valid_compression = frozenset(("zstd", "snappy", "gzip", "brotli", "lz4", "none"))
    if compression_raw not in valid_compression:
        print(
            f"Invalid sequence_cache.parquet_compression in {cfg_path}: "
            f"{compression_raw!r}",
            file=sys.stderr,
        )
        return 1
    if n_max <= 0:
        print("sequence_cache.n_max_per_run must be a positive integer", file=sys.stderr)
        return 1
    if max_workers < 1:
        print("sequence_cache.max_workers must be >= 1", file=sys.stderr)
        return 1

    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        return 1

    parquet_compression = None if compression_raw == "none" else compression_raw
    cache_root = tetramer_cache_dataset_root(cache_dir, n_max)
    complete_marker = cache_root / "_complete"
    cache_root.mkdir(parents=True, exist_ok=True)

    jobs = _collect_jobs(repo_root, data_dir, fasta_dir_key)
    if args.first_run_per_study:
        seen: set[str] = set()
        filtered: List[Tuple[str, str, Path]] = []
        for study_name, run, fasta_gz in jobs:
            if study_name in seen:
                continue
            seen.add(study_name)
            filtered.append((study_name, run, fasta_gz))
        jobs = filtered

    if not jobs:
        print(f"Error: no eligible runs under {data_dir}", file=sys.stderr)
        return 1

    n_eligible = len(jobs)
    all_jobs = jobs
    _, n_skipped_upfront = split_jobs_by_cache_state(
        all_jobs, cache_root, force=args.force
    )
    if n_skipped_upfront and not args.force:
        print(
            f"Resuming: {n_skipped_upfront}/{n_eligible} partition(s) up-to-date "
            f"(FASTA not newer than cache); {n_eligible - n_skipped_upfront} run(s) remaining.",
            flush=True,
        )
    if n_skipped_upfront == n_eligible and not args.force:
        complete_marker.write_text("ok\n", encoding="utf-8")
        print(
            f"Tetramer cache complete: all {n_eligible} partition(s) up-to-date under {cache_root}.",
            flush=True,
        )
        return 0

    start = time.perf_counter()
    n_rows_total = 0
    n_written = 0
    n_skipped_existing = 0
    n_skipped_selection = 0
    n_missing_fasta = 0

    print(
        f"Building tetramer cache ({n_eligible} runs) -> {cache_root}; "
        f"n_max={n_max}, sample_mode={selection['sample_mode']}, "
        f"max_workers={max_workers}, force={args.force}.",
        flush=True,
    )

    worker_kwargs = {
        "cache_root_s": str(cache_root),
        "n_max": n_max,
        "seq_offset": selection["seq_offset"],
        "min_seqs": selection["min_seqs"],
        "sample_mode": selection["sample_mode"],
        "sampling_seed": selection["sampling_seed"],
        "parquet_compression": parquet_compression,
        "force": args.force,
    }

    def _handle(status: str, n_rows: int) -> None:
        nonlocal n_rows_total, n_written, n_skipped_existing, n_skipped_selection, n_missing_fasta
        if status == "written":
            n_written += 1
            n_rows_total += n_rows
        elif status == "skipped_existing":
            n_skipped_existing += 1
        elif status == "skipped_selection":
            n_skipped_selection += 1
        elif status == "missing_fasta":
            n_missing_fasta += 1

    if max_workers == 1:
        for study_name, run, fasta_gz in tqdm(
            all_jobs, total=n_eligible, desc="tetramer cache", unit="run"
        ):
            try:
                status, n_rows = _build_one_run_partition(
                    fasta_path_s=str(fasta_gz),
                    study_name=study_name,
                    run=run,
                    **worker_kwargs,
                )
            except (OSError, ValueError) as exc:
                print(f"\nError: {exc}", file=sys.stderr)
                return 1
            _handle(status, n_rows)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(
                    _build_one_run_partition,
                    fasta_path_s=str(fasta_gz),
                    study_name=study_name,
                    run=run,
                    **worker_kwargs,
                )
                for study_name, run, fasta_gz in all_jobs
            ]
            for fut in tqdm(
                as_completed(futures),
                total=n_eligible,
                desc="tetramer cache",
                unit="run",
            ):
                try:
                    status, n_rows = fut.result()
                except (OSError, ValueError) as exc:
                    print(f"\nWorker failed: {exc}", file=sys.stderr)
                    return 1
                _handle(status, n_rows)

    if n_written == 0 and n_skipped_existing == 0:
        print("Error: no runs were written or reused in the tetramer cache", file=sys.stderr)
        return 1

    complete_marker.write_text("ok\n", encoding="utf-8")
    elapsed = time.perf_counter() - start
    print(f"Wrote {n_rows_total} rows across {n_written} runs to {cache_root}")
    if n_skipped_existing:
        print(f"Skipped up-to-date partitions: {n_skipped_existing}")
    if n_skipped_selection:
        print(f"Skipped (selection / too few sequences): {n_skipped_selection}")
    if n_missing_fasta:
        print(f"Skipped (missing FASTA): {n_missing_fasta}")
    print(f"Elapsed: {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
