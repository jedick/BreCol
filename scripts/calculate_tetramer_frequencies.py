#!/usr/bin/env python3
"""
Compute tetramer frequency profiles per sequencing run (Run), incrementally.

For each CSV row under data/ with sample_used=TRUE (case-insensitive), reads the
matching hive partition from the tetramer Parquet cache (build_tetramer_cache.py),
sums selected sequence-level counts for the run, converts to percentages (rounded
to 3 decimals), and appends new rows to paths.tetramer_frequencies_csv.

That CSV is feature-only: one Run column plus 256 lexicographic ACGT tetramer columns.
Labels/splits come from study CSV metadata via shared utilities downstream.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from shared_utilities import RUN_PATTERN, TETRAMERS
from cache_operations import TetramerCacheReader, tetramer_cache_dataset_root


def row_is_sample_used(row: Mapping[str, object]) -> bool:
    return (row.get("sample_used") or "").strip().casefold() == "true"


def percentages_from_counts(counts: Sequence[int]) -> List[float]:
    total = sum(counts)
    if total == 0:
        return [0.0] * 256
    return [round(100.0 * c / total, 3) for c in counts]


def load_existing_runs(output_path: Path) -> set[str]:
    if not output_path.is_file():
        return set()
    runs: set[str] = set()
    with open(output_path, newline="") as in_f:
        reader = csv.DictReader(in_f)
        for row in reader:
            run = (row.get("Run") or "").strip()
            if run:
                runs.add(run)
    return runs


def _resolve_repo_path(repo_root: Path, raw: str) -> Path:
    p = Path(str(raw).strip())
    return p if p.is_absolute() else repo_root / p


def _default_paths_from_defaults_yaml(
    repo_root: Path,
) -> Tuple[Path, Path, Path, int]:
    """Return (data_dir, cache_root, tetramer_frequencies_csv, n_max)."""
    cfg_path = repo_root / "defaults.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    paths = cfg["paths"]
    n_max = int(cfg["sequence_cache"]["n_max_per_run"])
    cache_dir = _resolve_repo_path(repo_root, str(paths["tetramer_cache_dir"]))
    cache_root = tetramer_cache_dataset_root(cache_dir, n_max)
    return (
        _resolve_repo_path(repo_root, str(paths["data_dir"])),
        cache_root,
        _resolve_repo_path(repo_root, str(paths["tetramer_frequencies_csv"])),
        n_max,
    )


def counts_from_cache_partition(
    reader: TetramerCacheReader,
    study_name: str,
    run: str,
) -> Tuple[Optional[List[int]], Optional[str]]:
    try:
        _, matrix = reader.load_run(study_name, run)
    except OSError as exc:
        return None, str(exc)
    if matrix.shape[0] == 0:
        return None, "no data rows"
    totals = matrix.sum(axis=0).astype(np.int64).tolist()
    return totals, None


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir, cache_root, output_path, _n_max = _default_paths_from_defaults_yaml(repo_root)
    complete_marker = cache_root / "_complete"

    if not complete_marker.is_file():
        raise SystemExit(
            f"Tetramer cache not built: {cache_root} (run: make tetramer_cache)"
        )

    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        return 1

    data_files = sorted(data_dir.rglob("*.csv"))
    if not data_files:
        print(f"Error: no CSV files under {data_dir}", file=sys.stderr)
        return 1

    reader = TetramerCacheReader(cache_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["Run", *list(TETRAMERS)]
    rows_written = 0
    rows_already_in_output = 0
    rows_missing_counts = 0
    rows_zero_kmers = 0
    status_width = 100

    def show_run_progress(run_i: int, n_runs: int, run: str, note: str = "") -> None:
        tail = f"  {note}" if note else ""
        line = f"  {run_i}/{n_runs} runs  (current: {run}){tail}"
        sys.stdout.write("\r" + line.ljust(status_width))
        sys.stdout.flush()

    existing_runs = load_existing_runs(output_path)
    output_exists = output_path.is_file()
    output_needs_header = (not output_exists) or output_path.stat().st_size == 0
    with open(output_path, "a", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        if output_needs_header:
            writer.writeheader()

        for csv_path in data_files:
            study_name = csv_path.stem
            rel_path = csv_path.relative_to(data_dir)
            data_file = rel_path.as_posix()

            with open(csv_path, newline="") as in_f:
                reader_csv = csv.DictReader(in_f)
                rows = list(reader_csv)

            n_runs = sum(
                1
                for row in rows
                if RUN_PATTERN.match((row.get("Run") or "").strip())
                and row_is_sample_used(row)
            )
            print(f"Study data from {data_file}: {n_runs} runs", flush=True)

            study_written = 0
            study_missing = 0
            study_zero = 0
            run_i = 0

            for row in rows:
                run = (row.get("Run") or "").strip()
                if not RUN_PATTERN.match(run):
                    continue
                if not row_is_sample_used(row):
                    continue

                run_i += 1
                show_run_progress(run_i, n_runs, run)

                if run in existing_runs:
                    rows_already_in_output += 1
                    show_run_progress(run_i, n_runs, run, "skipped: row already exists")
                    continue

                counts, err = counts_from_cache_partition(reader, study_name, run)
                if err is not None or counts is None:
                    print(
                        f"Warning: missing tetramer cache for {study_name}/{run}: {err}",
                        file=sys.stderr,
                    )
                    rows_missing_counts += 1
                    study_missing += 1
                    show_run_progress(run_i, n_runs, run, "skipped: no cache partition")
                    continue

                if sum(counts) == 0:
                    rows_zero_kmers += 1
                    study_zero += 1
                    print(
                        f"Warning: zero tetramers in cache for {study_name}/{run}",
                        file=sys.stderr,
                    )

                pct = percentages_from_counts(counts)
                out_row: Dict[str, object] = {"Run": run}
                for kmer, val in zip(TETRAMERS, pct):
                    out_row[kmer] = val
                writer.writerow(out_row)
                rows_written += 1
                study_written += 1
                existing_runs.add(run)
                show_run_progress(run_i, n_runs, run, "wrote row")

            sys.stdout.write("\n")
            sys.stdout.flush()
            print(
                f"  Finished: wrote {study_written}, "
                f"skipped (missing cache) {study_missing}, "
                f"zero tetramer totals {study_zero}",
                flush=True,
            )

    print(f"Appended {rows_written} new rows to {output_path}")
    print(f"Rows already in output: {rows_already_in_output}")
    if rows_missing_counts:
        print(f"Skipped (missing cache partitions): {rows_missing_counts}")
    if rows_zero_kmers:
        print(f"Runs with zero tetramer totals: {rows_zero_kmers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
