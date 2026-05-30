#!/usr/bin/env python3
"""
Build a feature-only HyenaDNA tensor cache per sequencing run (FASTA -> .pt).

This cache is task-agnostic: each run file stores token tensors only, and labels/splits
are joined later by train_hyenadna.py via shared metadata utilities.

Tensor files live under paths.hyenadna_run_tensors_dir as <Run>.pt files. A summary
CSV (sequence_counts.csv) is written next to those .pt files with one row
per study and one column per set position (set_0, set_1, ...) recording the total
number of sequences packed into each set across all runs of that study.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import yaml
from tqdm import tqdm

from hyenadna_fasta_data import (  # noqa: E402
    iter_fasta_sequences,
    make_character_tokenizer,
    run_to_tensors,
    split_sequences_into_sets,
)
from cache_operations import load_sequence_row_selection
from shared_utilities import build_run_table, fasta_path_for_run, resolve_repo_path


def _format_run_tensor_summary(
    *,
    written: int,
    skip_short_seq_offset: int,
    skip_min_seqs: int,
    skip_other: int,
    seq_offset: int,
    min_seqs: int,
) -> str:
    total_skipped = skip_short_seq_offset + skip_min_seqs + skip_other
    parts: List[str] = []
    if seq_offset > 0 and skip_short_seq_offset:
        parts.append(
            f"{skip_short_seq_offset} shorter than seq_offset ({seq_offset})"
        )
    if min_seqs > 0 and skip_min_seqs:
        parts.append(
            f"{skip_min_seqs} without min_seqs ({min_seqs}) after offset"
        )
    if skip_other:
        parts.append(f"{skip_other} other")
    msg = f"\nWrote/updated {written} run tensors; skipped {total_skipped} runs"
    if parts:
        msg += f" ({', '.join(parts)})"
    return msg + "."


def _load_defaults(defaults_path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise SystemExit(f"{defaults_path} must contain a YAML mapping.")
    return cfg


def _read_existing_n_per_set(out_path: Path) -> Optional[List[int]]:
    """Return the saved per-set sequence counts from an existing .pt blob, or None.

    None means the field is missing (e.g. a .pt produced by an older build) and the
    caller should treat the run as not contributing to the summary CSV.
    """
    try:
        blob = torch.load(out_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(blob, dict):
        return None
    val = blob.get("n_per_set")
    if not isinstance(val, list):
        return None
    try:
        return [int(x) for x in val]
    except (TypeError, ValueError):
        return None


def _build_one_run_tensor(
    *,
    repo_root_s: str,
    fasta_dir_key: str,
    run_tensors_root_s: str,
    run: str,
    study_name: str,
    num_sets: int,
    max_length: int,
    seq_offset: int,
    min_seqs: int,
    force: bool,
) -> Tuple[str, str, str, Optional[List[int]]]:
    repo_root = Path(repo_root_s)
    run_tensors_root = Path(run_tensors_root_s)
    out_path = run_tensors_root / f"{run}.pt"
    if out_path.is_file() and not force:
        return ("skipped_existing", run, "", _read_existing_n_per_set(out_path))

    fasta_gz = fasta_path_for_run(repo_root, fasta_dir_key, study_name, run)
    if not fasta_gz.is_file():
        return ("skipped_missing_fasta", run, f"missing_fasta:{fasta_gz}", None)

    tokenizer = make_character_tokenizer(max_length)
    sequences = list(iter_fasta_sequences(fasta_gz))
    n_raw = len(sequences)
    if seq_offset > 0 and n_raw < seq_offset:
        return (
            "skipped_short_seq_offset",
            run,
            f"n_sequences={n_raw}<{seq_offset}",
            None,
        )
    if seq_offset > 0:
        sequences = sequences[seq_offset:]
    n_after_offset = len(sequences)
    if min_seqs > 0 and n_after_offset < min_seqs:
        return (
            "skipped_min_seqs",
            run,
            f"n_sequences_after_offset={n_after_offset}<{min_seqs}",
            None,
        )

    sets = split_sequences_into_sets(sequences, max_length, num_sets)
    n_per_set = [len(s) for s in sets]

    input_ids, attention_mask, n_valid = run_to_tensors(
        sequences,
        tokenizer=tokenizer,
        max_length=max_length,
        num_sets=num_sets,
    )
    if input_ids is None or n_valid == 0:
        return ("skipped_empty", run, "no_tokenized_sequence_content", None)

    torch.save(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "n_sets": int(n_valid),
            "n_raw_sequences": int(n_raw),
            "n_after_offset_sequences": int(n_after_offset),
            "n_per_set": [int(x) for x in n_per_set],
        },
        out_path,
    )
    return ("written", run, "", n_per_set)


def _build_tensor_cache(
    *,
    repo_root: Path,
    defaults_path: Path,
    force: bool,
) -> None:
    cfg = _load_defaults(defaults_path)
    paths_cfg = cfg.get("paths")
    if not isinstance(paths_cfg, dict):
        raise SystemExit(f"{defaults_path} must define paths as a mapping.")

    fasta_dir_key = str(paths_cfg["fasta_dir"]).strip()
    hyenadna_run_tensors_cfg = cfg.get("hyenadna_run_tensors")
    if not isinstance(hyenadna_run_tensors_cfg, dict):
        raise SystemExit(
            f"{defaults_path} must define hyenadna_run_tensors as a mapping."
        )
    selection = load_sequence_row_selection(cfg)
    seq_offset = selection["seq_offset"]
    min_seqs = selection["min_seqs"]
    run_tensors_key = str(
        paths_cfg.get("hyenadna_run_tensors_dir", "outputs/hyenadna_run_tensors")
    ).strip()
    run_tensors_root = resolve_repo_path(repo_root, run_tensors_key)

    run_tensors_root.mkdir(parents=True, exist_ok=True)

    num_sets = int(hyenadna_run_tensors_cfg["num_sets"])
    max_length = int(hyenadna_run_tensors_cfg["max_length"])
    max_workers = int(hyenadna_run_tensors_cfg.get("max_workers", 1))
    if num_sets <= 0:
        raise SystemExit("hyenadna_run_tensors.num_sets must be > 0.")
    if max_length <= 0:
        raise SystemExit("hyenadna_run_tensors.max_length must be > 0.")
    if max_workers <= 0:
        raise SystemExit("hyenadna_run_tensors.max_workers must be >= 1.")

    run_df = build_run_table(config_path=defaults_path)
    runs_frame = run_df[["Run", "study_name"]].drop_duplicates(subset=["Run"])
    run_to_study: Dict[str, str] = dict(
        zip(runs_frame["Run"].astype(str), runs_frame["study_name"].astype(str))
    )

    written = 0
    skipped: List[Dict[str, str]] = []
    skip_short_seq_offset = 0
    skip_min_seqs = 0
    skip_other = 0
    n_existing_without_counts = 0
    sequence_counts_totals: Dict[str, List[int]] = defaultdict(lambda: [0] * num_sets)
    n_total = len(runs_frame)
    print(
        f"\nBuilding HyenaDNA run tensors ({n_total} runs) -> {run_tensors_root} "
        f"(num_sets={num_sets}, max_length={max_length}, "
        f"seq_offset={seq_offset}, min_seqs={min_seqs}, "
        f"max_workers={max_workers})",
        flush=True,
    )

    jobs = [
        (str(row["Run"]).strip(), str(row["study_name"]).strip())
        for _, row in runs_frame.iterrows()
    ]
    worker_kwargs = {
        "repo_root_s": str(repo_root),
        "fasta_dir_key": fasta_dir_key,
        "run_tensors_root_s": str(run_tensors_root),
        "num_sets": num_sets,
        "max_length": max_length,
        "seq_offset": seq_offset,
        "min_seqs": min_seqs,
        "force": force,
    }

    def _record(
        status: str,
        done_run: str,
        reason: str,
        n_per_set: Optional[List[int]],
    ) -> None:
        nonlocal written, skip_short_seq_offset, skip_min_seqs, skip_other
        nonlocal n_existing_without_counts
        if status == "written":
            written += 1
        elif status == "skipped_existing":
            if n_per_set is None:
                n_existing_without_counts += 1
        elif status == "skipped_short_seq_offset":
            skip_short_seq_offset += 1
            skipped.append({"run": done_run, "reason": reason})
        elif status == "skipped_min_seqs":
            skip_min_seqs += 1
            skipped.append({"run": done_run, "reason": reason})
        elif status.startswith("skipped") and reason:
            skip_other += 1
            skipped.append({"run": done_run, "reason": reason})
        if n_per_set is not None:
            study = run_to_study.get(done_run)
            if study is not None:
                totals = sequence_counts_totals[study]
                for i in range(min(len(n_per_set), len(totals))):
                    totals[i] += int(n_per_set[i])

    if max_workers == 1:
        for run, study_name in tqdm(jobs, total=n_total, desc="FASTA -> tensors", unit="run"):
            _record(*_build_one_run_tensor(run=run, study_name=study_name, **worker_kwargs))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(
                    _build_one_run_tensor,
                    run=run,
                    study_name=study_name,
                    **worker_kwargs,
                )
                for run, study_name in jobs
            ]
            for fut in tqdm(as_completed(futures), total=n_total, desc="FASTA -> tensors", unit="run"):
                _record(*fut.result())

    print(
        _format_run_tensor_summary(
            written=written,
            skip_short_seq_offset=skip_short_seq_offset,
            skip_min_seqs=skip_min_seqs,
            skip_other=skip_other,
            seq_offset=seq_offset,
            min_seqs=min_seqs,
        ),
        flush=True,
    )

    _write_sequence_counts_summary(
        repo_root=repo_root,
        paths_cfg=paths_cfg,
        run_tensors_root=run_tensors_root,
        num_sets=num_sets,
        sequence_counts_totals=sequence_counts_totals,
    )

    if n_existing_without_counts:
        print(
            f"Note: {n_existing_without_counts} existing run tensor(s) lack 'n_per_set' "
            "metadata (built before this field was added); their sequences are not "
            "reflected in sequence_counts.csv. Re-run with --force to refresh.",
            flush=True,
        )
    # Makefile tracks the HyenaDNA run tensors directory mtime as the cache target signal.
    os.utime(run_tensors_root, None)


def _write_sequence_counts_summary(
    *,
    repo_root: Path,
    paths_cfg: Dict[str, Any],
    run_tensors_root: Path,
    num_sets: int,
    sequence_counts_totals: Dict[str, List[int]],
) -> None:
    """Write sequence_counts.csv (one row per study from datasets.csv)."""
    datasets_csv_path = resolve_repo_path(
        repo_root, str(paths_cfg.get("datasets_csv", "datasets.csv")).strip()
    )
    studies_in_order = (
        pd.read_csv(datasets_csv_path, usecols=["study_name"], dtype=str)["study_name"]
        .astype(str)
        .tolist()
    )
    set_columns = [f"set_{i}" for i in range(num_sets)]
    rows: List[Dict[str, Any]] = []
    for study in studies_in_order:
        totals = sequence_counts_totals.get(study, [0] * num_sets)
        row: Dict[str, Any] = {"study": study}
        for i, col in enumerate(set_columns):
            row[col] = int(totals[i]) if i < len(totals) else 0
        rows.append(row)
    summary_df = pd.DataFrame(rows, columns=["study", *set_columns])
    csv_out_path = run_tensors_root / "sequence_counts.csv"
    summary_df.to_csv(csv_out_path, index=False)
    print(
        f"Wrote sequence counts ({len(summary_df)} studies) -> {csv_out_path}",
        flush=True,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild existing run tensor files.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = Path(__file__).resolve().parent.parent
    defaults_path = repo_root / "defaults.yaml"

    _build_tensor_cache(
        repo_root=repo_root,
        defaults_path=defaults_path,
        force=bool(args.force),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
