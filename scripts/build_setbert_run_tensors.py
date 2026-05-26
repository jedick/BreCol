#!/usr/bin/env python3
"""
Build a feature-only SetBERT token tensor cache per sequencing run (FASTA -> .pt).

Mirrors scripts/build_run_tensors.py for HyenaDNA, but uses the DNABERT k-mer tokenizer
bundled with the SetBERT checkpoint. Each run file stores token tensors only; labels
and splits are joined later by scripts/train_setbert.py via shared metadata utilities.

For each data/ CSV row with sample_used=TRUE, the builder:
  1. counts FASTA records and skips runs that fail sequence_cache offset / min_seqs;
  2. picks ``setbert.set_size`` 0-based row indices via select_row_indices_0based
     (without replacement, deterministic per Run);
  3. iterates the FASTA and randomly trims each selected sequence to a length drawn
     uniformly in [setbert.min_sequence_length, setbert.max_sequence_length] using the
     per-Run RNG seeded by ``setbert.truncation_seed`` (same convention as
     build_setbert_embeddings.py);
  4. tokenizes with the SetBERT-bundled DNABERT tokenizer;
  5. right-pads each token row to the run's max token length and writes a small .pt
     file at paths.setbert_run_tensors_dir/<Run>.pt.

Resumable: a per-run .pt file is reused when its mtime is >= the source FASTA mtime.
Use --force to rebuild every Run.

Downstream: scripts/train_setbert.py.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

from cache_operations import (
    load_sequence_row_selection,
    row_is_sample_used,
)
from setbert_data import (
    load_setbert_model,
    load_setbert_section,
    pad_set_to_token_len,
    resolve_device,
    select_trimmed_set_for_run,
    tokenize_sequences,
)
from shared_utilities import RUN_PATTERN, fasta_path_for_run, resolve_repo_path


def _load_cfg(defaults_path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise SystemExit(f"{defaults_path} must contain a YAML mapping.")
    return cfg


def _collect_jobs(
    repo_root: Path, data_dir: Path, fasta_dir_key: str
) -> List[Tuple[str, str, Path]]:
    """Return [(study_name, run, fasta_gz_path), ...] in study order, sample_used=TRUE only."""
    jobs: List[Tuple[str, str, Path]] = []
    for csv_path in sorted(data_dir.rglob("*.csv")):
        study_name = csv_path.stem
        with open(csv_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                run = (row.get("Run") or "").strip()
                if not RUN_PATTERN.match(run):
                    continue
                if not row_is_sample_used(row):
                    continue
                fasta_gz = fasta_path_for_run(repo_root, fasta_dir_key, study_name, run)
                jobs.append((study_name, run, fasta_gz))
    return jobs


def _is_up_to_date(pt_path: Path, fasta_gz: Path) -> bool:
    if not pt_path.is_file() or not fasta_gz.is_file():
        return False
    return fasta_gz.stat().st_mtime <= pt_path.stat().st_mtime


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild existing per-run .pt files.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = Path(__file__).resolve().parent.parent
    defaults_path = repo_root / "defaults.yaml"

    try:
        cfg = _load_cfg(defaults_path)
        paths = cfg["paths"]
        data_dir = resolve_repo_path(repo_root, str(paths["data_dir"]))
        fasta_dir_key = str(paths["fasta_dir"]).strip()
        run_tensors_root = resolve_repo_path(
            repo_root, str(paths["setbert_run_tensors_dir"])
        )
        selection = load_sequence_row_selection(cfg)
        settings = load_setbert_section(cfg)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"Invalid pipeline config in {defaults_path}: {exc}", file=sys.stderr)
        return 1

    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        return 1

    set_size = int(settings["set_size"])
    device = resolve_device(settings["device_raw"])

    print(
        f"Loading SetBERT {settings['pretrained_repo']}@{settings['pretrained_revision']} "
        f"for tokenizer on {device}",
        flush=True,
    )
    try:
        _model, tokenizer, embed_dim, pad_token_id, kmer = load_setbert_model(
            pretrained_repo=settings["pretrained_repo"],
            pretrained_revision=settings["pretrained_revision"],
            sequence_encoder_chunk_size=settings["sequence_encoder_chunk_size"],
            device=torch.device("cpu"),
            eval_mode=True,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Failed to load SetBERT model: {exc}", file=sys.stderr)
        return 1
    # Free the encoder weights immediately; the cache build only needs the tokenizer.
    del _model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        f"  embed_dim={embed_dim}, kmer={kmer}, pad_token_id={pad_token_id}",
        flush=True,
    )

    run_tensors_root.mkdir(parents=True, exist_ok=True)

    jobs = _collect_jobs(repo_root, data_dir, fasta_dir_key)
    if not jobs:
        print(f"Error: no eligible runs under {data_dir}", file=sys.stderr)
        return 1
    n_eligible = len(jobs)

    pending: List[Tuple[str, str, Path]] = []
    n_up_to_date = 0
    for study_name, run, fasta_gz in jobs:
        out_path = run_tensors_root / f"{run}.pt"
        if not args.force and _is_up_to_date(out_path, fasta_gz):
            n_up_to_date += 1
            continue
        pending.append((study_name, run, fasta_gz))

    if n_up_to_date:
        print(
            f"Resuming: {n_up_to_date}/{n_eligible} runs already cached "
            f"under {run_tensors_root.name}/; {len(pending)} remaining.",
            flush=True,
        )

    if not pending:
        print(
            f"SetBERT run tensors complete: all {n_eligible} runs present in "
            f"{run_tensors_root}.",
            flush=True,
        )
        return 0

    print(
        f"\nBuilding SetBERT run tensors ({len(pending)} runs) -> {run_tensors_root} "
        f"(set_size={set_size}, trim=[{settings['min_sequence_length']},"
        f"{settings['max_sequence_length']}], kmer={kmer}, pad_token_id={pad_token_id})",
        flush=True,
    )

    written = 0
    n_missing_fasta = 0
    n_skipped_selection = 0
    n_skipped_tokenize = 0
    start = time.perf_counter()

    pbar = tqdm(total=len(pending), desc="FASTA -> setbert tensors", unit="run")
    for study_name, run, fasta_gz in pending:
        if not fasta_gz.is_file():
            n_missing_fasta += 1
            pbar.update(1)
            continue
        try:
            seq_index, sequences, n_raw = select_trimmed_set_for_run(
                fasta_gz,
                seq_offset=selection["seq_offset"],
                min_seqs=selection["min_seqs"],
                set_size=set_size,
                sample_mode=selection["sample_mode"],
                sampling_seed=selection["sampling_seed"],
                truncation_seed=settings["truncation_seed"],
                min_sequence_length=settings["min_sequence_length"],
                max_sequence_length=settings["max_sequence_length"],
                run=run,
            )
        except (OSError, ValueError) as exc:
            pbar.write(f"\n{study_name}/{run}: {exc}", file=sys.stderr)
            pbar.update(1)
            n_skipped_selection += 1
            continue
        if seq_index is None or sequences is None:
            n_skipped_selection += 1
            pbar.update(1)
            continue

        try:
            token_rows = tokenize_sequences(sequences, tokenizer)
        except ValueError as exc:
            pbar.write(f"\n{study_name}/{run}: {exc}", file=sys.stderr)
            pbar.update(1)
            n_skipped_tokenize += 1
            continue

        target_token_len = max(len(row) for row in token_rows)
        token_array = pad_set_to_token_len(
            token_rows,
            target_token_len=target_token_len,
            pad_token_id=pad_token_id,
        )
        input_ids = torch.from_numpy(token_array)

        n_after_offset = max(0, n_raw - selection["seq_offset"])
        out_path = run_tensors_root / f"{run}.pt"
        torch.save(
            {
                "input_ids": input_ids,
                "sequence_index_1based": torch.from_numpy(seq_index),
                "set_size": int(set_size),
                "token_len": int(target_token_len),
                "pad_token_id": int(pad_token_id),
                "kmer": int(kmer),
                "pretrained_repo": settings["pretrained_repo"],
                "pretrained_revision": settings["pretrained_revision"],
                "study_name": str(study_name),
                "n_raw_sequences": int(n_raw),
                "n_after_offset_sequences": int(n_after_offset),
                "seq_offset": int(selection["seq_offset"]),
                "min_seqs": int(selection["min_seqs"]),
                "sample_mode": str(selection["sample_mode"]),
                "sampling_seed": int(selection["sampling_seed"]),
                "truncation_seed": int(settings["truncation_seed"]),
                "min_sequence_length": int(settings["min_sequence_length"]),
                "max_sequence_length": int(settings["max_sequence_length"]),
            },
            out_path,
        )
        written += 1
        pbar.update(1)
    pbar.close()

    elapsed = time.perf_counter() - start
    print(f"\nWrote/updated {written} run tensors under {run_tensors_root}")
    if n_skipped_selection:
        print(f"Skipped (selection / too few sequences): {n_skipped_selection}")
    if n_skipped_tokenize:
        print(f"Skipped (tokenization error): {n_skipped_tokenize}")
    if n_missing_fasta:
        print(f"Skipped (missing FASTA): {n_missing_fasta}")
    print(f"Elapsed: {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
