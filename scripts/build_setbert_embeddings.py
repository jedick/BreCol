#!/usr/bin/env python3
"""
Build run-level SetBERT [CLS] embeddings from FASTA files.

For each data/ CSV row with sample_used=TRUE, samples `set_size` sequences from
fasta/<study_name>/<Run>.fasta.gz (sequence_cache row selection), randomly trims each
sequence to defaults.yaml ``setbert.min_sequence_length`` .. ``setbert.max_sequence_length``,
tokenizes with the DNABERT tokenizer bundled in the SetBERT checkpoint, and forwards
through SetBERT to obtain a single embed_dim-sized [CLS] vector per run. Embeddings are
appended row-by-row to paths.setbert_embeddings_csv as ``Run, dim_0, ..., dim_{D-1}``
(D = ``model.config.embed_dim``).

When ``setbert.store_sequence_embeddings`` is true, the per-sequence contextualized
embeddings (output["sequences"]) are also written as hive-partitioned Parquet under
paths.setbert_embedding_cache_dir/n<set_size>/study_name=<x>/Run=<y>/.

Downstream: scripts/fit_classifier.py --setbert.
"""

from __future__ import annotations

import csv
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

from cache_operations import (
    cache_dataset_root,
    load_sequence_row_selection,
    row_is_sample_used,
    run_partition_dir,
    write_embedding_run_partition,
)
from setbert_data import (
    build_batch_tokens,
    load_setbert_model,
    load_setbert_section,
    resolve_device,
    select_trimmed_set_for_run,
    tokenize_sequences,
)
from shared_utilities import RUN_PATTERN, fasta_path_for_run


def _resolve_repo_path(repo_root: Path, raw: object) -> Path:
    p = Path(str(raw).strip())
    return p if p.is_absolute() else repo_root / p


# ----- Existing-CSV bookkeeping -----


def _load_existing_runs(csv_path: Path) -> set[str]:
    if not csv_path.is_file():
        return set()
    runs: set[str] = set()
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            run = (row.get("Run") or "").strip()
            if run:
                runs.add(run)
    return runs


def _write_run_rows(
    csv_path: Path,
    *,
    rows: Sequence[Tuple[str, np.ndarray]],
    fieldnames: Sequence[str],
) -> None:
    """Append one CSV row per (Run, embedding) pair. Writes header lazily."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = (not csv_path.is_file()) or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if needs_header:
            writer.writerow(fieldnames)
        for run, emb in rows:
            if emb.ndim != 1:
                raise ValueError("Per-run embedding must be a 1-D array.")
            writer.writerow([run, *(f"{float(v):.6f}" for v in emb.tolist())])


# ----- Forward pass -----


def _forward_batch(
    model: torch.nn.Module,
    batch_tokens_np: np.ndarray,
    *,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Run SetBERT forward on a (B, set_size, T) int array. Return {'class','sequences'} numpy."""
    input_ids = torch.from_numpy(batch_tokens_np).to(device)
    with torch.inference_mode():
        out = model(sequence_tokens=input_ids)
    cls = out["class"].to(dtype=torch.float32).cpu().numpy()
    seqs = out["sequences"].to(dtype=torch.float32).cpu().numpy()
    return {"class": cls, "sequences": seqs}


# ----- Job collection -----


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


# ----- Main -----


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = repo_root / "defaults.yaml"
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        paths = cfg["paths"]
        data_dir = _resolve_repo_path(repo_root, str(paths["data_dir"]))
        fasta_dir_key = str(paths["fasta_dir"]).strip()
        csv_out_path = _resolve_repo_path(
            repo_root, str(paths["setbert_embeddings_csv"])
        )
        seq_cache_cache_dir = _resolve_repo_path(
            repo_root, str(paths["setbert_embedding_cache_dir"])
        )
        selection = load_sequence_row_selection(cfg)
        settings = load_setbert_section(cfg)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Invalid pipeline config in {cfg_path}: {exc}", file=sys.stderr)
        return 1

    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        return 1

    set_size = settings["set_size"]
    # Length policy: when min == max the per-sequence length is fixed (paper-faithful) and only
    # the trim offset is randomized. When min < max, both the length and offset are drawn
    # per sequence via the per-Run NumPy Generator in _select_trimmed_set_for_run.

    device = resolve_device(settings["device_raw"])
    print(
        f"Loading SetBERT {settings['pretrained_repo']}@{settings['pretrained_revision']} "
        f"on {device}",
        flush=True,
    )
    try:
        model, tokenizer, embed_dim, pad_token_id, kmer = load_setbert_model(
            pretrained_repo=settings["pretrained_repo"],
            pretrained_revision=settings["pretrained_revision"],
            sequence_encoder_chunk_size=settings["sequence_encoder_chunk_size"],
            device=device,
            eval_mode=True,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Failed to load SetBERT model: {exc}", file=sys.stderr)
        return 1
    print(
        f"  embed_dim={embed_dim}, kmer={kmer}, pad_token_id={pad_token_id}, "
        f"chunk_size={model.config.sequence_encoder_chunk_size}",
        flush=True,
    )

    jobs = _collect_jobs(repo_root, data_dir, fasta_dir_key)
    if not jobs:
        print(f"Error: no eligible runs under {data_dir}", file=sys.stderr)
        return 1
    n_eligible = len(jobs)

    existing_runs = _load_existing_runs(csv_out_path)
    pending_jobs = [job for job in jobs if job[1] not in existing_runs]
    n_skipped_existing = n_eligible - len(pending_jobs)
    if n_skipped_existing:
        print(
            f"Resuming: {n_skipped_existing}/{n_eligible} runs already in "
            f"{csv_out_path.name}; {len(pending_jobs)} remaining.",
            flush=True,
        )
    if not pending_jobs:
        print(
            f"SetBERT embeddings complete: all {n_eligible} runs present in "
            f"{csv_out_path}.",
            flush=True,
        )
        return 0

    store_sequences = settings["store_sequence_embeddings"]
    parquet_compression = (
        None
        if settings["parquet_compression"] == "none"
        else settings["parquet_compression"]
    )
    seq_cache_root: Optional[Path] = None
    if store_sequences:
        seq_cache_root = cache_dataset_root(seq_cache_cache_dir, set_size)
        seq_cache_root.mkdir(parents=True, exist_ok=True)
        print(
            f"Per-sequence SetBERT embeddings will be written under {seq_cache_root}",
            flush=True,
        )

    print(
        f"Building SetBERT embeddings ({len(pending_jobs)} runs) -> {csv_out_path}; "
        f"set_size={set_size} trim=[{settings['min_sequence_length']},"
        f"{settings['max_sequence_length']}] run_batch_size={settings['run_batch_size']} "
        f"store_sequences={store_sequences}.",
        flush=True,
    )

    fieldnames = ["Run", *(f"dim_{i}" for i in range(embed_dim))]
    run_batch_size = settings["run_batch_size"]

    start = time.perf_counter()
    n_written = 0
    n_skipped_selection = 0
    n_missing_fasta = 0
    n_seq_partitions_written = 0

    pbar = tqdm(total=len(pending_jobs), desc="setbert embeddings", unit="run")
    pending_iter = iter(pending_jobs)
    while True:
        # Build one batch: collect up to run_batch_size eligible runs (skip ineligible inline)
        batch: List[Tuple[str, str, np.ndarray, List[List[int]]]] = []
        # tuple = (study_name, run, sequence_index_1based, token_rows)
        for study_name, run, fasta_gz in pending_iter:
            if not fasta_gz.is_file():
                n_missing_fasta += 1
                pbar.update(1)
                continue
            try:
                seq_index, sequences, _n_raw = select_trimmed_set_for_run(
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
                n_skipped_selection += 1
                continue
            batch.append((study_name, run, seq_index, token_rows))
            if len(batch) >= run_batch_size:
                break

        if not batch:
            break

        batch_tokens_np = build_batch_tokens(
            [token_rows for *_, token_rows in batch],
            pad_token_id=pad_token_id,
        )
        try:
            out = _forward_batch(model, batch_tokens_np, device=device)
        except (RuntimeError, ValueError) as exc:
            run_names = ",".join(run for _, run, *_ in batch)
            pbar.write(
                f"\nForward pass failed for batch [{run_names}]: {exc}", file=sys.stderr
            )
            return 1

        cls_arr = out["class"]
        seqs_arr = out["sequences"] if store_sequences else None

        rows_to_write: List[Tuple[str, np.ndarray]] = []
        for i, (study_name, run, seq_index, _tokens) in enumerate(batch):
            rows_to_write.append((run, cls_arr[i]))
            if store_sequences and seqs_arr is not None and seq_cache_root is not None:
                part_dir = run_partition_dir(seq_cache_root, study_name, run)
                if part_dir.is_dir():
                    shutil.rmtree(part_dir)
                write_embedding_run_partition(
                    cache_root=seq_cache_root,
                    study_name=study_name,
                    run=run,
                    sequence_index_1based=seq_index,
                    embeddings=seqs_arr[i],
                    compression=parquet_compression,
                )
                n_seq_partitions_written += 1
        _write_run_rows(csv_out_path, rows=rows_to_write, fieldnames=fieldnames)
        n_written += len(rows_to_write)
        pbar.update(len(batch))

    pbar.close()

    elapsed = time.perf_counter() - start
    print(f"Wrote {n_written} run rows to {csv_out_path}")
    if n_skipped_selection:
        print(f"Skipped (selection / too few sequences): {n_skipped_selection}")
    if n_missing_fasta:
        print(f"Skipped (missing FASTA): {n_missing_fasta}")
    if store_sequences:
        print(
            f"Wrote {n_seq_partitions_written} per-sequence partition(s) under "
            f"{seq_cache_root}"
        )
    print(f"Elapsed: {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
