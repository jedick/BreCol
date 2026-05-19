#!/usr/bin/env python3
"""
Build a hive-partitioned per-sequence HyenaDNA embedding cache from FASTA files.

For each data/ CSV row with sample_used=TRUE, reads fasta/<study_name>/<Run>.fasta.gz,
applies sequence_cache row selection from defaults.yaml, runs the pretrained backbone (use_head=False),
and writes Parquet under paths.embedding_cache_dir.

Downstream: run_uc_cap_pipeline.py with --emb.

Use --first-run-per-study for a quick sanity check. Use --force to rebuild all partitions.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from hyenadna import HyenaDNAPreTrainedModel
from tqdm import tqdm

from cache_operations import (
    cache_dataset_root,
    count_fasta_records,
    iter_selected_fasta_sequences,
    load_sequence_row_selection,
    partition_is_up_to_date,
    row_is_sample_used,
    run_partition_dir,
    select_row_indices_0based,
    skip_reason,
    split_jobs_by_cache_state,
    write_embedding_run_partition,
)
from hyenadna_fasta_data import fasta_path_for_run, model_max_length, resolve_repo_path
from hyenadna_sequence_embeddings import embeddings_from_sequences
from shared_utilities import RUN_PATTERN

_WORKER: Dict[str, Any] = {}


def _resolve_repo_path(repo_root: Path, raw: str) -> Path:
    p = Path(str(raw).strip())
    return p if p.is_absolute() else repo_root / p


def _load_sequence_cache_cfg(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    cache_cfg = cfg.get("sequence_cache")
    if not isinstance(cache_cfg, dict):
        raise KeyError("sequence_cache must be a mapping")
    return dict(cache_cfg)


def _load_embedding_settings(
    cfg: Mapping[str, Any],
    train_hyenadna_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    cache_cfg = _load_sequence_cache_cfg(cfg)
    model_name = str(cache_cfg["model"]).strip()
    max_tokens = model_max_length(model_name, None)
    checkpoint_dir_raw = cache_cfg.get("checkpoint_dir")
    if checkpoint_dir_raw is None:
        checkpoint_dir_raw = train_hyenadna_cfg.get("checkpoint_dir", "checkpoints")
    download_pretrained = cache_cfg.get("download_pretrained")
    if download_pretrained is None:
        download_pretrained = train_hyenadna_cfg.get("download_pretrained", False)
    device_raw = str(cache_cfg.get("device") or "").strip().lower()
    sequence_reduce = str(cache_cfg.get("sequence_reduce", "masked_mean")).strip().lower()
    if sequence_reduce not in ("masked_mean", "last"):
        raise ValueError("sequence_cache.sequence_reduce must be masked_mean or last")
    return {
        "n_max": int(cache_cfg["n_max_per_run"]),
        "compression_raw": str(cache_cfg["parquet_compression"]).strip(),
        "max_workers": int(cache_cfg.get("max_workers", 1)),
        "embedding_batch_size": int(cache_cfg["embedding_batch_size"]),
        "model_name": model_name,
        "max_tokens": max_tokens,
        "checkpoint_dir": str(checkpoint_dir_raw).strip(),
        "download_pretrained": bool(download_pretrained),
        "device_raw": device_raw,
        "sequence_reduce": sequence_reduce,
    }


def _resolve_device(device_raw: str) -> torch.device:
    if device_raw in ("", "auto", "null"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_raw)


def _load_model_on_device(
    repo_root: Path,
    *,
    model_name: str,
    checkpoint_dir: str,
    download_pretrained: bool,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint_path = resolve_repo_path(repo_root, checkpoint_dir)
    model = HyenaDNAPreTrainedModel.from_pretrained(
        str(checkpoint_path),
        model_name,
        download=download_pretrained,
        config=None,
        device=str(device),
        use_head=False,
        n_classes=2,
        head_pooling_mode="pool",
    )
    model = model.to(device)
    model.eval()
    return model


def selected_sequences_from_fasta(
    fasta_gz: Path,
    *,
    seq_offset: int,
    min_seqs: int,
    n_max: int,
    sample_mode: str,
    sampling_seed: int,
    run: str,
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:

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
    seq_rows: List[str] = []
    for seq_index, seq in iter_selected_fasta_sequences(fasta_gz, wanted, max_index):
        index_rows.append(seq_index + 1)
        seq_rows.append(seq)

    if len(seq_rows) != int(indices_0.size):
        return None, None

    return np.asarray(index_rows, dtype=np.int32), seq_rows


def _init_worker(
    repo_root_s: str,
    model_name: str,
    checkpoint_dir: str,
    download_pretrained: bool,
    max_tokens: int,
    embedding_batch_size: int,
    sequence_reduce: str,
) -> None:
    from hyenadna_fasta_data import make_character_tokenizer

    repo_root = Path(repo_root_s)
    device = torch.device("cpu")
    model = _load_model_on_device(
        repo_root,
        model_name=model_name,
        checkpoint_dir=checkpoint_dir,
        download_pretrained=download_pretrained,
        device=device,
    )
    _WORKER.clear()
    _WORKER.update(
        {
            "model": model,
            "tokenizer": make_character_tokenizer(max_tokens),
            "device": device,
            "max_tokens": max_tokens,
            "embedding_batch_size": embedding_batch_size,
            "sequence_reduce": sequence_reduce,
        }
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
    # GPU path passes these; CPU workers use _WORKER
    model: Optional[torch.nn.Module] = None,
    tokenizer: Optional[object] = None,
    device: Optional[torch.device] = None,
    max_tokens: Optional[int] = None,
    embedding_batch_size: Optional[int] = None,
    sequence_reduce: Optional[str] = None,
) -> Tuple[str, int]:
    fasta_gz = Path(fasta_path_s)
    cache_root = Path(cache_root_s)

    if not fasta_gz.is_file():
        return ("missing_fasta", 0)
    if not force and partition_is_up_to_date(cache_root, study_name, run, fasta_gz):
        return ("skipped_existing", 0)

    try:
        seq_index, sequences = selected_sequences_from_fasta(
            fasta_gz,
            seq_offset=seq_offset,
            min_seqs=min_seqs,
            n_max=n_max,
            sample_mode=sample_mode,
            sampling_seed=sampling_seed,
            run=run,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"{study_name}/{run}: {exc}") from exc

    if seq_index is None or sequences is None or not sequences:
        return ("skipped_selection", 0)

    if model is None:
        model = _WORKER["model"]
        tokenizer = _WORKER["tokenizer"]
        device = _WORKER["device"]
        max_tokens = int(_WORKER["max_tokens"])
        embedding_batch_size = int(_WORKER["embedding_batch_size"])
        sequence_reduce = str(_WORKER["sequence_reduce"])

    assert model is not None
    assert tokenizer is not None
    assert device is not None
    assert max_tokens is not None
    assert embedding_batch_size is not None
    assert sequence_reduce is not None

    embeddings = embeddings_from_sequences(
        model,
        sequences,
        tokenizer,
        max_tokens=max_tokens,
        batch_size=embedding_batch_size,
        device=device,
        sequence_reduce=sequence_reduce,
        run=run,
        sequence_indices_1based=[int(x) for x in seq_index.tolist()],
    )

    if force:
        part_dir = run_partition_dir(cache_root, study_name, run)
        if part_dir.is_dir():
            shutil.rmtree(part_dir)

    write_embedding_run_partition(
        cache_root=cache_root,
        study_name=study_name,
        run=run,
        sequence_index_1based=seq_index,
        embeddings=embeddings,
        compression=parquet_compression,
    )
    return ("written", int(embeddings.shape[0]))


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
        "--force",
        action="store_true",
        help="Rebuild Parquet partitions even when FASTA is not newer.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = repo_root / "defaults.yaml"
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        train_hyenadna_cfg = cfg["train_hyenadna"]
        settings = _load_embedding_settings(cfg, train_hyenadna_cfg)
        paths = cfg["paths"]
        data_dir = _resolve_repo_path(repo_root, str(paths["data_dir"]))
        fasta_dir_key = str(paths["fasta_dir"]).strip()
        cache_dir = _resolve_repo_path(repo_root, str(paths["embedding_cache_dir"]))
        selection = load_sequence_row_selection(cfg)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Invalid pipeline config in {cfg_path}: {exc}", file=sys.stderr)
        return 1

    valid_compression = frozenset(("zstd", "snappy", "gzip", "brotli", "lz4", "none"))
    compression_raw = settings["compression_raw"]
    if compression_raw not in valid_compression:
        print(
            f"Invalid sequence_cache.parquet_compression in {cfg_path}: {compression_raw!r}",
            file=sys.stderr,
        )
        return 1
    n_max = settings["n_max"]
    max_workers = settings["max_workers"]
    embedding_batch_size = settings["embedding_batch_size"]
    if n_max <= 0:
        print("sequence_cache.n_max_per_run must be a positive integer", file=sys.stderr)
        return 1
    if max_workers < 1:
        print("sequence_cache.max_workers must be >= 1", file=sys.stderr)
        return 1
    if embedding_batch_size < 1:
        print("sequence_cache.embedding_batch_size must be >= 1", file=sys.stderr)
        return 1

    device = _resolve_device(settings["device_raw"])
    use_run_parallelism = device.type == "cpu" and max_workers > 1
    effective_workers = max_workers if use_run_parallelism else 1

    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        return 1

    parquet_compression = None if compression_raw == "none" else compression_raw
    cache_root = cache_dataset_root(cache_dir, n_max)
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
            f"Embedding cache complete: all {n_eligible} partition(s) up-to-date under {cache_root}.",
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
        f"Building embedding cache ({n_eligible} runs) -> {cache_root}; "
        f"model={settings['model_name']} context={settings['max_tokens']} "
        f"n_max={n_max} batch_size={embedding_batch_size} "
        f"sequence_reduce={settings['sequence_reduce']} device={device} "
        f"run_parallelism={effective_workers} force={args.force}.",
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

    if use_run_parallelism:
        init_args = (
            str(repo_root),
            settings["model_name"],
            settings["checkpoint_dir"],
            settings["download_pretrained"],
            settings["max_tokens"],
            embedding_batch_size,
            settings["sequence_reduce"],
        )
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            initializer=_init_worker,
            initargs=init_args,
        ) as ex:
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
                desc="embedding cache",
                unit="run",
            ):
                try:
                    status, n_rows = fut.result()
                except (OSError, ValueError) as exc:
                    print(f"\nWorker failed: {exc}", file=sys.stderr)
                    return 1
                _handle(status, n_rows)
    else:
        from hyenadna_fasta_data import make_character_tokenizer

        model = _load_model_on_device(
            repo_root,
            model_name=settings["model_name"],
            checkpoint_dir=settings["checkpoint_dir"],
            download_pretrained=settings["download_pretrained"],
            device=device,
        )
        tokenizer = make_character_tokenizer(settings["max_tokens"])
        gpu_kwargs = {
            **worker_kwargs,
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
            "max_tokens": settings["max_tokens"],
            "embedding_batch_size": embedding_batch_size,
            "sequence_reduce": settings["sequence_reduce"],
        }
        for study_name, run, fasta_gz in tqdm(
            all_jobs, total=n_eligible, desc="embedding cache", unit="run"
        ):
            try:
                status, n_rows = _build_one_run_partition(
                    fasta_path_s=str(fasta_gz),
                    study_name=study_name,
                    run=run,
                    **gpu_kwargs,
                )
            except (OSError, ValueError) as exc:
                print(f"\nError: {exc}", file=sys.stderr)
                return 1
            _handle(status, n_rows)

    if n_written == 0 and n_skipped_existing == 0:
        print("Error: no runs were written or reused in the embedding cache", file=sys.stderr)
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
