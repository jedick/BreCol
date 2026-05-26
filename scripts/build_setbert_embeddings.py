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
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

# SetBERT's DNABERT backbone wraps sequence-encoder calls in torch.utils.checkpoint, which
# emits a UserWarning at inference time because gradients are disabled. The warning is benign
# (we never train via this script) and would otherwise be printed once per chunked forward.
warnings.filterwarnings(
    "ignore",
    message="None of the inputs have requires_grad=True. Gradients will be None",
    category=UserWarning,
    module="torch.utils.checkpoint",
)

from cache_operations import (
    cache_dataset_root,
    count_fasta_records,
    iter_selected_fasta_sequences,
    load_sequence_row_selection,
    partition_is_up_to_date,
    row_is_sample_used,
    run_partition_dir,
    run_sampling_seed,
    select_row_indices_0based,
    skip_reason,
    write_embedding_run_partition,
)
from shared_utilities import RUN_PATTERN, fasta_path_for_run, resolve_repo_path


# ----- Per-run trimming RNG -----


def _run_trim_rng(truncation_seed: int, run: str) -> "np.random.Generator":
    """Deterministic per-Run NumPy Generator for random trim-window offsets."""
    return np.random.default_rng(run_sampling_seed(truncation_seed, run))


def _trim_sequence(seq: str, *, target_len: int, rng: "np.random.Generator") -> str:
    """Return a window of `target_len` chars from `seq` (random offset). Shorter seqs unchanged."""
    n = len(seq)
    if n <= target_len:
        return seq
    offset = int(rng.integers(0, n - target_len + 1))
    return seq[offset : offset + target_len]


# ----- Tokenization + padding -----


def _tokenize_sequences(
    sequences: Sequence[str],
    tokenizer,
) -> List[List[int]]:
    """Tokenize each DNA string to a list of int token ids using the DNABERT tokenizer."""
    out: List[List[int]] = []
    for seq in sequences:
        tokens = tokenizer(seq)
        if not tokens:
            raise ValueError("DNABERT tokenizer returned an empty token list.")
        out.append([int(t) for t in tokens])
    return out


def _pad_set_to_token_len(
    token_rows: Sequence[Sequence[int]],
    *,
    target_token_len: int,
    pad_token_id: int,
) -> np.ndarray:
    """Right-pad each token row to `target_token_len`. Returns int64 array (n, target_token_len)."""
    n = len(token_rows)
    out = np.full((n, target_token_len), pad_token_id, dtype=np.int64)
    for i, row in enumerate(token_rows):
        L = len(row)
        if L > target_token_len:
            raise ValueError(
                f"Token row length {L} exceeds target {target_token_len}; "
                "increase setbert.max_sequence_length or check tokenizer."
            )
        out[i, :L] = row
    return out


# ----- Per-run selection and trimming -----


def _select_trimmed_set_for_run(
    fasta_gz: Path,
    *,
    seq_offset: int,
    min_seqs: int,
    set_size: int,
    sample_mode: str,
    sampling_seed: int,
    truncation_seed: int,
    min_sequence_length: int,
    max_sequence_length: int,
    run: str,
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """Return (1-based sequence indices, trimmed sequence strings) for one run, or (None, None)."""
    n_raw = count_fasta_records(fasta_gz)
    if skip_reason(n_raw, seq_offset=seq_offset, min_seqs=min_seqs) is not None:
        return None, None
    pool_after_offset = n_raw - seq_offset
    if pool_after_offset < set_size:
        return None, None

    indices_0 = select_row_indices_0based(
        n_raw,
        seq_offset=seq_offset,
        min_seqs=min_seqs,
        n_max=set_size,
        sample_mode=sample_mode,
        sampling_seed=sampling_seed,
        run=run,
    )
    if indices_0 is None or int(indices_0.size) != set_size:
        return None, None

    wanted = {int(i) for i in indices_0.tolist()}
    max_index = max(wanted)

    seq_by_index: Dict[int, str] = {}
    for seq_index, seq in iter_selected_fasta_sequences(fasta_gz, wanted, max_index):
        seq_by_index[seq_index] = seq

    if len(seq_by_index) != set_size:
        return None, None

    rng = _run_trim_rng(truncation_seed, run)
    index_rows: List[int] = []
    trimmed: List[str] = []
    for idx0 in indices_0.tolist():
        seq = seq_by_index[int(idx0)]
        target_len = int(rng.integers(min_sequence_length, max_sequence_length + 1))
        trimmed.append(_trim_sequence(seq, target_len=target_len, rng=rng))
        index_rows.append(int(idx0) + 1)
    return np.asarray(index_rows, dtype=np.int32), trimmed


# ----- Config loading -----


def _resolve_repo_path(repo_root: Path, raw: object) -> Path:
    p = Path(str(raw).strip())
    return p if p.is_absolute() else repo_root / p


def _load_setbert_section(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    section = cfg.get("setbert")
    if not isinstance(section, dict):
        raise SystemExit("defaults.yaml must define a `setbert` mapping.")
    try:
        pretrained_repo = str(section["pretrained_repo"]).strip()
        pretrained_revision = str(section["pretrained_revision"]).strip()
        set_size = int(section["set_size"])
        min_sequence_length = int(section["min_sequence_length"])
        max_sequence_length = int(section["max_sequence_length"])
        truncation_seed = int(section["truncation_seed"])
        sequence_encoder_chunk_size = int(section["sequence_encoder_chunk_size"])
        run_batch_size = int(section["run_batch_size"])
        device_raw = str(section.get("device") or "").strip().lower()
        store_sequence_embeddings = bool(section["store_sequence_embeddings"])
        parquet_compression = str(section["parquet_compression"]).strip()
    except KeyError as exc:
        raise SystemExit(f"setbert missing required key: {exc.args[0]!r}") from exc
    if set_size <= 0:
        raise SystemExit("setbert.set_size must be a positive integer.")
    if min_sequence_length <= 0 or max_sequence_length <= 0:
        raise SystemExit("setbert.min_sequence_length / max_sequence_length must be positive.")
    if min_sequence_length > max_sequence_length:
        raise SystemExit(
            "setbert.min_sequence_length must be <= setbert.max_sequence_length."
        )
    if sequence_encoder_chunk_size < 0:
        raise SystemExit("setbert.sequence_encoder_chunk_size must be >= 0.")
    if run_batch_size < 1:
        raise SystemExit("setbert.run_batch_size must be >= 1.")
    valid_compression = frozenset(("zstd", "snappy", "gzip", "brotli", "lz4", "none"))
    if parquet_compression not in valid_compression:
        raise SystemExit(
            f"setbert.parquet_compression must be one of {sorted(valid_compression)}; "
            f"got {parquet_compression!r}."
        )
    return {
        "pretrained_repo": pretrained_repo,
        "pretrained_revision": pretrained_revision,
        "set_size": set_size,
        "min_sequence_length": min_sequence_length,
        "max_sequence_length": max_sequence_length,
        "truncation_seed": truncation_seed,
        "sequence_encoder_chunk_size": sequence_encoder_chunk_size,
        "run_batch_size": run_batch_size,
        "device_raw": device_raw,
        "store_sequence_embeddings": store_sequence_embeddings,
        "parquet_compression": parquet_compression,
    }


def _resolve_device(device_raw: str) -> torch.device:
    if device_raw in ("", "auto", "null", "none"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_raw)


def _load_model_on_device(
    *,
    pretrained_repo: str,
    pretrained_revision: str,
    sequence_encoder_chunk_size: int,
    device: torch.device,
) -> Tuple[torch.nn.Module, Any, int, int]:
    """Load SetBERT from HF Hub. Return (model, tokenizer, embed_dim, pad_token_id)."""
    from setbert import SetBert

    model = SetBert.from_pretrained(pretrained_repo, revision=pretrained_revision)
    model.config.sequence_encoder_chunk_size = int(sequence_encoder_chunk_size)
    model = model.to(device)
    model.eval()
    tokenizer = model.sequence_encoder.tokenizer
    embed_dim = int(model.config.embed_dim)
    pad_token_id = int(model.config.pad_token_id)
    if pad_token_id != int(tokenizer.vocab["[PAD]"]):
        raise SystemExit(
            "Model pad_token_id and tokenizer [PAD] id disagree; cannot build padding mask."
        )
    return model, tokenizer, embed_dim, pad_token_id


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


def _build_batch_tokens(
    batch_token_rows: Sequence[Sequence[Sequence[int]]],
    *,
    pad_token_id: int,
) -> np.ndarray:
    """Stack a list of (set_size,) lists of token rows into a (B, set_size, T_max) int array."""
    batch_size = len(batch_token_rows)
    set_sizes = {len(rows) for rows in batch_token_rows}
    if len(set_sizes) != 1:
        raise ValueError("All runs in a batch must share the same set_size.")
    set_size = set_sizes.pop()
    target_token_len = max(
        len(tok) for rows in batch_token_rows for tok in rows
    )
    out = np.full(
        (batch_size, set_size, target_token_len), pad_token_id, dtype=np.int64
    )
    for b, rows in enumerate(batch_token_rows):
        out[b] = _pad_set_to_token_len(
            rows, target_token_len=target_token_len, pad_token_id=pad_token_id
        )
    return out


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
        settings = _load_setbert_section(cfg)
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

    device = _resolve_device(settings["device_raw"])
    print(
        f"Loading SetBERT {settings['pretrained_repo']}@{settings['pretrained_revision']} "
        f"on {device}",
        flush=True,
    )
    try:
        model, tokenizer, embed_dim, pad_token_id = _load_model_on_device(
            pretrained_repo=settings["pretrained_repo"],
            pretrained_revision=settings["pretrained_revision"],
            sequence_encoder_chunk_size=settings["sequence_encoder_chunk_size"],
            device=device,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Failed to load SetBERT model: {exc}", file=sys.stderr)
        return 1
    print(
        f"  embed_dim={embed_dim}, kmer={tokenizer.kmer}, pad_token_id={pad_token_id}, "
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
                seq_index, sequences = _select_trimmed_set_for_run(
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
                token_rows = _tokenize_sequences(sequences, tokenizer)
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

        batch_tokens_np = _build_batch_tokens(
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
