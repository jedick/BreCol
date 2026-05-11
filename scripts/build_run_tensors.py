#!/usr/bin/env python3
"""
Build a feature-only HyenaDNA tensor cache per sequencing run (FASTA -> .pt).

This cache is task-agnostic: each run file stores token tensors only, and labels/splits
are joined later by train_hyenadna.py via shared metadata utilities.

Tensor files live under paths.run_tensors_dir as <Run>.pt files.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from tqdm import tqdm

from hyenadna_fasta_data import (  # noqa: E402
    fasta_path_for_run,
    iter_fasta_sequences,
    make_character_tokenizer,
    resolve_repo_path,
    run_to_tensors,
)
from shared_utilities import build_run_table


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
    sample_mode: str,
    sampling_seed: int,
    deduplicate_sequences: bool,
    max_sequences_per_run: Optional[int],
    force: bool,
) -> Tuple[str, str, str]:
    repo_root = Path(repo_root_s)
    run_tensors_root = Path(run_tensors_root_s)
    out_path = run_tensors_root / f"{run}.pt"
    if out_path.is_file() and not force:
        return ("skipped_existing", run, "")

    fasta_gz = fasta_path_for_run(repo_root, fasta_dir_key, study_name, run)
    if not fasta_gz.is_file():
        return ("skipped_missing_fasta", run, f"missing_fasta:{fasta_gz}")

    tokenizer = make_character_tokenizer(max_length)
    sequences = list(iter_fasta_sequences(fasta_gz))
    n_raw = len(sequences)
    if seq_offset > 0 and n_raw < seq_offset:
        return (
            "skipped_short_seq_offset",
            run,
            f"n_sequences={n_raw}<{seq_offset}",
        )
    if seq_offset > 0:
        sequences = sequences[seq_offset:]
    n_after_offset = len(sequences)
    if min_seqs > 0 and n_after_offset < min_seqs:
        return (
            "skipped_min_seqs",
            run,
            f"n_sequences_after_offset={n_after_offset}<{min_seqs}",
        )
    if deduplicate_sequences:
        deduped: List[str] = []
        seen = set()
        for seq in sequences:
            if seq in seen:
                continue
            seen.add(seq)
            deduped.append(seq)
        sequences = deduped
    n_after_dedup = len(sequences)

    if max_sequences_per_run is not None and n_after_dedup > max_sequences_per_run:
        if sample_mode == "head":
            sequences = sequences[:max_sequences_per_run]
        elif sample_mode == "random":
            run_seed = int(
                hashlib.sha256(run.encode("utf-8")).hexdigest()[:8],
                16,
            )
            rng = random.Random(int(sampling_seed) + run_seed)
            picked = sorted(rng.sample(range(n_after_dedup), max_sequences_per_run))
            sequences = [sequences[i] for i in picked]
        else:
            raise SystemExit(
                f"Unknown run_tensors.sample_mode {sample_mode!r} (use head or random)."
            )
    n_selected = len(sequences)

    input_ids, attention_mask, n_valid = run_to_tensors(
        sequences,
        tokenizer=tokenizer,
        max_length=max_length,
        num_sets=num_sets,
    )
    if input_ids is None or n_valid == 0:
        return ("skipped_empty", run, "no_tokenized_sequence_content")

    torch.save(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "n_sets": int(n_valid),
            "n_raw_sequences": int(n_raw),
            "n_after_offset_sequences": int(n_after_offset),
            "n_after_dedup_sequences": int(n_after_dedup),
            "n_selected_sequences": int(n_selected),
            "sample_mode": str(sample_mode),
            "sampling_seed": int(sampling_seed),
            "deduplicate_sequences": bool(deduplicate_sequences),
            "max_sequences_per_run": (
                None if max_sequences_per_run is None else int(max_sequences_per_run)
            ),
        },
        out_path,
    )
    return ("written", run, "")


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
    run_tensors_cfg = cfg.get("run_tensors")
    if not isinstance(run_tensors_cfg, dict):
        raise SystemExit(f"{defaults_path} must define run_tensors as a mapping.")
    run_tensors_key = str(paths_cfg.get("run_tensors_dir", "outputs/run_tensors")).strip()
    run_tensors_root = resolve_repo_path(repo_root, run_tensors_key)

    run_tensors_root.mkdir(parents=True, exist_ok=True)

    num_sets = int(run_tensors_cfg["num_sets"])
    max_length = int(run_tensors_cfg["max_length"])
    seq_offset = int(run_tensors_cfg.get("seq_offset", 0))
    min_seqs = int(run_tensors_cfg.get("min_seqs", 0))
    sample_mode = str(run_tensors_cfg.get("sample_mode", "head")).strip().lower()
    sampling_seed = int(run_tensors_cfg.get("sampling_seed", 0))
    deduplicate_sequences = bool(run_tensors_cfg.get("deduplicate_sequences", False))
    raw_max_sequences = run_tensors_cfg.get("max_sequences_per_run", None)
    max_sequences_per_run: Optional[int]
    if raw_max_sequences in (None, "null"):
        max_sequences_per_run = None
    else:
        max_sequences_per_run = int(raw_max_sequences)
    max_workers = int(run_tensors_cfg.get("max_workers", 1))
    if num_sets <= 0:
        raise SystemExit("run_tensors.num_sets must be > 0.")
    if max_length <= 0:
        raise SystemExit("run_tensors.max_length must be > 0.")
    if seq_offset < 0:
        raise SystemExit("run_tensors.seq_offset must be >= 0.")
    if min_seqs < 0:
        raise SystemExit("run_tensors.min_seqs must be >= 0.")
    if sample_mode not in ("head", "random"):
        raise SystemExit("run_tensors.sample_mode must be 'head' or 'random'.")
    if max_sequences_per_run is not None and max_sequences_per_run <= 0:
        raise SystemExit("run_tensors.max_sequences_per_run must be > 0 when set.")
    if max_workers <= 0:
        raise SystemExit("run_tensors.max_workers must be >= 1.")

    run_df = build_run_table(config_path=defaults_path)
    runs_frame = run_df[["Run", "study_name"]].drop_duplicates(subset=["Run"])

    written = 0
    skipped: List[Dict[str, str]] = []
    skip_short_seq_offset = 0
    skip_min_seqs = 0
    skip_other = 0
    n_total = len(runs_frame)
    print(
        f"\nBuilding run tensors ({n_total} runs) -> {run_tensors_root} "
        f"(num_sets={num_sets}, max_length={max_length}, "
        f"seq_offset={seq_offset}, min_seqs={min_seqs}, "
        f"sample_mode={sample_mode}, max_sequences_per_run={max_sequences_per_run}, "
        f"deduplicate_sequences={deduplicate_sequences}, "
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
        "sample_mode": sample_mode,
        "sampling_seed": sampling_seed,
        "deduplicate_sequences": deduplicate_sequences,
        "max_sequences_per_run": max_sequences_per_run,
        "force": force,
    }

    if max_workers == 1:
        for run, study_name in tqdm(jobs, total=n_total, desc="FASTA -> tensors", unit="run"):
            status, done_run, reason = _build_one_run_tensor(
                run=run,
                study_name=study_name,
                **worker_kwargs,
            )
            if status == "written":
                written += 1
            elif status == "skipped_short_seq_offset":
                skip_short_seq_offset += 1
                skipped.append({"run": done_run, "reason": reason})
            elif status == "skipped_min_seqs":
                skip_min_seqs += 1
                skipped.append({"run": done_run, "reason": reason})
            elif status.startswith("skipped") and reason:
                skip_other += 1
                skipped.append({"run": done_run, "reason": reason})
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
                status, done_run, reason = fut.result()
                if status == "written":
                    written += 1
                elif status == "skipped_short_seq_offset":
                    skip_short_seq_offset += 1
                    skipped.append({"run": done_run, "reason": reason})
                elif status == "skipped_min_seqs":
                    skip_min_seqs += 1
                    skipped.append({"run": done_run, "reason": reason})
                elif status.startswith("skipped") and reason:
                    skip_other += 1
                    skipped.append({"run": done_run, "reason": reason})

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
    # Makefile tracks run_tensors directory mtime as the cache target signal.
    os.utime(run_tensors_root, None)


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
