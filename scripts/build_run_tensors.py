#!/usr/bin/env python3
"""
Build a feature-only HyenaDNA tensor cache per sequencing run (FASTA -> .pt).

This cache is task-agnostic: each run file stores token tensors only, and labels/splits
are joined later by train_hyenadna.py via shared metadata utilities.

Tensor files always live under paths.run_tensors_dir/<1-based cache index>/ (see experiments.yaml
run_tensors rows merged over defaults.yaml build_run_tensors).
"""

from __future__ import annotations

import argparse
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
    run_to_tensors,
)
from shared_utilities import build_run_table
from hyenadna_tensor_cache import (
    base_run_tensors_root,
    merged_build_run_tensors_for_cache,
    run_tensors_rows,
)


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
        },
        out_path,
    )
    return ("written", run, "")


def _build_cache_index(
    *,
    repo_root: Path,
    defaults_path: Path,
    experiments_path: Path,
    cache_index: int,
    force: bool,
) -> None:
    cfg = _load_defaults(defaults_path)
    paths_cfg = cfg.get("paths")
    if not isinstance(paths_cfg, dict):
        raise SystemExit(f"{defaults_path} must define paths as a mapping.")

    fasta_dir_key = str(paths_cfg["fasta_dir"]).strip()
    build_cfg = merged_build_run_tensors_for_cache(
        defaults_path,
        experiments_path,
        cache_1based=cache_index,
    )
    run_tensors_root = base_run_tensors_root(repo_root, defaults_path) / str(cache_index)

    run_tensors_root.mkdir(parents=True, exist_ok=True)

    num_sets = int(build_cfg["num_sets"])
    max_length = int(build_cfg["max_length"])
    max_workers = int(build_cfg.get("max_workers", 1))
    if num_sets <= 0:
        raise SystemExit("build_run_tensors.num_sets must be > 0.")
    if max_length <= 0:
        raise SystemExit("build_run_tensors.max_length must be > 0.")
    if max_workers <= 0:
        raise SystemExit("build_run_tensors.max_workers must be >= 1.")

    run_df = build_run_table(config_path=defaults_path)
    runs_frame = run_df[["Run", "study_name"]].drop_duplicates(subset=["Run"])

    written = 0
    skipped: List[Dict[str, str]] = []
    n_total = len(runs_frame)
    print(
        f"\nBuilding run tensors ({n_total} runs) -> {run_tensors_root} "
        f"(cache={cache_index}, num_sets={num_sets}, max_length={max_length}, "
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
            elif status.startswith("skipped") and reason:
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
                elif status.startswith("skipped") and reason:
                    skipped.append({"run": done_run, "reason": reason})

    print(
        f"\nWrote/updated {written} run tensors; skipped {len(skipped)} runs.",
        flush=True,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild existing run tensor files.",
    )
    parser.add_argument(
        "--cache",
        type=int,
        default=None,
        metavar="N",
        help=(
            "1-based experiments.yaml run_tensors row index (writes under paths.run_tensors_dir/N/). "
            "Required."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = Path(__file__).resolve().parent.parent
    defaults_path = repo_root / "defaults.yaml"
    experiments_path = repo_root / "experiments.yaml"

    rows = run_tensors_rows(experiments_path)
    cache_arg = args.cache
    if cache_arg is None:
        raise SystemExit("Pass --cache <n> (1-based index matching experiments.yaml run_tensors).")
    if cache_arg < 1 or cache_arg > len(rows):
        raise SystemExit(
            f"--cache {cache_arg} out of range; experiments.yaml has {len(rows)} run_tensors row(s)."
        )

    _build_cache_index(
        repo_root=repo_root,
        defaults_path=defaults_path,
        experiments_path=experiments_path,
        cache_index=cache_arg,
        force=bool(args.force),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
