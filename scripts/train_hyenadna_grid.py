#!/usr/bin/env python3
"""Run train_hyenadna for every (CACHE, EXPT) pair allowed by tensor vs experiment geometry."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterator, Tuple

import yaml

from hyenadna_fasta_data import merge_train_hyenadna_config, model_max_length
from hyenadna_tensor_cache import merged_build_run_tensors_for_cache, run_tensors_rows


def iter_allowed_train_hyenadna_pairs(
    defaults_path: Path,
    experiments_path: Path,
) -> Iterator[Tuple[int, int]]:
    """
    Yield (cache_index, expt_index) both 1-based, such that cache tensors can serve the experiment.

    Requires cache max_length >= experiment max_length and cache num_sets >= experiment num_sets.
    """
    rows = run_tensors_rows(experiments_path)
    cfg = yaml.safe_load(experiments_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        cfg = {}
    sec = cfg.get("train_hyenadna") or {}
    if not isinstance(sec, dict):
        raise SystemExit("experiments.yaml train_hyenadna must be a mapping when present.")
    experiments = sec.get("experiments") or []
    if not isinstance(experiments, list) or not experiments:
        raise SystemExit(
            "train_hyenadna grid requires train_hyenadna.experiments in experiments.yaml."
        )

    for ci, _row in enumerate(rows, start=1):
        cache_cfg = merged_build_run_tensors_for_cache(
            defaults_path,
            experiments_path,
            cache_1based=ci,
        )
        c_ns = int(cache_cfg["num_sets"])
        c_ml = int(cache_cfg["max_length"])

        for ei in range(1, len(experiments) + 1):
            merged_tr, _name, _tpl = merge_train_hyenadna_config(
                defaults_path,
                experiments_path,
                expt=ei,
                cache_1based=ci,
            )
            model_name = str(merged_tr["model"]).strip()
            e_ns = int(merged_tr["num_sets"])
            max_raw = merged_tr.get("max_length")
            e_ml = model_max_length(model_name, int(max_raw) if max_raw is not None else None)

            if c_ml >= e_ml and c_ns >= e_ns:
                yield (ci, ei)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    defaults_path = repo_root / "defaults.yaml"
    experiments_path = repo_root / "experiments.yaml"
    script = repo_root / "scripts" / "train_hyenadna.py"
    pairs = list(iter_allowed_train_hyenadna_pairs(defaults_path, experiments_path))
    if not pairs:
        raise SystemExit("No allowed CACHE×EXPT pairs (check run_tensors vs train_hyenadna settings).")
    print(
        f"\ntrain_hyenadna grid: {len(pairs)} allowed cache×experiment pair(s)",
        flush=True,
    )
    for ci, ei in pairs:
        print(f"\n--- cache={ci} expt={ei} ---", flush=True)
        subprocess.check_call(
            [sys.executable, str(script), "--cache", str(ci), "--expt", str(ei)],
            cwd=str(repo_root),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
