#!/usr/bin/env python3
"""
Extract frozen HyenaDNA run-level embeddings from cached run tensors.

Reads paths.run_tensors_dir/<Run>.pt and writes one consolidated CSV per feature set:
paths.embeddings_dir/{num_sets}sets_{max_length}L.csv

Configuration source:
- defaults.yaml extract_embeddings (baseline)
- experiments.yaml extract_embeddings row via --feat (1-based)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from hyenadna import HyenaDNAPreTrainedModel
from tqdm import tqdm

from hyenadna_fasta_data import model_max_length, resolve_repo_path
from shared_utilities import build_run_table


def _parse_argv(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feat",
        type=int,
        default=None,
        help=(
            "Optional 1-based feature-set index from experiments.yaml extract_embeddings. "
            "Omit for defaults.yaml baseline."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.feat is not None and args.feat < 1:
        raise SystemExit("--feat must be >= 1 when provided.")
    return args


def _resolve_extract_embeddings_config(
    defaults_cfg: Mapping[str, Any],
    experiments_cfg: Mapping[str, Any],
    *,
    feat: Optional[int],
) -> Tuple[Dict[str, Any], Optional[int]]:
    baseline = defaults_cfg.get("extract_embeddings")
    if not isinstance(baseline, dict):
        raise SystemExit("defaults.yaml extract_embeddings must be a mapping.")
    if feat is None:
        return dict(baseline), None
    rows = experiments_cfg.get("extract_embeddings") or []
    if not isinstance(rows, list):
        raise SystemExit("experiments.yaml extract_embeddings must be a list.")
    if feat > len(rows):
        raise SystemExit(
            f"--feat {feat} is out of range. experiments.yaml defines {len(rows)} extract_embeddings rows."
        )
    row = rows[feat - 1]
    if not isinstance(row, dict):
        raise SystemExit("experiments.yaml extract_embeddings entries must be mappings.")
    return {**dict(baseline), **dict(row)}, int(feat)


def _feature_tag(num_sets: int, max_length: int) -> str:
    return f"{int(num_sets)}sets_{int(max_length)}L"


def _embedding_csv_path(repo_root: Path, paths_cfg: Mapping[str, Any], *, tag: str) -> Path:
    embeddings_dir = resolve_repo_path(repo_root, str(paths_cfg["embeddings_dir"]).strip())
    return embeddings_dir / f"{tag}.csv"


def _masked_mean_by_set(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    set_mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
    denom = set_mask.sum(dim=1).clamp_min(1.0)
    return (hidden_states * set_mask).sum(dim=1) / denom


def main(argv: Optional[Sequence[str]] = None) -> int:
    cli = _parse_argv(argv)
    repo_root = Path(__file__).resolve().parent.parent
    defaults_path = repo_root / "defaults.yaml"
    experiments_path = repo_root / "experiments.yaml"

    defaults_cfg = yaml.safe_load(defaults_path.read_text(encoding="utf-8")) or {}
    experiments_cfg: Mapping[str, Any] = {}
    if experiments_path.is_file():
        experiments_cfg = yaml.safe_load(experiments_path.read_text(encoding="utf-8")) or {}

    if not isinstance(defaults_cfg, dict):
        raise SystemExit(f"{defaults_path} must contain a YAML mapping.")
    paths_cfg = defaults_cfg.get("paths")
    if not isinstance(paths_cfg, dict):
        raise SystemExit(f"{defaults_path} must define paths as a mapping.")
    run_tensors_cfg = defaults_cfg.get("run_tensors")
    if not isinstance(run_tensors_cfg, dict):
        raise SystemExit(f"{defaults_path} must define run_tensors as a mapping.")
    train_hyenadna_cfg = defaults_cfg.get("train_hyenadna")
    if not isinstance(train_hyenadna_cfg, dict):
        raise SystemExit(f"{defaults_path} must define train_hyenadna as a mapping.")

    merged, feat_index = _resolve_extract_embeddings_config(
        defaults_cfg,
        experiments_cfg,
        feat=cli.feat,
    )
    model_name = str(merged["model"]).strip()
    num_sets = int(merged["num_sets"])
    max_len = model_max_length(model_name, merged.get("max_length"))
    head_mode = str(merged.get("head_pooling_mode", "pool")).strip().lower()
    if head_mode != "pool":
        raise SystemExit("extract_embeddings.head_pooling_mode must be 'pool'.")
    if num_sets <= 0:
        raise SystemExit("extract_embeddings.num_sets must be > 0.")

    cache_num_sets = int(run_tensors_cfg["num_sets"])
    cache_max_len = int(run_tensors_cfg["max_length"])
    if num_sets > cache_num_sets:
        raise SystemExit(
            f"extract_embeddings.num_sets ({num_sets}) exceeds run_tensors.num_sets ({cache_num_sets})."
        )
    if max_len > cache_max_len:
        raise SystemExit(
            f"extract_embeddings.max_length ({max_len}) exceeds run_tensors.max_length ({cache_max_len})."
        )

    run_tensors_root = resolve_repo_path(repo_root, str(paths_cfg["run_tensors_dir"]).strip())
    if not run_tensors_root.is_dir():
        raise SystemExit(f"Missing run tensors directory: {run_tensors_root}")

    device_raw = str(merged.get("device") or "").strip().lower()
    device = torch.device(device_raw if device_raw and device_raw != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    checkpoint_dir_raw = merged.get("checkpoint_dir")
    if checkpoint_dir_raw is None:
        checkpoint_dir_raw = train_hyenadna_cfg.get("checkpoint_dir", "checkpoints")
    download_pretrained = bool(
        merged.get(
            "download_pretrained",
            train_hyenadna_cfg.get("download_pretrained", False),
        )
    )
    checkpoint_dir = resolve_repo_path(repo_root, checkpoint_dir_raw)

    model = HyenaDNAPreTrainedModel.from_pretrained(
        str(checkpoint_dir),
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

    run_df = build_run_table(config_path=defaults_path)
    runs = (
        run_df.loc[:, ["Run"]]
        .drop_duplicates(subset=["Run"])
        .reset_index(drop=True)
    )
    start = cache_max_len - max_len
    if start < 0:
        raise SystemExit("Resolved max_length exceeds run_tensors.max_length.")

    out_runs: list[str] = []
    out_embeddings: list[np.ndarray] = []
    skipped_missing = 0
    skipped_empty = 0

    label = "baseline" if feat_index is None else f"feat={feat_index}"
    print(
        f"\nExtract embeddings ({label}) | model={model_name} num_sets={num_sets} max_length={max_len}",
        flush=True,
    )

    with torch.no_grad():
        for _, row in tqdm(runs.iterrows(), total=len(runs), desc="Extract", unit="run"):
            run = str(row["Run"]).strip()
            pt_path = run_tensors_root / f"{run}.pt"
            if not pt_path.is_file():
                skipped_missing += 1
                continue
            try:
                blob = torch.load(pt_path, map_location="cpu", weights_only=False)
            except TypeError:
                blob = torch.load(pt_path, map_location="cpu")
            n_valid = min(int(blob.get("n_sets", 0)), num_sets, cache_num_sets)
            if n_valid <= 0:
                skipped_empty += 1
                continue
            x = blob["input_ids"][:num_sets, start:cache_max_len].to(device)
            mask = blob["attention_mask"][:num_sets, start:cache_max_len].to(device)
            hidden = model(x[:n_valid])
            per_set = _masked_mean_by_set(hidden, mask[:n_valid])
            run_emb = per_set.mean(dim=0).to(dtype=torch.float32).cpu().numpy()
            out_runs.append(run)
            out_embeddings.append(run_emb)

    if not out_embeddings:
        raise SystemExit("No embeddings were extracted from run tensors.")

    mat = np.vstack(out_embeddings).astype(np.float32, copy=False)
    feature_cols = [f"embed_{i}" for i in range(mat.shape[1])]
    out_df = pd.DataFrame(mat, columns=feature_cols)
    out_df.insert(0, "Run", out_runs)
    out_df = out_df.drop_duplicates(subset=["Run"], keep="first")

    tag = _feature_tag(num_sets, max_len)
    out_csv = _embedding_csv_path(repo_root, paths_cfg, tag=tag)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    print(
        f"Wrote {len(out_df)} run embeddings to {out_csv} "
        f"(dim={mat.shape[1]}, skipped_missing={skipped_missing}, skipped_empty={skipped_empty}).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
