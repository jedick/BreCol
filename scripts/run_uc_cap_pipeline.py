#!/usr/bin/env python3
"""
Run UC/CAP with decoupled sequence budgets for clustering vs abundance profiling.

UC (unsupervised clustering):
  - Fit MiniBatchKMeans on the first n_uc sequences per train run from the tetramer cache.

CAP (cluster abundance profiles):
  - Assign n_cap sequences per run to nearest centroid and aggregate K-dimensional
    cluster count/abundance vectors per run.

Input sequence features are tetramer count vectors (256 columns) from the partitioned
Parquet cache under paths.tetramer_cache_dir.

Configuration is read from <repo>/defaults.yaml (run_uc_cap_pipeline baseline list,
merged in order, then overlaid by experiments.yaml when --feat is set).
The only CLI flag is --feat.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from shared_utilities import TETRAMERS, build_run_table
from tetramer_cache_io import TetramerCacheReader, tetramer_cache_dataset_root


class SequenceTransform:
    """Feature transform for sequence-level 4-mer count vectors."""

    def __init__(self, normalize: bool, log1p: bool):
        self.normalize = normalize
        self.log1p = log1p

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = X.astype(np.float64, copy=True)
        if self.normalize:
            totals = Z.sum(axis=1, keepdims=True)
            np.divide(Z, totals, out=Z, where=(totals > 0))
        if self.log1p:
            Z = np.log1p(Z)
        return Z


def _default_cache_root(repo_root: Path, defaults_cfg: Mapping[str, Any]) -> Path:
    try:
        paths_cfg = defaults_cfg["paths"]
        cache_dir_rel = str(paths_cfg["tetramer_cache_dir"]).strip()
        n_max = int(defaults_cfg["tetramer_cache"]["n_max_per_run"])
    except (TypeError, KeyError, ValueError) as exc:
        raise SystemExit(f"Invalid pipeline config for tetramer cache path: {exc}") from exc
    cache_dir = repo_root / cache_dir_rel
    return tetramer_cache_dataset_root(cache_dir, n_max)


def fit_uc_model(
    X: np.ndarray,
    *,
    n_clusters: int,
    random_state: int,
    transform: SequenceTransform,
    pca_components: Optional[int],
    pca_variance: float,
    batch_size: int,
    max_iter: int,
) -> Tuple[MiniBatchKMeans, PCA, np.ndarray]:
    X_t = transform.transform(X)
    if pca_components is not None:
        pca = PCA(n_components=pca_components, random_state=random_state)
    else:
        pca = PCA(n_components=pca_variance, random_state=random_state)
    X_fit = pca.fit_transform(X_t)

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=batch_size,
        max_iter=max_iter,
        n_init="auto",
    )
    km.fit(X_fit)
    labels = km.predict(X_fit)
    return km, pca, labels


def assign_matrix(
    X_counts: np.ndarray,
    *,
    transform: SequenceTransform,
    pca: PCA,
    km: MiniBatchKMeans,
) -> np.ndarray:
    X_t = transform.transform(X_counts)
    X_t = pca.transform(X_t)
    return km.predict(X_t)


def update_run_cluster_counts(
    run_cluster_counts: Dict[Tuple[str, str], np.ndarray],
    run_key: Tuple[str, str],
    cluster_ids: np.ndarray,
    n_clusters: int,
) -> None:
    acc = run_cluster_counts.get(run_key)
    if acc is None:
        acc = np.zeros(n_clusters, dtype=np.int64)
        run_cluster_counts[run_key] = acc
    acc += np.bincount(cluster_ids, minlength=n_clusters).astype(np.int64, copy=False)


def build_cap_from_cache(
    *,
    cache: TetramerCacheReader,
    run_keys: Sequence[Tuple[str, str]],
    n_cap: int,
    transform: SequenceTransform,
    pca: PCA,
    km: MiniBatchKMeans,
    n_clusters: int,
) -> Dict[Tuple[str, str], np.ndarray]:
    run_cluster_counts: Dict[Tuple[str, str], np.ndarray] = {}
    n_runs = len(run_keys)
    for i, (study_name, run) in enumerate(run_keys, start=1):
        line = f"  CAP: {i}/{n_runs} runs (current: {study_name}/{run})"
        sys.stdout.write("\r" + line.ljust(100))
        sys.stdout.flush()
        _, X = cache.load_run(study_name, run)
        if X.shape[0] == 0:
            continue
        n = min(n_cap, X.shape[0])
        cluster_ids = assign_matrix(X[:n], transform=transform, pca=pca, km=km)
        update_run_cluster_counts(
            run_cluster_counts, (study_name, run), cluster_ids, n_clusters
        )
    sys.stdout.write("\n")
    return run_cluster_counts


def make_cap_dataframe(
    run_cluster_counts: Dict[Tuple[str, str], np.ndarray],
    n_clusters: int,
    cap_transform: str,
    clr_pseudocount: float,
) -> pd.DataFrame:
    cluster_cols = [f"cluster_{i:03d}" for i in range(n_clusters)]
    rows: List[Dict[str, object]] = []
    for (study_name, run), counts in sorted(run_cluster_counts.items()):
        total = int(counts.sum())
        abund = counts.astype(np.float64)
        if total > 0:
            abund /= total
        if cap_transform == "clr":
            abund = np.log(abund + clr_pseudocount)
            abund = abund - abund.mean()
        row: Dict[str, object] = {
            "study_name": study_name,
            "Run": run,
            "n_assigned_sequences": total,
        }
        row.update({cluster_cols[i]: float(abund[i]) for i in range(n_clusters)})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_cap_sparsity(
    run_cluster_counts: Dict[Tuple[str, str], np.ndarray],
) -> Tuple[float, float, int, int]:
    if not run_cluster_counts:
        return 0.0, 0.0, 0, 0
    nnz = np.asarray(
        [int(np.count_nonzero(counts)) for counts in run_cluster_counts.values()],
        dtype=np.int64,
    )
    return float(nnz.mean()), float(np.median(nnz)), int(nnz.min()), int(nnz.max())


def _parse_n_cap(raw: object) -> int:
    if isinstance(raw, int):
        value = raw
    elif isinstance(raw, str):
        text = raw.strip().lower()
        if text == "all":
            raise SystemExit(
                "n_cap='all' is no longer supported; raise tetramer_cache.n_max_per_run "
                "or lower n_cap so CAP fits in the cache."
            )
        value = int(text)
    else:
        raise SystemExit(f"n_cap must be an int, got {type(raw).__name__}")
    if value <= 0:
        raise SystemExit("n_cap must be positive.")
    return value


_FEAT_HELP = (
    "Optional 1-based index into experiments.yaml run_uc_cap_pipeline (selects a CAP "
    "feature-set row, merged over the defaults.yaml baseline). Omit to run the baseline only."
)


def _parse_feature_cli(argv: Optional[Sequence[str]]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feat", type=int, default=None, help=_FEAT_HELP)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.feat is not None and args.feat <= 0:
        raise SystemExit("--feat must be a positive integer (1-based index), or omit it.")
    return int(args.feat) if args.feat is not None else 0


def _shallow_merge_uc_cap(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update(overlay)
    return out


def merge_defaults_uc_cap_baseline_fragments(baseline_rows: List[Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for i, frag in enumerate(baseline_rows):
        if not isinstance(frag, dict):
            raise SystemExit(
                f"defaults.yaml run_uc_cap_pipeline[{i}] must be a mapping, got {type(frag).__name__}."
            )
        merged = {**merged, **frag}
    return merged


def load_merged_uc_cap_config(repo_root: Path, *, feat: int) -> Tuple[Dict[str, Any], int]:
    defaults_path = repo_root / "defaults.yaml"
    experiments_path = repo_root / "experiments.yaml"
    try:
        defaults_cfg = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"Failed to read {defaults_path}: {exc}") from exc

    try:
        baseline_rows = defaults_cfg["run_uc_cap_pipeline"]
    except (KeyError, TypeError) as exc:
        raise SystemExit(f"defaults.yaml missing run_uc_cap_pipeline: {exc}") from exc
    if not isinstance(baseline_rows, list) or not baseline_rows:
        raise SystemExit(
            "defaults.yaml run_uc_cap_pipeline must be a non-empty list of mappings."
        )
    base = merge_defaults_uc_cap_baseline_fragments(baseline_rows)

    if feat == 0:
        return base, 0

    if not experiments_path.is_file():
        raise SystemExit(f"--feat {feat} requires {experiments_path}")
    try:
        experiments_cfg = yaml.safe_load(experiments_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"Failed to read {experiments_path}: {exc}") from exc

    grid = experiments_cfg.get("run_uc_cap_pipeline")
    if not isinstance(grid, list) or not grid:
        raise SystemExit(f"No run_uc_cap_pipeline list in {experiments_path}")
    if feat > len(grid):
        raise SystemExit(
            f"--feat {feat} is out of range; experiments.yaml defines {len(grid)} feature-set rows."
        )
    row = grid[feat - 1]
    if not isinstance(row, dict):
        raise SystemExit(f"run_uc_cap_pipeline[{feat - 1}] must be a mapping.")
    return _shallow_merge_uc_cap(base, row), feat


def _uc_refit_error(uc_dir: Path, uc_assign_out: Path, model_out: Path, reason: str) -> None:
    raise SystemExit(
        f"{reason}\n"
        "Delete the cached UC artifacts and re-run this script, for example:\n"
        f"  rm -f {uc_assign_out} {model_out}\n"
        f"(or remove the whole directory {uc_dir}/ if you prefer a clean slate.)"
    )


def _write_uc_assignments(
    path: Path,
    rows: Sequence[Tuple[str, str, int, int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["study_name", "Run", "sequence_index", "cluster_id"])
        writer.writerows(rows)


def _fit_uc_from_cache(
    *,
    cache: TetramerCacheReader,
    train_keys: Sequence[Tuple[str, str]],
    n_uc: int,
    n_clusters: int,
    random_state: int,
    transform: SequenceTransform,
    pca_components: Optional[int],
    pca_variance: float,
    batch_size: int,
    max_iter: int,
) -> Tuple[MiniBatchKMeans, PCA, List[Tuple[str, str, int, int]]]:
    X_parts: List[np.ndarray] = []
    run_slices: List[Tuple[str, str, np.ndarray, int]] = []
    for study_name, run in train_keys:
        seq_index, X = cache.load_run(study_name, run)
        if X.shape[0] == 0:
            continue
        n = min(n_uc, X.shape[0])
        if n == 0:
            continue
        X_parts.append(X[:n])
        run_slices.append((study_name, run, seq_index[:n], n))
    if not X_parts:
        raise SystemExit("No UC training sequences found in tetramer cache for train runs.")
    X_all = np.vstack(X_parts)
    km, pca, labels = fit_uc_model(
        X_all,
        n_clusters=n_clusters,
        random_state=random_state,
        transform=transform,
        pca_components=pca_components,
        pca_variance=pca_variance,
        batch_size=batch_size,
        max_iter=max_iter,
    )
    assignment_rows: List[Tuple[str, str, int, int]] = []
    offset = 0
    for study_name, run, seq_index, n in run_slices:
        for i in range(n):
            assignment_rows.append(
                (study_name, run, int(seq_index[i]), int(labels[offset + i]))
            )
        offset += n
    return km, pca, assignment_rows


def run_pipeline_from_merged(
    repo_root: Path,
    *,
    config_path: Path,
    merged: Dict[str, Any],
    feature_index: int,
) -> int:
    paths_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))["paths"]

    def _resolve_cfg_path(raw: object) -> Path:
        p = Path(str(raw).strip())
        return p if p.is_absolute() else repo_root / p

    try:
        n_uc = int(merged["n_uc"])
        n_cap = _parse_n_cap(merged["n_cap"])
        n_clusters = int(merged["n_clusters"])
        random_state = int(merged["random_state"])
        pca_variance = float(merged["pca_variance"])
        pca_components = merged.get("pca_components")
        if pca_components is not None:
            pca_components = int(pca_components)
        seq_normalize = bool(merged["seq_normalize"])
        seq_log1p = bool(merged["seq_log1p"])
        cap_transform = str(merged["cap_transform"]).strip()
        clr_pseudocount = float(merged["clr_pseudocount"])
        batch_size = int(merged["batch_size"])
        max_iter = int(merged["max_iter"])
        out_dir = _resolve_cfg_path(paths_cfg["tetramer_uc_cap_root"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid merged run_uc_cap_pipeline config: {exc}") from exc

    if cap_transform not in ("none", "clr"):
        raise SystemExit(f"cap_transform must be 'none' or 'clr', got {cap_transform!r}")

    if n_uc <= 0:
        raise SystemExit("n_uc must be positive.")
    if n_clusters <= 1:
        raise SystemExit("n_clusters must be > 1.")
    if pca_components is not None and pca_components <= 0:
        raise SystemExit("pca_components must be positive when provided.")
    if not (0.0 < pca_variance <= 1.0):
        raise SystemExit("pca_variance must be in (0, 1].")
    if clr_pseudocount <= 0:
        raise SystemExit("clr_pseudocount must be positive.")
    if batch_size <= 0 or max_iter <= 0:
        raise SystemExit("batch_size and max_iter must be positive.")

    defaults_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    n_max = int(defaults_cfg["tetramer_cache"]["n_max_per_run"])
    if n_uc > n_max:
        raise SystemExit(
            f"n_uc ({n_uc}) exceeds tetramer_cache.n_max_per_run ({n_max})."
        )
    if n_cap > n_max:
        raise SystemExit(
            f"n_cap ({n_cap}) exceeds tetramer_cache.n_max_per_run ({n_max})."
        )
    cache_root = _default_cache_root(repo_root, defaults_cfg)
    complete_marker = cache_root / "_complete"
    if not complete_marker.is_file():
        raise SystemExit(
            f"Tetramer cache not built: {cache_root} (run: make tetramer_cache)"
        )

    prefix = f"F{feature_index} " if feature_index > 0 else ""
    print(f"{prefix}n_uc={n_uc} n_clusters={n_clusters} n_cap={n_cap}", flush=True)

    start = time.perf_counter()
    transform = SequenceTransform(normalize=seq_normalize, log1p=seq_log1p)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = TetramerCacheReader(cache_root)
    run_meta = build_run_table(config_path=config_path)
    train_runs = set(run_meta.loc[run_meta["split"] == "train", "Run"])
    train_keys = [
        (str(row["study_name"]), str(row["Run"]))
        for _, row in run_meta.loc[run_meta["Run"].isin(train_runs)]
        .drop_duplicates(subset=["study_name", "Run"])
        .sort_values(["study_name", "Run"])
        .iterrows()
    ]
    all_keys = [
        (str(row["study_name"]), str(row["Run"]))
        for _, row in run_meta.drop_duplicates(subset=["study_name", "Run"])
        .sort_values(["study_name", "Run"])
        .iterrows()
    ]

    uc_dir = out_dir / f"uc{n_uc}_k{n_clusters}"
    uc_dir.mkdir(parents=True, exist_ok=True)
    uc_assign_out = uc_dir / "uc_assignments.csv"
    model_out = uc_dir / "uc_model.pkl"
    reuse_uc = uc_assign_out.is_file() and model_out.is_file()

    if reuse_uc:
        print("Reusing existing UC model and assignments", flush=True)
        with open(model_out, "rb") as f:
            model_payload = pickle.load(f)
        km = model_payload["kmeans"]
        pca = model_payload["pca"]
        if int(km.n_clusters) != n_clusters:
            _uc_refit_error(
                uc_dir,
                uc_assign_out,
                model_out,
                f"Saved UC model has n_clusters={km.n_clusters} but config requests "
                f"n_clusters={n_clusters}.",
            )
        saved_train_runs = model_payload.get("split_train_runs")
        if saved_train_runs is None:
            _uc_refit_error(
                uc_dir,
                uc_assign_out,
                model_out,
                "Existing UC model was fit without shared train-only runs metadata.",
            )
        if set(saved_train_runs) != train_runs:
            _uc_refit_error(
                uc_dir,
                uc_assign_out,
                model_out,
                "Existing UC model train split does not match the current shared split.",
            )
        saved_transform = model_payload.get("transform", {})
        if (
            saved_transform.get("normalize") != transform.normalize
            or saved_transform.get("log1p") != transform.log1p
        ):
            _uc_refit_error(
                uc_dir,
                uc_assign_out,
                model_out,
                "Existing UC model sequence transform settings do not match config.",
            )
        uc_n_seqs = sum(1 for _ in uc_assign_out.open(encoding="utf-8")) - 1
    else:
        print(
            f"UC fit (train runs): {len(train_keys)} runs, up to {n_uc} sequences each",
            flush=True,
        )
        km, pca, assignment_rows = _fit_uc_from_cache(
            cache=cache,
            train_keys=train_keys,
            n_uc=n_uc,
            n_clusters=n_clusters,
            random_state=random_state,
            transform=transform,
            pca_components=pca_components,
            pca_variance=pca_variance,
            batch_size=batch_size,
            max_iter=max_iter,
        )
        _write_uc_assignments(uc_assign_out, assignment_rows)
        uc_n_seqs = len(assignment_rows)
        with open(model_out, "wb") as f:
            pickle.dump(
                {
                    "kmeans": km,
                    "pca": pca,
                    "transform": {
                        "normalize": transform.normalize,
                        "log1p": transform.log1p,
                    },
                    "split_train_runs": sorted(train_runs),
                    "tetramers": list(TETRAMERS),
                },
                f,
            )

    uc_unique_clusters = int(
        pd.read_csv(uc_assign_out, usecols=["cluster_id"])["cluster_id"].nunique()
    )
    print(f"UC training sequences: {uc_n_seqs}", flush=True)
    print(f"PCA components retained: {pca.n_components_}", flush=True)
    pca_components_used = int(pca.n_components_)

    print("Building CAP", flush=True)
    run_cluster_counts = build_cap_from_cache(
        cache=cache,
        run_keys=all_keys,
        n_cap=n_cap,
        transform=transform,
        pca=pca,
        km=km,
        n_clusters=n_clusters,
    )

    cap_df = make_cap_dataframe(
        run_cluster_counts,
        n_clusters=n_clusters,
        cap_transform=cap_transform,
        clr_pseudocount=clr_pseudocount,
    )
    cap_df = cap_df.merge(
        run_meta[["cancer_type", "study_name", "Run", "sample_label", "split"]]
        .drop_duplicates(subset=["study_name", "Run"]),
        on=["study_name", "Run"],
        how="left",
    )
    cap_df["split"] = cap_df["split"].fillna("unsplit")

    cap_n_assigned_min = int(cap_df["n_assigned_sequences"].min())
    cap_n_assigned_max = int(cap_df["n_assigned_sequences"].max())
    cap_nnz_mean, cap_nnz_median, cap_nnz_min, cap_nnz_max = summarize_cap_sparsity(
        run_cluster_counts
    )

    cap_name = (
        f"cap{n_cap}" if cap_transform == "none" else f"cap{n_cap}_{cap_transform}"
    )
    cap_out = uc_dir / f"{cap_name}.csv"
    config_out = uc_dir / f"{cap_name}.json"

    cap_df.to_csv(cap_out, index=False)
    split_counts = cap_df["split"].value_counts().to_dict()

    datasets_csv_abs = _resolve_cfg_path(paths_cfg["datasets_csv"])
    data_dir_abs = _resolve_cfg_path(paths_cfg["data_dir"])
    with open(config_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tetramer_cache_root": str(cache_root),
                "datasets_csv": str(datasets_csv_abs),
                "data_dir": str(data_dir_abs),
                "n_uc": n_uc,
                "n_cap": n_cap,
                "n_clusters": n_clusters,
                "random_state": random_state,
                "pca_components": pca_components,
                "pca_variance": pca_variance,
                "pca_components_used": pca_components_used,
                "seq_normalize": transform.normalize,
                "seq_log1p": transform.log1p,
                "cap_transform": cap_transform,
                "clr_pseudocount": clr_pseudocount,
                "batch_size": batch_size,
                "max_iter": max_iter,
                "cap_output_csv": str(cap_out),
                "uc_assignments_csv": str(uc_assign_out),
                "uc_model_pkl": str(model_out),
                "uc_unique_clusters": uc_unique_clusters,
                "cap_runs": int(len(cap_df)),
                "cap_split_counts": split_counts,
                "cap_n_assigned_min": cap_n_assigned_min,
                "cap_n_assigned_max": cap_n_assigned_max,
                "cap_nonzero_clusters_mean": cap_nnz_mean,
                "cap_nonzero_clusters_median": cap_nnz_median,
                "cap_nonzero_clusters_min": cap_nnz_min,
                "cap_nonzero_clusters_max": cap_nnz_max,
                "feature_index": feature_index,
            },
            f,
            indent=2,
        )

    elapsed = time.perf_counter() - start
    print(
        f"UC clusters used: {uc_unique_clusters}/{n_clusters}",
        flush=True,
    )
    print(
        "CAP nonzero clusters per run "
        f"(mean/median/min/max): {cap_nnz_mean:.2f}/{cap_nnz_median:.1f}/"
        f"{cap_nnz_min}/{cap_nnz_max}",
        flush=True,
    )
    print(f"Output directory: {uc_dir}")
    print(f"Elapsed: {elapsed:.2f}s")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / "defaults.yaml"
    feat = _parse_feature_cli(argv)
    merged, feature_index = load_merged_uc_cap_config(repo_root, feat=feat)
    if feature_index == 0:
        print("Baseline config (defaults.yaml only)", flush=True)
    return run_pipeline_from_merged(
        repo_root,
        config_path=config_path,
        merged=merged,
        feature_index=feature_index,
    )


if __name__ == "__main__":
    raise SystemExit(main())
