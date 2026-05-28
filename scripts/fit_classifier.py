#!/usr/bin/env python3
"""
Train a classifier on run-level tetramer frequencies or UC/CAP cluster features.

Modes (mutually exclusive):
  --tetramer   Inputs: paths.tetramer_frequencies_csv (256 ACGT tetramer columns).
  --uc_cap     Inputs: CAP CSV from run_uc_cap_pipeline (cluster_* columns).
               Optional --emb selects embedding-based CAP outputs (embedding_uc_cap_root).

Both modes use scripts/shared_utilities.py for train/val/test/holdout, support the same
model families (baseline, knn, random_forest, svm), hyperparameter grids from
defaults.yaml fit_classifier (YAML lists; validation tuning_metric, default auc),
and optional experiment overlays from experiments.yaml (fit_classifier.experiments).

UC/CAP mode:
  --feat N   1-based index into experiments.yaml run_uc_cap_pipeline (merged over the
             defaults.yaml baseline). Omit --feat to use baseline-only config.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

# sklearn PCA n_components: None (all), int (fixed count), or float in (0, 1] (variance ratio).
PcaNComponents = Optional[Union[int, float]]

import numpy as np
import pandas as pd
import yaml
from shared_utilities import (
    HOLDOUT,
    TEST,
    TRAIN,
    VAL,
    binary_auc_from_scores,
    build_run_task_table,
    require_binary_classes,
    TETRAMERS,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ----- Constants -----
MODEL_CHOICES = ("baseline", "knn", "random_forest", "svm")
TUNING_METRIC_CHOICES = ("auc", "f1")


@dataclass(frozen=True)
class TaskSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_holdout: np.ndarray
    y_holdout: np.ndarray
    study_test: np.ndarray
    study_holdout: np.ndarray

    @property
    def y_development(self) -> np.ndarray:
        return np.concatenate((self.y_train, self.y_val, self.y_test))


@dataclass(frozen=True)
class ModelGrids:
    knn_n_neighbors: List[int]
    knn_weights: List[str]
    rf_n_estimators: List[int]
    rf_max_depth: List[Optional[int]]
    rf_min_samples_leaf: List[int]
    svm_c: List[float]
    svm_gamma: List[Union[float, str]]
    svm_kernel: List[str]


@dataclass(frozen=True)
class TuningResult:
    best_params: Dict[str, object]
    validation_score: float


@dataclass(frozen=True)
class EvaluationResult:
    test_auc: float
    holdout_auc: float
    test_per_study: Dict[str, Dict[str, object]]
    holdout_per_study: Dict[str, Dict[str, object]]


# ----- UC/CAP path resolution (aligned with helpers/list_uc_cap_feature_outputs.py) -----


def _merge_uc_cap_row(
    defaults_cfg: Mapping[str, Any],
    experiments_cfg: Mapping[str, Any],
    *,
    feat: Optional[int],
) -> Dict[str, Any]:
    """feat None = defaults run_uc_cap_pipeline list only; feat >= 1 adds experiments row."""
    baseline = defaults_cfg.get("run_uc_cap_pipeline")
    if not isinstance(baseline, list):
        raise SystemExit("defaults.yaml run_uc_cap_pipeline must be a list")
    merged: Dict[str, Any] = {}
    for frag in baseline:
        if not isinstance(frag, dict):
            raise SystemExit("defaults.yaml run_uc_cap_pipeline entries must be mappings")
        merged = {**merged, **frag}
    if feat is None:
        return merged
    if feat < 1:
        raise SystemExit("--feat must be >= 1 when provided.")
    rows = experiments_cfg.get("run_uc_cap_pipeline") or []
    if not isinstance(rows, list):
        raise SystemExit("experiments.yaml run_uc_cap_pipeline must be a list")
    if feat > len(rows):
        raise SystemExit(
            f"--feat {feat} is out of range. experiments.yaml defines {len(rows)} UC/CAP rows."
        )
    row = rows[feat - 1]
    if not isinstance(row, dict):
        raise SystemExit("experiments.yaml run_uc_cap_pipeline entries must be mappings")
    return {**merged, **row}


def _cap_csv_path(
    repo_root: Path,
    paths_cfg: Mapping[str, Any],
    merged: Dict[str, Any],
    *,
    use_embeddings: bool,
) -> Path:
    uc_root = str(
        paths_cfg["embedding_uc_cap_root" if use_embeddings else "tetramer_uc_cap_root"]
    ).strip()
    n_uc = int(merged["n_uc"])
    n_clusters = int(merged["n_clusters"])
    n_cap = int(merged["n_cap"])
    tag = str(n_cap)
    return repo_root / uc_root / f"uc{n_uc}_k{n_clusters}" / f"cap{tag}.csv"


def _load_cap_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit("Input CSV has no data rows.")
    if "Run" not in df.columns:
        raise SystemExit("CSV missing required column: Run")
    if not any(c.startswith("cluster_") for c in df.columns):
        raise SystemExit("CSV has no cluster feature columns (expected prefix 'cluster_').")
    if df["Run"].isna().any():
        raise SystemExit("Found empty Run values in CSV.")
    return df


# ----- Config loading -----


def _load_experiment_args(
    root: Path,
    *,
    expt: int,
    features: str,
    feat: Optional[int],
    results_json_cli: Optional[str],
    use_embeddings: bool = False,
) -> SimpleNamespace:
    defaults_path = root / "defaults.yaml"
    experiments_path = root / "experiments.yaml"
    try:
        defaults_cfg = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))
        experiments_cfg = (
            yaml.safe_load(experiments_path.read_text(encoding="utf-8"))
            if experiments_path.is_file()
            else {}
        )
    except OSError as exc:
        raise SystemExit(f"Failed to read config file: {exc}") from exc

    try:
        defaults = dict(defaults_cfg["fit_classifier"])
        paths_cfg = defaults_cfg["paths"]
        experiments_section = experiments_cfg.get("fit_classifier", {})
        experiments = experiments_section.get("experiments", [])
        results_json_template = experiments_section.get("results_json_template")
    except (TypeError, KeyError) as exc:
        raise SystemExit(f"Invalid defaults/experiment configuration: {exc}") from exc

    if not isinstance(experiments, list):
        raise SystemExit(f"Invalid experiments list in {experiments_path}")

    experiment_name = None
    if expt == 0:
        selected: Dict[str, Any] = {}
    else:
        if not experiments:
            raise SystemExit(f"No experiments found in {experiments_path}")
        if expt > len(experiments):
            raise SystemExit(
                f"--expt {expt} is out of range. experiments.yaml defines {len(experiments)} experiments."
            )
        selected = experiments[expt - 1]
        experiment_name = selected.get("name")

    if not isinstance(selected, dict):
        raise SystemExit("Selected experiment entry must be a mapping.")
    overrides = selected.get("overrides", {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise SystemExit(f"Experiment {expt} overrides must be a mapping.")

    config = {**defaults, **overrides}
    if expt != 0 and config.get("results_json") is None and results_json_template is not None:
        if not isinstance(results_json_template, str) or not results_json_template.strip():
            raise SystemExit(
                "experiments.yaml results_json_template must be a non-empty string."
            )
        if not isinstance(experiment_name, str) or not experiment_name.strip():
            raise SystemExit(
                "Each experiment must define a non-empty 'name' when using results_json_template."
            )
        config["results_json"] = results_json_template.format(
            name=experiment_name.strip(),
            features=features,
        )

    results_json = config.get("results_json")
    if results_json_cli is not None:
        results_json = results_json_cli

    if features == "tetramer":
        csv_path = root / str(paths_cfg["tetramer_frequencies_csv"]).strip()
    elif features == "uc_cap":
        merged_uc = _merge_uc_cap_row(defaults_cfg, experiments_cfg, feat=feat)
        csv_path = _cap_csv_path(
            root, paths_cfg, merged_uc, use_embeddings=use_embeddings
        )
    else:
        raise SystemExit(f"Unknown features mode: {features!r}")

    use_scaler = bool(config["use_scaler"])
    use_clr = bool(config["use_clr"])

    args_dict: Dict[str, Any] = {
        "features": features,
        "feat_index": feat,
        "use_embeddings": use_embeddings,
        "csv": csv_path,
        "model": str(config["model"]).strip(),
        "task": str(config["task"]).strip(),
        "random_state": int(config["random_state"]),
        "tuning_metric": str(config["tuning_metric"]).strip(),
        "no_scaler": not use_scaler,
        "no_clr": not use_clr,
        "clr_pseudocount": float(config["clr_pseudocount"]),
        "n_components_grid": _parse_pca_n_components_grid_list(
            config["n_components_grid"], name="n_components_grid"
        ),
        "knn_n_neighbors": _parse_int_grid_list(
            config["knn_n_neighbors_grid"], name="knn_n_neighbors_grid"
        ),
        "knn_weights": _parse_str_grid_list(
            config["knn_weights_grid"], name="knn_weights_grid"
        ),
        "rf_n_estimators": _parse_int_grid_list(
            config["rf_n_estimators_grid"], name="rf_n_estimators_grid"
        ),
        "rf_max_depth": _parse_optional_int_grid_list(
            config["rf_max_depth_grid"], name="rf_max_depth_grid"
        ),
        "rf_min_samples_leaf": _parse_int_grid_list(
            config["rf_min_samples_leaf_grid"], name="rf_min_samples_leaf_grid"
        ),
        "svm_c": _parse_float_grid_list(config["svm_c_grid"], name="svm_c_grid"),
        "svm_gamma": _parse_svm_gamma_grid_list(config["svm_gamma_grid"], name="svm_gamma_grid"),
        "svm_kernel": _parse_svm_kernel_grid_list(
            config["svm_kernel_grid"], name="svm_kernel_grid"
        ),
        "results_json": results_json,
        "experiment_index": expt,
        "experiment_overrides": dict(overrides),
        "log_prefix": (f"E{expt}" if expt > 0 else ""),
    }
    return SimpleNamespace(**args_dict)


def _print_experiment_line(args: argparse.Namespace) -> None:
    expt = int(getattr(args, "experiment_index", 0))
    model = str(getattr(args, "model", ""))
    task = str(getattr(args, "task", ""))
    prefix = str(getattr(args, "log_prefix", ""))
    features = str(getattr(args, "features", ""))
    feat = getattr(args, "feat_index", None)
    if expt == 0:
        print("Default config", flush=True)
    else:
        print(_prefixed(prefix, f"Config - model: {model}, task: {task}"), flush=True)
    if features == "uc_cap":
        fi = "baseline" if feat is None else str(feat)
        emb = "embeddings" if getattr(args, "use_embeddings", False) else "tetramer"
        print(_prefixed(prefix, f"UC/CAP feature set: {fi} ({emb})"), flush=True)


def _validate_basic_args(args: argparse.Namespace) -> None:
    if args.clr_pseudocount <= 0:
        raise SystemExit("--clr-pseudocount must be positive.")
    if args.model not in MODEL_CHOICES:
        raise SystemExit(f"Unknown model {args.model!r}. Expected one of {MODEL_CHOICES}.")
    if args.tuning_metric not in TUNING_METRIC_CHOICES:
        raise SystemExit(
            f"Unknown tuning_metric {args.tuning_metric!r}. "
            f"Expected one of {TUNING_METRIC_CHOICES}."
        )


def _prefixed(prefix: str, text: str) -> str:
    return f"{prefix} {text}" if prefix else text


def _normalize_grid(raw: object, *, name: str) -> List[object]:
    """Accept a YAML list or a single scalar; reject comma-separated strings."""
    if isinstance(raw, str):
        raise SystemExit(
            f"{name} must be a YAML list or scalar, not a string "
            f"(got {raw!r}; use a list like [5, 15])."
        )
    if isinstance(raw, (list, tuple)):
        if not raw:
            raise SystemExit(f"{name} must not be empty.")
        return list(raw)
    return [raw]


def _coerce_optional_int(value: object, *, name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "none":
        return None
    if isinstance(value, bool):
        raise SystemExit(f"{name}: booleans are not valid integers.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"{name}: expected an integer or none, got {value!r}."
        ) from exc


def _parse_int_grid_list(raw: object, *, name: str) -> List[int]:
    out: List[int] = []
    for item in _normalize_grid(raw, name=name):
        if isinstance(item, bool):
            raise SystemExit(f"{name}: booleans are not valid integers.")
        try:
            out.append(int(item))
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"{name}: expected integers, got {item!r}.") from exc
    return out


def _parse_float_grid_list(raw: object, *, name: str) -> List[float]:
    out: List[float] = []
    for item in _normalize_grid(raw, name=name):
        try:
            out.append(float(item))
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"{name}: expected floats, got {item!r}.") from exc
    return out


def _parse_str_grid_list(raw: object, *, name: str) -> List[str]:
    out: List[str] = []
    for item in _normalize_grid(raw, name=name):
        if not isinstance(item, str) or not item.strip():
            raise SystemExit(f"{name}: expected non-empty strings, got {item!r}.")
        out.append(item.strip())
    return out


def _parse_optional_int_grid_list(raw: object, *, name: str) -> List[Optional[int]]:
    return [
        _coerce_optional_int(item, name=name) for item in _normalize_grid(raw, name=name)
    ]


def _coerce_pca_n_components(value: object, *, name: str) -> PcaNComponents:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "none":
        return None
    if isinstance(value, bool):
        raise SystemExit(f"{name}: booleans are not valid PCA n_components values.")
    if isinstance(value, int):
        if value < 1:
            raise SystemExit(f"{name}: PCA component count must be >= 1, got {value}.")
        return value
    if isinstance(value, float):
        if 0.0 < value <= 1.0:
            return float(value)
        raise SystemExit(
            f"{name}: explained-variance ratio must be in (0, 1], got {value}."
        )
    raise SystemExit(
        f"{name}: expected none, an integer >= 1, or a float in (0, 1]; got {value!r}."
    )


def _parse_pca_n_components_grid_list(raw: object, *, name: str) -> List[PcaNComponents]:
    return [
        _coerce_pca_n_components(item, name=name) for item in _normalize_grid(raw, name=name)
    ]


def _parse_svm_gamma_grid_list(raw: object, *, name: str) -> List[Union[float, str]]:
    out: List[Union[float, str]] = []
    for item in _normalize_grid(raw, name=name):
        if isinstance(item, str):
            lv = item.strip().lower()
            if lv in ("scale", "auto"):
                out.append(lv)
                continue
        try:
            out.append(float(item))
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"{name}: each entry must be scale, auto, or a float; got {item!r}."
            ) from exc
    return out


def _parse_svm_kernel_grid_list(raw: object, *, name: str) -> List[str]:
    allowed = {"rbf", "linear", "poly", "sigmoid"}
    kernels = _parse_str_grid_list(raw, name=name)
    bad = [k for k in kernels if k not in allowed]
    if bad:
        raise SystemExit(f"Unsupported SVM kernel(s) in {name}: {bad}. Allowed: {sorted(allowed)}.")
    return kernels


def _build_model_grids(args: argparse.Namespace) -> ModelGrids:
    return ModelGrids(
        knn_n_neighbors=list(args.knn_n_neighbors),
        knn_weights=list(args.knn_weights),
        rf_n_estimators=list(args.rf_n_estimators),
        rf_max_depth=list(args.rf_max_depth),
        rf_min_samples_leaf=list(args.rf_min_samples_leaf),
        svm_c=list(args.svm_c),
        svm_gamma=list(args.svm_gamma),
        svm_kernel=list(args.svm_kernel),
    )


# ----- Data loading and task splits -----


def _load_tetramer_features(csv_path: Path) -> pd.DataFrame:
    """Load the tetramer feature CSV; return a table with Run and 256 tetramer columns."""
    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit("No data rows in tetramer CSV.")
    required = {"Run", *TETRAMERS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(
            f"Tetramer CSV is missing {len(missing)} required columns "
            f"(first few: {missing[:5]!r})."
        )
    out = df.copy()
    out["Run"] = out["Run"].astype(str).str.strip()
    if (out["Run"] == "").any():
        raise SystemExit("Found empty 'Run' values in tetramer CSV.")
    return out


def _build_task_splits(
    task_df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build split → (X, y, study) arrays from a table with task_label, split, study_name."""
    frames = {
        TRAIN: task_df[task_df["split"] == TRAIN],
        VAL: task_df[task_df["split"] == VAL],
        TEST: task_df[task_df["split"] == TEST],
        HOLDOUT: task_df[task_df["split"] == HOLDOUT],
    }
    for split_name in (TRAIN, VAL, TEST):
        if frames[split_name].empty:
            raise SystemExit(f"No {split_name} rows found after task filtering.")
    return {
        split_name: (
            frame.loc[:, list(feature_cols)].to_numpy(dtype=np.float64, copy=False),
            frame["task_label"].to_numpy(dtype=object),
            frame["study_name"].to_numpy(dtype=object),
        )
        for split_name, frame in frames.items()
    }


def _task_splits_from_table(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
) -> TaskSplits:
    split_xy = _build_task_splits(df, feature_cols=feature_cols)
    X_train, y_train, _ = split_xy[TRAIN]
    X_val, y_val, _ = split_xy[VAL]
    X_test, y_test, study_test = split_xy[TEST]
    X_holdout, y_holdout, study_holdout = split_xy[HOLDOUT]
    return TaskSplits(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        X_holdout=X_holdout,
        y_holdout=y_holdout,
        study_test=study_test,
        study_holdout=study_holdout,
    )


# ----- Feature preprocessing and pipelines -----


class CLRTransformer(BaseEstimator, TransformerMixin):
    """Centered log-ratio transform for compositional nonnegative features."""

    def __init__(self, pseudocount: float = 1e-6):
        self.pseudocount = float(pseudocount)

    def fit(self, X, y=None):  # noqa: ARG002
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.pseudocount <= 0:
            raise ValueError("pseudocount must be positive.")
        if np.any(X < 0):
            raise ValueError("CLR expects nonnegative compositions before pseudocount.")
        log_x = np.log(X + self.pseudocount)
        return log_x - np.mean(log_x, axis=1, keepdims=True)


def make_pipeline(
    *,
    model: str,
    use_clr: bool,
    pseudocount: float,
    use_scaler: bool,
    pca_n_components: PcaNComponents,
    random_state: int,
) -> Pipeline:
    steps: List[Tuple[str, object]] = []
    if use_clr:
        steps.append(("clr", CLRTransformer(pseudocount=pseudocount)))
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    if model in ("knn", "svm"):
        steps.append(
            (
                "pca",
                PCA(
                    n_components=pca_n_components,
                    svd_solver="full",
                    random_state=random_state,
                ),
            )
        )
    if model == "knn":
        steps.append(("clf", KNeighborsClassifier()))
    elif model == "random_forest":
        steps.append(("clf", RandomForestClassifier(random_state=random_state)))
    elif model == "svm":
        steps.append(
            (
                "clf",
                SVC(random_state=random_state),
            )
        )
    elif model == "baseline":
        steps.append(("clf", DummyClassifier(strategy="most_frequent")))
    else:
        raise SystemExit(f"Unknown model value: {model!r}")
    return Pipeline(steps)


# ----- Validation tuning -----


def _evaluation_scores(pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        proba = pipe.predict_proba(X)
        classes = clf.classes_
        if classes.size != 2:
            raise SystemExit("Binary classification expected.")
        pos = list(classes).index(classes[1])
        return proba[:, pos]
    if hasattr(pipe, "decision_function"):
        return np.asarray(pipe.decision_function(X), dtype=np.float64).ravel()
    raise SystemExit("Classifier has neither predict_proba nor decision_function.")


def _score_val(y_true: np.ndarray, y_pred: np.ndarray, tuning_metric: str) -> float:
    if tuning_metric == "f1":
        classes = np.unique(np.asarray(y_true, dtype=object))
        if classes.size != 2:
            return float("nan")
        return float(
            f1_score(
                y_true,
                y_pred,
                pos_label=classes[1],
                average="binary",
                zero_division=0,
            )
        )
    raise SystemExit(f"Unknown tuning_metric: {tuning_metric!r}")


def _validation_score(pipe: Pipeline, splits: TaskSplits, tuning_metric: str) -> float:
    if tuning_metric == "auc":
        y_score = _evaluation_scores(pipe, splits.X_val)
        auc = binary_auc_from_scores(splits.y_val, y_score)
        return float(auc) if np.isfinite(auc) else float("-inf")
    return _score_val(splits.y_val, pipe.predict(splits.X_val), tuning_metric)


def tune_knn_on_val(
    pipe: Pipeline,
    splits: TaskSplits,
    n_components_list: Sequence[PcaNComponents],
    n_neighbors_list: Sequence[int],
    weights_list: Sequence[str],
    tuning_metric: str,
) -> TuningResult:
    best_score = -1.0
    best_params: Dict[str, object] = {}
    for n_components, n_neighbors, weights in itertools.product(
        n_components_list,
        n_neighbors_list,
        weights_list,
    ):
        pipe.set_params(
            pca__n_components=n_components,
            clf__n_neighbors=n_neighbors,
            clf__weights=weights,
        )
        pipe.fit(splits.X_train, splits.y_train)
        score = _validation_score(pipe, splits, tuning_metric)
        if score > best_score:
            best_score = score
            best_params = {
                "n_components": n_components,
                "n_neighbors": n_neighbors,
                "weights": weights,
            }
    return TuningResult(best_params=best_params, validation_score=best_score)


def tune_random_forest_on_val(
    pipe: Pipeline,
    splits: TaskSplits,
    n_estimators_list: Sequence[int],
    max_depth_list: Sequence[Optional[int]],
    min_samples_leaf_list: Sequence[int],
    tuning_metric: str,
) -> TuningResult:
    best_score = -1.0
    best_params: Dict[str, object] = {}
    for n_estimators, max_depth, min_samples_leaf in itertools.product(
        n_estimators_list,
        max_depth_list,
        min_samples_leaf_list,
    ):
        pipe.set_params(
            clf__n_estimators=n_estimators,
            clf__max_depth=max_depth,
            clf__min_samples_leaf=min_samples_leaf,
        )
        pipe.fit(splits.X_train, splits.y_train)
        score = _validation_score(pipe, splits, tuning_metric)
        if score > best_score:
            best_score = score
            best_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
            }
    return TuningResult(best_params=best_params, validation_score=best_score)


def tune_svm_on_val(
    pipe: Pipeline,
    splits: TaskSplits,
    n_components_list: Sequence[PcaNComponents],
    c_list: Sequence[float],
    gamma_list: Sequence[Union[float, str]],
    kernel_list: Sequence[str],
    tuning_metric: str,
) -> TuningResult:
    best_score = -1.0
    best_params: Dict[str, object] = {}
    for n_components, c_val, gamma, kernel in itertools.product(
        n_components_list,
        c_list,
        gamma_list,
        kernel_list,
    ):
        pipe.set_params(
            pca__n_components=n_components,
            clf__C=c_val,
            clf__gamma=gamma,
            clf__kernel=kernel,
        )
        pipe.fit(splits.X_train, splits.y_train)
        score = _validation_score(pipe, splits, tuning_metric)
        if score > best_score:
            best_score = score
            best_params = {
                "n_components": n_components,
                "C": c_val,
                "gamma": gamma,
                "kernel": kernel,
            }
    return TuningResult(best_params=best_params, validation_score=best_score)


def _tune_model_on_validation(
    pipe: Pipeline,
    *,
    args: argparse.Namespace,
    splits: TaskSplits,
    grids: ModelGrids,
    n_components_grid: Sequence[PcaNComponents],
) -> TuningResult:
    prefix = str(getattr(args, "log_prefix", ""))
    if args.model == "baseline":
        print(
            _prefixed(
                prefix,
                "Grid - baseline: DummyClassifier(strategy=most_frequent) (no hyperparameter search)",
            ),
            flush=True,
        )
        pipe.fit(splits.X_train, splits.y_train)
        score = _validation_score(pipe, splits, args.tuning_metric)
        return TuningResult(
            best_params={"strategy": "most_frequent"},
            validation_score=score,
        )

    if args.model == "knn":
        print(
            _prefixed(
                prefix,
                f"Grid - n_components: {list(n_components_grid)}, "
                f"n_neighbors: {grids.knn_n_neighbors}, weights: {grids.knn_weights}",
            ),
            flush=True,
        )
        result = tune_knn_on_val(
            pipe,
            splits,
            n_components_list=n_components_grid,
            n_neighbors_list=grids.knn_n_neighbors,
            weights_list=grids.knn_weights,
            tuning_metric=args.tuning_metric,
        )
        pipe.set_params(
            pca__n_components=result.best_params["n_components"],
            clf__n_neighbors=result.best_params["n_neighbors"],
            clf__weights=result.best_params["weights"],
        )
        return result

    if args.model == "random_forest":
        print(
            _prefixed(
                prefix,
                f"Grid - n_estimators: {grids.rf_n_estimators}, "
                f"max_depth: {grids.rf_max_depth}, "
                f"min_samples_leaf: {grids.rf_min_samples_leaf}",
            ),
            flush=True,
        )
        result = tune_random_forest_on_val(
            pipe,
            splits,
            n_estimators_list=grids.rf_n_estimators,
            max_depth_list=grids.rf_max_depth,
            min_samples_leaf_list=grids.rf_min_samples_leaf,
            tuning_metric=args.tuning_metric,
        )
        pipe.set_params(
            clf__n_estimators=result.best_params["n_estimators"],
            clf__max_depth=result.best_params["max_depth"],
            clf__min_samples_leaf=result.best_params["min_samples_leaf"],
        )
        return result

    if args.model == "svm":
        print(
            _prefixed(
                prefix,
                f"Grid - n_components: {list(n_components_grid)}, "
                f"C: {grids.svm_c}, gamma: {grids.svm_gamma}, kernel: {grids.svm_kernel}",
            ),
            flush=True,
        )
        result = tune_svm_on_val(
            pipe,
            splits,
            n_components_list=n_components_grid,
            c_list=grids.svm_c,
            gamma_list=grids.svm_gamma,
            kernel_list=grids.svm_kernel,
            tuning_metric=args.tuning_metric,
        )
        pipe.set_params(
            pca__n_components=result.best_params["n_components"],
            clf__C=result.best_params["C"],
            clf__gamma=result.best_params["gamma"],
            clf__kernel=result.best_params["kernel"],
        )
        return result

    raise SystemExit(f"Unhandled model for tuning: {args.model!r}")


# ----- Evaluation and reporting -----


def _per_study_block(
    *,
    task: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: Optional[np.ndarray],
    study: np.ndarray,
) -> Dict[str, Dict[str, object]]:
    """Per-study metrics keyed by study_name (first-occurrence order in `study`).

    cancer_diagnosis → {'auc', 'n', 'class_counts'} per study (auc may be None
    if a study has only one class in its evaluated runs).
    cancer_type      → {'acc', 'n', 'class_counts'} per study.
    """
    if task not in ("cancer_diagnosis", "cancer_type"):
        return {}
    out: Dict[str, Dict[str, object]] = {}
    if len(study) == 0:
        return out
    y_true_obj = np.asarray(y_true, dtype=object)
    study_obj = np.asarray(study, dtype=object)
    for s in pd.unique(study_obj):
        mask = study_obj == s
        y_t_sub = y_true_obj[mask]
        n = int(mask.sum())
        classes, counts = np.unique(y_t_sub, return_counts=True)
        class_counts = {str(c): int(k) for c, k in zip(classes, counts)}
        entry: Dict[str, object] = {"n": n, "class_counts": class_counts}
        if task == "cancer_diagnosis":
            auc = binary_auc_from_scores(y_t_sub, np.asarray(y_score)[mask])
            entry["auc"] = _float_for_json(float(auc))
        else:  # cancer_type
            if y_pred is None:
                raise SystemExit("cancer_type per-study accuracy requires y_pred.")
            y_p_sub = np.asarray(y_pred, dtype=object)[mask]
            correct = int(np.sum(y_p_sub == y_t_sub))
            entry["acc"] = (correct / n) if n > 0 else None
        out[str(s)] = entry
    return out


def _evaluate_model(
    pipe: Pipeline, splits: TaskSplits, *, task: str
) -> EvaluationResult:
    pipe.fit(splits.X_train, splits.y_train)
    test_scores = _evaluation_scores(pipe, splits.X_test)
    test_pred = pipe.predict(splits.X_test) if task == "cancer_type" else None
    test_auc = binary_auc_from_scores(splits.y_test, test_scores)
    test_per_study = _per_study_block(
        task=task,
        y_true=splits.y_test,
        y_score=test_scores,
        y_pred=test_pred,
        study=splits.study_test,
    )
    holdout_auc = float("nan")
    holdout_per_study: Dict[str, Dict[str, object]] = {}
    if len(splits.y_holdout) > 0:
        hold_scores = _evaluation_scores(pipe, splits.X_holdout)
        hold_pred = (
            pipe.predict(splits.X_holdout) if task == "cancer_type" else None
        )
        holdout_auc = binary_auc_from_scores(splits.y_holdout, hold_scores)
        holdout_per_study = _per_study_block(
            task=task,
            y_true=splits.y_holdout,
            y_score=hold_scores,
            y_pred=hold_pred,
            study=splits.study_holdout,
        )
    return EvaluationResult(
        test_auc=test_auc,
        holdout_auc=holdout_auc,
        test_per_study=test_per_study,
        holdout_per_study=holdout_per_study,
    )


def _label_counts(y: np.ndarray) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for lab in y:
        out[str(lab)] = out.get(str(lab), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _print_dataset_summary(splits: TaskSplits, *, prefix: str = "") -> None:
    dev_counts = _label_counts(splits.y_development)
    holdout_counts = _label_counts(splits.y_holdout)
    dev_counts_line = ", ".join(f"{k}: {v}" for k, v in dev_counts.items())
    holdout_counts_line = ", ".join(f"{k}: {v}" for k, v in holdout_counts.items())
    print(
        _prefixed(
            prefix,
            f"Sizes - development: {len(splits.y_development)}, "
            f"holdout: {len(splits.y_holdout)}, features: {splits.X_train.shape[1]}",
        ),
        flush=True,
    )
    print(
        _prefixed(prefix, f"Development - {dev_counts_line}"),
        flush=True,
    )
    print(
        _prefixed(
            prefix,
            f"  Splits - train: {len(splits.y_train)}, "
            f"val: {len(splits.y_val)}, test: {len(splits.y_test)}",
        ),
        flush=True,
    )
    print(
        _prefixed(prefix, f"Holdout - {holdout_counts_line}"),
        flush=True,
    )


def _print_evaluation(model: str, result: EvaluationResult, *, prefix: str = "") -> None:
    del model
    test_value = f"{result.test_auc:.6f}" if np.isfinite(result.test_auc) else "nan"
    holdout_value = f"{result.holdout_auc:.6f}" if np.isfinite(result.holdout_auc) else "nan"
    print(_prefixed(prefix, "Evaluation (binary AUC):"), flush=True)
    print(_prefixed(prefix, f"  test: {test_value}"), flush=True)
    print(_prefixed(prefix, f"  holdout: {holdout_value}"), flush=True)


def _float_for_json(x: float) -> Optional[float]:
    if not math.isfinite(x):
        return None
    return float(x)


def _format_hyperparameters(params: Dict[str, object]) -> str:
    return ", ".join(f"{k}: {v}" for k, v in params.items())


def _jsonify_for_results(obj: object) -> object:
    """Recursively convert values to JSON-serializable types (e.g. numpy scalars)."""
    if isinstance(obj, dict):
        return {str(k): _jsonify_for_results(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify_for_results(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def _resolve_path_under_repo(repo_root: Path, raw: object) -> Path:
    p = Path(str(raw).strip())
    return p if p.is_absolute() else repo_root / p


def _results_json_out_path(
    repo_root: Path,
    raw: Optional[str],
    *,
    features: str,
    task: str,
    model: str,
) -> Optional[Path]:
    if raw is None:
        return None
    if raw == "":
        defaults_path = repo_root / "defaults.yaml"
        try:
            paths_cfg = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))["paths"]
            scratch_key = paths_cfg["results_scratch_dir"]
        except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
            raise SystemExit(
                f"Cannot read paths.results_scratch_dir from {defaults_path}: {exc}"
            ) from exc
        scratch_base = _resolve_path_under_repo(repo_root, scratch_key)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"{features}_{task}_{model}_{ts}.json"
        return scratch_base / name
    return Path(raw).expanduser()


def _write_results_json(
    path: Path,
    *,
    args: argparse.Namespace,
    n_components_grid: Sequence[PcaNComponents],
    tuning: TuningResult,
    evaluation: EvaluationResult,
    n_features: int,
) -> None:
    # Resolved from defaults.yaml / experiments.yaml (and CLI paths); excludes task,
    # model, and fit-time search outcomes (those live under tuning / data / metrics).
    config: Dict[str, object] = {
        "features": getattr(args, "features", ""),
        "feat_index": getattr(args, "feat_index", None),
        "csv": str(Path(args.csv).resolve()),
        "random_state": args.random_state,
        "no_scaler": args.no_scaler,
        "no_clr": args.no_clr,
        "clr_pseudocount": args.clr_pseudocount,
        "n_components_grid": _jsonify_for_results(list(args.n_components_grid)),
        "knn_n_neighbors": _jsonify_for_results(list(args.knn_n_neighbors)),
        "knn_weights": _jsonify_for_results(list(args.knn_weights)),
        "rf_n_estimators": _jsonify_for_results(list(args.rf_n_estimators)),
        "rf_max_depth": _jsonify_for_results(list(args.rf_max_depth)),
        "rf_min_samples_leaf": _jsonify_for_results(list(args.rf_min_samples_leaf)),
        "svm_c": _jsonify_for_results(list(args.svm_c)),
        "svm_gamma": _jsonify_for_results(list(args.svm_gamma)),
        "svm_kernel": _jsonify_for_results(list(args.svm_kernel)),
        "tuning_metric": args.tuning_metric,
    }
    ex_idx = getattr(args, "experiment_index", None)
    if ex_idx is not None and int(ex_idx) > 0:
        config["experiment_index"] = int(ex_idx)
        ex_over = getattr(args, "experiment_overrides", None)
        if isinstance(ex_over, dict) and ex_over:
            config["experiment_overrides"] = _jsonify_for_results(dict(ex_over))

    data_block: Dict[str, object] = {
        "n_features": int(n_features),
    }
    tuning_block: Dict[str, object] = {
        "split": "validation",
        "metric": args.tuning_metric,
        "score": float(tuning.validation_score),
        "n_components_grid": _jsonify_for_results(list(n_components_grid)),
        "best_hyperparameters": _jsonify_for_results(dict(tuning.best_params)),
    }
    test_block: Dict[str, object] = {"auc": _float_for_json(evaluation.test_auc)}
    if evaluation.test_per_study:
        test_block["per_study"] = _jsonify_for_results(evaluation.test_per_study)
    holdout_block: Dict[str, object] = {"auc": _float_for_json(evaluation.holdout_auc)}
    if evaluation.holdout_per_study:
        holdout_block["per_study"] = _jsonify_for_results(evaluation.holdout_per_study)
    metrics_block: Dict[str, object] = {
        "test": test_block,
        "holdout": holdout_block,
    }
    payload = {
        "script": Path(__file__).name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "model": args.model,
        "results_json": str(path.resolve()),
        "config": config,
        "data": data_block,
        "tuning": tuning_block,
        "metrics": metrics_block,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def run_classifier(args: argparse.Namespace, root: Path) -> int:
    _validate_basic_args(args)
    _print_experiment_line(args)
    prefix = str(getattr(args, "log_prefix", ""))
    results_json_path = _results_json_out_path(
        root,
        args.results_json,
        features=str(args.features),
        task=args.task,
        model=args.model,
    )
    config_path = root / "defaults.yaml"

    # Labels and splits come from study CSVs via shared_utilities; feature CSVs
    # provide only the Run identifier and feature columns.
    run_task_df = build_run_task_table(args.task, config_path=config_path)

    if args.features == "tetramer":
        feat_df = _load_tetramer_features(args.csv)
        merged = run_task_df.merge(
            feat_df[["Run"] + list(TETRAMERS)], on="Run", how="inner"
        )
        feature_cols = list(TETRAMERS)
    elif args.features == "uc_cap":
        cap_df = _load_cap_csv(Path(args.csv))
        feature_cols = [c for c in cap_df.columns if c.startswith("cluster_")]
        merged = run_task_df.merge(
            cap_df[["Run"] + feature_cols], on="Run", how="inner"
        )
    else:
        raise SystemExit(f"Unknown features mode: {args.features!r}")

    n_features = len(feature_cols)
    splits = _task_splits_from_table(merged, feature_cols=feature_cols)
    require_binary_classes(splits.y_train, split_name="train split", task=args.task)
    require_binary_classes(splits.y_val, split_name="validation split", task=args.task)
    require_binary_classes(splits.y_test, split_name="test split", task=args.task)
    _print_dataset_summary(splits, prefix=prefix)

    grids = _build_model_grids(args)
    if args.model in ("random_forest", "baseline"):
        n_components_grid: List[PcaNComponents] = []
        print(_prefixed(prefix, "PCA: not used for this model"), flush=True)
    else:
        n_components_grid = list(args.n_components_grid)
        if not n_components_grid:
            raise SystemExit("n_components_grid must not be empty for KNN and SVM.")
        use_clr = not args.no_clr
        print(
            _prefixed(
                prefix,
                f"PCA n_components grid: {n_components_grid}, "
                f"CLR: {'on' if use_clr else 'off'}",
            ),
            flush=True,
        )
    pipe = make_pipeline(
        model=args.model,
        use_clr=not args.no_clr,
        pseudocount=args.clr_pseudocount,
        use_scaler=not args.no_scaler,
        pca_n_components=(n_components_grid[0] if n_components_grid else None),
        random_state=args.random_state,
    )
    tuning = _tune_model_on_validation(
        pipe,
        args=args,
        splits=splits,
        grids=grids,
        n_components_grid=n_components_grid,
    )
    print(_prefixed(prefix, "Best validation:"), flush=True)
    print(_prefixed(prefix, f"  {args.tuning_metric}: {tuning.validation_score:.6f}"), flush=True)
    print(
        _prefixed(prefix, f"  Hyperparameters - {_format_hyperparameters(tuning.best_params)}"),
        flush=True,
    )

    evaluation = _evaluate_model(pipe, splits, task=str(args.task))
    _print_evaluation(args.model, evaluation, prefix=prefix)

    if results_json_path is not None:
        _write_results_json(
            results_json_path,
            args=args,
            n_components_grid=n_components_grid,
            tuning=tuning,
            evaluation=evaluation,
            n_features=n_features,
        )
    return 0


def _parse_main_argv(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mx = parser.add_mutually_exclusive_group(required=True)
    mx.add_argument("--tetramer", action="store_true", help="Use tetramer frequency features.")
    mx.add_argument("--uc_cap", action="store_true", help="Use UC/CAP cluster features.")
    parser.add_argument(
        "--emb",
        action="store_true",
        help="With --uc_cap: use embedding-based CAP CSVs (embedding_uc_cap_root).",
    )
    parser.add_argument(
        "--expt",
        type=int,
        default=None,
        help=(
            "Optional 1-based experiment index from experiments.yaml fit_classifier. "
            "If omitted, use defaults.yaml fit_classifier only."
        ),
    )
    parser.add_argument(
        "--feat",
        type=int,
        default=None,
        help=(
            "1-based feature-set index for --uc_cap (experiments.yaml run_uc_cap_pipeline). "
            "Omit for baseline-only merged config from defaults.yaml."
        ),
    )
    parser.add_argument(
        "--results-json",
        type=str,
        nargs="?",
        const="",
        default=argparse.SUPPRESS,
        help=(
            "Override results JSON path from config. With no path, writes under results/scratch/ "
            "with an auto-generated name ({features}_{task}_{model}_{utc}.json). "
            "Omit entirely to use defaults.yaml / experiments only."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.expt is not None and args.expt <= 0:
        raise SystemExit(
            "--expt must be a positive integer (1-based experiment index), or omit for defaults."
        )
    if args.tetramer and args.feat is not None:
        raise SystemExit("--feat is not valid with --tetramer.")
    if args.uc_cap and args.feat is not None and args.feat < 1:
        raise SystemExit("--feat must be >= 1 when provided.")
    if args.emb and not args.uc_cap:
        raise SystemExit("--emb is only valid with --uc_cap.")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    cli = _parse_main_argv(argv)
    expt = int(cli.expt) if cli.expt is not None else 0
    if cli.tetramer:
        features = "tetramer"
    elif cli.uc_cap:
        features = "uc_cap"
    else:
        raise SystemExit("Specify --tetramer or --uc_cap.")
    if hasattr(cli, "results_json"):
        results_json_cli: Optional[str] = cli.results_json
    else:
        results_json_cli = None

    args = _load_experiment_args(
        repo_root,
        expt=expt,
        features=features,
        feat=cli.feat,
        results_json_cli=results_json_cli,
        use_embeddings=bool(cli.emb),
    )
    return run_classifier(args, repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
