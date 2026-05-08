#!/usr/bin/env python3
"""experiments.yaml run_tensors rows merged over defaults.yaml build_run_tensors (shared helpers)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

def resolve_repo_path(repo_root: Path, raw: object) -> Path:
    p = Path(str(raw).strip())
    return p if p.is_absolute() else repo_root / p


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def run_tensors_rows(experiments_path: Path) -> List[Dict[str, Any]]:
    """Return experiments.yaml run_tensors list entries (each row shallow-merged over defaults)."""
    if not experiments_path.is_file():
        raise SystemExit(
            f"Missing {experiments_path}; it must define a non-empty run_tensors list "
            "(tensor caches live under paths.run_tensors_dir/<1-based index>/)."
        )
    cfg = _load_yaml_mapping(experiments_path)
    lst = cfg.get("run_tensors")
    if not isinstance(lst, list) or len(lst) == 0:
        raise SystemExit(
            f"{experiments_path} must define run_tensors as a non-empty list "
            "(each entry merges over defaults.yaml build_run_tensors)."
        )
    out: List[Dict[str, Any]] = []
    for item in lst:
        if not isinstance(item, dict):
            raise SystemExit("run_tensors list entries must be mappings.")
        out.append(dict(item))
    return out


def base_run_tensors_root(repo_root: Path, defaults_path: Path) -> Path:
    cfg = _load_yaml_mapping(defaults_path)
    paths_cfg = cfg.get("paths")
    if not isinstance(paths_cfg, dict):
        raise SystemExit(f"{defaults_path} must define paths as a mapping.")
    key = str(paths_cfg.get("run_tensors_dir", "outputs/run_tensors")).strip()
    return resolve_repo_path(repo_root, key)


def default_build_run_tensors(defaults_path: Path) -> Dict[str, Any]:
    cfg = _load_yaml_mapping(defaults_path)
    sec = cfg.get("build_run_tensors")
    if not isinstance(sec, dict):
        raise SystemExit(f"{defaults_path} must define build_run_tensors as a mapping.")
    return dict(sec)


def merged_build_run_tensors_for_cache(
    defaults_path: Path,
    experiments_path: Path,
    *,
    cache_1based: int,
) -> Dict[str, Any]:
    """1-based cache index into experiments.yaml run_tensors; shallow-merge over defaults."""
    base = default_build_run_tensors(defaults_path)
    rows = run_tensors_rows(experiments_path)
    if cache_1based < 1 or cache_1based > len(rows):
        raise SystemExit(
            f"--cache {cache_1based} out of range; experiments.yaml has {len(rows)} run_tensors row(s)."
        )
    row = rows[cache_1based - 1]
    return {**base, **row}
