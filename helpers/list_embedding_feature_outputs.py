#!/usr/bin/env python3
"""List consolidated embedding CSV paths for Makefile FEAT rules."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


def _feature_tag(merged: Mapping[str, Any]) -> str:
    num_sets = int(merged["num_sets"])
    max_length = int(merged["max_length"])
    return f"{num_sets}sets_{max_length}L"


def _embedding_csv_path(
    repo_root: Path,
    defaults_cfg: Mapping[str, Any],
    merged: Mapping[str, Any],
) -> str:
    embeddings_dir = str(defaults_cfg["paths"]["embeddings_dir"]).strip()
    path = repo_root / embeddings_dir / f"{_feature_tag(merged)}.csv"
    return path.as_posix()


def _load_defaults(repo_root: Path) -> Dict[str, Any]:
    path = repo_root / "defaults.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_experiments(repo_root: Path) -> Dict[str, Any]:
    path = repo_root / "experiments.yaml"
    if not path.is_file():
        raise SystemExit(f"Missing {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Print embedding CSV paths under the repo: all experiment rows (default), "
            "or a single baseline path with --baseline."
        )
    )
    parser.add_argument(
        "repo_root",
        type=Path,
        help="Repository root (directory containing defaults.yaml).",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Print only the baseline embeddings path from defaults.yaml.",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root
    defaults_cfg = _load_defaults(repo_root)
    baseline = defaults_cfg.get("extract_embeddings")
    if not isinstance(baseline, dict):
        raise SystemExit("defaults.yaml extract_embeddings must be a mapping")

    if args.baseline:
        print(_embedding_csv_path(repo_root, defaults_cfg, baseline))
        return 0

    experiments_cfg = _load_experiments(repo_root)
    rows = experiments_cfg.get("extract_embeddings") or []
    if not isinstance(rows, list):
        raise SystemExit("experiments.yaml extract_embeddings must be a list")
    paths: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise SystemExit("experiments.yaml extract_embeddings entries must be mappings")
        merged = {**baseline, **row}
        paths.append(_embedding_csv_path(repo_root, defaults_cfg, merged))
    print(" ".join(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
