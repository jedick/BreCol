"""FASTA iteration and run eligibility shared by tetramer and embedding cache builders."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterable, List, Mapping, Tuple


def row_is_sample_used(row: Mapping[str, object]) -> bool:
    return (row.get("sample_used") or "").strip().casefold() == "true"


def count_fasta_records(fasta_gz: Path) -> int:
    """Count FASTA records by header lines only (no sequence assembly)."""
    n = 0
    with gzip.open(fasta_gz, "rt", encoding="ascii", errors="replace") as handle:
        for raw in handle:
            if raw.strip().startswith(">"):
                n += 1
    return n


def iter_selected_fasta_sequences(
    fasta_gz: Path,
    wanted: set[int],
    max_index: int,
) -> Iterable[Tuple[int, str]]:
    """Yield (0-based sequence index, sequence) for selected indices only."""
    seq_index = -1
    collecting = False
    chunks: List[str] = []
    with gzip.open(fasta_gz, "rt", encoding="ascii", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if collecting and chunks:
                    yield seq_index, "".join(chunks)
                seq_index += 1
                if seq_index > max_index:
                    return
                collecting = seq_index in wanted
                chunks = []
                continue
            if collecting:
                chunks.append(line)
        if collecting and chunks:
            yield seq_index, "".join(chunks)
