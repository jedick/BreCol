"""SetBERT (vendored, flat layout) for the BreCol pipeline.

This package consolidates the runtime code from upstream
`dbtk-setbert <https://github.com/DLii-Research/setbert>`_,
`dbtk-dnabert <https://github.com/DLii-Research/dbtk-dnabert>`_, and
`deepbio-toolkit <https://github.com/DLii-Research/deepbio-toolkit>`_ into a
single ``setbert`` package containing only what
``scripts/train_setbert.py`` and ``scripts/build_setbert_embeddings.py``
actually exercise. See the README for the lineage and pruning details.
"""

from __future__ import annotations

import importlib.metadata as _metadata

from .dnabert import DnaBert, DnaBertForEmbedding
from .models import SetBert, SetBertConfig

try:
    __version__ = _metadata.version("setbert")
except _metadata.PackageNotFoundError:  # editable install before metadata is built
    __version__ = "0.0.0"

__all__ = [
    "SetBert",
    "SetBertConfig",
    "DnaBert",
    "DnaBertForEmbedding",
]
