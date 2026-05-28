"""Internal helpers used across the flattened SetBERT package."""

from __future__ import annotations

import sys


def export(fn):
    """Append ``fn``'s name to its module's ``__all__`` and return it.

    Based on Duncan Booth's idea (http://groups.google.com/group/comp.lang.python/msg/11cbb03e09611b8a)
    via Dave Angel (http://groups.google.com/group/comp.lang.python/msg/3d400fb22d8a42e1).
    """
    mod = sys.modules[fn.__module__]
    if hasattr(mod, "__all__"):
        name = fn.__name__
        all_ = mod.__all__
        if name not in all_:
            all_.append(name)
    else:
        mod.__all__ = [fn.__name__]  # type: ignore[attr-defined]
    return fn


def deprecated(reason):
    """No-op stand-in for ``@deprecated`` decorators on legacy layers.

    The upstream deepbio-toolkit shipped a stub identical to this one (the
    real warning-emitting body had been commented out). We keep the stub so
    the decorator syntax in ``layers.py`` keeps working without adding the
    PyPI ``Deprecated`` dependency.
    """
    def decorator(f):
        return f
    return decorator
