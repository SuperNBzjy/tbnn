"""Compatibility helpers for deprecated NumPy aliases and distutils shims."""

from __future__ import annotations

import importlib
import warnings
from typing import Any

import numpy as np


def _has_legacy_alias(name: str) -> bool:
    """Return ``True`` when ``numpy.<name>`` is still provided upstream."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", FutureWarning)
        return hasattr(np, name)


def _ensure_alias(name: str, target: Any) -> None:
    """Ensure ``numpy.<name>`` exists and points to *target*."""

    # NumPy 1.20+ emits FutureWarnings when the deprecated aliases are touched.
    # Using :func:`hasattr` inside a warnings context keeps the import noise-free
    # for users on intermediary releases while still reinstating the alias when
    # it has been removed entirely (NumPy 1.24+).
    if not _has_legacy_alias(name):
        setattr(np, name, target)


_ALIAS_TARGETS = {
    # Historically these aliases mirrored the corresponding Python built-in
    # types.  We keep that behaviour so code such as ``np.zeros(..., dtype=np.int)``
    # continues to work without modification.
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "str": str,
    "unicode": str,
    "long": int,
}

for _alias, _target in _ALIAS_TARGETS.items():
    _ensure_alias(_alias, _target)


def _ensure_numpy_distutils_shim() -> None:
    """Provide minimal ``numpy.distutils`` configuration attributes when missing."""

    try:
        np_config = importlib.import_module("numpy.distutils.__config__")
    except Exception:  # pragma: no cover - depends on external optional module
        return

    for _name in ("blas_opt_info", "lapack_opt_info"):
        if not hasattr(np_config, _name):
            setattr(np_config, _name, {})


_ensure_numpy_distutils_shim()

# ``np.bool_`` remains available as the NumPy boolean scalar type.  Having the
# ``np.bool`` alias present again ensures both spellings are usable when
# interacting with downstream libraries such as Theano.
del _alias, _target
