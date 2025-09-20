"""Compatibility helpers for deprecated NumPy aliases.

NumPy 1.24 removed a handful of historical alias names such as ``np.bool`` and
``np.int``.  Theano, which powers the neural network implementation in this
package, still expects those aliases to be present.  This module reinstates the
aliases when running on modern NumPy versions so the rest of the codebase can be
imported without modification.
"""

from __future__ import annotations

import numpy as np


def _ensure_alias(name: str, target) -> None:
    """Ensure ``numpy.<name>`` exists and points to *target*.

    The helper mirrors the behaviour from older NumPy versions without
    overwriting the attribute when it is still provided upstream.
    """

    if not hasattr(np, name):
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

# ``np.bool_`` remains available as the NumPy boolean scalar type.  Having the
# ``np.bool`` alias present again ensures both spellings are usable when
# interacting with downstream libraries such as Theano.
del _alias, _target
