###############################################################
#
# Copyright 2017 Sandia Corporation. Under the terms of
# Contract DE-AC04-94AL85000 with Sandia Corporation, the
# U.S. Government retains certain rights in this software.
# This software is distributed under the BSD-3-Clause license.
#
##############################################################

import importlib
import sys

import numpy as np


def test_tbnn_import_restores_numpy_bool(monkeypatch):
    """Importing :mod:`tbnn` should recreate legacy NumPy aliases."""

    monkeypatch.delattr(np, "bool", raising=False)
    assert not hasattr(np, "bool")

    if "tbnn" in sys.modules:
        tbnn_module = sys.modules["tbnn"]
        compat_module = sys.modules.get("tbnn._compat")
        if compat_module is not None:
            importlib.reload(compat_module)
        else:
            compat_module = importlib.import_module("tbnn._compat")
        importlib.reload(tbnn_module)
    else:
        importlib.import_module("tbnn")

    assert hasattr(np, "bool")
    assert getattr(np, "bool") in (bool, np.bool_)
