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
import types

import numpy as np


def _install_stubbed_dependencies(monkeypatch):
    fake_lasagne = types.ModuleType("lasagne")
    fake_layers = types.ModuleType("lasagne.layers")

    class _MergeLayer:
        def __init__(self, incomings, **kwargs):
            self.incomings = incomings
            self.kwargs = kwargs

    class _InputLayer:
        def __init__(self, *args, **kwargs):
            pass

    class _DenseLayer:
        def __init__(self, *args, **kwargs):
            pass

    fake_layers.MergeLayer = _MergeLayer
    fake_layers.InputLayer = _InputLayer
    fake_layers.DenseLayer = _DenseLayer
    fake_layers.get_output = lambda *args, **kwargs: None
    fake_layers.get_all_layers = lambda *args, **kwargs: [types.SimpleNamespace(input_var=None)] * 3

    fake_updates = types.ModuleType("lasagne.updates")

    def _noop(*args, **kwargs):
        return None

    fake_updates.adam = fake_updates.sgd = fake_updates.momentum = fake_updates.rmsprop = _noop

    fake_objectives = types.ModuleType("lasagne.objectives")
    fake_objectives.squared_error = _noop
    fake_objectives.aggregate = lambda value, mode=None: value

    fake_init = types.ModuleType("lasagne.init")

    class _HeUniform:
        def __init__(self, gain=None):
            self.gain = gain

    fake_init.HeUniform = _HeUniform

    fake_nonlinearities = types.ModuleType("lasagne.nonlinearities")

    class _LeakyRectify:
        def __init__(self, leakiness="0.1"):
            self.leakiness = leakiness

        def __call__(self, value):
            return value

    fake_nonlinearities.LeakyRectify = _LeakyRectify

    fake_lasagne.layers = fake_layers
    fake_lasagne.updates = fake_updates
    fake_lasagne.objectives = fake_objectives
    fake_lasagne.init = fake_init
    fake_lasagne.nonlinearities = fake_nonlinearities

    for name, module in {
        "lasagne": fake_lasagne,
        "lasagne.layers": fake_layers,
        "lasagne.updates": fake_updates,
        "lasagne.objectives": fake_objectives,
        "lasagne.init": fake_init,
        "lasagne.nonlinearities": fake_nonlinearities,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    fake_theano = types.ModuleType("theano")
    fake_tensor = types.ModuleType("theano.tensor")

    fake_tensor.dmatrix = lambda name: name
    fake_tensor.dtensor3 = lambda name: name
    fake_tensor.batched_tensordot = lambda *args, **kwargs: None

    def _fake_function(*args, **kwargs):
        def _runner(*_args, **_kwargs):
            return None

        return _runner

    def _fake_shared(value):
        return types.SimpleNamespace(get_value=lambda: value, set_value=lambda new_value: None)

    fake_theano.tensor = fake_tensor
    fake_theano.function = _fake_function
    fake_theano.shared = _fake_shared
    fake_theano.config = types.SimpleNamespace(floatX="float32")

    monkeypatch.setitem(sys.modules, "theano", fake_theano)
    monkeypatch.setitem(sys.modules, "theano.tensor", fake_tensor)


def test_tbnn_import_restores_numpy_bool(monkeypatch):
    """Importing :mod:`tbnn` should recreate legacy NumPy aliases."""

    _install_stubbed_dependencies(monkeypatch)
    monkeypatch.delattr(np, "bool", raising=False)
    assert not hasattr(np, "bool")

    for module_name in ["tbnn", "tbnn._compat"]:
        if module_name in sys.modules:
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    tbnn = importlib.import_module("tbnn")
    importlib.reload(tbnn)

    assert hasattr(np, "bool")
    assert getattr(np, "bool") in (bool, np.bool_)
