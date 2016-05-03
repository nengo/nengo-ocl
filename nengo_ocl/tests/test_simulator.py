import numpy as np
import pyopencl as cl
import pytest

import nengo
from nengo.builder import Model
from nengo.builder.operator import DotInc, PreserveValue, Reset
from nengo.builder.signal import Signal

import nengo_ocl


ctx = cl.create_some_context()


def OclSimulator(*args, **kwargs):
    return nengo_ocl.Simulator(*args, context=ctx, **kwargs)


def test_multidotinc_compress():
    a = Signal([0, 0])
    b = Signal([0, 0])
    A = Signal([[1, 2], [0, 1]])
    B = Signal([[2, 1], [-1, 1]])
    x = Signal([1, 1])
    y = Signal([1, -1])

    m = Model(dt=0)
    m.operators += [Reset(a), DotInc(A, x, a), DotInc(B, y, a)]
    m.operators += [PreserveValue(b), DotInc(A, y, b), DotInc(B, x, b)]

    with OclSimulator(None, model=m) as sim:
        sim.step()
        assert np.allclose(sim.signals[a], [4, -1])
        assert np.allclose(sim.signals[b], [2, -1])
        sim.step()
        assert np.allclose(sim.signals[a], [4, -1])
        assert np.allclose(sim.signals[b], [4, -2])


def test_version(monkeypatch):
    from nengo_ocl.version import latest_nengo_version_info

    monkeypatch.setattr(nengo.version, 'version_info', (2, 0, 0))

    with nengo.Network() as model:
        nengo.Ensemble(10, 1)

    with pytest.raises(ValueError):
        OclSimulator(model)

    next_version = list(latest_nengo_version_info)
    next_version[-1] += 1
    monkeypatch.setattr(nengo.version, 'version_info', tuple(next_version))
    with pytest.warns(UserWarning):
        OclSimulator(model)
