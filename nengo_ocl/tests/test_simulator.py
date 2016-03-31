import numpy as np

import pyopencl as cl

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
