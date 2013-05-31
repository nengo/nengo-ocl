import numpy as np
from base import Model
import sim_ocl
import pyopencl as cl

import test_sim_npy
import test_matmul

ctx = cl.create_some_context()

def test_probe_with_base(show=False):
    def Simulator(*args, **kwargs):
        return sim_ocl.Simulator(ctx, *args, **kwargs)
    return test_sim_npy.test_probe_with_base(show=show, Simulator=Simulator)


def test_matrix_mult(show=False):
    def Simulator(*args, **kwargs):
        return sim_ocl.Simulator(ctx, *args, **kwargs)
    return test_matmul.test_matrix_mult_example(
            D1=5, D2=5, D3=5, N=200,
            show=show, Simulator=Simulator)


def test_filter():
    dt = .001
    m = Model(dt)
    one = m.signal(value=1.0)
    steps = m.signal()
    simtime = m.signal()

    # -- hold all constants on the line
    m.filter(1.0, one, one)

    # -- steps counts by 1.0
    m.filter(1.0, steps, steps)
    m.filter(1.0, one, steps)

    # simtime <- dt * steps
    m.filter(dt, steps, simtime)

    sim = sim_ocl.Simulator(ctx, m, n_prealloc_probes=1000)
    sim.alloc_signals()
    sim.alloc_filters()
    pf = sim.plan_filters()
    pc = sim.plan_copy_sigs()

    for i in range(10):
        pc()
        pf()

    assert np.allclose(sim.sigs.buf.get(),
            [1.0, 10.0, .009])

