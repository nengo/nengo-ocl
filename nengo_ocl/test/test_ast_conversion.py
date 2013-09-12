
import numpy as np

import nengo
from nengo.core import Signal, Direct, Encoder
# from nengo.tests.helpers import NengoTestLoader
# from nengo.tests.helpers import load_nengo_tests

# from .. import sim_npy
# from .. import sim_ocl

import nengo_ocl
from nengo_ocl.tricky_imports import unittest
from nengo_ocl import sim_ocl
from nengo_ocl import ast_conversion

import pyopencl as cl
ctx = cl.create_some_context()

def OclSimulator(*args, **kwargs):
    rval = sim_ocl.Simulator(ctx, *args, **kwargs)
    rval.plan_all()
    return rval

class TestAstConversion(unittest.TestCase):

    def test_product(self):
        n = 100
        x = np.random.randn(n, 2)

        model = nengo.Model("Product")

        @ast_conversion.OCL_Function
        def product(x):
            return x[0] * x[1]

        signals = []
        directs = []
        for i in xrange(n):
            s = model.add(Signal(n=2, name='s%d' % i))
            d = model.add(Direct(n_in=2, n_out=1, fn=product))
            e = model.add(Encoder(s, d, np.eye(2)))

            signals.append(s)
            directs.append(d)

        sim = model.simulator(sim_class=OclSimulator)
        for xx, s in zip(x, signals):
            sim.signals[sim.copied(s)] = xx
        sim.step()

        for xx, d in zip(x, directs):
            assert np.allclose(xx, sim.signals[sim.copied(d.output_signal)])
