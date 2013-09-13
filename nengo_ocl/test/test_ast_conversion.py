
import numpy as np

import nengo
import nengo.core
import nengo.simulator
from nengo.core import Signal

import nengo_ocl
from nengo_ocl.tricky_imports import unittest
from nengo_ocl import sim_ocl
from nengo_ocl import ast_conversion

import pyopencl as cl
ctx = cl.create_some_context()

def OclSimulator(*args, **kwargs):
    rval = sim_ocl.Simulator(ctx, *args, **kwargs)
    return rval

class TestAstConversion(unittest.TestCase):
    def test_product(self):

        @ast_conversion.OCL_Function
        def product(x):
            return x[0] * x[1]

        FunctionObject = lambda: None
        FunctionObject.fn = product

        n = 100
        x = np.random.randn(n, 2)
        y = map(product, x)

        model = nengo.Model("Product")

        output_signals = []
        for i, xx in enumerate(x):
            s = model.add(Signal(n=1, name="output_%d" % i))
            model._operators.append(nengo.simulator.SimDirect(
                    s, nengo.core.Constant(xx, name="input_%d" % i), FunctionObject))
            output_signals.append(s)

        sim = model.simulator(sim_class=OclSimulator)
        sim.step()

        for s, yy in zip(output_signals, y):
            out = sim.signals[sim.copied(s)]
            print yy
            print out
            assert np.allclose(out, yy, rtol=1e-5, atol=1e-8)

