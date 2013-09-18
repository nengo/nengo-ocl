
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

def OclSimulator(model):
    return sim_ocl.Simulator(model, ctx)

class TestAstConversion(unittest.TestCase):
    def _test_fn(self, fn, in_dims, low=-10, high=10):
        """Test an arbitrary function"""

        seed = sum(map(ord, fn.__name__)) % 2**30
        rng = np.random.RandomState(seed)

        FunctionObject = lambda: None
        FunctionObject.fn = fn

        n = 200
        x = rng.uniform(low=low, high=high, size=(n, in_dims))
        y = map(fn, x)
        out_dims = np.asarray(y[0]).size

        model = nengo.Model(fn.__name__)

        output_signals = []
        for i, xx in enumerate(x):
            s = model.add(Signal(n=out_dims, name="output_%d" % i))
            model._operators.append(nengo.simulator.SimDirect(
                    s, nengo.core.Constant(xx, name="input_%d" % i), FunctionObject))
            output_signals.append(s)

        sim = model.simulator(sim_class=OclSimulator)
        sim.step()

        for s, yy in zip(output_signals, y):
            out = sim.signals[sim.copied(s)]
            ### use slightly loose tols since OCL uses singles
            assert np.allclose(out, yy, rtol=1e-5, atol=1e-6)

    def test_raw(self):
        """Test a raw (Numpy) function"""
        self._test_fn(np.sin, 1)

    def test_closures(self):
        """Test a function defined using closure variables"""

        mult = 1.23
        power = 3.2
        def func(x):
            return mult * x**power

        self._test_fn(func, 1, low=0)

    def test_product(self):
        def product(x):
            return x[0] * x[1]

        self._test_fn(product, 2)
