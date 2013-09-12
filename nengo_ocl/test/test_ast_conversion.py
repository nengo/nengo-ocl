
import numpy as np

import nengo
from nengo.core import Signal, Direct

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

        n = 100
        x = np.random.randn(n, 2)
        y = map(product, x)

        model = nengo.Model("Product")

        directs = []
        for i in xrange(n):
            d = model.add(Direct(n_in=2, n_out=1, fn=product))
            directs.append(d)

        sim = model.simulator(sim_class=OclSimulator)
        for d, xx in zip(directs, x):
            sim.signals[sim.copied(d.input_signal)] = xx

        sim.step()

        for d, yy in zip(directs, y):
            out = sim.signals[sim.copied(d.output_signal)]
            assert np.allclose(out, yy, rtol=1e-5, atol=1e-8)


if __name__ == '__main__':
    test = TestAstConversion()
    test.test_product()
