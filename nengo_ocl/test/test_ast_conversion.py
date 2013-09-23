
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
import logging

ctx = cl.create_some_context()
logger = logging.getLogger(__name__)

def OclSimulator(model):
    return sim_ocl.Simulator(model, ctx)

class TestAstConversion(unittest.TestCase):
    def _test_fn(self, fn, in_dims, low=-10, high=10):
        """Test an arbitrary function"""

        seed = sum(map(ord, fn.__name__)) % 2**30
        rng = np.random.RandomState(seed)

        FunctionObject = lambda: None
        FunctionObject.fn = fn

        # print "running numpy"
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

        # print "running simulator"
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

    def test_all_functions(self):
        import math
        dfuncs = ast_conversion.direct_funcs
        ifuncs = ast_conversion.indirect_funcs
        # functions = (dfuncs.keys() + ifuncs.keys())
        functions = (ifuncs.keys())
        # functions = [math.atan2]
        # functions = [math.erfc]
        all_passed = True
        for fn in functions:
            try:
                if fn in ast_conversion.direct_funcs:
                    self._test_fn(fn, 1)
                else:
                    ### get lambda function
                    lambda_fn = ifuncs[fn]
                    while lambda_fn.__name__ != '<lambda>':
                        lambda_fn = ifuncs[lambda_fn]
                    dims = lambda_fn.func_code.co_argcount

                    if dims == 1:
                        wrapper = fn
                    elif dims == 2:
                        def wrapper(x):
                            return fn(x[0], x[1])
                    else:
                        raise ValueError(
                            "Cannot test functions with more than 2 arguments")
                    self._test_fn(wrapper, dims)
                logger.info("Function `%s` passed" % fn.__name__)
            # except None:
            #     pass
            except Exception as e:
                all_passed = False
                logger.warning("Function `%s` failed with message \"%s\""
                               % (fn.__name__, e.message))

        self.assertTrue(all_passed, "Some functions failed, "
                        "see logger warnings for details")
