
import numpy as np
import math
import inspect

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
queue = cl.CommandQueue(ctx)

def OclSimulator(model):
    return sim_ocl.Simulator(model, ctx)

class _ArgGen(object):
    def __init__(self, low=-10, high=10, integer=False):
        self.low = low
        self.high = high
        self.integer = integer

    def gen(self, n, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        if self.integer:
            return rng.randint(low=self.low, high=self.high, size=n)
        else:
            return rng.uniform(low=self.low, high=self.high, size=n)


class TestAstConversion(unittest.TestCase):
    def _test_fn(self, fn, in_dims, arggens=None, in_nengo=True):
        """Test an arbitrary function"""

        if arggens is None or isinstance(arggens, list) and len(arggens) == 0:
            arggens = [_ArgGen() for i in xrange(in_dims)]
        elif not isinstance(arggens, list):
            arggens = [arggens]
        else:
            assert len(arggens) == in_dims

        seed = sum(map(ord, fn.__name__)) % 2**30
        rng = np.random.RandomState(seed)

        FunctionObject = lambda: None
        FunctionObject.fn = fn

        n = 20
        x = zip(*[arggen.gen(n, rng=rng) for arggen in arggens])
        y = map(fn, x)
        out_dims = np.asarray(y[0]).size

        if in_nengo:
            out = self._test_in_nengo(fn, x, out_dims)
        else:
            out = self._test_directly(fn, x, out_dims)

        assert len(y) == len(out), "INTERNAL: len of ref and sim must match"
        for xx, yy, oo in zip(x, y, out):
            ## use slightly loose tols since OCL uses singles
            assert np.allclose(oo, yy, rtol=1e-4, atol=1e-4), (
                "results outside of tolerances")

    def _test_in_nengo(self, fn, x, out_dims):
        FunctionObject = lambda: None
        FunctionObject.fn = fn

        model = nengo.Model(fn.__name__)

        output_signals = []
        for i, xx in enumerate(x):
            s = model.add(Signal(n=out_dims, name="output_%d" % i))
            model._operators.append(nengo.simulator.SimDirect(
                    s, nengo.core.Constant(xx, name="input_%d" % i), FunctionObject))
            output_signals.append(s)

        sim = model.simulator(sim_class=OclSimulator)
        sim.step()

        out = []
        for s in output_signals:
            out.append(sim.signals[sim.copied(s)])

        return out

    # def _test_directly(self, fn, x, out_dims):
    #     source = ast_conversion.strip_leading_whitespace(inspect.getsource(fn))
    #     globals_dict = fn.func_globals
    #     closure_dict = (
    #         dict(zip(fn.func_code.co_freevars,
    #                  [c.cell_contents for c in fn.func_closure]))
    #         if fn.func_closure is not None else {})

    #     ot = ast_conversion.OCL_Translator(source, globals_dict, closure_dict)

    #     on = ast_conversion.OUTPUT_NAME;
    #     programCode = "__kernel void dummy(__global float* x, __global float* " + on + ") \n" \
    #         + "{\n    " \
    #         + "\n".join(ot.body) \
    #         + "\n}"

    #     program = cl.Program(ctx, programCode).build()

    #     mf = cl.mem_flags

    #     out = []
    #     for i, xx in enumerate(x):
    #         a = np.array(xx, dtype=np.float32)
    #         retval = np.empty(out_dims, dtype=np.float32)
    #         a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    #         dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, retval.nbytes)
    #         program.dummy(queue, a.shape, None, a_buf, dest_buf) # "dummy" is defined in getCLPRogram
    #         cl.enqueue_read_buffer(queue, dest_buf, retval).wait()
    #         out.append(retval)

    #     return out

    def test_raw(self):
        """Test a raw function"""
        self._test_fn(np.sin, 1)

    def test_closures(self):
        """Test a function defined using closure variables"""

        mult = 1.23
        power = 3.2
        def func(x):
            return mult * x[0]**power

        self._test_fn(func, 1, _ArgGen(0))

    def test_product(self):
        def product(x):
            return x[0] * x[1]

        self._test_fn(product, 2)

    def test_function_maps(self):
        """Test the function maps in ast_converter.py"""

        AG = _ArgGen
        A = AG()
        Aint = AG(integer=True)
        arggens = {
            math.acos: AG(-1, 1),
            math.acosh: AG(1, 10),
            math.atanh: AG(-1, 1),
            math.asin: AG(-1, 1),
            # math.fmod: AG(0), # this one really just doesn't like 0
            math.gamma: AG(0),
            math.lgamma: AG(0),
            math.log: AG(0),
            math.log10: AG(0),
            math.log1p: AG(-1),
            math.pow: [A, Aint],
            math.sqrt: AG(0),
            np.arccos: AG(-1, 1),
            np.arcsin: AG(-1, 1),
            np.arccosh: AG(1, 10),
            np.arctanh: AG(-1, 1),
            np.log: AG(0),
            np.log10: AG(0),
            np.log1p: AG(-1, 10),
            np.log2: AG(0),
            np.sqrt: AG(0),
            ### two-argument functions
            math.ldexp: [A, Aint],
            np.ldexp: [A, Aint],
            np.power: [AG(0), A],
        }

        dfuncs = ast_conversion.direct_funcs
        ifuncs = ast_conversion.indirect_funcs
        functions = (dfuncs.keys() + ifuncs.keys())
        all_passed = True
        for fn in functions:
            try:
                if fn in ast_conversion.direct_funcs:
                    def wrapper(x):
                        return fn(x[0])
                    self._test_fn(wrapper, 1, arggens.get(fn, None))
                else:
                    ### get lambda function
                    lambda_fn = ifuncs[fn]
                    while lambda_fn.__name__ != '<lambda>':
                        lambda_fn = ifuncs[lambda_fn]
                    dims = lambda_fn.func_code.co_argcount

                    if dims == 1:
                        def wrapper(x):
                            return fn(x[0])
                    elif dims == 2:
                        def wrapper(x):
                            return fn(x[0], x[1])
                    else:
                        raise ValueError(
                            "Cannot test functions with more than 2 arguments")
                    self._test_fn(wrapper, dims, arggens.get(fn, None))
                logger.info("Function `%s` passed" % fn.__name__)
            except Exception as e:
                all_passed = False
                logger.warning("Function `%s` failed with:\n    %s: %s"
                               % (fn.__name__, e.__class__.__name__, e.message))

        self.assertTrue(all_passed, "Some functions failed, "
                        "see logger warnings for details")

    def test_lambda(self):
        # Test various ways of using lambda functions

        a = lambda x: x[0]**2
        self._test_fn(a, 1)

        class Foo:
            def __init__(self, my_fn):
                self.fn = my_fn

        F = Foo(my_fn=lambda x: x[0]**2)
        b = F.fn
        self._test_fn(b, 1)

        def bar(fn):
            return fn

        c = bar(lambda x: x[0]**2)
        self._test_fn(c, 1)

        def egg(fn1, fn2):
            return fn1

        # this shouldn't convert to OCL b/c it has two lambdas on one line
        d = egg(lambda x: x[0]**2, lambda y: y[0]**3)
        self._test_fn(d, 1) # this should pass because a warning
                            # is issued and fn kept in python
        try:
            of = ast_conversion.OCL_Function(d)
            of.translator()
            assert False, ("This should fail because we don't support conversion"
                           "to OCL with multiple lambda functions in a source line")
        except NotImplementedError:
            pass
