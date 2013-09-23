
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

class TestAstConversion(unittest.TestCase):
    def _test_fn(self, fn, in_dims, low=-10, high=10, in_nengo=True, int_inds=[]):
        """Test an arbitrary function"""

        seed = sum(map(ord, fn.__name__)) % 2**30
        rng = np.random.RandomState(seed)

        FunctionObject = lambda: None
        FunctionObject.fn = fn

        # print "running numpy"
        n = 200
        x = rng.uniform(low=low, high=high, size=(n, in_dims))

        # some functions only accept integers for some inputs
        for i in int_inds:
            print("integer index: " + str(i))
            x[:,i] = np.round(x[:,i])

        y = map(fn, x)
        out_dims = np.asarray(y[0]).size

        if in_nengo:
            out = self._test_in_nengo(fn, x, out_dims)
        else:
            out = self._test_directly(fn, x, out_dims)

        for yy, oo in zip(y, out):
            ## use slightly loose tols since OCL uses singles
            assert np.allclose(oo, yy, rtol=1e-4, atol=1e-4), str(fn)

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

        # print "running simulator"
        sim = model.simulator(sim_class=OclSimulator)
        sim.step()

        out = []
        for s in output_signals:
            out.append(sim.signals[sim.copied(s)])

        return out

    def _test_directly(self, fn, x, out_dims):
        source = ast_conversion.strip_leading_whitespace(inspect.getsource(fn))
        globals_dict = fn.func_globals
        closure_dict = (
            dict(zip(fn.func_code.co_freevars,
                     [c.cell_contents for c in fn.func_closure]))
            if fn.func_closure is not None else {})

        ot = ast_conversion.OCL_Translator(source, globals_dict, closure_dict)

        on = ast_conversion.OUTPUT_NAME;
        programCode = "__kernel void dummy(__global float* x, __global float* " + on + ") \n" \
            + "{\n    " \
            + "\n".join(ot.body) \
            + "\n}"

        program = cl.Program(ctx, programCode).build()

        mf = cl.mem_flags

        out = []
        for i, xx in enumerate(x):
            a = np.array(xx, dtype=np.float32)
            retval = np.empty(out_dims, dtype=np.float32)
            a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
            dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, retval.nbytes)
            program.dummy(queue, a.shape, None, a_buf, dest_buf) # "dummy" is defined in getCLPRogram
            cl.enqueue_read_buffer(queue, dest_buf, retval).wait()
            out.append(retval)

        return out

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

    def test_lambda(self):
        # Test various ways of using lambda functions

        a = lambda x: x**2
        self._test_fn(a, 1, low=-10, high=10)

        class Foo:
            def __init__(self, fn):
                self.fn = fn

        F = Foo(lambda x: x**2)
        b = F.fn
        self._test_fn(b, 1, low=-10, high=10)

        def bar(fn):
            return fn

        c = bar(lambda x: x**2)
        self._test_fn(c, 1, low=-10, high=10)

        def egg(fn1, fn2):
            return fn1


        # this shouldn't convert to OCL
        d = egg(lambda x: x**2, lambda y: y**3)
        self._test_fn(d, 1) # this should pass because a warning is issued and fn kept in python

        try: # this
            of = ast_conversion.OCL_Function(d)
            of.translator()
            assert False, "This should fail because we don't support conversion to OCL with multiple lambda functions in a source line"
        except NotImplementedError:
            pass

    def test_function_map(self):

        two_arg_functions = {math.ldexp, math.pow, math.copysign, math.atan2, math.fmod, math.hypot}

        positive_range = [1e-10, 10]
        custom_ranges = {
            math.gamma: positive_range,
            math.sqrt: positive_range,
            math.log: positive_range,
            math.log10: positive_range,
            math.log1p: [-.99, 10],
            math.acos: [-1, 1],
            math.acosh: [1, 10],
            math.atanh: [-1+1e-10, 1-1e-10],
            math.asin: [-1, 1],
            math.fmod: positive_range, # this one really just doesn't like 0
            math.gamma: positive_range,
            math.lgamma: positive_range,
            np.arccos: [-1, 1],
            np.arcsin: [-1, 1],
            np.arccosh: [1, 10],
            np.arctanh: [-1+1e-10, 1-1e-10],
            np.log: positive_range,
            np.log10: positive_range,
            np.log1p: [-.99, 10],
            np.log2: positive_range,
            np.sqrt: positive_range,

        }

        #TODO: this makes _test_fn round input values, which isn't useful very often -- figure out a good way to deal with integer args
        integer_arg_inds = {
            math.pow: [1]}

        for pyfun, mapping in ast_conversion.function_map.items():

            int_inds = integer_arg_inds.get(pyfun, [])

            num_args = 1 #default
            if isinstance(pyfun, np.ufunc):
                num_args = pyfun.nargs - 1
            else:
                if pyfun in two_arg_functions:
                    num_args = 2

            if num_args == 1:
                def fn(x):
                    return pyfun(x)
            else:
                def fn(x):
                    return pyfun(x[0], x[1])

            r = custom_ranges.get(pyfun, [-10, 10])
            in_nengo = pyfun not in custom_ranges

            self._test_fn(fn, num_args, low=r[0], high=r[1], in_nengo=in_nengo, int_inds=int_inds)
