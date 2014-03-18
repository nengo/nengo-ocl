import logging
import math
import numpy as np
import pyopencl as cl
import pytest

import nengo
import nengo_ocl


logger = logging.getLogger(__name__)
ctx = cl.create_some_context()


def OclSimulator(*args, **kwargs):
    return nengo_ocl.sim_ocl.Simulator(
        *args, context=ctx, ocl_only=True, **kwargs)


def pytest_funcarg__Simulator(request):
    return OclSimulator


def _test_fn(Simulator, fn, size_in=0):
    seed = sum(map(ord, fn.__name__)) % 2**30
    rng = np.random.RandomState(seed + 1)

    # make input
    x = rng.uniform(size=size_in)

    # make model
    model = nengo.Model("test_%s" % fn.__name__, seed=seed)
    with model:
        v = nengo.Node(output=fn, size_in=size_in)
        vp = nengo.Probe(v)

        if size_in > 0:
            u = nengo.Node(output=x)
            nengo.Connection(u, v, filter=0)

    # run model
    sim = Simulator(model)
    sim.run(0.005)

    # compare output
    t = sim.trange()
    y = np.array([fn(tt, x) if size_in > 0 else fn(tt) for tt in t])
    z = sim.data(vp)

    y.shape = z.shape
    assert np.allclose(z[1:], y[1:])
    # assert np.allclose(z[1:], y[:-1])


def test_t(Simulator):
    _test_fn(Simulator, lambda t: t)


@pytest.mark.parametrize("size_in", [1, 3, 5])
def test_identity(Simulator, size_in):
    _test_fn(Simulator, lambda t, x: x, size_in=1)


def test_raw(Simulator):
    _test_fn(Simulator, np.sin)


@pytest.mark.parametrize("size_in", [1, 3])
def test_closures(Simulator, size_in):
    """Test a function defined using closure variables"""

    mult = 1.23
    power = 3.2
    def closures(t, x):
        return mult * x**power

    _test_fn(Simulator, closures, size_in=size_in)


def test_product(Simulator):
    def product(t, x):
        return x[0] * x[1]

    _test_fn(Simulator, product, size_in=2)


def test_lambda_simple(Simulator):
    fn = lambda t, x: x[0]**2
    _test_fn(Simulator, fn, size_in=1)


def test_lambda_class(Simulator):
    class Foo:
        def __init__(self, my_fn):
            self.fn = my_fn

    F = Foo(my_fn=lambda t, x: x[0]**2)
    b = F.fn
    _test_fn(Simulator, b, size_in=1)


def test_lambda_wrapper(Simulator):
    def bar(fn):
        return fn

    c = bar(lambda t, x: x[0]**2)
    _test_fn(Simulator, c, size_in=1)


def test_lambda_double(Simulator):
    def egg(fn1, fn2):
        return fn1

    # this shouldn't convert to OCL b/c it has two lambdas on one line
    d = egg(lambda t, x: x[0]**2, lambda t, y: y[0]**3)
    with pytest.raises(NotImplementedError):
        _test_fn(Simulator, d, size_in=1)


# def test_functions(self):
#     """Test the function maps in ast_converter.py"""

#     AG = _ArgGen
#     A = AG()
#     Aint = AG(integer=True)
#     arggens = {
#         math.acos: AG(-1, 1),
#         math.acosh: AG(1, 10),
#         math.atanh: AG(-1, 1),
#         math.asin: AG(-1, 1),
#         # math.fmod: AG(0), # this one really just doesn't like 0
#         math.gamma: AG(0),
#         math.lgamma: AG(0),
#         math.log: AG(0),
#         math.log10: AG(0),
#         math.log1p: AG(-1),
#         math.pow: [A, Aint],
#         math.sqrt: AG(0),
#         np.arccos: AG(-1, 1),
#         np.arcsin: AG(-1, 1),
#         np.arccosh: AG(1, 10),
#         np.arctanh: AG(-1, 1),
#         np.log: AG(0),
#         np.log10: AG(0),
#         np.log1p: AG(-1, 10),
#         np.log2: AG(0),
#         np.sqrt: AG(0),
#         ### two-argument functions
#         math.ldexp: [A, Aint],
#         np.ldexp: [A, Aint],
#         np.power: [AG(0), A],
#     }

#     dfuncs = ast_conversion.direct_funcs
#     ifuncs = ast_conversion.indirect_funcs
#     functions = (dfuncs.keys() + ifuncs.keys())
#     all_passed = True
#     for fn in functions:
#         try:
#             if fn in ast_conversion.direct_funcs:
#                 def wrapper(x):
#                     return fn(x[0])
#                 self._test_fn(wrapper, 1, arggens.get(fn, None))
#             else:
#                 ### get lambda function
#                 lambda_fn = ifuncs[fn]
#                 while lambda_fn.__name__ != '<lambda>':
#                     lambda_fn = ifuncs[lambda_fn]
#                 dims = lambda_fn.func_code.co_argcount

#                 if dims == 1:
#                     def wrapper(x):
#                         return fn(x[0])
#                 elif dims == 2:
#                     def wrapper(x):
#                         return fn(x[0], x[1])
#                 else:
#                     raise ValueError(
#                         "Cannot test functions with more than 2 arguments")
#                 self._test_fn(wrapper, dims, arggens.get(fn, None))
#             logger.info("Function `%s` passed" % fn.__name__)
#         except Exception as e:
#             all_passed = False
#             logger.warning("Function `%s` failed with:\n    %s: %s"
#                            % (fn.__name__, e.__class__.__name__, e.message))

#     self.assertTrue(all_passed, "Some functions failed, "
#                     "see logger warnings for details")


# def test_vector_functions(self):
#     d = 5
#     boolean = [any, all, np.any, np.all]
#     funcs = ast_conversion.vector_funcs.keys()
#     all_passed = True
#     for fn in funcs:
#         try:
#             if fn in boolean:
#                 def wrapper(x):
#                     return fn(np.asarray(x) > 0)
#             else:
#                 def wrapper(x):
#                     return fn(x)

#             self._test_fn(wrapper, d)

#             logger.info("Function `%s` passed" % fn.__name__)
#         except Exception as e:
#             all_passed = False
#             logger.warning("Function `%s` failed with:\n    %s: %s"
#                            % (fn.__name__, e.__class__.__name__, e.message))

#     self.assertTrue(all_passed, "Some functions failed, "
#                     "see logger warnings for details")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
