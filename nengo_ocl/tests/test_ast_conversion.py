import math
import numpy as np
import pytest

import nengo
from nengo.dists import Uniform

import nengo_ocl
import nengo_ocl.ast_conversion as ast_conversion
from nengo_ocl.ast_conversion import OCL_Function


@pytest.fixture(scope="session")
def OclOnlySimulator(request, ctx):
    """A nengo_ocl.Simulator that only allows OCL python functions"""

    def OclOnlySimulator(*args, **kwargs):
        return nengo_ocl.Simulator(*args, context=ctx, if_python_code="error", **kwargs)

    return OclOnlySimulator


def test_slice():
    s = slice(1, None)

    def func(x):
        return x[s]

    ocl_fn = OCL_Function(func, in_dims=(3,))
    assert ocl_fn.code  # assert that we can make the code (no exceptions)


@pytest.mark.xfail  # see https://github.com/nengo/nengo_ocl/issues/54
def test_nested():
    f = lambda x: x ** 2

    def func(x):
        return f(x)

    ocl_fn = OCL_Function(func, in_dims=(3,))
    print(ocl_fn.init)
    print(ocl_fn.code)


def _test_node(OclOnlySimulator, fn, size_in=0):
    seed = sum(map(ord, fn.__name__)) % 2 ** 30
    rng = np.random.RandomState(seed + 1)

    # make input
    x = rng.uniform(size=size_in)

    # make model
    model = nengo.Network("test_%s" % fn.__name__, seed=seed)
    with model:
        v = nengo.Node(output=fn, size_in=size_in)
        vp = nengo.Probe(v)

        if size_in > 0:
            u = nengo.Node(output=x)
            nengo.Connection(u, v, synapse=0)

    # run model
    with OclOnlySimulator(model) as sim:
        sim.run(0.005)

    # compare output
    t = sim.trange()
    y = np.array([fn(tt, x) if size_in > 0 else fn(tt) for tt in t])
    z = sim.data[vp]

    y.shape = z.shape
    assert np.allclose(z[1:], y[1:])
    # assert np.allclose(z[1:], y[:-1])


def test_t(OclOnlySimulator):
    _test_node(OclOnlySimulator, lambda t: t)


@pytest.mark.parametrize("size_in", [1, 3, 5])
def test_identity(OclOnlySimulator, size_in):
    _test_node(OclOnlySimulator, lambda t, x: x, size_in=size_in)


def test_raw(OclOnlySimulator):
    _test_node(OclOnlySimulator, np.sin)


@pytest.mark.parametrize("size_in", [1, 3])
def test_closures(OclOnlySimulator, size_in):
    """Test a function defined using closure variables"""

    mult = 1.23
    power = 3.2

    def closures(t, x):
        return mult * x ** power

    _test_node(OclOnlySimulator, closures, size_in=size_in)


def test_product(OclOnlySimulator):
    def product(t, x):
        return x[0] * x[1]

    _test_node(OclOnlySimulator, product, size_in=2)


def test_lambda_simple(OclOnlySimulator):
    fn = lambda t, x: x[0] ** 2
    _test_node(OclOnlySimulator, fn, size_in=1)


def test_lambda_class(OclOnlySimulator):
    class Foo:
        def __init__(self, my_fn):
            self.fn = my_fn

    F = Foo(my_fn=lambda t, x: x[0] ** 2)
    b = F.fn
    _test_node(OclOnlySimulator, b, size_in=1)


def test_lambda_wrapper(OclOnlySimulator):
    def bar(fn):
        return fn

    c = bar(lambda t, x: x[0] ** 2)
    _test_node(OclOnlySimulator, c, size_in=1)


def test_lambda_double(OclOnlySimulator):
    def egg(fn1, fn2):
        return fn1

    # this shouldn't convert to OCL b/c it has two lambdas on one line
    d = egg(lambda t, x: x[0] ** 2, lambda t, y: y[0] ** 3)
    with pytest.raises(NotImplementedError):
        _test_node(OclOnlySimulator, d, size_in=1)


@pytest.mark.parametrize("synapse", [None, nengo.Lowpass(tau=0.005)])
def test_direct_connection(OclOnlySimulator, synapse):
    """Test a direct-mode connection"""

    model = nengo.Network("test_connection", seed=124)
    with model:
        a = nengo.Ensemble(1, dimensions=1, neuron_type=nengo.Direct())
        b = nengo.Ensemble(1, dimensions=1, neuron_type=nengo.Direct())
        nengo.Connection(a, b, function=lambda x: x ** 2, synapse=synapse)

    OclOnlySimulator(model)


def _test_conn(OclOnlySimulator, fn, size_in, dist_in=None, n=1):
    seed = sum(map(ord, fn.__name__)) % 2 ** 30
    rng = np.random.RandomState(seed + 1)

    # make input
    if dist_in is None:
        dist_in = [Uniform(-10, 10) for i in range(size_in)]
    elif not isinstance(dist_in, (list, tuple)):
        dist_in = [dist_in]
    assert len(dist_in) == size_in

    x = list(zip(*[d.sample(n, rng=rng) for d in dist_in]))  # preserve types
    # x = zip(*[arggen.gen(n, rng=rng) for arggen in arggens])
    y = [fn(xx) for xx in x]

    y = np.array([fn(xx) for xx in x])
    if y.ndim < 2:
        y.shape = (n, 1)
    size_out = y.shape[1]

    # make model
    model = nengo.Network("test_%s" % fn.__name__, seed=seed)
    with model:
        probes = []
        for i in range(n):
            u = nengo.Node(output=x[i])
            v = nengo.Ensemble(1, dimensions=size_in, neuron_type=nengo.Direct())
            w = nengo.Ensemble(1, dimensions=size_out, neuron_type=nengo.Direct())
            nengo.Connection(u, v, synapse=None)
            nengo.Connection(v, w, synapse=None, function=fn, eval_points=x)
            probes.append(nengo.Probe(w))

    # run model
    with OclOnlySimulator(model) as sim:
        sim.step()
        # sim.step()
        # sim.step()

    # compare output
    z = np.array([sim.data[p][-1] for p in probes])
    assert np.allclose(z, y, atol=2e-7)


def test_sin_conn(OclOnlySimulator):
    _test_conn(OclOnlySimulator, np.sin, 1, n=10)


def test_functions(OclOnlySimulator, capsys, n_points=10):  # noqa: C901
    """Test the function maps in ast_converter.py"""
    # TODO: split this into one test per function using py.test utilities

    U = Uniform
    ignore = [math.ldexp, math.pow, np.ldexp]
    arggens = {
        math.acos: U(-1, 1),
        math.acosh: U(1, 10),
        math.atanh: U(-1, 1),
        math.asin: U(-1, 1),
        # math.fmod: U(0), # this one really just doesn't like 0
        math.gamma: U(0, 10),
        math.lgamma: U(0, 10),
        math.log: U(0, 10),
        math.log10: U(0, 10),
        math.log1p: U(-1, 10),
        math.sqrt: U(0, 10),
        np.arccos: U(-1, 1),
        np.arcsin: U(-1, 1),
        np.arccosh: U(1, 10),
        np.arctanh: U(-1, 1),
        np.log: U(0, 10),
        np.log10: U(0, 10),
        np.log1p: U(-1, 10),
        np.log2: U(0, 10),
        np.sqrt: U(0, 10),
        # two-argument functions
        # math.ldexp: [U(-10, 10), U(-10, 10, integer=True)],
        # math.pow: [U(-10, 10), U(-10, 10, integer=True)],
        # np.ldexp: [U(-10, 10), U(-10, 10, integer=True)],
        np.power: [U(0, 10), U(-10, 10)],
    }

    dfuncs = ast_conversion.direct_funcs
    ifuncs = ast_conversion.indirect_funcs
    functions = list(dfuncs.keys()) + list(ifuncs.keys())
    functions = [f for f in functions if f not in ignore]
    all_passed = True
    for fn in functions:
        try:
            if fn in ast_conversion.direct_funcs:

                def wrapper(x):
                    return fn(x[0])

                wrapper.__name__ = fn.__name__ + "_" + wrapper.__name__
                _test_conn(
                    OclOnlySimulator,
                    wrapper,
                    1,
                    dist_in=arggens.get(fn, None),
                    n=n_points,
                )
            else:
                # get lambda function
                lambda_fn = ifuncs[fn]
                while lambda_fn.__name__ != "<lambda>":
                    lambda_fn = ifuncs[lambda_fn]
                dims = lambda_fn.__code__.co_argcount

                if dims == 1:

                    def wrapper(x):
                        return fn(x[0])

                elif dims == 2:

                    def wrapper(x):
                        return fn(x[0], x[1])

                else:
                    raise ValueError("Cannot test functions with more than 2 arguments")
                _test_conn(
                    OclOnlySimulator,
                    wrapper,
                    dims,
                    dist_in=arggens.get(fn, None),
                    n=n_points,
                )
            print("Function `%s` passed" % fn.__name__)
        except Exception as e:
            all_passed = False
            with capsys.disabled():
                print(
                    "Function `%s` failed with:\n    %s%s"
                    % (fn.__name__, e.__class__.__name__, e.args)
                )

    assert all_passed, "Some functions failed, " "see logger warnings for details"


def test_vector_functions(OclOnlySimulator, capsys):
    d = 5
    boolean = [any, all, np.any, np.all]
    funcs = ast_conversion.vector_funcs.keys()
    all_passed = True
    for fn in funcs:
        try:
            if fn in boolean:

                def wrapper(x):
                    return fn(np.asarray(x) > 0)

            else:

                def wrapper(x):
                    return fn(x)

            _test_conn(OclOnlySimulator, wrapper, d, n=10)

            print("Function `%s` passed" % fn.__name__)
        except Exception as e:
            all_passed = False
            with capsys.disabled():
                print(
                    "Function `%s` failed with:\n    %s: %s"
                    % (fn.__name__, e.__class__.__name__, e)
                )

    assert all_passed, "Some functions failed, " "see logger warnings for details"
