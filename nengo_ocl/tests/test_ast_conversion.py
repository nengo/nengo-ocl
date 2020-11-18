"""Test ast_conversion.py"""

# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=redefined-outer-name,broad-except

import math

import nengo
import numpy as np
import pytest
from nengo.dists import Uniform

import nengo_ocl
import nengo_ocl.ast_conversion as ast_conversion
from nengo_ocl.ast_conversion import OclFunction


@pytest.fixture(scope="session")
def OclOnlySimulator(request, ctx):
    """A nengo_ocl.Simulator that only allows OCL python functions"""

    def ocl_only_sim(*args, **kwargs):
        return nengo_ocl.Simulator(*args, context=ctx, if_python_code="error", **kwargs)

    return ocl_only_sim


def test_slice():
    s = slice(1, None)

    def func(x):
        return x[s]

    ocl_fn = OclFunction(func, in_dims=(3,))
    assert ocl_fn.code  # assert that we can make the code (no exceptions)


@pytest.mark.xfail  # see https://github.com/nengo/nengo_ocl/issues/54
def test_nested():
    f = lambda x: x ** 2

    def func(x):
        return f(x)

    ocl_fn = OclFunction(func, in_dims=(3,))
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
                    return fn(x[0])  # pylint: disable=cell-var-from-loop

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
                        return fn(x[0])  # pylint: disable=cell-var-from-loop

                elif dims == 2:

                    def wrapper(x):
                        return fn(x[0], x[1])  # pylint: disable=cell-var-from-loop

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
                    # pylint: disable=cell-var-from-loop
                    return fn(np.asarray(x) > 0)

            else:

                def wrapper(x):
                    return fn(x)  # pylint: disable=cell-var-from-loop

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


def print_various_ocl():  # noqa: C901
    # pylint: disable=chained-comparison
    # pylint: disable=comparison-with-itself
    # pylint: disable=function-redefined
    # pylint: disable=unnecessary-lambda
    # pylint: disable=using-constant-test

    def ocl_f(*args, **kwargs):
        ocl_fn = OclFunction(*args, **kwargs)
        print(ocl_fn.init)
        print(ocl_fn.code)
        print("")
        return ocl_fn

    print("*" * 5 + "Raw" + "*" * 50)
    ocl_f(np.sin, in_dims=(1,))

    print("*" * 5 + "Multi sine" + "*" * 50)
    ocl_f(np.sin, in_dims=(3,))

    print("*" * 5 + "List-return" + "*" * 50)

    def func(t):
        return [1, 2, 3]

    ocl_f(func, in_dims=(1,))

    print("*" * 5 + "Multi-arg" + "*" * 50)

    def func(t, x):
        return t + x[:2] + x[2]

    ocl_f(func, in_dims=(1, 3))

    print("*" * 5 + "Simplify" + "*" * 50)

    def func(y):
        return y + np.sin([1, 2, 3])

    ocl_f(func, in_dims=(1,))

    multiplier = 3842.012

    def square(x):
        # pylint: disable=undefined-variable

        # print("wow: %f, %d, %s" % (x[0], 9, "hello"))

        if 1 + (2 == 2):
            y = 2.0 * x
            z -= 4 + (3 if x > 99 else 2)  # noqa: F821
        elif x == 2:
            y *= 9.12 if 3 > 4 else 0
            z = 4 * (x - 2)
        else:
            y = 9 * x
            z += x ** (-1.1)

        return np.sin(multiplier * (y * z) + np.square(y))

    ocl_f(square, in_dims=1)

    print("*" * 5 + "Vector lambda" + "*" * 50)
    insert = -0.5
    func = lambda x: x + 3 if all(x > 2) else x - 1
    ocl_f(func, in_dims=3)

    if 0:
        print("*" * 5 + "Large input" + "*" * 50)
        insert = -0.5
        func = lambda x: [x[1] * x[1051], x[3] * x[62]]
        ocl_f(func, in_dims=1100)

    print("*" * 5 + "List comprehension" + "*" * 50)
    insert = -0.5
    func = lambda x: [np.maximum(0.1, np.sin(2)) * x[4 - i] for i in range(5)]
    ocl_f(func, in_dims=5)

    print("*" * 5 + "Unary minus" + "*" * 50)
    insert = -0.5

    def function(x):
        return x * -insert

    ocl_f(function, in_dims=1)

    print("*" * 5 + "Subtract" + "*" * 50)

    def function(x):
        return np.subtract(x[1], x[0])

    ocl_f(function, in_dims=2)

    print("*" * 5 + "List" + "*" * 50)

    def function(y):
        z = y[0] * y[1]
        return [y[1], z]

    ocl_f(function, in_dims=2)

    print("*" * 5 + "Array" + "*" * 50)
    value = np.arange(3)

    def function(y):
        return value

    ocl_f(function, in_dims=value.size)

    print("*" * 5 + "AsArray" + "*" * 50)

    def function(y):
        return np.asarray([y[0], y[1], 3])

    ocl_f(function, in_dims=2)

    print("*" * 5 + "IfExp" + "*" * 50)

    def function(y):
        return 5 if y > 3 else 0

    ocl_f(function, in_dims=1)

    print("*" * 5 + "Sign" + "*" * 50)

    def function(y):
        return np.sign(y)

    ocl_f(function, in_dims=1)

    print("*" * 5 + "Radians" + "*" * 50)
    power = 2

    def function(y):
        return np.radians(y ** power)

    ocl_f(function, in_dims=1)

    print("*" * 5 + "Boolop" + "*" * 50)
    power = 3.2

    def function(y):
        if y > 3 and y < 5:
            return y ** power
        else:
            return np.sign(y)

    ocl_f(function, in_dims=1)

    print("*" * 5 + "Nested return" + "*" * 50)
    power = 3.2

    def function(y):
        if y > 3 and y < 5:
            return y ** power

        return np.sign(y)

    ocl_f(function, in_dims=1)

    print("*" * 5 + "Math constants" + "*" * 50)

    def function(y):
        return np.sin(np.pi * y) + np.e

    ocl_f(function, in_dims=1)

    print("*" * 5 + "Vector functions" + "*" * 50)
    ocl_f(lambda x: x[len(x) // 2 :], in_dims=4)
    ocl_f(lambda x: np.sum(x), in_dims=3)
    ocl_f(lambda x: np.mean(x), in_dims=3)
    ocl_f(lambda x: x.min(), in_dims=4)
    ocl_f(lambda x: np.sqrt((x ** 2).mean()), in_dims=5)


if __name__ == "__main__":
    print_various_ocl()
