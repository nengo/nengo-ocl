# pylint: disable=missing-module-docstring,missing-function-docstring

import logging

import nengo
import numpy as np
import pyopencl as cl
import pytest
from nengo.utils.filter_design import ss2tf
from nengo.utils.stdlib import Timer

import nengo_ocl
from nengo_ocl import raggedarray as ra
from nengo_ocl.clra_nonlinearities import (
    plan_copy,
    plan_elementwise_inc,
    plan_lif,
    plan_lif_rate,
    plan_linearfilter,
    plan_reset,
    plan_slicedcopy,
)
from nengo_ocl.clraggedarray import CLRaggedArray as CLRA
from nengo_ocl.clraggedarray import to_device
from nengo_ocl.raggedarray import RaggedArray

logger = logging.getLogger(__name__)
RA = lambda arrays, dtype=np.float32: RaggedArray(arrays, dtype=dtype)


def not_close(a, b, rtol=1e-3, atol=1e-3):
    return np.abs(a - b) > atol + rtol * np.abs(b)


@pytest.mark.parametrize("upsample", [1, 4])
def test_lif_step(ctx, upsample):
    """Test the lif nonlinearity, comparing one step with the Numpy version."""
    rng = np.random

    dt = 1e-3
    n_neurons = [12345, 23456, 34567]
    J = RA([rng.normal(scale=1.2, size=n) for n in n_neurons])
    V = RA([rng.uniform(low=0, high=1, size=n) for n in n_neurons])
    W = RA([rng.uniform(low=-5 * dt, high=5 * dt, size=n) for n in n_neurons])
    OS = RA([np.zeros(n) for n in n_neurons])

    ref = 2e-3
    taus = rng.uniform(low=15e-3, high=80e-3, size=len(n_neurons))
    amp = 1.0

    queue = cl.CommandQueue(ctx)
    clJ = CLRA(queue, J)
    clV = CLRA(queue, V)
    clW = CLRA(queue, W)
    clOS = CLRA(queue, OS)
    clTaus = CLRA(queue, RA([t * np.ones(n) for t, n in zip(taus, n_neurons)]))

    # simulate host
    nls = [nengo.LIF(tau_ref=ref, tau_rc=taus[i]) for i, n in enumerate(n_neurons)]
    for i, nl in enumerate(nls):
        if upsample <= 1:
            nl.step(dt, J[i], OS[i], voltage=V[i], refractory_time=W[i])
        else:
            s = np.zeros_like(OS[i])
            for j in range(upsample):
                nl.step(dt / upsample, J[i], s, voltage=V[i], refractory_time=W[i])
                OS[i] = (1.0 / dt) * ((OS[i] > 0) | (s > 0))

    # simulate device
    plan = plan_lif(queue, dt, clJ, clV, clW, clOS, ref, clTaus, amp, upsample=upsample)
    plan()

    # print not close values
    a, b = V, clV
    for i, _ in enumerate(a):
        nc, _ = not_close(a[i], b[i]).nonzero()
        if len(nc) > 0:
            j = nc[0]
            print("i", i, "j", j)
            print("J", J[i][j], clJ[i][j])
            print("V", V[i][j], clV[i][j])
            print("W", W[i][j], clW[i][j])
            print("...", len(nc) - 1, "more")

    n_spikes = np.sum([np.sum(os) for os in OS])
    if n_spikes < 1.0:
        logger.warning("LIF spiking mechanism was not tested!")
    assert ra.allclose(J, clJ.to_host())
    assert ra.allclose(V, clV.to_host())
    assert ra.allclose(W, clW.to_host())
    assert ra.allclose(OS, clOS.to_host())


@pytest.mark.parametrize("heterogeneous", [False, True])
def test_lif_speed(ctx, rng, heterogeneous):
    """Test the speed of the lif nonlinearity

    heterogeneous: if true, use a wide range of population sizes.
    """
    dt = 1e-3
    ref = 2e-3
    tau = 20e-3
    amp = 1.0

    n_iters = 10
    if heterogeneous:
        n_neurons = [1.0e5] * 50 + [1e3] * 5000
    else:
        n_neurons = [1.1e5] * 50
    n_neurons = list(map(int, n_neurons))

    J = RA([rng.randn(n) for n in n_neurons], dtype=np.float32)
    V = RA([rng.uniform(low=0, high=1, size=n) for n in n_neurons], dtype=np.float32)
    W = RA(
        [rng.uniform(low=-10 * dt, high=10 * dt, size=n) for n in n_neurons],
        dtype=np.float32,
    )
    OS = RA([np.zeros(n) for n in n_neurons], dtype=np.float32)

    queue = cl.CommandQueue(
        ctx, properties=cl.command_queue_properties.PROFILING_ENABLE
    )

    clJ = CLRA(queue, J)
    clV = CLRA(queue, V)
    clW = CLRA(queue, W)
    clOS = CLRA(queue, OS)

    for i, blockify in enumerate([False, True]):
        plan = plan_lif(
            queue, dt, clJ, clV, clW, clOS, ref, tau, amp, blockify=blockify
        )

        with Timer() as timer:
            for _ in range(n_iters):
                plan()

        print("plan %d: blockify = %s, dur = %0.3f" % (i, blockify, timer.duration))


@pytest.mark.parametrize("blockify", [False, True])
def test_lif_rate(ctx, blockify):
    """Test the `lif_rate` nonlinearity"""
    rng = np.random
    dt = 1e-3

    n_neurons = [123459, 23456, 34567]
    J = RA([rng.normal(loc=1, scale=10, size=n) for n in n_neurons])
    R = RA([np.zeros(n) for n in n_neurons])

    ref = 2e-3
    taus = list(rng.uniform(low=15e-3, high=80e-3, size=len(n_neurons)))
    amp = 1.0

    queue = cl.CommandQueue(ctx)
    clJ = CLRA(queue, J)
    clR = CLRA(queue, R)
    clTaus = CLRA(queue, RA([t * np.ones(n) for t, n in zip(taus, n_neurons)]))

    # simulate host
    nls = [nengo.LIFRate(tau_ref=ref, tau_rc=taus[i]) for i, n in enumerate(n_neurons)]
    for i, nl in enumerate(nls):
        nl.step(dt, J[i], R[i])

    # simulate device
    plan = plan_lif_rate(queue, dt, clJ, clR, ref, clTaus, amp, blockify=blockify)
    plan()

    rate_sum = np.sum([np.sum(r) for r in R])
    if rate_sum < 1.0:
        logger.warning("LIF rate was not tested above the firing threshold!")
    assert ra.allclose(J, clJ.to_host())
    assert ra.allclose(R, clR.to_host())


def test_copy(ctx, rng):
    sizes = [(10, 1), (40, 64), (457, 342), (1, 100)]
    X = RA([rng.normal(size=size) for size in sizes])
    incs = rng.randint(0, 2, size=len(sizes)).astype(np.int32)

    queue = cl.CommandQueue(ctx)
    clX = CLRA(queue, X)
    clY = CLRA(queue, RA([np.zeros_like(x) for x in X]))

    # compute on device
    plan = plan_copy(queue, clX, clY, incs)
    plan()

    # check result
    for x, y in zip(X, clY.to_host()):
        assert np.allclose(y, x)


@pytest.mark.parametrize("use_alpha", [False, True])
@pytest.mark.parametrize("inc", [True, False])
def test_elementwise_inc(inc, use_alpha, ctx, rng, allclose):
    Xsizes = [(3, 3), (32, 64), (457, 342), (1, 100)]
    Asizes = [(3, 3), (1, 1), (457, 342), (100, 1)]

    A = [rng.normal(size=size).astype(np.float32) for size in Asizes]
    X = [rng.normal(size=size).astype(np.float32) for size in Xsizes]
    AX = [a * x for a, x in zip(A, X)]

    if use_alpha:
        alpha = rng.normal(size=len(Xsizes)).astype(np.float32)
        AX = [alph * ax for alph, ax in zip(alpha, AX)]
    else:
        alpha = None

    Y = [rng.normal(size=ax.shape).astype(np.float32) for ax in AX]
    if inc:
        Yref = [y + ax for y, ax in zip(Y, AX)]
    else:
        Yref = AX

    queue = cl.CommandQueue(ctx)
    clA = CLRA(queue, RA(A))
    clX = CLRA(queue, RA(X))
    clY = CLRA(queue, RA(Y))

    # compute on device
    plan = plan_elementwise_inc(queue, clA, clX, clY, alpha=alpha, inc=inc)
    plan()

    # check result
    for k, (yref, y) in enumerate(zip(Yref, clY.to_host())):
        assert allclose(y, yref, atol=2e-7), "Array %d not close" % k


def test_elementwise_inc_outer(ctx, rng):
    Xsizes = [3, 9, 15, 432]
    Asizes = [9, 42, 10, 124]
    alpha = rng.normal(size=len(Xsizes)).astype(np.float32)
    A = RA([rng.normal(size=size).astype(np.float32) for size in Asizes])
    X = RA([rng.normal(size=size).astype(np.float32) for size in Xsizes])
    AX = RA([alph * np.outer(a, x) for alph, a, x in zip(alpha, A, X)])
    Y = RA([rng.normal(size=ax.shape) for ax in AX])
    Yref = RA([y + ax for y, ax in zip(Y, AX)])

    queue = cl.CommandQueue(ctx)
    clA = CLRA(queue, A)
    clX = CLRA(queue, X)
    clY = CLRA(queue, Y)

    # compute on device
    plan = plan_elementwise_inc(queue, clA, clX, clY, alpha=alpha, outer=True)
    plan()

    # check result
    for yref, y in zip(Yref, clY.to_host()):
        assert np.allclose(y, yref, atol=2e-7)


def test_reset(ctx, rng):
    # Yshapes = [(100,), (10, 17), (3, 3)]
    Yshapes = [(1000000,), (1000, 1700), (3, 3)]
    values = rng.uniform(size=len(Yshapes)).astype(np.float32)

    queue = cl.CommandQueue(ctx)
    clY = CLRA(queue, RA([np.zeros(shape) for shape in Yshapes]))
    clvalues = to_device(queue, values)

    plan = plan_reset(queue, clY, clvalues)
    with Timer() as t:
        plan()

    print(t.duration)

    # with Timer() as t:
    #     for i in range(len(clY)):
    #         cl.enqueue_fill_buffer(
    #             queue, clY.cl_buf.data, values[i],
    #             clY.starts[i], clY.shape0s[i] * clY.shape1s[i])
    #     queue.finish()

    # print(t.duration)

    for y, v in zip(clY, values):
        assert np.all(y == v)


def test_slicedcopy(ctx, rng):
    sizes = rng.randint(20, 200, size=10)
    A = RA([rng.normal(size=size) for size in sizes])
    B = RA([rng.normal(size=size) for size in sizes])
    incs = rng.randint(0, 2, size=len(sizes))

    Ainds = []
    Binds = []
    for size in sizes:
        r = np.arange(size, dtype=np.int32)
        u = rng.choice([0, 1, 2])
        if u == 0:
            Ainds.append(r)
            Binds.append(r)
        elif u == 1:
            Ainds.append(r[:10])
            Binds.append(r[-10:])
        elif u == 2:
            n = rng.randint(2, size - 2)
            Ainds.append(rng.permutation(size)[:n])
            Binds.append(rng.permutation(size)[:n])

    Ainds = RA(Ainds, dtype=np.int32)
    Binds = RA(Binds, dtype=np.int32)

    queue = cl.CommandQueue(ctx)
    clA = CLRA(queue, A)
    clB = CLRA(queue, B)
    clAinds = CLRA(queue, Ainds)
    clBinds = CLRA(queue, Binds)

    # compute on host
    for i in range(len(sizes)):
        if incs[i]:
            B[i][Binds[i]] += A[i][Ainds[i]]
        else:
            B[i][Binds[i]] = A[i][Ainds[i]]

    # compute on device
    plan = plan_slicedcopy(queue, clA, clB, clAinds, clBinds, incs)
    plan()

    # check result
    for y, yy in zip(B, clB.to_host()):
        assert np.allclose(y, yy)


@pytest.mark.parametrize(
    "n_per_kind",
    [(100000, 0, 0), (0, 100000, 0), (0, 0, 100000), (10000, 10000, 10000)],
)
def test_linearfilter(ctx, n_per_kind, rng):
    kinds = (
        nengo.synapses.LinearFilter((2.0,), (1.0,), analog=False),
        nengo.synapses.Lowpass(0.005),
        nengo.synapses.Alpha(0.005),
    )
    assert len(n_per_kind) == len(kinds)
    kinds_n = [(kind, n) for kind, n in zip(kinds, n_per_kind) if n > 0]

    dt = 0.001
    steps = list()
    for kind, n in kinds_n:
        state = kind.make_state((n,), (n,), dt, dtype=np.float32)
        step = kind.make_step((n,), (n,), dt, rng=None, state=state)
        steps.append(step)

    # Nengo 3 uses state space filters. For now, convert back to transfer function.
    # Getting rid of this conversion would require a new plan_linearfilter.
    dens = list()
    nums = list()
    for f in steps:
        if type(f).__name__ == "NoX":  # special case for a feedthrough
            den = np.array([1.0])
            num = f.D
        else:
            num, den = ss2tf(f.A, f.B, f.C, f.D)

        # This preprocessing copied out of nengo2.8/synapses.LinearFilter.make_step
        num = num.flatten()

        assert den[0] == 1.0
        num = num[1:] if num[0] == 0 else num
        den = den[1:]  # drop first element (equal to 1)
        num, den = num.astype(np.float32), den.astype(np.float32)
        dens.append(den)
        nums.append(num)

    A = RA(dens)
    B = RA(nums)

    X = RA([rng.normal(size=n) for kind, n in kinds_n])
    Y = RA([np.zeros(n) for kind, n in kinds_n])
    Xbuf = RA([np.zeros(shape) for shape in zip(B.sizes, X.sizes)])
    Ybuf = RA([np.zeros(shape) for shape in zip(A.sizes, Y.sizes)])

    queue = cl.CommandQueue(ctx)
    clA = CLRA(queue, A)
    clB = CLRA(queue, B)
    clX = CLRA(queue, X)
    clY = CLRA(queue, Y)
    clXbuf = CLRA(queue, Xbuf)
    clYbuf = CLRA(queue, Ybuf)

    n_calls = 3
    plans = plan_linearfilter(queue, clX, clY, clA, clB, clXbuf, clYbuf)
    with Timer() as timer:
        for _ in range(n_calls):
            for plan in plans:
                plan()

    print(timer.duration)

    for i, [kind, n] in enumerate(kinds_n):
        n = min(n, 100)
        state = kind.make_state((n,), (n,), dt, dtype=np.float32)
        step = kind.make_step((n,), (n,), dt, rng=None, state=state)

        x = X[i][:n].T
        y = np.zeros_like(x)
        for _ in range(n_calls):
            y[:] = step(0, x)

        z = clY[i][:n].T
        assert np.allclose(z, y, atol=1e-7, rtol=1e-5), kind


@pytest.mark.parametrize(
    "neuron_type", (nengo.neurons.RectifiedLinear(), nengo.neurons.Sigmoid())
)
def test_static_neurons(plt, rng, neuron_type):
    with nengo.Network(seed=0) as model:
        u = nengo.Node(nengo.processes.WhiteNoise(scale=False))
        a = nengo.Ensemble(31, 1, neuron_type=neuron_type)
        nengo.Connection(u, a, synapse=None)

        xp = nengo.Probe(a.neurons, "input")
        yp = nengo.Probe(a.neurons)

    with nengo_ocl.Simulator(model) as sim:
        sim.run(1.0)

    x = sim.data[xp].ravel()
    y = sim.data[yp].ravel()
    r = neuron_type.rates(x, 1.0, 0.0).ravel()

    # --- plot
    (i,) = ((x > -10) & (x < 10)).nonzero()
    n_show = 100
    if len(i) > n_show:
        i = rng.choice(i, size=n_show, replace=False)

    plt.plot(x[i], r[i], "kx")
    plt.plot(x[i], y[i], ".")

    assert np.allclose(y, r, atol=1e-4, rtol=1e-3)
