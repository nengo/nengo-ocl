import logging
import numpy as np
import pyopencl as cl
import pytest

from nengo.neurons import LIF, LIFRate
from nengo.utils.compat import range
from nengo.utils.stdlib import Timer

from nengo_ocl import raggedarray as ra
from nengo_ocl.raggedarray import RaggedArray as RA
from nengo_ocl.clraggedarray import CLRaggedArray as CLRA, to_device

from nengo_ocl.clra_nonlinearities import (
    plan_lif, plan_lif_rate, plan_elementwise_inc, plan_reset, plan_slicedcopy)


ctx = cl.create_some_context()
logger = logging.getLogger(__name__)


def not_close(a, b, rtol=1e-3, atol=1e-3):
    return np.abs(a - b) > atol + rtol * np.abs(b)


@pytest.mark.parametrize(
    "upsample, n_elements", [(1, 0), (1, 6), (4, 0), (4, 7)])
def test_lif_step(upsample, n_elements):
    """Test the lif nonlinearity, comparing one step with the Numpy version."""
    rng = np.random

    dt = 1e-3
    n_neurons = [12345, 23456, 34567]
    J = RA([rng.normal(scale=1.2, size=n) for n in n_neurons])
    V = RA([rng.uniform(low=0, high=1, size=n) for n in n_neurons])
    W = RA([rng.uniform(low=-5 * dt, high=5 * dt, size=n) for n in n_neurons])
    OS = RA([np.zeros(n) for n in n_neurons])

    ref = 2e-3
    taus = list(rng.uniform(low=15e-3, high=80e-3, size=len(n_neurons)))

    queue = cl.CommandQueue(ctx)
    clJ = CLRA(queue, J)
    clV = CLRA(queue, V)
    clW = CLRA(queue, W)
    clOS = CLRA(queue, OS)
    clTau = CLRA(queue, RA(taus))

    # simulate host
    nls = [LIF(tau_ref=ref, tau_rc=taus[i])
           for i, n in enumerate(n_neurons)]
    for i, nl in enumerate(nls):
        if upsample <= 1:
            nl.step_math(dt, J[i], OS[i], V[i], W[i])
        else:
            s = np.zeros_like(OS[i])
            for j in range(upsample):
                nl.step_math(dt / upsample, J[i], s, V[i], W[i])
                OS[i] = (1./dt) * ((OS[i] > 0) | (s > 0))

    # simulate device
    plan = plan_lif(queue, clJ, clV, clW, clV, clW, clOS, ref, clTau, dt,
                    n_elements=n_elements, upsample=upsample)
    plan()

    if 1:
        a, b = V, clV
        for i in range(len(a)):
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
        logger.warn("LIF spiking mechanism was not tested!")
    assert ra.allclose(J, clJ.to_host())
    assert ra.allclose(V, clV.to_host())
    assert ra.allclose(W, clW.to_host())
    assert ra.allclose(OS, clOS.to_host())


@pytest.mark.parametrize("heterogeneous", [False, True])
def test_lif_speed(rng, heterogeneous):
    """Test the speed of the lif nonlinearity

    heterogeneous: if true, use a wide range of population sizes.
    """
    dt = 1e-3
    ref = 2e-3
    tau = 20e-3

    if heterogeneous:
        n_neurons = [1.0e5] * 5 + [1e3] * 50
    else:
        n_neurons = [1.1e5] * 5
    n_neurons = list(map(int, n_neurons))

    J = RA([rng.randn(n) for n in n_neurons])
    V = RA([rng.uniform(low=0, high=1, size=n) for n in n_neurons])
    W = RA([rng.uniform(low=-10 * dt, high=10 * dt, size=n)
            for n in n_neurons])
    OS = RA([np.zeros(n) for n in n_neurons])

    queue = cl.CommandQueue(
        ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    clJ = CLRA(queue, J)
    clV = CLRA(queue, V)
    clW = CLRA(queue, W)
    clOS = CLRA(queue, OS)

    # n_elements = [0, 1, 2, 5, 10, 50]
    n_elements = [0, 1, 2, 5]
    for i, nel in enumerate(n_elements):
        plan = plan_lif(queue, clJ, clV, clW, clV, clW, clOS, ref, tau, dt,
                        n_elements=nel)

        with Timer() as timer:
            for j in range(1000):
                plan()

        print("plan %d: n_elements = %d, dur = %0.3f"
              % (i, nel, timer.duration))


@pytest.mark.parametrize("n_elements", [0, 1, 10])
def test_lif_rate(n_elements):
    """Test the `lif_rate` nonlinearity"""
    rng = np.random
    dt = 1e-3

    n_neurons = [123459, 23456, 34567]
    J = RA([rng.normal(loc=1, scale=10, size=n) for n in n_neurons])
    R = RA([np.zeros(n) for n in n_neurons])

    ref = 2e-3
    taus = list(rng.uniform(low=15e-3, high=80e-3, size=len(n_neurons)))

    queue = cl.CommandQueue(ctx)
    clJ = CLRA(queue, J)
    clR = CLRA(queue, R)
    clTau = CLRA(queue, RA(taus))

    # simulate host
    nls = [LIFRate(tau_ref=ref, tau_rc=taus[i])
           for i, n in enumerate(n_neurons)]
    for i, nl in enumerate(nls):
        nl.step_math(dt, J[i], R[i])

    # simulate device
    plan = plan_lif_rate(queue, clJ, clR, ref, clTau, dt=dt,
                         n_elements=n_elements)
    plan()

    rate_sum = np.sum([np.sum(r) for r in R])
    if rate_sum < 1.0:
        logger.warn("LIF rate was not tested above the firing threshold!")
    assert ra.allclose(J, clJ.to_host())
    assert ra.allclose(R, clR.to_host())


def test_elementwise_inc(rng):
    Xsizes = [(3, 3), (32, 64), (457, 342), (1, 100)]
    Asizes = [(3, 3), (1, 1),   (457, 342), (100, 1)]
    A = RA([rng.normal(size=size) for size in Asizes])
    X = RA([rng.normal(size=size) for size in Xsizes])
    Y = RA([a * x for a, x in zip(A, X)])

    queue = cl.CommandQueue(ctx)
    clA = CLRA(queue, A)
    clX = CLRA(queue, X)
    clY = CLRA(queue, RA([np.zeros_like(y) for y in Y]))

    # compute on device
    plan = plan_elementwise_inc(queue, clA, clX, clY)
    plan()

    # check result
    for y, yy in zip(Y, clY.to_host()):
        assert np.allclose(y, yy)


def test_reset(rng):
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


def test_slicedcopy(rng):
    sizes = rng.randint(20, 200, size=10)
    A = RA([rng.normal(size=size) for size in sizes])
    B = RA([rng.normal(size=size) for size in sizes])
    incs = RA([rng.randint(0, 2) for _ in sizes])

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

    Ainds = RA(Ainds)
    Binds = RA(Binds)

    queue = cl.CommandQueue(ctx)
    clA = CLRA(queue, A)
    clB = CLRA(queue, B)
    clAinds = CLRA(queue, Ainds)
    clBinds = CLRA(queue, Binds)
    clincs = CLRA(queue, incs)

    # compute on host
    for i in range(len(sizes)):
        if incs[i]:
            B[i][Binds[i]] += A[i][Ainds[i]]
        else:
            B[i][Binds[i]] = A[i][Ainds[i]]

    # compute on device
    plan = plan_slicedcopy(queue, clA, clB, clAinds, clBinds, clincs)
    plan()

    # check result
    for y, yy in zip(B, clB.to_host()):
        assert np.allclose(y, yy)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
