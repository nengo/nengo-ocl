import logging
import numpy as np
import pyopencl as cl
import pytest

from nengo.neurons import LIF, LIFRate
from nengo.utils.compat import range

from nengo_ocl import raggedarray as ra
from nengo_ocl.raggedarray import RaggedArray as RA
from nengo_ocl.clraggedarray import CLRaggedArray as CLRA

from nengo_ocl.clra_nonlinearities import (
    plan_lif, plan_lif_rate, plan_elementwise_inc)


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
    from nengo.utils.testing import Timer

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

    n_elements = [0, 1, 2, 5, 10, 50]
    for i, nel in enumerate(n_elements):
        plan = plan_lif(queue, clJ, clV, clW, clV, clW, clOS, ref, tau, dt,
                        n_elements=nel)

        with Timer() as timer:
            for j in range(1000):
                plan(profiling=True)

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


def test_elementwise_inc():

    rng = np.random.RandomState(8)
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
