
import nose
import numpy as np

from nengo.objects import LIF, LIFRate, Direct

from ..ra_gemv import ragged_gather_gemv
from  .. import raggedarray as ra
from  ..raggedarray import RaggedArray as RA
from ..clraggedarray import CLRaggedArray as CLRA

from ..clra_nonlinearities import plan_lif

import pyopencl as cl
ctx = cl.create_some_context()

def not_close(a, b, rtol=1e-3, atol=1e-3):
    return np.abs(a - b) > atol + rtol * np.abs(b)

def test_lif_step():
    dt = 1e-3
    upsample = 2
    # t_final = 1.
    # t = dt * np.arange(np.round(t_final / dt))
    # nt = len(t)

    # n_neurons = [3, 3, 3]
    # n_neurons = [500, 200, 100]
    n_neurons = [12345, 23456, 34567]
    J = RA([np.random.randn(n) for n in n_neurons])
    V = RA([np.random.uniform(low=0, high=1, size=n) for n in n_neurons])
    W = RA([np.random.uniform(low=-10*dt, high=10*dt, size=n) for n in n_neurons])
    OS = RA([np.zeros(n) for n in n_neurons])

    ref = 2e-3
    tau = 20e-3
    # tau_array = RA([tau*np.ones(n) for n in n_neurons])

    queue = cl.CommandQueue(ctx)
    dJ = CLRA(queue, J)
    dV = CLRA(queue, V)
    dW = CLRA(queue, W)
    dOS = CLRA(queue, OS)
    # dTau = CLRA(queue, tau_array)

    ### simulate host
    nls = [LIF(n, tau_ref=ref, tau_rc=tau) for n in n_neurons]
    for i, nl in enumerate(nls):
        if upsample <= 1:
            nl.step_math0(dt, J[i], V[i], W[i], OS[i], upsample=upsample)
        else:
            s = np.zeros_like(OS[i])
            for j in xrange(upsample):
                nl.step_math0(dt/upsample, J[i], V[i], W[i], s)
                OS[i] = (OS[i] > 0.5) | (s > 0.5)

    ### simulate device
    plan = plan_lif(queue, dJ, dV, dW, dV, dW, dOS, ref, tau, dt, upsample=upsample)
    # plan = plan_lif(queue, dJ, dV, dW, dV, dW, dOS, ref, dTau, dt)
    plan()

    if 1:
        a, b = V, dV
        for i in xrange(len(a)):
            nc, _ = not_close(a[i], b[i]).nonzero()
            if len(nc) > 0:
                j = nc[0]
                print "i", i, "j", j
                print "J", J[i][j], dJ[i][j]
                print "V", V[i][j], dV[i][j]
                print "W", W[i][j], dW[i][j]
                print "...", len(nc) - 1, "more"

    print "number of spikes", np.sum([np.sum(OS[i]) for i in xrange(len(OS))])
    assert ra.allclose(J, dJ.to_host())
    assert ra.allclose(V, dV.to_host())
    assert ra.allclose(W, dW.to_host())
    assert ra.allclose(OS, dOS.to_host())

def test_lif_speed():
    # import time

    dt = 1e-3
    # t_final = 1.
    # t = dt * np.arange(np.round(t_final / dt))
    # nt = len(t)

    ref = 2e-3
    tau = 20e-3

    # n_neurons = [1.1e5] * 5
    n_neurons = [1.0e5] * 5 + [1e3]*50
    J = RA([np.random.randn(n) for n in n_neurons])
    V = RA([np.random.uniform(low=0, high=1, size=n) for n in n_neurons])
    W = RA([np.random.uniform(low=-10*dt, high=10*dt, size=n) for n in n_neurons])
    OS = RA([np.zeros(n) for n in n_neurons])

    queue = cl.CommandQueue(
        ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    dJ = CLRA(queue, J)
    dV = CLRA(queue, V)
    dW = CLRA(queue, W)
    dOS = CLRA(queue, OS)

    plan = plan_lif(queue, dJ, dV, dW, dV, dW, dOS, ref, tau, dt)
    # plan = plan_lif_flat(queue, dJ, dV, dW, dV, dW, dOS, ref, tau, dt)
    # plan = plan_lif_group(queue, dJ, dV, dW, dV, dW, dOS, ref, tau, dt)

    for i in range(1000):
        plan(profiling=True)

    print 'n_calls         ', plan.n_calls
    print 'queued -> submit', plan.atime
    print 'submit -> start ', plan.btime
    print 'start -> end    ', plan.ctime

    # timer = time.time()
    # for i in xrange(1000):
        # plan()
    # print "elapsed time:", time.time() - timer

# def test_lif0():
#     if profiling:
#         queue = cl.CommandQueue(
#             ctx,
#             properties=cl.command_queue_properties.PROFILING_ENABLE)
#     else:
#         queue = cl.CommandQueue(ctx)
#     N_neurons = 50000
#     N_iters = 50

#     J = to_device(queue, np.random.rand(N_neurons).astype('float32'))
#     V = to_device(queue, np.random.rand(N_neurons).astype('float32'))
#     RT = to_device(queue, np.random.rand(N_neurons).astype('float32'))
#     OS = to_device(queue, np.random.rand(N_neurons).astype('float32'))

#     plan = plan_lif(queue, V, RT, J, V, RT, OS,
#             .001, .02, .002, 1.0, 2)

#     for ii in range(N_iters):
#         plan(profiling=profiling)
#     print 'time per call', plan.ctime / plan.n_calls

