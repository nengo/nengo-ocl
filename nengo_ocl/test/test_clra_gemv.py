import nose
import numpy as np
from ..ra_gemv import ragged_gather_gemv
from  .. import raggedarray as ra
RA = ra.RaggedArray
from ..clraggedarray import CLRaggedArray as CLRA

from ..clra_gemv import plan_ragged_gather_gemv

import pyopencl as cl
ctx = cl.create_some_context()

def test_basic():
    # -- prepare initial conditions on host
    A = RA([ [[0.1, .2], [.3, .4]], [[.5, .6]]])
    X = RA([ [3, 5] ])

    A_js = RA([[1], [0]])
    X_js = RA([[0], [0]])

    Y = RA([[0.0], [2, 3],])

    # -- prepare initial conditions on device
    queue = cl.CommandQueue(ctx)

    clA = CLRA(queue, A)
    clX = CLRA(queue, X)

    clA_js = CLRA(queue, A_js)
    clX_js = CLRA(queue, X_js)

    clY = CLRA(queue, Y)

    plan = plan_ragged_gather_gemv(
        queue,
        .5, clA, clA_js, clX, clX_js, .1, clY)

    assert ra.allclose(A, clA)
    assert ra.allclose(X, clX)
    assert ra.allclose(Y, clY)
    assert ra.allclose(A_js, clA_js)
    assert ra.allclose(X_js, clX_js)

    # -- run host computation
    ragged_gather_gemv(
        .5, A, A_js, X, X_js, .1, Y,
        use_raw_fn=True)
    result1 = Y.buf

    # -- run cl computation
    plan()
    result2 = clY.buf

    # -- ensure they match
    print result1
    print result2
    assert ra.allclose(Y, clY)


def test_reduction_speed():
    queue = cl.CommandQueue(
        ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE)

    L = 512  # -- length of each vector
    N = 1000 # -- number of vectors

    A = CLRA(queue, RA([np.random.randn(1, L) for i in range(N)]))
    X = CLRA(queue, RA([np.random.randn(L, 1) for i in range(N)]))

    X_js = CLRA(queue, RA([[i] for i in range(N) + range(N)]))
    A_js = CLRA(queue, RA([[i] for i in range(N) + list(reversed(range(N)))]))

    Y = CLRA(queue, RA([[1.0] for i in range(2 * N)]))

    plan = plan_ragged_gather_gemv(queue,
                                   1.0, A, A_js, X, X_js,
                                   0.0, Y)
    for i in range(10):
        plan(profiling=True)

    print 'n_calls         ', plan.n_calls
    print 'queued -> submit', plan.atime
    print 'submit -> start ', plan.btime
    print 'start -> end    ', plan.ctime

