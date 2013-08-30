import nose
import numpy as np
from ..ra_gemv import ragged_gather_gemv
from ..raggedarray import RaggedArray as RA
from ..clraggedarray import CLRaggedArray as CLRA

from ..clra_gemv import plan_ragged_gather_gemv

import pyopencl as cl
ctx = cl.create_some_context()

def test_basic():
    A = RA([ [0.1, .2, .3, .4], [.5, .6]])
    Ms = [2, 1]
    Ns = [2, 2]
    X = RA([ [3, 5] ])

    X_js = RA([[0], [0]])
    A_js = RA([[1], [0]])

    Y = RA([[0.0], [2, 3],])

    print ragged_gather_gemv(.5, A, A_js, X, X_js, .1, Y)

    result1 = Y.buf

    queue = cl.CommandQueue(ctx)

    A = CLRA(queue, [[0.1, .2, .3, .4], [.5, .6]])
    Ms = [2, 1]
    Ns = [2, 2]
    X = CLRA(queue, [[3, 5]])

    X_js = CLRA(queue, [[0], [0]])
    A_js = CLRA(queue, [[1], [0]])

    Y = CLRA(queue, [[0.0], [2, 3],])

    plan = plan_ragged_gather_gemv(queue, Ms, Ns, .5, A, A_js, X, X_js, .1, Y)
    plan()
    result2 = Y.buf.get()
    assert np.allclose(result1, result2)


@nose.SkipTest
def test_reduction_speed():
    queue = cl.CommandQueue(
        ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE)

    L = 512  # -- length of each vector
    N = 1000 # -- number of vectors

    Arows = [1] * N
    Acols = [L] * N
    A = CLRA(queue, [np.random.randn(L) for i in range(N)])
    X = CLRA(queue, [np.random.randn(L) for i in range(N)])

    X_js = CLRA(queue, [[i] for i in range(N) + range(N)])
    A_js = CLRA(queue, [[i] for i in range(N) + list(reversed(range(N)))])

    Y = CLRA(queue, [[1.0] for i in range(2 * N)])

    plan = plan_ragged_gather_gemv(queue, Arows, Acols,
                                   1.0, A, A_js, X, X_js,
                                   0.0, Y)
    for i in range(10):
        plan(profiling=True)

    print 'n_calls         ', plan.n_calls
    print 'queued -> submit', plan.atime
    print 'submit -> start ', plan.btime
    print 'start -> end    ', plan.ctime

