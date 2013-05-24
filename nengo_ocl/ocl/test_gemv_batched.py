import numpy as np
from ..sim_npy import ragged_gather_gemv
from ..sim_npy import RaggedArray as RA
from ..sim_ocl import RaggedArray as CLRA

from gemv_batched import plan_ragged_gather_gemv

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

    print ragged_gather_gemv(Ms, Ns, .5, A, A_js, X, X_js, .1, Y)
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

