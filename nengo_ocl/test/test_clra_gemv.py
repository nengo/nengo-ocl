
import nose
import numpy as np

from nengo.tests.helpers import assert_allclose

from nengo_ocl.tricky_imports import unittest
from nengo_ocl.ra_gemv import ragged_gather_gemv
from  nengo_ocl import raggedarray as ra
RA = ra.RaggedArray
from nengo_ocl.clraggedarray import CLRaggedArray as CLRA

from nengo_ocl.clra_gemv import plan_ragged_gather_gemv
# from nengo_ocl.clra_gemv import plan_parallel_ragged_gather_gemv2
# from nengo_ocl.clra_gemv import plan_parallel_ragged_gather_gemv3

import pyopencl as cl
import logging

ctx = cl.create_some_context()
logger = logging.getLogger(__name__)

def allclose(raA, raB):
    assert len(raA) == len(raB)
    for i in xrange(len(raA)):
         if not np.allclose(raA[i], raB[i]):
             return False
    return True

class TestStuff(unittest.TestCase):

    def test_basic(self):
        # -- prepare initial conditions on host
        A = RA([ [[0.1, .2], [.3, .4]], [[.5, .6]]])
        X = RA([ [3, 5] ])
        Y = RA([[0.0], [2, 3],])
        A_js = RA([[1], [0]])
        X_js = RA([[0], [0]])
        alpha = 0.5
        beta = 0.1

        # -- prepare initial conditions on device
        queue = cl.CommandQueue(ctx)
        clA = CLRA(queue, A)
        clX = CLRA(queue, X)
        clY = CLRA(queue, Y)
        clA_js = CLRA(queue, A_js)
        clX_js = CLRA(queue, X_js)
        assert allclose(A, clA)
        assert allclose(X, clX)
        assert allclose(Y, clY)
        assert allclose(A_js, clA_js)
        assert allclose(X_js, clX_js)

        # -- run cl computation
        plan = plan_ragged_gather_gemv(
            queue, alpha, clA, clA_js, clX, clX_js, beta, clY)

        plan()

        # -- ensure they match
        for i in xrange(len(A_js)):
            aj, xj = int(A_js[i]), int(X_js[i])
            ref = alpha*np.dot(A[aj], X[xj]) + beta*Y[i]
            sim = clY[i]
            assert np.allclose(ref, sim)

    def _test_random(self, k=4, p=1, m=10, n=10):
        """
        Parameters
        ----------
        k : number of operations (length of A_js)
        p : number of dots per operation (width of A_js)
        m : output dimensions
        n : input dimensions
        """

        rng = np.random.RandomState(3294)

        aa = [rng.normal(size=(m, n)) for i in xrange(k)]
        xx = [rng.normal(size=n) for i in xrange(k)]
        yy = [rng.normal(size=m) for i in xrange(k)]
        ajs = [rng.randint(k, size=p) for i in xrange(k)]
        xjs = [rng.randint(k, size=p) for i in xrange(k)]

        A = RA(aa)
        X = RA(xx)
        Y = RA(yy)
        A_js = RA(ajs)
        X_js = RA(xjs)
        alpha = 0.5
        beta = 0.1

        # -- prepare initial conditions on device
        queue = cl.CommandQueue(ctx)
        clA = CLRA(queue, A)
        clX = CLRA(queue, X)
        clY = CLRA(queue, Y)
        clA_js = CLRA(queue, A_js)
        clX_js = CLRA(queue, X_js)
        assert allclose(A, clA)
        assert allclose(X, clX)
        assert allclose(Y, clY)
        assert allclose(A_js, clA_js)
        assert allclose(X_js, clX_js)

        # -- run cl computation
        prog = plan_ragged_gather_gemv(
            queue, alpha, clA, clA_js, clX, clX_js, beta, clY)

        print '-' * 5 + ' Plans ' + '-' * 45
        for plan in prog.plans:
            print plan
        prog()

        # -- ensure they match
        for i in xrange(k):
            ref = beta*Y[i]
            for aj, xj in zip(A_js[i], X_js[i]):
                ref += alpha*np.dot(A[aj], X[xj])
            sim = clY[i]
            assert np.allclose(ref, sim, atol=1e-3, rtol=1e-3)
            # assert_allclose(self, logger, ref, sim, atol=1e-3, rtol=1e-3)

    def test_random_small(self):
        self._test_random(k=4, m=10, n=10)

    def test_random_large(self):
        self._test_random(k=10, m=550, n=550)

    def test_many_dots_small(self):
        self._test_random(k=4, p=4, m=10, n=10)

    def test_many_dots_large(self):
        # self._test_random(k=4, p=4, m=550, n=550)
        self._test_random(k=4, p=4, m=2000, n=1000)

if __name__ == '__main__':
   unittest.main()

