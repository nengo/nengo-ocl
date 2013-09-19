
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
        self._test_random(k=4, p=4, m=1000, n=1000)

class TestSpeed(unittest.TestCase):

    def setUp(self):
        queue = cl.CommandQueue(
            ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

        L = 512  # -- length of each vector
        N = 1000 # -- number of vectors

        np.random.seed(50)

        A = CLRA(queue, RA([np.random.randn(1, L) for i in range(N)]))
        X = CLRA(queue, RA([np.random.randn(L, 1) for i in range(N)]))

        X_js = CLRA(queue,
                RA([[i] for i in range(N) + range(N)]))
        A_js = CLRA(queue,
                RA([[i] for i in range(N) + list(reversed(range(N)))]))

        Y = CLRA(queue, RA([[1.0] for i in range(2 * N)]))
        self.__dict__.update(locals())
        del self.self

    def test_gemv(self, runs=100):

        plan = plan_ragged_gather_gemv(self.queue,
                                       1.0,
                                       self.A, self.A_js,
                                       self.X, self.X_js,
                                       0.0,
                                       self.Y)
        for i in range(runs):
            plan(profiling=True)

        print 'n_calls(gemv)             ', plan.n_calls
        print 'avg. queued -> submit (ms)', (1000 * plan.atime / runs)
        print 'avg. submit -> start  (ms)', (1000 * plan.btime / runs)
        print 'avg. start -> end     (ms)', (1000 * plan.ctime / runs)

        print "Output: "
        print self.Y.buf
        print self.Y.buf.sum()


    # -- Using same data and number of runs as test_reduction_speed() performs
    #    dot-products #in groups within separate work-items.
    #    -  group_size corresponds to the size of the separate chunks
    #       that Y is split into.
    #       That is, if N = 1000 and group_size = 32,
    #       then each kernel will do 1024/32=32 dot-products
    #       32 is optimal size on my GT650M
    def test_gemv2(self,
            group_size=128,
            runs=100):

        plan = plan_parallel_ragged_gather_gemv2(
            self.queue,
            1.0, self.A, self.A_js, self.X, self.X_js,
            0.0, self.Y, group_size=group_size)

        for i in xrange(runs):
            plan(profiling=True)

        print 'n_calls(gemv2)            ', plan.n_calls
        print 'avg. queued -> submit (ms)', (1000 * plan.atime / runs)
        print 'avg. submit -> start  (ms)', (1000 * plan.btime / runs)
        print 'avg. start -> end     (ms)', (1000 * plan.ctime / runs)

        print "Output: "
        print self.Y.buf
        print self.Y.buf.sum()

    def test_gemv3(self,
            group_size=128,
            runs=100):
        plan = plan_parallel_ragged_gather_gemv3(
            self.queue,
            1.0, self.A, self.A_js, self.X, self.X_js,
            0.0, self.Y, group_size=group_size)

        for i in xrange(runs):
            plan(profiling=True)

        print 'n_calls(gemv2)            ', plan.n_calls
        print 'avg. queued -> submit (ms)', (1000 * plan.atime / runs)
        print 'avg. submit -> start  (ms)', (1000 * plan.btime / runs)
        print 'avg. start -> end     (ms)', (1000 * plan.ctime / runs)

        print "Output: "
        print self.Y.buf
        print self.Y.buf.sum()


if __name__ == '__main__':
   unittest.main()

