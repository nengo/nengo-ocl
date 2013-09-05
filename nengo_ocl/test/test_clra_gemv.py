from nengo_ocl.tricky_imports import unittest

import nose
import numpy as np
from nengo_ocl.ra_gemv import ragged_gather_gemv
from  nengo_ocl import raggedarray as ra
RA = ra.RaggedArray
from nengo_ocl.clraggedarray import CLRaggedArray as CLRA

from nengo_ocl.clra_gemv import plan_ragged_gather_gemv
from nengo_ocl.clra_gemv import plan_parallel_ragged_gather_gemv2
from nengo_ocl.clra_gemv import plan_parallel_ragged_gather_gemv3

import pyopencl as cl
ctx = cl.create_some_context()

class TestStuff(unittest.TestCase):

    def test_basic(self):
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

