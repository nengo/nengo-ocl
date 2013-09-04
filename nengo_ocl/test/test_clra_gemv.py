from nengo_ocl.tricky_imports import unittest

import nose
import numpy as np
from nengo_ocl.ra_gemv import ragged_gather_gemv
from  nengo_ocl import raggedarray as ra
RA = ra.RaggedArray
from nengo_ocl.clraggedarray import CLRaggedArray as CLRA

from nengo_ocl.clra_gemv import plan_ragged_gather_gemv
from nengo_ocl.clra_gemv import plan_parallel_ragged_gather_gemv2

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


    def test_reduction_speed(self, runs=10):
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

        plan = plan_ragged_gather_gemv(queue,
                                       1.0, A, A_js, X, X_js,
                                       0.0, Y)
        for i in range(runs):
            plan(profiling=True)

        print 'n_calls(gemv)             ', plan.n_calls
        print 'avg. queued -> submit (ms)', (1000 * plan.atime / runs)
        print 'avg. submit -> start  (ms)', (1000 * plan.btime / runs)
        print 'avg. start -> end     (ms)', (1000 * plan.ctime / runs)

        print "Output: "
        print Y.buf
        print Y.buf.sum()


    # -- Using same data and number of runs as test_reduction_speed() performs
    #    dot-products #in groups within separate work-items.
    #    -  group_size corresponds to the size of the separate chunks
    #       that Y is split into.
    #       That is, if N = 1000 and group_size = 32,
    #       then each kernel will do 1024/32=32 dot-products
    #       32 is optimal size on my GT650M
    def test_parallel_reduction_speed(self,
            group_size=128,
            runs=1000):
        queue = cl.CommandQueue(
            ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

        np.random.seed(50)

        L = 512  # -- length of each vector
        N = 1000 # -- number of vectors

        # adjacency matrix for synapse weights. Row is a soma, column soma
        A = CLRA(queue, RA([np.random.randn(1, L) for i in range(N)]))
        X = CLRA(queue, RA([np.random.randn(L, 1) for i in range(N)]))
        # X potential on axons. X*A = synapse strength times current strength.
        # Dot product from row of A to row of X to get total current that
        # goes into a soma.  List of lists of size 1.
        # Xjs can be different lengths too!

        X_js = CLRA(queue, RA([[i] for i in range(N) + range(N)]))
        A_js = CLRA(queue, RA([[i] for i in range(N) + list(reversed(range(N)))]))

        Y = CLRA(queue, RA([[1.0] for i in range(2 * N)]))

        plan = plan_parallel_ragged_gather_gemv2(
            queue,
            1.0, A, A_js, X, X_js,
            0.0, Y, group_size=group_size)

        for i in xrange(runs):
            plan(profiling=True)

        print 'n_calls(gemv2)            ', plan.n_calls
        print 'avg. queued -> submit (ms)', (1000 * plan.atime / runs)
        print 'avg. submit -> start  (ms)', (1000 * plan.btime / runs)
        print 'avg. start -> end     (ms)', (1000 * plan.ctime / runs)

        print "Output: "
        print Y.buf
        print Y.buf.sum()


if __name__ == '__main__':
   unittest.main()

