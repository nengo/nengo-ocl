import numpy as np
from nengo_ocl.tricky_imports import unittest

import nengo
from nengo.nonlinearities import LIF, LIFRate, Direct

import pyopencl as cl
ctx = cl.create_some_context()

import logging
logger = logging.getLogger(__name__)
# nengo.log(True)

from nengo_ocl.ra_gemv import ragged_gather_gemv
from nengo_ocl import raggedarray as ra
from nengo_ocl.raggedarray import RaggedArray as RA
from nengo_ocl.clraggedarray import CLRaggedArray as CLRA

from nengo_ocl.clra_nonlinearities import *

def not_close(a, b, rtol=1e-3, atol=1e-3):
    return np.abs(a - b) > atol + rtol * np.abs(b)

class Test_CLRA_Nonlinearities(unittest.TestCase):
    def test_lif_step0(self):
        self.test_lif_step(upsample=1, n_elements=0)

    def test_lif_step1(self):
        self.test_lif_step(upsample=4, n_elements=0)

    def test_lif_step2(self):
        self.test_lif_step(upsample=4, n_elements=7)

    def test_lif_step(self, upsample=1, n_elements=0):
        """Test the lif nonlinearity, comparing one step with the Numpy version."""
        dt = 1e-3
        # n_neurons = [3, 3, 3]
        n_neurons = [12345, 23456, 34567]
        N = len(n_neurons)
        J = RA([np.random.normal(scale=1.2, size=n) for n in n_neurons])
        V = RA([np.random.uniform(low=0, high=1, size=n) for n in n_neurons])
        W = RA([np.random.uniform(low=-5*dt, high=5*dt, size=n) for n in n_neurons])
        OS = RA([np.zeros(n) for n in n_neurons])

        ref = 2e-3
        # tau = 20e-3
        # refs = list(np.random.uniform(low=1.7e-3, high=4.2e-3, size=len(n_neurons)))
        taus = list(np.random.uniform(low=15e-3, high=80e-3, size=len(n_neurons)))

        queue = cl.CommandQueue(ctx)
        clJ = CLRA(queue, J)
        clV = CLRA(queue, V)
        clW = CLRA(queue, W)
        clOS = CLRA(queue, OS)
        # clRef = CLRA(queue, RA(refs))
        clTau = CLRA(queue, RA(taus))

        ### simulate host
        nls = [LIF(n, tau_ref=ref, tau_rc=taus[i])
               for i, n in enumerate(n_neurons)]
        for i, nl in enumerate(nls):
            if upsample <= 1:
                nl.step_math0(dt, J[i], V[i], W[i], OS[i])
            else:
                s = np.zeros_like(OS[i])
                for j in xrange(upsample):
                    nl.step_math0(dt/upsample, J[i], V[i], W[i], s)
                    OS[i] = (OS[i] > 0.5) | (s > 0.5)

        ### simulate device
        plan = plan_lif(queue, clJ, clV, clW, clV, clW, clOS, ref, clTau, dt,
                        n_elements=n_elements, upsample=upsample)
        plan()

        if 1:
            a, b = V, clV
            for i in xrange(len(a)):
                nc, _ = not_close(a[i], b[i]).nonzero()
                if len(nc) > 0:
                    j = nc[0]
                    print "i", i, "j", j
                    print "J", J[i][j], clJ[i][j]
                    print "V", V[i][j], clV[i][j]
                    print "W", W[i][j], clW[i][j]
                    print "...", len(nc) - 1, "more"

        n_spikes = np.sum([np.sum(os) for os in OS])
        if n_spikes < 1.0:
            logger.warn("LIF spiking mechanism was not tested!")
        assert ra.allclose(J, clJ.to_host())
        assert ra.allclose(V, clV.to_host())
        assert ra.allclose(W, clW.to_host())
        assert ra.allclose(OS, clOS.to_host())

    def test_lif_speed(self, heterogeneous=True):
        """Test the speed of the lif nonlinearity

        heterogeneous: if true, use a wide range of population sizes.
        """

        dt = 1e-3
        ref = 2e-3
        tau = 20e-3

        if heterogeneous:
            n_neurons = [1.0e5] * 5 + [1e3]*50
        else:
            n_neurons = [1.1e5] * 5

        J = RA([np.random.randn(n) for n in n_neurons])
        V = RA([np.random.uniform(low=0, high=1, size=n) for n in n_neurons])
        W = RA([np.random.uniform(low=-10*dt, high=10*dt, size=n) for n in n_neurons])
        OS = RA([np.zeros(n) for n in n_neurons])

        queue = cl.CommandQueue(
            ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        clJ = CLRA(queue, J)
        clV = CLRA(queue, V)
        clW = CLRA(queue, W)
        clOS = CLRA(queue, OS)

        n_elements = [0, 2, 5, 10]
        for i, nel in enumerate(n_elements):
            plan = plan_lif(queue, clJ, clV, clW, clV, clW, clOS, ref, tau, dt,
                            n_elements=nel)

            for j in range(1000):
                plan(profiling=True)

            print "plan %d: n_elements = %d" % (i, nel)
            print 'n_calls         ', plan.n_calls
            print 'queued -> submit', plan.atime
            print 'submit -> start ', plan.btime
            print 'start -> end    ', plan.ctime

    def test_lif_rate(self, n_elements=0):
        """Test the `lif_rate` nonlinearity"""
        # n_neurons = [3, 3, 3]
        n_neurons = [123459, 23456, 34567]
        N = len(n_neurons)
        J = RA([np.random.normal(loc=1, scale=10, size=n) for n in n_neurons])
        R = RA([np.zeros(n) for n in n_neurons])

        ref = 2e-3
        taus = list(np.random.uniform(low=15e-3, high=80e-3, size=len(n_neurons)))

        queue = cl.CommandQueue(ctx)
        clJ = CLRA(queue, J)
        clR = CLRA(queue, R)
        clTau = CLRA(queue, RA(taus))

        ### simulate host
        nls = [LIF(n, tau_ref=ref, tau_rc=taus[i])
               for i, n in enumerate(n_neurons)]
        for i, nl in enumerate(nls):
            R[i] = nl.rates(J[i].flatten()).reshape((-1,1))

        ### simulate device
        plan = plan_lif_rate(queue, clJ, clR, ref, clTau, n_elements=n_elements)
        plan()

        rate_sum = np.sum([np.sum(r) for r in R])
        if rate_sum < 1.0:
            logger.warn("LIF rate was not tested above the firing threshold!")
        assert ra.allclose(J, clJ.to_host())
        assert ra.allclose(R, clR.to_host())

if __name__ == '__main__':
   unittest.main()
