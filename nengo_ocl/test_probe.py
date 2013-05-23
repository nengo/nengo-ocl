import time

import matplotlib.pyplot as plt
import numpy as np
import pyopencl as cl

ctx = cl.create_some_context()

from nengo import nef_theano as nef

from ocl import array
from ocl.plan import Prog
from ocl.lif import plan_lif
from ocl.gemv_batched import plan_misc_gemv

import base


class LIFMultiEnsemble(object):
    """
    Object that allocates a bunch of buffers for
    * lif neurons
    * the signals they represent
    * the encoders and decoders for relating the above

    """
    def __init__(self, n_populations, n_neurons_per, n_signals, signal_size,
            n_dec_per_signal=1,
            n_enc_per_population=1,
            lif_tau_rc=0.02,
            lif_tau_ref=0.002,
            lif_V_threshold=1.0,
            lif_output_pstc=0.01,
            lif_upsample=2,
            noise=None,
            queue=None):
        self.__dict__.update(locals())
        del self.self

        self.lif_v = array.zeros(queue,
                shape=(n_populations, n_neurons_per),
                dtype='float32')

        self.lif_rt = array.zeros(queue,
                shape=(n_populations, n_neurons_per),
                dtype='float32')

        self.lif_ic = array.zeros(queue,
                shape=(n_populations, n_neurons_per),
                dtype='float32')

        self.lif_bias = array.zeros(queue,
                shape=(n_populations, n_neurons_per),
                dtype='float32')

        self.lif_output = array.zeros(queue,
                shape=(n_populations, n_neurons_per),
                dtype='float32')

        self.lif_output_filter = array.zeros(queue,
                shape=(n_populations, n_neurons_per),
                dtype='float32')

        self.signals = array.zeros(queue,
                shape=(n_signals, signal_size),
                dtype='float32')

        self.encoders = array.zeros(queue,
                shape=(n_populations, n_neurons_per, n_enc_per_population,
                    signal_size),
                dtype='float32')

        self.encoders_signal_idx = array.zeros(queue,
                shape=(n_populations, n_enc_per_population),
                dtype='int32')

        self.decoders = array.zeros(queue,
                shape=(n_signals, signal_size, n_dec_per_signal,
                    n_neurons_per),
                dtype='float32')

        self.decoders_population_idx = array.zeros(queue,
                shape=(n_signals, n_dec_per_signal),
                dtype='int32')

    def realloc_for_GPU(self):
        self.realloc_encoders_for_GPU()
        self.realloc_decoders_for_GPU()

    def realloc_encoders_for_GPU(self):
        encoders = self.encoders.get(self.queue)
        encoders = encoders.transpose(3, 2, 0, 1).copy()
        cmajor_cl_encoders = array.to_device(self.queue, encoders)
        self.encoders = cmajor_cl_encoders.transpose(2, 3, 1, 0)

    def realloc_decoders_for_GPU(self):
        decoders = self.decoders.get(self.queue)
        decoders = decoders.transpose(3, 2, 0, 1).copy()
        cmajor_cl_decoders = array.to_device(self.queue, decoders)
        self.decoders = cmajor_cl_decoders.transpose(2, 3, 1, 0)

    def neuron_plan(self, dt):
        # XXX add support for built-in filtering
        rval = plan_lif(self.queue,
                V=self.lif_v,
                RT=self.lif_rt,
                J=self.lif_ic,
                OV=self.lif_v,
                ORT=self.lif_rt,
                OS=self.lif_output,
                OSfilt=self.lif_output_filter,
                dt=dt,
                tau_rc=self.lif_tau_rc,
                tau_ref=self.lif_tau_ref,
                V_threshold=self.lif_V_threshold,
                upsample=self.lif_upsample,
                pstc=self.lif_output_pstc,
                )
        return rval

    def decoder_plan(self, signals_beta):
        # XXX: do not overwrite the entire signals vector
        #       when decoding, some of the signals are not
        #       decoded from these populations.
        if self.n_dec_per_signal != 1:
            raise NotImplementedError()
        rval = plan_misc_gemv(self.queue,
                alpha=1.0,
                A=self.decoders[:, :, 0],
                X=self.lif_output_filter,
                Xi=self.decoders_population_idx[:, 0],
                beta=signals_beta,
                Y=self.signals,
                tag='decoders')
        return rval

    def encoder_plan(self):
        if self.n_enc_per_population != 1:
            raise NotImplementedError()
        rval = plan_misc_gemv(self.queue,
                alpha=1.0,
                A=self.encoders[:, :, 0],
                X=self.signals,
                Xi=self.encoders_signal_idx[:, 0],
                beta=1.0,
                Y_in=self.lif_bias,
                Y=self.lif_ic,
                tag='encoders')
        return rval

    def prog(self, dt):
        """
        A single loop through the simulator does the following things in order:

        """
        return Prog([
            self.encoder_plan(),
            self.neuron_plan(dt=dt),
            self.decoder_plan(signals_beta=0.0),
        ])


def test_one_random():
    queue = cl.CommandQueue(ctx,
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    n_ensembles = 6000
    size = 100
    rank = 2
    mdl = LIFMultiEnsemble(
            n_populations=n_ensembles,
            n_neurons_per=size,
            n_signals=n_ensembles,
            signal_size=rank,
            lif_upsample=2,
            queue=queue)

    # -- re-arrange memory for GPU
    encoders = np.random.randn(*mdl.encoders.shape)
    mdl.encoders.set(encoders.astype('float32'))

    # -- re-arrange memory for GPU
    decoders = np.random.randn(*mdl.decoders.shape)
    mdl.decoders.set(decoders.astype('float32'))

    # -- set each population to be self-connected
    idx = np.arange(n_ensembles, dtype='int32')
    mdl.decoders_population_idx.set(idx)
    mdl.encoders_signal_idx.set(idx)

    mdl.realloc_encoders_for_GPU()
    mdl.realloc_decoders_for_GPU()

    prog = mdl.prog(dt=0.001)

    prog.call_n_times(1000)

    clrk = cl.enqueue_nd_range_kernel
    qs, ks, gs, ls = prog.queues, prog.kerns, prog.gsize, prog.lsize
    evs = []
    for ii in range(250):
        evs.append(map(clrk, qs, ks, gs, ls))
    evs[-1][-1].wait()

    ctime = {}
    for evlist in evs:
        for ev, plan in zip(evlist, prog.plans):
            ctime.setdefault(plan, 0)
            ctime[plan] += ev.profile.end - ev.profile.start

    for plan, ticks in ctime.items():
        print ticks * 1e-9, plan


def test_speed():
    """
    Produce some speed comparisons with the Java Nengo software.

    (This was done with nengo-1.4, in prep for Frontiers article.)
    """
    queue = cl.CommandQueue(ctx)
    nengo_1s = {}
    nengo_1s[(10, 10, 1)] = 0.802000045776
    nengo_1s[(10, 10, 2)] = 0.666999816895
    nengo_1s[(10, 10, 50)] = 1.08800005913
    nengo_1s[(10, 100, 1)] = 1.15799999237
    nengo_1s[(10, 100, 2)] = 1.37699985504
    nengo_1s[(10, 100, 50)] = 1.72799992561

    nengo_1s[(100, 10, 1)] = 1.28299999237
    nengo_1s[(100, 10, 2)] = 1.36100006104
    nengo_1s[(100, 10, 50)] = 3.16299986839
    nengo_1s[(100, 100, 1)] = 6.14800000191
    nengo_1s[(100, 100, 2)] = 6.14300012589
    nengo_1s[(100, 100, 50)] = 9.89700007439

    nengo_1s[(1000, 10, 1)] = 7.7009999752
    nengo_1s[(1000, 10, 2)] = 8.11899995804
    nengo_1s[(1000, 10, 50)] = 25.9370000362
    nengo_1s[(1000, 100, 1)] = 50.736000061
    nengo_1s[(1000, 100, 2)] = 52.0779998302
    nengo_1s[(1000, 100, 50)] = 76.2179999352
    for n_ensembles in [10, 100, 1000, 10000]:
        for size in [10, 100, 1000]:
            for rank in [1, 2, 50]:
                if n_ensembles * size * rank > 10 * 1000 * 1000:
                    continue
                mdl = LIFMultiEnsemble(
                        n_populations=n_ensembles,
                        n_neurons_per=size,
                        n_signals=n_ensembles,
                        signal_size=rank,
                        lif_upsample=2,
                        queue=queue)

                encoders = np.random.randn(*mdl.encoders.shape)
                decoders = np.random.randn(*mdl.decoders.shape)
                mdl.encoders.set(encoders.astype('float32'))
                mdl.decoders.set(decoders.astype('float32'))
                idx = np.arange(n_ensembles, dtype='int32')
                mdl.decoders_population_idx.set(idx)
                mdl.encoders_signal_idx.set(idx)

                mdl.realloc_encoders_for_GPU()
                mdl.realloc_decoders_for_GPU()

                prog = mdl.prog(dt=0.001)

                prog() # once for allocation & warmup
                t0 = time.time()
                prog.call_n_times(1000)
                t1 = time.time()
                elapsed = (t1 - t0)

                key = (n_ensembles, size, rank)
                if key in nengo_1s:
                    nengo_walltime = nengo_1s[key]
                    speedup = nengo_walltime / elapsed
                else:
                    nengo_walltime = 'N/A'
                    speedup = 'N/A'

                print '%8i %8i %8i: %10i neurons took %4.4f (vs. %s, %sx) seconds' %(
                        n_ensembles, size, rank, n_ensembles * size,
                        elapsed, nengo_walltime, speedup)

