import time

import matplotlib.pyplot as plt
import numpy as np
import pyopencl as cl

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

from base import LIFMultiEnsemble
from nengo import nef_theano as nef


def test_speed():
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

                mdl._randomize_decoders(rng=np.random)
                mdl._randomize_encoders(rng=np.random)
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

def make_net(N):
    net = nef.Network('Runtime Test', seed=123)
    print 'make_input'
    net.make_input('in', value=np.sin)
    print 'make A'
    net.make('A', 1000, 1)
    print 'make B'
    net.make('B', 1000, 1)
    print 'make C'
    net.make('C', 1000, 1)
    print 'make D'
    net.make('D', 1000, 1)

    # some functions to use in our network
    def pow(x):
        return [xval**2 for xval in x]

    def mult(x):
        return [xval*2 for xval in x]

    print 'connecting in -> A'
    net.connect('in', 'A')
    print 'connecting other ones...'
    net.connect('A', 'B')
    net.connect('A', 'C', func=pow)
    net.connect('A', 'D', func=mult)
    # -- don't support this yet because
    #    plan_misc_gemv doesn't support multiple decoders per signal
    #    (see #XXX)
    #net.connect('D', 'B', func=pow) # throw in some recurrency whynot

    Ap = net.make_probe('A', dt_sample=.01, pstc=0.01)
    Bp = net.make_probe('B', dt_sample=.01)

    return net, Ap, Bp


def test_ref():
    # Makes the net and runs it
    net, Ap, Bp = make_net(1000)

    t0 = time.time()
    net.run(1.0)
    t1 = time.time()
    print 'time', (t1 - t0)

    plt.plot(Ap.get_data())
    plt.show()


def test_net_by_hand():
    net, Ap, Bp = make_net(1000)
    net.run(0.01)

    lme = LIFMultiEnsemble(
            n_populations=4,
            n_neurons_per=1000,
            n_signals=4,
            signal_size=1,
            queue=queue,
            )
    pop = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
    }
    sig = {
        'sin(t)': 0,
        'A.X': 1,
        'A.mult': 2,
        'A.pow': 3,
    }

    def get_decoders(obj, signalname):
        origin = net.get_object(obj).origin[signalname]
        rval = origin.decoders.get_value().astype('float32')
        r = origin.ensemble.radius
        rval = rval * r / net.dt
        rval.shape = rval.shape[:-1]
        return rval

    def get_encoders(obj):
        ensemble = net.get_object(obj)
        encoders = ensemble.shared_encoders.get_value().astype('float32')
        # -- N.B. shared encoders already have "alpha" factored in
        return encoders

    def get_bias(obj):
        ensemble = net.get_object(obj)
        return ensemble.bias.astype('float32')

    # -- bring in neuron bias terms
    bias = lme.lif_bias.get()
    for name in 'A', 'B', 'C', 'D':
        bias[pop[name], :] = get_bias(name)
    lme.lif_bias.set(bias)

    # -- bring in encoders
    encoders = lme.encoders.get()
    for name in 'A', 'B', 'C', 'D':
        encoders[pop[name], :, 0, :] = get_encoders(name)
    lme.encoders.set(encoders)

    # -- enumerate incoming signals (for encoding)
    signal_idx = lme.encoders_signal_idx.get()
    signal_idx[pop['A'], 0] = sig['sin(t)']
    signal_idx[pop['B'], 0] = sig['A.X']
    signal_idx[pop['C'], 0] = sig['A.pow']
    signal_idx[pop['D'], 0] = sig['A.mult']
    lme.encoders_signal_idx.set(signal_idx)

    # -- bring in decoder weights
    decoders = lme.decoders.get()
    # -- leave decoders['sin(t)'] at zero
    #decoders[0, :, 0, :] = net.get_object('A').decoders
    decoders[sig['A.X'], :, 0, :] = get_decoders('A', 'X')
    decoders[sig['A.pow'], :, 0, :] = get_decoders('A', 'pow')
    decoders[sig['A.mult'], :, 0, :] = get_decoders('A', 'mult')
    lme.decoders.set(decoders)

    # -- enumerate incoming populations (for decoding)
    pop_idx = lme.decoders_population_idx.get()
    pop_idx[sig['sin(t)'], 0] = 0 # sin   <- N/A
    pop_idx[sig['A.X'], 0] = pop['A']
    pop_idx[sig['A.pow'], 0] = pop['A']
    pop_idx[sig['A.mult'], 0] = pop['A']
    lme.decoders_population_idx.set(pop_idx)

    signals = []
    spikes = []
    prog = lme.prog(dt=net.dt)
    signals_t = lme.signals.get()
    t0 = time.time()
    for ii in range(1000):
        signals_t[0] = np.sin(ii / 1000.0)
        lme.signals.set(signals_t, queue=queue)
        prog()
        if ii % 10 == 0:
            queue.finish()
            # N.B. prog currently overwrites the signals_t[0]
            #      with 0 as the last thing it does, while
            #      writing the other 3 signals. That's not
            #      ideal, but a known issue.
            signals_t = lme.signals.get(queue=queue)
            signals_t[0] = np.sin(ii / 1000.0)
            signals.append(signals_t.copy())
            spikes.append(lme.lif_output_filter.get(queue=queue))
    queue.finish()
    t1 = time.time()
    print 'time', (t1 - t0)

    signals = np.asarray(signals)
    spikes = np.asarray(spikes)

    plt.subplot(2, 1, 1)

    plt.title('4 signals over time')
    plt.plot(signals[:, 0], label='zero * sin(t)')
    plt.plot(signals[:, 1], label='Asin(t)')
    plt.plot(signals[:, 2], label='Asin(t)^2')
    plt.plot(signals[:, 3], label='Asin(t)*2')
    plt.legend(loc='upper left')
    plt.subplot(2, 1, 2)

    plt.title('spikes')
    plt.plot(spikes[:, 0, 0])
    plt.plot(spikes[:, 0, 1])
    plt.plot(spikes[:, 0, 2])
    plt.plot(spikes[:, 0, 3])
    plt.plot(spikes[:, 0, 4])
    plt.show()

