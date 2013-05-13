import time

import matplotlib.pyplot as plt
import numpy as np
import pyopencl as cl

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

from base import LIFMultiEnsemble
from nengo import nef_theano as nef


def test_speed():
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
                        queue=queue)

                mdl._randomize_decoders(rng=np.random)
                mdl._randomize_encoders(rng=np.random)
                prog = mdl.prog(dt=0.001)

                prog() # once for allocation & warmup
                t0 = time.time()
                for i in range(200):
                    map(cl.enqueue_nd_range_kernel,
                        prog.queues, prog.kerns, prog.gsize, prog.lsize)
                queue.finish()
                t1 = time.time()
                elapsed = (t1 - t0)
                print '%8i %8i %8i: %i neurons took %s seconds' %(
                        n_ensembles, size, rank, n_ensembles * size, elapsed)

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

    lme = LIFMultiEnsemble(
            n_populations=4,
            n_neurons_per=1000,
            n_signals=4,
            signal_size=1,
            queue=queue,
            )

    if 0:
        print 'A encoded_input'
        print net.get_object('A').encoded_input

        print 'A decoded_input'
        print net.get_object('A').decoded_input

        print 'A origin'
        print net.get_object('A').origin['X'].decoders.get_value()

        print 'B encoded_input'
        print net.get_object('B').encoded_input

        print 'B decoded_input'
        print net.get_object('B').decoded_input

    def get_decoders(obj, signalname):
        origin = net.get_object(obj).origin[signalname]
        rval = origin.decoders.get_value().astype('float32')
        rval.shape = rval.shape[:-1]
        return rval

    def get_encoders(obj):
        ensemble = net.get_object(obj)
        encoders = ensemble.shared_encoders.get_value().astype('float32')
        alpha = ensemble.alpha.astype('float32')
        print 'encoders', encoders.shape, alpha.shape
        return alpha[:, :, None] * encoders

    def get_bias(obj):
        ensemble = net.get_object(obj)
        return ensemble.bias.astype('float32')

    bias = lme.lif_bias.get()
    net.get_object('A').encoded_input
    bias[0, :] = get_bias('A')
    bias[1, :] = get_bias('B')
    bias[2, :] = get_bias('C')
    bias[3, :] = get_bias('D')
    lme.lif_bias.set(bias)

    encoders = lme.encoders.get()
    net.get_object('A').encoded_input
    encoders[0, :, 0, :] = get_encoders('A')
    encoders[1, :, 0, :] = get_encoders('B')
    encoders[2, :, 0, :] = get_encoders('C')
    encoders[3, :, 0, :] = get_encoders('D')
    lme.encoders.set(encoders)

    signal_idx = lme.encoders_signal_idx.get()
    signal_idx[0, 0] = 0 # A <- in
    signal_idx[1, 0] = 1 # B <- Adec
    signal_idx[2, 0] = 2 # C <- Apow
    signal_idx[3, 0] = 3 # D <- Amult
    lme.encoders_signal_idx.set(signal_idx)

    decoders = lme.decoders.get()
    # -- leave them at zero
    #decoders[0, :, 0, :] = net.get_object('A').decoders
    decoders[1, :, 0, :] = get_decoders('A', 'X')
    decoders[2, :, 0, :] = get_decoders('A', 'pow')
    decoders[3, :, 0, :] = get_decoders('A', 'mult')
    lme.decoders.set(decoders)

    pop_idx = lme.decoders_population_idx.get()
    pop_idx[0, 0] = 0 # sin   <- N/A
    pop_idx[1, 0] = 1 # Adec  <- A
    pop_idx[2, 0] = 1 # Apow  <- A
    pop_idx[3, 0] = 1 # Amult <- A
    lme.decoders_population_idx.set(pop_idx)

    signals = []
    prog = lme.prog(dt=net.dt)
    signals_t = lme.signals.get()
    t0 = time.time()
    print 'signals', lme.signals.data
    for ii in range(1000):
        signals_t[0] = np.sin(ii / 1000.0)
        lme.signals.set(signals_t)
        prog()
        queue.finish()
        signals_t = lme.signals.get()
        signals.append(signals_t)
    t1 = time.time()
    print 'time', (t1 - t0)

    signals = np.asarray(signals)
    plt.plot(signals[:, 0, 0])
    plt.plot(signals[:, 1, 0])
    plt.show()

