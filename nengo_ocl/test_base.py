import numpy as np

from nengo import nef_theano as nef
from base import Model
from base import net_get_decoders
from base import net_get_bias
from base import net_get_encoders
from base import nengo2model

from sim_npy import Simulator




def test_nengo2model1():

    net = nef.Network('Runtime Test', dt=0.001, seed=123)
    print 'make_input'
    net.make_input('in', value=np.sin)
    print 'make A'
    net.make('A', 1000, 1)
    print 'connecting in -> A'
    net.connect('in', 'A')
    net_A_probe = net.make_probe('A', dt_sample=0.01, pstc=0.01)

    model, memo = nengo2model(net)
    sim = Simulator(model)

    net.run(1.0)
    sim.run_steps(1000)

    Adec = memo(net.get_object('A.X'))

    sim_data = sim.signal(Adec)
    net_data = net_A_probe.get_data()

    assert sim_data.shape == net_data.shape
    assert np.allclose(sim_data, net_data)




def test_whole_model_1(show=False):
    #
    # Step 1. BUILD A MODEL IN NENGO_THEANO
    #
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
    net.connect('D', 'B', func=pow) # throw in some recurrency

    Ap = net.make_probe('A', dt_sample=.01, pstc=0.01)
    Bp = net.make_probe('B', dt_sample=.01)

    #
    # Step 2. BUILD THE SAME MODEL WITH NENGO_OCL
    #
    def get_bias(*a):
        bias = net_get_bias(net, *a)
        if bias.shape[0] != 1:
            raise NotImplementedError()
        return bias[0]
    def get_encoders(*a):
        encoders = net_get_encoders(net, *a)
        if encoders.shape[0] != 1:
            raise NotImplementedError()
        return encoders[0]

    get_decoders = lambda *a: net_get_decoders(net, *a)

    m = Model(net.dt)
    one = m.signal(value=1.0)
    steps = m.signal()
    simtime = m.signal()
    sint = m.signal()
    Adec = m.signal()
    Amult = m.signal()
    Apow = m.signal()
    Bin = m.signal()
    Bdec = m.signal()
    Cdec = m.signal()
    Ddec = m.signal()
    Dpow = m.signal()

    A = m.population(n=1000, bias=get_bias('A'))
    B = m.population(n=1000, bias=get_bias('B'))
    C = m.population(n=1000, bias=get_bias('C'))
    D = m.population(n=1000, bias=get_bias('D'))

    # set up linear filters (exp decay)
    # for the signals
    # XXX use their pstc constants
    pstc = 0.01
    decay = np.exp(-m.dt / pstc)
    m.filter(decay, Adec, Adec)
    m.transform(1 - decay, Adec, Adec)

    m.filter(decay, Amult, Amult)
    m.transform(1 - decay, Amult, Amult)

    m.filter(decay, Apow, Apow)
    m.transform(1 - decay, Apow, Apow)

    m.filter(decay, Dpow, Dpow)
    m.transform(1 - decay, Dpow, Dpow)

    # -- hold all constants on the line
    m.filter(1.0, one, one)

    # -- steps counts by 1.0
    m.filter(1.0, steps, steps)
    m.filter(1.0, one, steps)

    # simtime <- dt * steps
    m.filter(net.dt, steps, simtime)

    # combine decoded [current] inputs to B
    m.transform(1 - decay, Adec, Bin)
    m.transform(1 - decay, Dpow, Bin)
    # -- then smooth out Bin
    m.filter(decay, Bin, Bin)

    m.custom_transform(np.sin, simtime, sint)

    m.encoder(sint, A, weights=get_encoders('A'))
    m.encoder(Bin, B, weights=get_encoders('B'))
    m.encoder(Adec, C, weights=get_encoders('C'))
    m.encoder(Adec, D, weights=get_encoders('D'))

    m.decoder(A, Adec, weights=get_decoders('A', 'X'))
    m.decoder(B, Bdec, weights=get_decoders('B', 'X'))
    m.decoder(C, Cdec, weights=get_decoders('C', 'X'))
    m.decoder(D, Ddec, weights=get_decoders('D', 'X'))
    m.decoder(A, Apow, weights=get_decoders('A', 'pow'))
    m.decoder(A, Amult, weights=get_decoders('A', 'mult'))
    m.decoder(D, Dpow, weights=get_decoders('D', 'pow'))

    m.signal_probe(sint, dt=0.01)
    m.signal_probe(Adec, dt=0.01)
    m.signal_probe(Apow, dt=0.01)
    m.signal_probe(Amult, dt=0.01)

    sim = Simulator(m, n_prealloc_probes=1000)
    sim.alloc_all()
    for i in range(1000):
        sim.do_all()
        #print 'one', sim.sigs[sim.sidx[one]]
        assert sim.sigs[sim.sidx[one]] == [1.0]
        #print 'simtime', sim.sidx[simtime], sim.sigs[sim.sidx[simtime]]
        assert sim.sigs[sim.sidx[simtime]] == [i * net.dt]
        #print 'sint', sim.sidx[sint], sim.sigs[sim.sidx[sint]]
        assert sim.sigs[sim.sidx[sint]] == [np.sin(i * net.dt)]

    sint_data = sim.signal(sint)
    Adec_data = sim.signal(Adec)
    Apow_data = sim.signal(Apow)
    Amult_data = sim.signal(Amult)

    assert sint_data.shape == (100, 1)
    assert Adec_data.shape == (100, 1)
    assert Apow_data.shape == (100, 1)
    assert Amult_data.shape == (100, 1)

    assert np.allclose(sint_data,
            np.sin(np.arange(100) * .01).reshape(100, 1))

    Adec_mse = np.mean((Adec_data - sint_data) ** 2)
    print 'Adec MSE', Adec_mse
    assert Adec_mse < 0.01  # getting .124 May 9 2013

    if show:
        from matplotlib import pyplot as plt
        plt.title('4 signals over time')
        plt.plot(sim.signal(sint), label='zero * sin(t)')
        plt.plot(sim.signal(Adec), label='Asin(t)')
        plt.plot(sim.signal(Apow), label='Asin(t)^2')
        plt.plot(sim.signal(Amult), label='Asin(t)*2')
        plt.legend(loc='upper left')
        plt.show()




