import numpy as np

from nengo import nef_theano as nef
from base import Model
from base import net_get_decoders
from base import net_get_bias
from base import net_get_encoders

from sim_npy import Simulator

def make_net(N):
    """
    Helper method for other tests - constructs a simple
    network with 4 populations and a few toy signals.
    """
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

    return net, Ap, Bp

def test_probe_with_base():
    net, Ap, Bp = make_net(1000)
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

    print sim.signal(sint)
    print sim.signal(Adec)


