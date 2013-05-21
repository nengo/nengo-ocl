import numpy as np

def allsame(seq):
    return len(set(seq)) < 2


class Signal(object):
    def __init__(self, n=1, pstc=0):
        self.n = n
        self.pstc = pstc


class Population(object):
    def __init__(self, n):
        self.n = n


class Transform(object):
    def __init__(self, insigs, outsig):
        self.insigs = insigs
        self.outsig = outsig


class Encoder(object):
    def __init__(self, sig, pop):
        self.sig = sig
        self.pop = pop


class Decoder(object):
    def __init__(self, pop, sig):
        self.pop = pop
        self.sig = sig


class RaggedArrayD(object):
    # a linear buffer that is partitioned into
    # sections of various lengths.
    # 
    def __init__(self, keys, lens, buf=None):
        chunks = {}
        offset = 0
        for k, l, in zip(keys, lens):
            chunks[k] = (offset, l)
            offset += l
        self.chunks = chunks
        self.keys = keys
        if buf is None:
            self.buf = np.zeros(offset)
        else:
            self.buf = buf

        self.idx = dict((k, i) for i, k in enumerate(keys))
        self.offsets = dict((k, chunks[k][0]) for k in keys)
        self.lens = dict((k, chunks[k][1]) for k in keys)


class Model(object):
    def __init__(self):
        self.signals = []
        self.populations = []
        self.encoders = []
        self.decoders = []
        self.transforms = []

    def signal(self, *args, **kwargs):
        rval = Signal(*args, **kwargs)
        self.signals.append(rval)
        return rval

    def population(self, *args, **kwargs):
        rval = Population(*args, **kwargs)
        self.populations.append(rval)
        return rval

    def encoder(self, sig, pop):
        rval = Encoder(sig, pop)
        self.encoders.append(rval)
        return rval

    def decoder(self, pop, sig):
        rval = Decoder(pop, sig)
        self.decoders.append(rval)
        return rval

    def transform(self, insigs, outsig):
        rval = Transform(insigs, outsig)
        self.transforms.append(rval)
        return rval


class Simulator(object):
    def __init__(self, model):
        self.model = model

    def alloc_signals(self):
        self.asigs = RaggedArrayD(
            self.model.signals,
            [1] * len(self.model.signals))

    def alloc_populations(self):
        self.apops = RaggedArrayD(
            self.model.populations,
            [p.n for p in self.model.populations])

    def alloc_transforms(self):
        asigs = self.asigs

        isigs = [[asigs.offsets[k]] for k in asigs.keys]
        for tf in self.model.transforms:
            isigs[asigs.idx[tf.outsig]].extend(
                [asigs.idx[insig] for insig in tf.insigs])

        def flatten(ll):
            rval = []
            for l in ll:
                rval.extend(l)
            return rval

        tf_ipos = RaggedArrayD(asigs.keys, map(len, isigs))
        tf_ipos.buf = np.asarray(flatten(isigs), dtype='int')

        tf_weights = RaggedArrayD(asigs.keys, map(len, isigs))
        # XXX
        # the  weights should be the pstc corrected for dt
        # in the case of the self-connections  (filters)
        # and it should be the multipliers on other signals
        # in the case of general transforms
        tf_weights.buf += .1  # -- this should be pstc corrected for dt

        # XXX
        # this is the other part of the filter, which should
        # also be corrected for dt.
        tf_beta = np.zeros(len(isigs)) + .9

        self.tf_ipos = tf_ipos
        self.tf_weights = tf_weights
        self.tf_beta = tf_beta

    def alloc_encoders(self):
        self.enc_weights = RaggedArrayD(
            self.model.encoders,
            [enc.pop.n for enc in self.model.encoders])

    def alloc_decoders(self):
        self.dec_weights = RaggedArrayD(
            self.model.decoders,
            [dec.pop.n for dec in self.model.encoders])

    def alloc_all(self):
        self.alloc_signals()
        self.alloc_populations()
        self.alloc_transforms()
        self.alloc_encoders()
        self.alloc_decoders()





def test_matrix_mult_example():
    # construct good way to do the 
    # examples/matrix_multiplication.py model
    # from nengo_theano

    # Adjust these values to change the matrix dimensions
    #  Input matrix A is D1xD2
    #  Input matrix B is D2xD3
    #  intermediate tensor of products D1 x D2 x D3
    #  result D is D1xD3

    D1 = 1
    D2 = 2
    D3 = 3


    m = Model()

    A = {}
    A_in = {}
    A_dec = {}
    for i in range(D1):
        for j in range(D2):
            A_in[(i, j)] = m.signal()
            A[(i, j)] = m.population(50)
            A_dec[(i, j)] = m.signal()

            m.encoder(A_in[(i, j)], A[(i, j)])
            m.decoder(A[(i, j)], A_dec[(i, j)])

    B = {}
    B_in = {}
    B_dec = {}
    for i in range(D2):
        for j in range(D3):
            B_in[(i, j)] = m.signal()
            B[(i, j)] = m.population(50)
            B_dec[(i, j)] = m.signal()

            m.encoder(B_in[(i, j)], B[(i, j)])
            m.decoder(B[(i, j)], B_dec[(i, j)])

    C = {}
    C_in = {}
    C_dec = {}
    for i in range(D1):
        for j in range(D2):
            for k in range(D3):
                C[(i, j, k)] = m.population(200)
                C_in[(i, j, k)] = m.signal()
                C_dec[(i, j, k)] = m.signal()
                m.transform(
                    [A_dec[(i, j)], B_dec[(j, k)]],
                    C_in[(i, j, k)])
                m.encoder(C_in[(i, j, k)], C[(i, j, k)])
                m.decoder(C[(i, j, k)], C_dec[(i, j, k)])

    D = {}
    D_in = {}
    D_dec = {}
    for i in range(D1):
        for j in range(D3):
            D[(i, j)] = m.population(50)
            D_in[(i, j)] = m.signal()
            D_dec[(i, j)] = m.signal()
            m.transform(
                [C_dec[(i, k, j)] for k in range(D2)],
                D_in[(i, j)])
            m.encoder([D_in[(i, j)]], D[(i, j)])
            m.decoder(D[(i, j)], D_dec[(i, j)])

    sim = Simulator(m)

    sim.alloc_all()
