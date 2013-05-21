
import numpy as np

class DuplicateFilter(Exception):
    pass


def allsame(seq):
    return len(set(seq)) < 2


class Signal(object):
    def __init__(self, n=1):
        self.n = n


class Population(object):
    def __init__(self, n):
        self.n = n


class Transform(object):
    def __init__(self, insig, outsig):
        self.insig = insig
        self.outsig = outsig


class Filter(object):
    def __init__(self, sig):
        self.sig = sig


class Encoder(object):
    def __init__(self, sig, pop):
        self.sig = sig
        self.pop = pop
        assert isinstance(sig, Signal)
        assert isinstance(pop, Population)


class Decoder(object):
    def __init__(self, pop, sig):
        self.pop = pop
        self.sig = sig

class Model(object):
    def __init__(self):
        self.signals = []
        self.populations = []
        self.encoders = []
        self.decoders = []
        self.transforms = []
        self.filters = []

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

    def transform(self, insig, outsig):
        rval = Transform(insig, outsig)
        self.transforms.append(rval)
        return rval

    def filter(self, insig, outsig):
        rval = Filter(insig, outsig)
        self.filters.append(rval)
        return rval



class RaggedArray(object):
    # a linear buffer that is partitioned into
    # sections of various lengths.
    # 
    def __init__(self, listoflists):

        starts = []
        lens = []
        buf = []

        for l in listoflists:
            starts.append(len(buf))
            lens.append(len(l))
            buf.extend(l)

        self.starts = starts
        self.lens = lens
        self.buf = np.asarray(buf)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, item):
        o = self.starts[item]
        n = self.lens[item]
        rval = self.buf[o: o + n]
        if len(rval) != n:
            raise ValueError('buf not long enough')
        return rval

    def __setitem__(self, item, val):
        o = self.starts[item]
        n = self.lens[item]
        if len(val) != n:
            raise ValueError('wrong len')
        rval = self.buf[o: o + n]
        if len(rval) != n:
            raise ValueError('buf not long enough')
        self.buf[o: o + n] = val


def ragged_gather_gemv(Ms, Ns, alpha, A, A_js, X, X_js,
                       beta, Y, Y_in=None):
    """
    """
    try:
        float(alpha)
        alpha = [alpha] * len(Y)
    except TypeError:
        pass

    try:
        float(beta)
        beta = [beta] * len(Y)
    except TypeError:
        pass

    if Y_in is None:
        Y_in = Y

    for i in xrange(len(Y)):
        try:
            y_i = beta[i] * Y_in[i]  # -- ragged getitem
        except:
            print i, beta, Y_in
            raise
        alpha_i = alpha[i]

        x_js_i = X_js[i] # -- ragged getitem
        A_js_i = A_js[i] # -- ragged getitem
        
        for xi, ai in zip(x_js_i, A_js_i):
            x_ij = X[xi] # -- ragged getitem
            M_i = Ms[ai]
            N_i = Ns[ai]
            assert N_i == len(x_ij)
            A_ij = A[ai].reshape(N_i, M_i) # -- ragged getitem
            try:
                y_i += alpha_i * np.dot(x_ij, A_ij.reshape(N_i, M_i))
            except:
                print i, xi, ai, A_ij, x_ij
                raise

        Y[i] = y_i


class Simulator(object):
    def __init__(self, model):
        self.model = model
        self.sidx = dict((s, i) for (i, s) in enumerate(model.signals))
        self.pidx = dict((p, i) for (i, p) in enumerate(model.populations))
        self.eidx = dict((s, i) for (i, s) in enumerate(model.encoders))
        self.didx = dict((p, i) for (i, p) in enumerate(model.decoders))

        if not all(s.n == 1 for s in self.model.signals):
            raise NotImplementedError()

    def alloc_signals(self):
        self.sigs = RaggedArray([[0] for s in self.model.signals])
        
        # filtered, transformed signals
        self.tf_sigs = RaggedArray([[0] for s in self.model.signals])

    def alloc_populations(self):
        def pop_props():
            return RaggedArray(
                [[0] * p.n for p in self.model.populations])
        # -- lif-specific stuff
        self.pop_ic = pop_props()
        self.pop_jbias = pop_props()
        self.pop_voltage = pop_props()
        self.pop_rt = pop_props()
        self.pop_output = pop_props()

    def alloc_filters_and_transforms(self):
        transforms = self.model.transforms
        filters = self.model.filters
        signals = self.model.signals

        filtersig = {}
        for filt in filters:
            if filt.sig in filtersig:
                raise DuplicateFilter('signal has multiple filters',
                                     filt.sig)
            # XXX use exp(), dt etc. to calculate filter coef
            filtersig[filt.sig] = .01

        self.tf_alpha = np.asarray([1 - filtersig.get(sig, 0)
                                    for sig in signals])
        self.tf_beta = np.asarray([filtersig.get(sig, 0)
                                   for sig in signals])

        self.tf_weights = RaggedArray([[0.1] for tf in transforms])
        self.tf_Ns = [1] * len(transforms)
        self.tf_Ms = [1] * len(transforms)

        # -- which transform(s) does each signal use
        tidx = dict((tf, i) for (i, tf) in enumerate(transforms))
        self.tf_weights_js = RaggedArray([
            [tidx[tf] for tf in transforms if tf.outsig == sig]
            for sig in self.model.signals
            ])

        # -- which corresponding(s) signal is transformed
        self.tf_signals_js = RaggedArray([
            [self.sidx[tf.insig] for tf in transforms if tf.outsig == sig]
            for sig in self.model.signals
            ])


    def alloc_encoders(self):
        encoders = self.model.encoders
        self.enc_weights = RaggedArray(
            [np.random.randn(enc.pop.n) for enc in encoders])

        self.enc_Ms = [enc.pop.n for enc in encoders]
        self.enc_Ns = [1] * len(encoders)

        # -- which encoder(s) does each population use
        self.enc_weights_js = [
            [self.eidx[enc] for enc in encoders if enc.pop == pop]
            for pop in self.model.populations]

        # -- and which corresponding signal does it encode
        self.enc_signals_js = [
            [self.sidx[enc.sig]
                for enc in encoders if enc.pop == pop]
            for pop in self.model.populations]

    def alloc_decoders(self):
        decoders = self.model.decoders
        self.dec_weights = RaggedArray(
            [np.random.randn(dec.pop.n) for dec in decoders])
        self.dec_Ms = [1] * len(decoders)
        self.dec_Ns = [dec.pop.n for dec in decoders]

        # -- which decoder(s) does each signal use
        self.dec_weights_js = [
            [self.didx[dec] for dec in decoders if dec.sig == sig]
            for sig in self.model.signals]

        # -- and which corresponding population does it decode
        self.dec_pops_js = [
            [self.pidx[dec.pop]
                for dec in decoders if dec.sig == sig]
            for sig in self.model.signals]

    def alloc_all(self):
        self.alloc_signals()
        self.alloc_populations()
        self.alloc_filters_and_transforms()
        self.alloc_encoders()
        self.alloc_decoders()


    def do_signal_transform(self):
        ragged_gather_gemv(
            Ms=self.tf_Ms,
            Ns=self.tf_Ns,
            alpha=self.tf_alpha,
            A=self.tf_weights,
            A_js=self.tf_weights_js,
            X=self.sigs,
            X_js=self.tf_signals_js,
            beta=self.tf_beta,
            Y=self.tf_sigs,
            )

    def do_encoders(self):
        ragged_gather_gemv(
            Ms=self.enc_Ms,
            Ns=self.enc_Ns,
            alpha=1.0,
            A=self.enc_weights,
            A_js=self.enc_weights_js,
            X=self.tf_sigs,
            X_js=self.enc_signals_js,
            beta=1.0,
            Y_in=self.pop_jbias,
            Y=self.pop_ic,
            )

    def do_populations(self):
        pass

    def do_decoders(self):
        ragged_gather_gemv(
            Ms=self.dec_Ms,
            Ns=self.dec_Ns,
            alpha=1.0,
            A=self.dec_weights,
            A_js=self.dec_weights_js,
            X=self.pop_output,
            X_js=self.dec_pops_js,
            beta=0.0,
            Y=self.sigs,
            )


    def step(self):
        self.do_signal_transform()
        self.do_encoders()
        self.do_populations()
        self.do_decoders()



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
                m.transform(A_dec[(i, j)], C_in[(i, j, k)])
                m.transform(B_dec[(i, j)], C_in[(i, j, k)])
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
            for k in range(D2):
                m.transform(C_dec[(i, k, j)], D_in[(i, j)])
            m.encoder(D_in[(i, j)], D[(i, j)])
            m.decoder(D[(i, j)], D_dec[(i, j)])

    sim = Simulator(m)

    sim.alloc_all()
    sim.step()
