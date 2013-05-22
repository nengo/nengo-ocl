import numpy as np


class DuplicateFilter(Exception):
    pass


def allsame(seq):
    return len(set(seq)) < 2


class Signal(object):
    def __init__(self, n=1):
        self.n = n


class Constant(Signal):
    def __init__(self, value):
        Signal.__init__(self, len(value))
        self.value = value


class Population(object):
    def __init__(self, n):
        self.n = n


class Transform(object):
    def __init__(self, alpha, insig, outsig):
        self.alpha = alpha
        self.insig = insig
        self.outsig = outsig

class CustomTransform(object):
    def __init__(self, func, insig, outsig):
        self.func = func
        self.insig = insig
        self.outsig = outsig

class Filter(object):
    def __init__(self, sig, beta):
        self.sig = sig
        self.beta = beta


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
        self.custom_transforms = []

    def signal(self, value=None):
        if value is None:
            rval = Signal()
        else:
            rval = Constant([value])
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

    def transform(self, alpha, insig, outsig):
        rval = Transform(alpha, insig, outsig)
        self.transforms.append(rval)
        return rval

    def custom_transform(self, func, insig, outsig):
        rval = CustomTransform(func, insig, outsig)
        self.custom_transforms.append(rval)
        return rval

    def filter(self, sig, beta):
        rval = Filter(sig, beta)
        for f in self.filters:
            if f.sig == sig:
                raise DuplicateFilter(
                        'signal has multiple filters',
                        sig)
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

    print alpha
    print A.buf, 'A'
    print X.buf, 'X'
    print Y_in.buf, 'in'

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
            print xi, x_ij, A_ij
            try:
                y_i += alpha_i * np.dot(x_ij, A_ij.reshape(N_i, M_i))
            except:
                print i, xi, ai, A_ij, x_ij
                raise

        Y[i] = y_i
    print Y.buf, 'out'


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
        self.sigs_ic = RaggedArray([getattr(s, 'value', [0])
            for s in self.model.signals])
        self.sigs = RaggedArray([getattr(s, 'value', [0])
            for s in self.model.signals])
        
        # filtered, transformed signals

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

    def alloc_transforms(self):
        transforms = self.model.transforms
        signals = self.model.signals
        filters = self.model.filters

        sigbeta = {}
        for filt in filters:
            # XXX use exp(), dt etc. to calculate filter coef
            sigbeta[filt.sig] = filt.beta
        for sig in signals:
            sigbeta.setdefault(sig, 0)
        self.filt_beta = np.asarray([sigbeta[sig] for sig in signals]) 

        self.tf_weights = RaggedArray([[tf.alpha] for tf in transforms])
        self.tf_Ns = [1] * len(transforms)
        self.tf_Ms = [1] * len(transforms)

        # -- which transform(s) does each signal use
        tidx = dict((tf, i) for (i, tf) in enumerate(transforms))
        self.tf_weights_js = RaggedArray([
            [tidx[tf] for tf in transforms if tf.outsig == sig]
            for sig in signals
            ])

        # -- which corresponding(s) signal is transformed
        self.tf_signals_js = RaggedArray([
            [self.sidx[tf.insig] for tf in transforms if tf.outsig == sig]
            for sig in signals
            ])

        print self.tf_weights.starts
        print self.tf_weights.lens
        print self.tf_weights.buf
        print self.tf_weights_js
        print self.tf_signals_js


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
        self.alloc_transforms()
        self.alloc_encoders()
        self.alloc_decoders()

    def do_transforms(self):
        # ADD linear combinations of signals to some signals
        ragged_gather_gemv(
            Ms=self.tf_Ms,
            Ns=self.tf_Ns,
            alpha=1.0,
            A=self.tf_weights,
            A_js=self.tf_weights_js,
            X=self.sigs_ic,
            X_js=self.tf_signals_js,
            beta=self.filt_beta,
            Y=self.sigs,
            )

    def do_custom_transforms(self):
        for ct in self.model.custom_transforms:
            ipos = self.sidx[ct.insig]
            opos = self.sidx[ct.outsig]
            self.sigs[opos] = ct.func(self.sigs[ipos])

    def do_encoders(self):
        ragged_gather_gemv(
            Ms=self.enc_Ms,
            Ns=self.enc_Ns,
            alpha=1.0,
            A=self.enc_weights,
            A_js=self.enc_weights_js,
            X=self.sigs,
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

    def do_all(self):
        self.do_transforms()
        self.do_custom_transforms()
        #self.do_encoders()
        #self.do_populations()
        #self.do_decoders()

    def step(self):
        do_all()






