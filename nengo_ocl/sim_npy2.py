"""
numpy Simulator in the style of the OpenCL one, to get design right.
"""
from collections import defaultdict
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
import itertools
import StringIO

import numpy as np

from nengo.objects import Signal
from nengo.objects import LIF, LIFRate, Direct

from sim_npy import ragged_gather_gemv
from sim_npy import RaggedArray

def isview(obj):
    return obj.base is not None and obj.base is not obj


def shape0(obj):
    try:
        return obj.shape[0]
    except IndexError:
        return 1


def shape1(obj):
    try:
        return obj.shape[1]
    except IndexError:
        return 1


def idxs(seq, offset):
    rval = dict((s, i + offset) for (i, s) in enumerate(seq))
    return rval, offset + len(rval)


def islif(obj):
    return isinstance(obj, LIF)


def islifrate(obj):
    return isinstance(obj, LIFRate)


def isconst(sig):
    # XXX has-value should not equate to is-constant
    #     there should be a way to provide initial values
    return hasattr(baseof(sig), 'value')


def baseof(sig):
    while sig.base is not sig:
        if sig.base is None:
            return sig
        sig = sig.base
    return sig


def ivalueof(sig):
    return sig.value


def stable_unique(seq):
    seen = set()
    rval = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            rval.append(item)
    return rval


class ViewBuilder(object):
    def __init__(self, bases, rarray):
        self.bases = bases
        self.sidx = dict((bb, ii) for ii, bb in enumerate(bases))
        assert len(self.bases) == len(self.sidx)
        self.rarray = rarray

        self.starts = []
        self.shape0s = []
        self.shape1s = []
        self.ldas = []
        self.names = []
        #self.orig_len = len(self.all_signals)
        #self.base_starts = self.all_data_A.starts

    def append_view(self, obj):
        if obj in self.bases:
            # -- it is not a view, but OK
            return
        if not isview(obj):
            # -- it is not a view, and not OK
            raise ValueError('can only append views of known signals', obj)

        if obj in self.sidx:
            raise KeyError('sidx already contains object', obj)

        idx = self.sidx[obj.base]
        self.starts.append(self.rarray.starts[idx] + obj.offset)
        self.shape0s.append(shape0(obj))
        self.shape1s.append(shape1(obj))
        if obj.ndim == 0:
            self.ldas.append(0)
        elif obj.ndim == 1:
            self.ldas.append(obj.shape[0])
        elif obj.ndim == 2:
            # -- N.B. the original indexing was
            #    based on ROW-MAJOR storage, and
            #    this simulator uses COL-MAJOR storage
            self.ldas.append(obj.elemstrides[0])
        else:
            raise NotImplementedError()
        self.names.append(getattr(obj, 'name', ''))
        self.sidx[obj] = len(self.sidx)

    def add_views_to(self, rarray):
        rarray.add_views(
            starts=self.starts,
            shape0s=self.shape0s,
            shape1s=self.shape1s,
            ldas=self.ldas,
            names=self.names)


class Simulator(object):

    def RaggedArray(self, *args, **kwargs):
        return RaggedArray(*args, **kwargs)


    def alloc_signal_data(self, sigseq):
        return self.RaggedArray(
            [np.zeros(ss.shape) + getattr(ss, 'value', np.zeros(ss.shape))
                for ss in sigseq],
            names=[getattr(ss, 'name', '') for ss in sigseq])

    def sig_gemv(self, seq, alpha, A_js_fn, X_js_fn, beta, Y_sig_fn,
                 Y_in_sig_fn=None):
        sidx = self.builder.sidx
        Y_sigs = [Y_sig_fn(item) for item in seq]
        if Y_in_sig_fn is None:
            Y_in_sigs = Y_sigs
        else:
            Y_in_sigs = [Y_in_sig_fn(item) for item in seq]
        Y_idxs = [sidx[sig] for sig in Y_sigs]
        Y_in_idxs = [sidx[sig] for sig in Y_in_sigs]

        # -- The following lines illustrate what we'd *like* to see...
        #
        # A_js = self.RaggedArray(
        #   [[sidx[ss] for ss in A_js_fn(item)] for item in seq])
        # X_js = self.RaggedArray(
        #   [[sidx[ss] for ss in X_js_fn(item)] for item in seq])
        #
        # -- ... but the simulator supports broadcasting. So in fact whenever
        #    a signal in X_js has shape (N, 1), the corresponding A_js signal
        #    can have shape (M, N) or (1, 1).
        #    Fortunately, scalar multiplication of X by A can be seen as
        #    Matrix multiplication of A by X, so we can still use gemv,
        #    we just need to reorder and transpose.
        A_js = []
        X_js = []
        for ii, item in enumerate(seq):
            A_js_i = []
            X_js_i = []
            A_sigs_i = A_js_fn(item)
            X_sigs_i = X_js_fn(item)
            assert len(A_sigs_i) == len(X_sigs_i)
            ysig = Y_sigs[ii]
            yidx = Y_idxs[ii]
            yM = self.all_data.shape0s[yidx]
            yN = self.all_data.shape1s[yidx]
            for asig, xsig in zip(A_sigs_i, X_sigs_i):
                aidx = sidx[asig]
                xidx = sidx[xsig]
                aM = self.all_data.shape0s[aidx]
                aN = self.all_data.shape1s[aidx]
                xM = self.all_data.shape0s[xidx]
                xN = self.all_data.shape1s[xidx]
                if aN == aM == 1:
                    A_js_i.append(xidx)
                    X_js_i.append(aidx)
                else:
                    if aN != xM:
                        raise ValueError('shape mismatch in sig_gemv',
                                         ((asig, aM, aN),
                                          (xsig, xM, xN),
                                          (ysig, yM, yN),
                                         ))
                    A_js_i.append(aidx)
                    X_js_i.append(xidx)

            A_js.append(A_js_i)
            X_js.append(X_js_i)


        return ragged_gather_gemv(
            Ms=self.all_data.shape0s,
            Ns=self.all_data.shape1s,
            alpha=alpha,
            A=self.all_data, A_js=A_js,
            X=self.all_data, X_js=X_js,
            beta=beta,
            Y=self.all_data[Y_idxs],
            Y_in=self.all_data[Y_in_idxs],
            )


    @staticmethod
    def orig_relevant_signals(model):
        for enc in model.encoders:
            yield (enc.weights_signal)
            yield (enc.sig)
            yield (enc.pop.input_signal)

        for nl in model.nonlinearities:
            yield nl.bias_signal
            yield nl.input_signal
            yield nl.output_signal

        for dec in model.decoders:
            yield (dec.weights_signal)
            yield (dec.sig)
            yield (dec.pop.input_signal)

        for filt in model.filters:
            yield (filt.alpha_signal)
            yield (filt.oldsig)
            yield (filt.newsig)

        for tf in model.transforms:
            yield (tf.alpha_signal)
            yield (tf.insig)
            yield (tf.outsig)


    def signals_iter(self):
        model = self.model

        for nl in model.nonlinearities:
            yield nl.bias_signal
            yield nl.input_signal
            yield nl.output_signal
            if isinstance(nl, Direct):
                pass
            elif isinstance(nl, LIF):
                yield self.lif_voltage[nl]
                yield self.lif_reftime[nl]
            elif isinstance(nl, LIFRate):
                pass

        for filt in model.filters:
            yield (filt.alpha_signal)
            yield (filt.oldsig)
            yield (filt.newsig)

        for tf in model.transforms:
            yield (tf.alpha_signal)
            yield (tf.insig)
            yield (tf.outsig)

        for enc in model.encoders:
            yield (enc.weights_signal)
            yield (enc.sig)
            yield (enc.pop.input_signal)

        for dec in model.decoders:
            yield dec.weights_signal
            yield dec.sig
            yield dec.pop.input_signal
        for sig in self.dec_outputs:
            yield sig

        for sig in self.outbufs:
            yield self.outbufs[sig]


    def __init__(self, model, n_prealloc_probes=1000, dtype='float32'):
        self.model = model
        self.n_prealloc_probes = n_prealloc_probes
        self.dtype = dtype
        self.bases = []
        self.sidx = []
        self.sim_step = 0

        self.lif_voltage = {}
        self.lif_reftime = {}
        self.dec_outputs = {}
        self.filter_outputs = {}    # -- values also in outbufs
        self.transform_outputs = {} # -- values also in outbufs
        self.outbufs = {}

        self.probe_output = dict((probe, []) for probe in model.probes)


        # create at least one buffer for every signal
        bases = stable_unique(baseof(sig) for sig in
                    self.orig_relevant_signals(model))

        def add_base_like(sig, suffix=''):
            N, = sig.shape
            a = Signal(N, name=sig.name + suffix)
            bases.append(a)
            return a

        # -- add some extra buffers for some signals:
        #    reset bias -> input current can be done in-place
        #    encoders -> input current can be done in-place
        #    nl outputs -> generally needs buffer
        for nl in model.nonlinearities:

            # -- also some neuron models need a few extra buffers
            if isinstance(nl, Direct):
                pass
            elif isinstance(nl, LIF):
                self.lif_voltage[nl] = add_base_like(nl.output_signal)
                self.lif_reftime[nl] = add_base_like(nl.output_signal)
            elif isinstance(nl, LIFRate):
                pass 
            else:
                raise NotImplementedError()

        #    decoder outputs -> also generally needs copy
        decoder_outputs_set = set(dec.sig for dec in model.decoders)
        self.decoder_outputs = [sig
                           for sig in self.orig_relevant_signals(self.model)
                           if sig in decoder_outputs_set]
        for sig in self.decoder_outputs:
            self.dec_outputs[sig] = add_base_like(sig)

        #    generally, filters and transforms are meant to
        #    write into a fresh "output buffer"
        #    which is then copied back over top of the old values
        #    There are cases where more can be done in-place, but we'll
        #    just do the general case for now.
        filt_and_tf_outputs = (
            [filt.newsig for filt in model.filters]
            + [tf.outsig for tf in model.transforms])
        for sig in stable_unique(filt_and_tf_outputs):
            self.outbufs[sig] = add_base_like(sig)

        for filt in self.model.filters:
            self.filter_outputs[filt] = self.outbufs[filt.newsig]
        for tf in self.model.transforms:
            self.transform_outputs[tf] = add_base_like(tf.outsig)



        # -- Choose a layout order for the constants.
        bases = sorted(bases, key=lambda bb: bb.size)

        self.bases[:] = bases
        self.all_data = self.alloc_signal_data(self.bases)
        self.builder = ViewBuilder(self.bases, self.all_data)
        for sig in stable_unique(self.signals_iter()):
            self.builder.append_view(sig)
        self.builder.add_views_to(self.all_data)


    def alloc_all(self):
        pass

    def do_transforms(self):
        """
        Combine the elements of input accumulator buffer (sigs_ic)
        into *add* them into sigs
        """
        transforms = self.model.transforms
        return self.sig_gemv(self.outbufs.keys(),
            1.0,
            lambda sig: [tf.alpha_signal
                         for tf in transforms if tf.outsig == sig],
            lambda sig: [tf.insig
                         for tf in transforms if tf.outsig == sig],
            1.0,
            lambda sig: self.outbufs[sig],
            )

    def do_filters(self):
        """
        Recombine the elements of previous signal buffer (sigs)
        and write them back to `sigs`
        """
        filters = self.model.filters
        return self.sig_gemv(
            self.outbufs.keys(),
            1.0,
            lambda sig: [tf.alpha_signal
                         for tf in filters if tf.newsig == sig],
            lambda sig: [tf.oldsig
                         for tf in filters if tf.newsig == sig],
            0.0,
            lambda sig: self.outbufs[sig],
            )

    def do_encoders(self):
        encoders = self.model.encoders

        return self.sig_gemv(
            self.model.nonlinearities,
            1.0,
            lambda pop: [enc.weights_signal
                         for enc in encoders if enc.pop == pop],
            lambda pop: [enc.sig
                         for enc in encoders if enc.pop == pop],
            1.0,
            lambda pop: pop.input_signal,
            lambda pop: pop.bias_signal,
            )

    def do_populations(self):
        dt = self.model.dt
        sidx = self.builder.sidx
        nls = sorted(self.model.nonlinearities, key=type)
        for nl_type, nl_group in itertools.groupby(nls, type):
            if nl_type == Direct:
                for nl in nl_group:
                    J = self.all_data[sidx[nl.input_signal]]
                    output = nl.fn(J)
                    self.all_data[sidx[nl.output_signal]][:] = output
            elif nl_type == LIF:
                for nl in nl_group:
                    J = self.all_data[sidx[nl.input_signal]]
                    voltage = self.all_data[sidx[self.lif_voltage[nl]]]
                    reftime = self.all_data[sidx[self.lif_reftime[nl]]]
                    output = self.all_data[sidx[nl.output_signal]]
                    nl.step_math0(dt, J, voltage, reftime, output,)
            else:
                raise NotImplementedError(nl_type)

    def do_decoders(self):
        decoders = self.model.decoders

        return self.sig_gemv(
            self.decoder_outputs,
            1.0, 
            lambda sig: [dec.weights_signal
                         for dec in decoders if dec.sig == sig],
            lambda sig: [dec.pop.output_signal
                         for dec in decoders if dec.sig == sig],
            0.0,
            lambda sig: sig,
            )

    def do_probes(self):
        probes = self.model.probes
        sidx = self.builder.sidx
        for probe in probes:
            period = int(probe.dt // self.model.dt)
            if self.sim_step % period == 0:
                self.probe_output[probe].append(
                    self.all_data[sidx[probe.sig]].copy()
                    )


    def do_back_copy(self):
        return self.sig_gemv(
            self.outbufs.keys(),
            1.0,
            lambda sig: [self.model.one],
            lambda sig: [self.outbufs[sig]],
            0.0,
            lambda sig: sig,
            )

    def do_all(self):
        #print '-' * 10 + 'A' * 10 + '-' * 10
        #print self.all_data_A
        #print '-' * 20
        self.do_encoders()
        #print self.pop_J
        self.do_populations()
        #print self.pop_J
        self.do_decoders()
        #print self.pop_J
        self.do_filters()
        #print self.pop_J
        self.do_transforms()
        #print self.pop_J
        self.do_back_copy()
        #print self.pop_J
        self.do_probes()
        #print self.pop_J

        #print '-' * 10 + 'B' * 10 + '-' * 10
        #print self.all_data_A
        #print '-' * 20

    def plan_all(self):
        pass

    def step(self):
        self.do_all()
        self.sim_step += 1

    def run_steps(self, N, verbose=False):
        for i in xrange(N):
            self.step()

    def signal(self, sig):
        probes = [sp for sp in self.model.probes if sp.sig == sig]
        if len(probes) == 0:
            raise KeyError()
        elif len(probes) > 1:
            raise KeyError()
        else:
            return self.signal_probe_output(probes[0])

    def probe_data(self, probe):
        return np.asarray(self.probe_output[probe])

