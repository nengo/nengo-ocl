"""
numpy Simulator in the style of the OpenCL one, to get design right.
"""
import itertools
import logging

from tricky_imports import OrderedDict

logger = logging.getLogger(__name__)
info = logger.info
warn = logger.warn
error = logger.error
critical = logger.critical

import numpy as np

from nengo.objects import Signal
from nengo.objects import SignalView
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
        rval = self.RaggedArray(
            [np.zeros(ss.shape) + getattr(ss, 'value', np.zeros(ss.shape))
                for ss in sigseq],
            names=[getattr(ss, 'name', '') for ss in sigseq])
        return rval

    def sig_gemv(self, seq, alpha, A_js_fn, X_js_fn, beta, Y_sig_fn,
                 Y_in_sig_fn=None,
                 verbose=0
                ):
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

        if verbose:
            print 'in sig_vemv'
            print 'print A', A_js
            print 'print X', X_js

        A_js = RaggedArray(A_js)
        X_js = RaggedArray(X_js)
        Y = self.all_data[Y_idxs]
        Y_in = self.all_data[Y_in_idxs]

        return self.plan_ragged_gather_gemv(
            Ms=self.all_data.shape0s,
            Ns=self.all_data.shape1s,
            alpha=alpha,
            A=self.all_data, A_js=A_js,
            X=self.all_data, X_js=X_js,
            beta=beta,
            Y=Y,
            Y_in=Y_in,
            )

    def plan_ragged_gather_gemv(self, *args, **kwargs):
        return (lambda: ragged_gather_gemv(*args, **kwargs))

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
            yield self.dec_outputs[sig]

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
        bases = stable_unique(sig.base for sig in
                    self.orig_relevant_signals(model))

        # -- the following bases cannot be modified during the simulator
        #    loop, because we need their old values at the end.
        self.filtered_bases_set = set(filt.oldsig.base for filt in model.filters)

        self.saved_for_filtering = OrderedDict()
        def save_for_filtering(sig):
            try:
                return self.saved_for_filtering[sig]
            except KeyError:
                newbase = add_base_like(sig, 'saved')
                self.saved_for_filtering[sig] = newbase
                return newbase


        def add_base_like(sig, suffix):
            N, = sig.shape
            a = Signal(N, name=sig.name + suffix)
            bases.append(a)
            return a

        def add_view_like(sig, newbase, suffix):
            return SignalView(newbase,
                shape=sig.shape,
                elemstrides=sig.elemstrides,
                offset=sig.offset,
                name=sig.name + suffix)

        # -- add some extra buffers for some signals:
        #    reset bias -> input current can be done in-place
        #    encoders -> input current can be done in-place
        #    nl outputs -> generally needs buffer
        for enc in model.encoders:
            if enc.pop.input_signal.base in self.filtered_bases_set:
                save_for_filtering(enc.pop.input_signal)

        for nl in model.nonlinearities:
            if nl.output_signal.base in self.filtered_bases_set:
                save_for_filtering(nl.output_signal)

            # -- also some neuron models need a few extra buffers
            if isinstance(nl, Direct):
                pass
            elif isinstance(nl, LIF):
                self.lif_voltage[nl] = add_base_like(nl.output_signal,
                                                     '.voltage')
                self.lif_reftime[nl] = add_base_like(nl.output_signal,
                                                     '.reftime')
            elif isinstance(nl, LIFRate):
                pass 
            else:
                raise NotImplementedError()


        if any(isview(dec.sig) for dec in model.decoders):
            # TODO: check for overlap
            warn("decoding to views without checking for overlap")

        #    decoder outputs -> also generally needs copy
        #    N.B. technically many decoders can decode the same signal
        #         this is defined to mean that we add together the several
        #         decoder outputs as if it were a large sparsely connected
        #         decoder
        decoder_output_bases = stable_unique(
            [dec.sig.base for dec in model.decoders])

        # XXX: confusing to have both
        #            self.decoder_outputs
        #      *and* self.dec_outputs
        self.decoder_outputs = stable_unique(
            [dec.sig for dec in model.decoders])

        for base in decoder_output_bases:
            self.dec_outputs[base] = add_base_like(base, '.dec_output')
        for sig in self.decoder_outputs:
            if sig not in self.dec_outputs:
                assert isview(sig)
                self.dec_outputs[sig] = add_view_like(
                    sig,
                    self.dec_outputs[sig.base],
                    '.dec_output')

        if 1:
            # -- sanity check
            #    Ensure that all decoder outputs are actually used, but more
            #    importantly that all transform inputs are decoder outputs.
            #    If you think you want to use a transform on something that
            #    isn't a decoder output, then you actually want a filter.
            #    A filter draws data from the original buffers. A transform
            #    draws from the decoder output buffers, and that's all the
            #    cases.
            pop_outputs_set = set(
                nl.output_signal for nl in model.nonlinearities)
            transform_inputs_set = set(
                tf.insig.base for tf in model.transforms)
            valid_transformable_things = pop_outputs_set
            valid_transformable_things.update(decoder_output_bases)
            if not transform_inputs_set.issubset(valid_transformable_things):
                print "Model error: Transform inputs != valid transformable"
                print "Decoder outputs:         ", decoder_output_bases
                print "Population outputs:      ", pop_outputs_set
                print "Transform inputs (bases):", transform_inputs_set
                assert 0
            del transform_inputs_set
            del valid_transformable_things
            del pop_outputs_set

        #    generally, filters and transforms are meant to
        #    write into a fresh "output buffer"
        #    which is then copied back over top of the old values
        #    There are cases where more can be done in-place, but we'll
        #    just do the general case for now.
        #
        #    -- N.B. that each of view of some common base gets its own
        #       allocated space in outbufs
        filt_and_tf_outputs = (
            [filt.newsig for filt in model.filters]
            + [tf.outsig for tf in model.transforms])
        for sig in stable_unique(filt_and_tf_outputs):
            self.outbufs[sig] = add_base_like(sig, '.outbuf')

        for filt in self.model.filters:
            self.filter_outputs[filt] = self.outbufs[filt.newsig]
        for tf in self.model.transforms:
            self.transform_outputs[tf] = self.outbufs[tf.outsig]

        # -- Choose a layout order for the constants.
        bases = stable_unique(sorted(bases, key=lambda bb: bb.size))
        self.bases[:] = bases
        self.all_data = self.alloc_signal_data(self.bases)
        self.builder = ViewBuilder(self.bases, self.all_data)
        for sig in stable_unique(self.signals_iter()):
            self.builder.append_view(sig)
        self.builder.add_views_to(self.all_data)

    def alloc_all(self):
        pass

    def __getitem__(self, item):
        """
        Return internally shaped signals, which are always 2d
        """
        return self.all_data[self.builder.sidx[item]]

    @property
    def signals(self):
        """Get/set [properly-shaped] signal value (either 0d, 1d, or 2d)
        """
        class Cls(object):
            def __iter__(_):
                return iter(stable_unique(self.signals_iter()))

            def __getitem__(_, item):
                raw = self.all_data[self.builder.sidx[item]]
                assert raw.ndim == 2
                if item.ndim == 0:
                    return raw[0, 0]
                elif item.ndim == 1:
                    return raw.ravel()
                elif item.ndim == 2:
                    return raw
                else:
                    raise NotImplementedError()

            def __setitem__(_, item, val):
                raw = self.all_data[self.builder.sidx[item]]
                assert raw.ndim == 2
                incoming = np.asarray(val)
                if item.ndim == 0:
                    assert incoming.size == 1
                    self.all_data[self.builder.sidx[item]][:] = incoming
                elif item.ndim == 1:
                    assert (item.size,) == incoming.shape
                    self.all_data[self.builder.sidx[item]][:] = incoming[:, None]
                elif item.ndim == 2:
                    assert item.shape == incoming.shape
                    self.all_data[self.builder.sidx[item]][:] = incoming
                else:
                    raise NotImplementedError()
        return Cls()

    def plan_encoders(self):
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

    def plan_populations(self):
        def fn():
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
        return fn

    def plan_decoders(self):
        decoders = self.model.decoders

        return self.sig_gemv(
            self.decoder_outputs,
            1.0, 
            lambda sig: [dec.weights_signal
                         for dec in decoders if dec.sig == sig],
            lambda sig: [dec.pop.output_signal
                         for dec in decoders if dec.sig == sig],
            0.0,
            lambda sig: self.dec_outputs[sig],
            )

    def plan_transforms(self, verbose=0):
        """
        Combine the elements of input accumulator buffer (sigs_ic)
        into *add* them into sigs
        """
        transforms = self.model.transforms
        return self.sig_gemv(self.outbufs.keys(),
            1.0,
            lambda sig: [tf.alpha_signal
                         for tf in transforms if tf.outsig == sig],
            lambda sig: [self.dec_outputs.get(tf.insig, tf.insig)
                         for tf in transforms if tf.outsig == sig],
            1.0,
            lambda sig: self.outbufs[sig],
            verbose=verbose
            )

    def plan_filters(self, verbose=0):
        """
        Recombine the elements of previous signal buffer (sigs)
        and write them back to `sigs`
        """
        filters = self.model.filters
        saved = self.saved_for_filtering
        return self.sig_gemv(
            self.outbufs.keys(),
            1.0,
            lambda sig: [filt.alpha_signal
                         for filt in filters if filt.newsig == sig],
            lambda sig: [saved.get(filt.oldsig, filt.oldsig)
                         for filt in filters if filt.newsig == sig],
            0.0,
            lambda sig: self.outbufs[sig],
            verbose=verbose
            )

    def plan_save_for_filters(self):
        saved = self.saved_for_filtering
        return self.sig_gemv(
            saved,
            1.0,
            lambda item: [self.model.one],
            lambda item: [item],
            0.0,
            lambda item: saved[item],
            verbose=0,
            )

    def plan_back_copy(self):
        # -- here we may have to serialize a little bit so that
        #    updates to views are copied back incrementally into
        #    any original signals

        # XXX This function should be more sophisticated, if sets
        #     of views are shown to be non-overlapping, then they
        #     can be updated at the same time.

        # -- by_base: map original base -> (view of base, outbuf for view)
        by_base = dict((sig_or_view.base, [])
                       for sig_or_view in self.outbufs.keys())

        for sig_or_view in self.outbufs:
            by_base[sig_or_view.base].append(
                (sig_or_view, self.outbufs[sig_or_view]))

        copy_fns = []
        beta = 0.0
        while by_base:
            bases = by_base.keys()
            copy_fns.append(
                self.sig_gemv(
                    bases,
                    1.0,
                    lambda base: [self.model.one],
                    lambda base: [by_base[base][-1][1]],
                    beta,
                    lambda base: by_base[base][-1][0],
                    verbose=0,
                    ))
            for base in by_base:
                by_base[base].pop()
            by_base = dict((k, v) for k, v in by_base.items() if v)
            beta = 1.0
        info('back_copy required %i passes' % len(copy_fns))
        return copy_fns

    def plan_probes(self):
        def fn():
            probes = self.model.probes
            #sidx = self.builder.sidx
            for probe in probes:
                period = int(probe.dt // self.model.dt)
                if self.sim_step % period == 0:
                    self.probe_output[probe].append(
                        self.signals[probe.sig].copy())
        return fn

    def plan_all(self):
        self._plan = [
            self.plan_save_for_filters(),
            self.plan_encoders(),
            self.plan_populations(),
            self.plan_decoders(),
            self.plan_filters(),
            self.plan_transforms(),
        ]
        self._plan.extend(self.plan_back_copy())
        self._plan.append(self.plan_probes())

    def step(self):
        for fn in self._plan:
            fn()
        self.sim_step += 1

    def run_steps(self, N, verbose=False):
        for i in xrange(N):
            self.step()

    # XXX there is both .signals and .signal and they are pretty different
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

