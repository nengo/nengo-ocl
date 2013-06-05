"""
numpy Simulator in the style of the OpenCL one, to get design right.
"""
from collections import defaultdict
import math
import numpy as np

from nengo.nonlinear import LIF, LIFRate


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

    def shallow_copy(self):
        rval = self.__class__.__new__(self.__class__)
        rval.starts = self.starts
        rval.lens = self.lens
        rval.buf = self.buf
        return rval

    def add_views(self, starts, lens):
        #assert start >= 0
        #assert start + length <= len(self.buf)
        # -- creates copies, same semantics
        #    as OCL version
        self.starts = self.starts + starts
        self.lens = self.lens + lens

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, item):
        if isinstance(item, (list, tuple)):
            rval = self.__class__.__new__(self.__class__)
            rval.starts = [self.starts[i] for i in item]
            rval.lens = [self.lens[i] for i in item]
            rval.buf = self.buf
            return rval
        else:
            o = self.starts[item]
            n = self.lens[item]
            rval = self.buf[o: o + n]
            if len(rval) != n:
                raise ValueError('buf not long enough')
            return rval

    def __setitem__(self, item, val):
        try:
            item = int(item)
        except TypeError:
            raise NotImplementedError()
        o = self.starts[item]
        n = self.lens[item]
        if len(val) != n:
            raise ValueError('wrong len')
        rval = self.buf[o: o + n]
        if len(rval) != n:
            raise ValueError('buf not long enough')
        # -- N.B. this is written for re-use by sim_ocl
        buf = self.buf
        buf[o: o + n] = val
        self.buf = buf


def raw_ragged_gather_gemv(BB,
        Ns, alphas,
        A_starts, A_data,
        A_js_starts,
        A_js_lens,
        A_js_data,
        X_starts,
        X_data,
        X_js_starts,
        X_js_data,
        betas,
        Y_in_starts,
        Y_in_data,
        Y_starts,
        Y_lens,
        Y_data):
    for bb in xrange(BB):
        alpha = alphas[bb]
        beta = betas[bb]
        n_dot_products = A_js_lens[bb]
        y_offset = Y_starts[bb]
        y_in_offset = Y_in_starts[bb]
        M = Y_lens[bb]
        for mm in xrange(M):
            Y_data[y_offset + mm] = beta * Y_in_data[y_in_offset + mm]

        for ii in xrange(n_dot_products):
            x_i = X_js_data[X_js_starts[bb] + ii]
            a_i = A_js_data[A_js_starts[bb] + ii]
            N_i = Ns[a_i]
            x_offset = X_starts[x_i]
            a_offset = A_starts[a_i]
            for mm in xrange(M):
                y_sum = 0.0
                for nn in xrange(N_i):
                    y_sum += X_data[x_offset + nn] * A_data[a_offset + nn * M + mm]
                Y_data[y_offset + mm] += alpha * y_sum


def ragged_gather_gemv(Ms, Ns, alpha, A, A_js, X, X_js,
                       beta, Y, Y_in=None,
                       use_raw_fn=False,
                      ):
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

    if use_raw_fn:
        # This is close to the OpenCL reference impl
        return raw_ragged_gather_gemv(
            len(Y),
            Ns,
            alpha,
            A.starts,
            A.buf,
            A_js.starts,
            A_js.lens,
            A_js.buf,
            X.starts,
            X.buf,
            X_js.starts,
            X_js.buf,
            beta,
            Y_in.starts,
            Y_in.buf,
            Y.starts,
            Y.lens,
            Y.buf)
    else:
        # -- less-close to the OpenCL impl
        #print alpha
        #print A.buf, 'A'
        #print X.buf, 'X'
        #print Y_in.buf, 'in'

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
                #print xi, x_ij, A_ij
                try:
                    y_i += alpha_i * np.dot(x_ij, A_ij.reshape(N_i, M_i))
                except:
                    print i, xi, ai, A_ij, x_ij
                    raise

            Y[i] = y_i


def lif_step(dt, J, voltage, refractory_time, spiked, tau_ref, tau_rc,
             upsample):
    """
    Replicates nengo.nonlinear.LIF's step_math0 function, except for
    a) not requiring a LIF instance and
    b) not adding the bias
    """
    if upsample != 1:
        raise NotImplementedError()

    # Euler's method
    dV = dt / tau_rc * (J - voltage)

    # increase the voltage, ignore values below 0
    v = np.maximum(voltage + dV, 0)

    # handle refractory period
    post_ref = 1.0 - (refractory_time - dt) / dt

    # set any post_ref elements < 0 = 0, and > 1 = 1
    v *= np.clip(post_ref, 0, 1)

    # determine which neurons spike
    # if v > 1 set spiked = 1, else 0
    spiked[:] = (v > 1) * 1.0

    old = np.seterr(all='ignore')
    try:

        # linearly approximate time since neuron crossed spike threshold
        overshoot = (v - 1) / dV
        spiketime = dt * (1.0 - overshoot)

        # adjust refractory time (neurons that spike get a new
        # refractory time set, all others get it reduced by dt)
        new_refractory_time = spiked * (spiketime + tau_ref) \
                              + (1 - spiked) * (refractory_time - dt)
    finally:
        np.seterr(**old)

    # return an ordered dictionary of internal variables to update
    # (including setting a neuron that spikes to a voltage of 0)

    voltage[:] = v * (1 - spiked)
    refractory_time[:] = new_refractory_time


def alloc_transform_helper(signals, transforms, sigs, sidx, RaggedArray,
        outsig_fn, insig_fn,
        ):
    # -- this is a function that is used to construct the
    #    so-called transforms and filters, which map from signals to signals.

    tidx = idxs(transforms)
    tf_by_outsig = defaultdict(list)
    for tf in transforms:
        tf_by_outsig[outsig_fn(tf)].append(tf)

    # N.B. the optimization below may not be valid
    # when alpha is not a scalar multiplier
    tf_weights = RaggedArray(
        [[tf.alpha] for tf in transforms])
    tf_Ns = [1] * len(transforms)
    tf_Ms = [1] * len(transforms)

    tf_sigs = sigs.shallow_copy()
    del sigs

    # -- which transform(s) does each signal use
    tf_weights_js = [[tidx[tf] for tf in tf_by_outsig[sig]]
        for sig in signals]

    # -- which corresponding(s) signal is transformed
    tf_signals_js = [[sidx[insig_fn(tf)] for tf in tf_by_outsig[sig]]
        for sig in signals]

    # -- Optimization:
    #    If any output signal is the sum of 
    #    consecutive weights with consecutive signals
    #    then turn that into *one* dot product
    #    of a new longer weight vector with a new
    #    longer signal vector.
    #    TODO: still do something like this for a 
    #    *part* of the wjs/sjs if possible
    #    TODO: sort the sjs or wjs to canonicalize things
    if 0:
        wstarts = []
        wlens = []
        sstarts = []
        slens = []
        for ii, (wjs, sjs) in enumerate(
                zip(tf_weights_js, tf_signals_js)):
            assert len(wjs) == len(sjs)
            K = len(wjs)
            if len(wjs) <= 1:
                continue
            if wjs != range(wjs[0], wjs[0] + len(wjs)):
                continue
            if sjs != range(sjs[0], sjs[0] + len(sjs)):
                continue
            # -- precondition satisfied
            # XXX length should be the sum of the lenghts
            #     of the tf_weights[i] for i in wjs
            wstarts.append(int(tf_weights.starts[wjs[0]]))
            wlens.append(K)
            new_w_j = len(tf_weights_js) + len(wstarts) - 1
            tf_Ns.append(K)
            tf_Ms.append(1)
            sstarts.append(int(tf_sigs.starts[sjs[0]]))
            slens.append(K)
            new_s_j = len(tf_signals_js) + len(sstarts) - 1
            tf_weights_js[ii] = [new_w_j]
            tf_signals_js[ii] = [new_s_j]

        tf_weights.add_views(wstarts, wlens)
        tf_sigs.add_views(sstarts, slens)

    if 0:
        for ii, (wjs, sjs) in enumerate(
                zip(tf_weights_js, tf_signals_js)):
            if wjs:
                print wjs, sjs

    return locals()


def idxs(seq, offset):
    rval = dict((s, i + offset) for (i, s) in enumerate(seq))
    return rval, offset + len(rval)


def islif(obj):
    return isinstance(obj, LIF)


def islifrate(obj):
    return isinstance(obj, LIFRate)


class Simulator(object):
    def __init__(self, model, n_prealloc_probes=1000):
        self.model = model
        self.nonlinearities = sorted(self.model.nonlinearities)

        self.bias_signals = [nl.bias_signal for nl in self.nonlinearities]
        bias_signals_set = set(self.bias_signals)

        self.input_signals = [nl.input_signal for nl in self.nonlinearities]

        self.output_signals = [nl.output_signal for nl in self.nonlinearities]

        self.const_signals = [sig for sig in model.signals
                if hasattr(sig, 'value') and not sig in bias_signals_set]

        self.all_signals = self.bias_signals + self.const_signals
        self.all_signals.extend(self.input_signals)
        self.all_signals.extend(self.output_signals)
        all_signals_set = set(self.all_signals)
        self.vector_signals = [sig for sig in model.signals
                if sig not in all_signals_set]
        self.all_signals.extend(self.vector_signals)

        self.all_signal_idxs, offset = idxs(self.all_signals, 0)
        assert offset == len(model.signals)

        # -- append a second version of vector_signals
        #    which are necessary for transforms
        self.tmp_vector_idxs, offset = idxs(self.vector_signals, offset)
        self.all_signals.extend(self.vector_signals)

        self.n_prealloc_probes = n_prealloc_probes
        self.sim_step = 0

    def RaggedArray(self, *args, **kwargs):
        return RaggedArray(*args, **kwargs)

    def alloc_signals(self):
        self.all_data_A = self.RaggedArray(
            [np.zeros(s.size) + getattr(s, 'value', 0)
                for s in self.all_signals])
        self.all_data_B = self.RaggedArray(
            [np.zeros(s.size) + getattr(s, 'value', 0)
                for s in self.all_signals])

    def alloc_probes(self):
        probes = self.model.probes
        dts = list(sorted(set([probe.dt for probe in probes])))
        self.sig_probes_output = {}
        self.sig_probes_buflen = {}
        self.sig_probes_Ms = self.RaggedArray([[1]])
        self.sig_probes_Ns = self.RaggedArray([[1]])
        self.sig_probes_A = self.RaggedArray([[1.0]])
        self.sig_probes_A_js = {}
        self.sig_probes_X_js = {}
        for dt in dts:
            sp_dt = [probe for probe in probes if probe.dt == dt]
            period = int(dt // self.model.dt)

            # -- allocate storage for the probe output
            sig_probes = self.RaggedArray([[0.0] for sp in sp_dt])
            buflen = len(sig_probes.buf)
            sig_probes.buf = np.zeros(
                    buflen * self.n_prealloc_probes,
                    dtype=sig_probes.buf.dtype)
            self.sig_probes_output[period] = sig_probes
            self.sig_probes_buflen[period] = buflen
            self.sig_probes_A_js[period] = self.RaggedArray(
                [[0] for sp in sp_dt])
            self.sig_probes_X_js[period] = self.RaggedArray(
                [[self.all_signal_idxs[sp.sig]] for sp in sp_dt])

    def alloc_populations(self):
        nls = self.nonlinearities
        sidx = self.all_signal_idxs
        pop_J_idxs = [sidx[nl.input_signal] for nl in nls]
        pop_bias_idxs = [sidx[nl.bias_signal] for nl in nls]
        pop_output_idxs = [sidx[nl.output_signal] for nl in nls]
        self.pop_bias = self.all_data_A[pop_bias_idxs]
        self.pop_J = self.all_data_B[pop_J_idxs]
        self.pop_output = self.all_data_B[pop_output_idxs]
        self.pidx, _ = idxs(self.nonlinearities, 0)

        lif_idxs = []
        direct_idxs = []
        for i, p in enumerate(nls):
            if islif(p):
                lif_idxs.append(i)
            else:
                direct_idxs.append(i)

        self.pop_lif_idxs = lif_idxs
        self.pop_direct_idxs = direct_idxs

        if lif_idxs:
            raise NotImplementedError()
            # sorting in the constructor was supposed to ensure this:
            assert lif_idxs == range(lif_idxs[0], lif_idxs[0] + len(lif_idxs))
            if lif_idxs and (nls[lif_idxs[0]] != nls[lif_idxs[-1]]):
                raise NotImplementedError('Non-homogeneous lif population')

            self.pop_lif_rep = nls[lif_idxs[0]]
            self.pop_lif_J = self.pop_J[lif_idxs]
            self.pop_lif_output = self.pop_output[lif_idxs]
            self.pop_lif_start = self.pop_lif_J.starts[0]
            lif_len = sum(self.pop_lif_J.lens)
            self.pop_lif_end = self.pop_lif_start + lif_len
            self.pop_lif_voltage = self.RaggedArray(
                [np.zeros(nls[idx].n_neurons) for idx in lif_idxs])
            self.pop_lif_reftime = self.RaggedArray(
                [np.zeros(nls[idx].n_neurons) for idx in lif_idxs])

        if direct_idxs:
            self.pop_direct_ins = self.pop_J[direct_idxs]
            self.pop_direct_outs = self.pop_output[direct_idxs]
            dtype = self.pop_direct_ins.buf.dtype
            # N.B. This uses numpy-based RaggedArray specifically.
            self.pop_direct_ins_npy = RaggedArray(
                [np.zeros(nls[di].n_in, dtype=dtype) for di in direct_idxs])
            self.pop_direct_outs_npy = RaggedArray(
                [np.zeros(nls[di].n_out, dtype=dtype) for di in direct_idxs])

    def alloc_transforms(self):
        sidx = self.all_signal_idxs
        transforms = self.model.transforms
        tidx, _ = idxs(transforms, 0)
        tf_by_outsig = defaultdict(list)
        for tf in transforms:
            tf_by_outsig[tf.outsig].append(tf)

        # N.B. the optimization below may not be valid
        # when alpha is not a scalar multiplier
        self.tf_Ns = [1] * len(transforms)
        self.tf_Ms = [1] * len(transforms)

        # -- which transform(s) does each signal use
        self.tf_weights = RaggedArray(
            [[float(tf.alpha)] for tf in transforms])
        self.tf_weights_js = self.RaggedArray(
                [[tidx[tf] for tf in tf_by_outsig[sig]]
                 for sig in self.vector_signals])

        # -- which corresponding(s) signal is transformed
        tmp_idxs = self.tmp_vector_idxs
        self.tf_signals = self.all_data_B
        self.tf_signals_js = self.RaggedArray(
                [[tmp_idxs[tf.insig] for tf in tf_by_outsig[sig]]
                 for sig in self.vector_signals])

        dst_signal_idxs = [self.all_signal_idxs[v] for v in self.vector_signals]
        self.tf_dest = self.all_data_B[dst_signal_idxs]

    def alloc_filters(self):
        sidx = self.all_signal_idxs
        filters = self.model.filters
        fidx, _ = idxs(filters, 0)
        f_by_newsig = defaultdict(list)
        for f in filters:
            f_by_newsig[f.newsig].append(f)

        # N.B. the optimization below may not be valid
        # when alpha is not a scalar multiplier
        self.f_Ns = [1] * len(filters)
        self.f_Ms = [1] * len(filters)

        # -- which transform(s) does each signal use
        self.f_weights = RaggedArray(
            [[float(f.alpha)] for f in filters])
        self.f_weights_js = self.RaggedArray(
                [[fidx[f] for f in f_by_newsig[sig]]
                 for sig in self.vector_signals])

        # -- which corresponding(s) signal is transformed
        self.f_signals = self.all_data_A
        self.f_signals_js = self.RaggedArray(
                [[sidx[f.oldsig] for f in f_by_newsig[sig]]
                 for sig in self.vector_signals])

        dst_signal_idxs = [sidx[v] for v in self.vector_signals]
        self.f_dest = self.all_data_B[dst_signal_idxs]

    def alloc_encoders(self):
        encoders = self.model.encoders
        # TODO: consider C vs. Fortran order
        eidx, _ = idxs(self.model.encoders, 0)

        self.enc_Ms = [enc.weights.shape[0] for enc in encoders]
        self.enc_Ns = [enc.weights.shape[1] for enc in encoders]

        # -- which encoder(s) does each population use
        self.enc_weights = self.RaggedArray(
            [enc.weights.flatten() for enc in encoders])
        self.enc_weights_js = self.RaggedArray([
            [eidx[enc] for enc in encoders if enc.pop == pop]
            for pop in self.model.nonlinearities])

        # -- and which corresponding signal does it encode
        self.enc_signals = self.all_data_A
        self.enc_signals_js = self.RaggedArray([
            [self.all_signal_idxs[enc.sig]
                for enc in encoders if enc.pop == pop]
            for pop in self.model.nonlinearities])

    def alloc_decoders(self):
        decoders = self.model.decoders
        didx, _ = idxs(self.model.decoders, 0)
        self.dec_Ms = [dec.weights.shape[0] for dec in decoders]
        self.dec_Ns = [dec.weights.shape[1] for dec in decoders]

        # -- which decoder(s) does each signal use
        self.dec_weights = self.RaggedArray(
            [dec.weights.flatten() for dec in decoders])
        self.dec_weights_js = self.RaggedArray([
            [didx[dec] for dec in decoders if dec.sig == sig]
            for sig in self.vector_signals])

        # -- and which corresponding population does it decode
        self.dec_pops = self.all_data_B
        self.dec_pops_js = self.RaggedArray([
            [self.all_signal_idxs[dec.pop.output_signal]
                for dec in decoders if dec.sig == sig]
            for sig in self.vector_signals])

        dst_idxs = [self.tmp_vector_idxs[sig]
            for sig in self.vector_signals]
        self.dec_dst = self.all_data_B[dst_idxs]

    def alloc_all(self):
        self.alloc_signals()
        self.alloc_probes()
        self.alloc_populations()
        self.alloc_transforms()
        self.alloc_filters()
        self.alloc_encoders()
        self.alloc_decoders()

    def do_transforms(self):
        """
        Combine the elements of input accumulator buffer (sigs_ic)
        into *add* them into sigs
        """
        # ADD linear combinations of signals to some signals
        ragged_gather_gemv(
            Ms=self.tf_Ms,
            Ns=self.tf_Ns,
            alpha=1.0,
            A=self.tf_weights,
            A_js=self.tf_weights_js,
            X=self.tf_signals,
            X_js=self.tf_signals_js,
            beta=1.0,
            Y=self.tf_dest,
            )

    def do_filters(self):
        """
        Recombine the elements of previous signal buffer (sigs)
        and write them back to `sigs`
        """
        # ADD linear combinations of signals to some signals
        ragged_gather_gemv(
            Ms=self.f_Ms,
            Ns=self.f_Ns,
            alpha=1.0,
            A=self.f_weights,
            A_js=self.f_weights_js,
            X=self.f_signals,
            X_js=self.f_signals_js,
            beta=0.0,
            Y=self.f_dest,
            )

    def do_encoders(self):
        ragged_gather_gemv(
            Ms=self.enc_Ms,
            Ns=self.enc_Ns,
            alpha=1.0,
            A=self.enc_weights,
            A_js=self.enc_weights_js,
            X=self.enc_signals,
            X_js=self.enc_signals_js,
            beta=1.0,
            Y_in=self.pop_bias,
            Y=self.pop_J,
            )

    def do_populations(self):
        dt = self.model.dt

        if self.pop_lif_idxs:
            lif_start = self.pop_lif_start
            lif_end = self.pop_lif_end
            lif_step(
                dt=dt,
                J=self.pop_lif_J.buf[lif_start:lif_end],
                voltage=self.pop_lif_voltage.buf,
                refractory_time=self.pop_lif_reftime.buf,
                spiked=self.pop_output.buf[lif_start:lif_end],
                tau_ref=self.pop_lif_rep.tau_ref,
                tau_rc=self.pop_lif_rep.tau_rc,
                upsample=self.pop_lif_rep.upsample
                )

        for ii in self.pop_direct_idxs:
            nl = self.nonlinearities[ii]
            popidx = self.pidx[nl]
            print 'do_populations: pop_J', self.pop_J[popidx]
            self.pop_output[popidx][...] = nl.fn(self.pop_J[popidx])

    def do_decoders(self):
        print 'do_decoders: pop_output.buf', self.pop_output.buf
        ragged_gather_gemv(
            Ms=self.dec_Ms,
            Ns=self.dec_Ns,
            alpha=1.0,
            A=self.dec_weights,
            A_js=self.dec_weights_js,
            X=self.dec_pops,
            X_js=self.dec_pops_js,
            beta=0.0,
            Y=self.dec_dst,
            )
        print 'sig tmp', self.dec_dst.buf

    def do_probes(self):
        for period in self.sig_probes_output:
            if 0 == self.sim_step % period:
                probe_out = self.sig_probes_output[period]
                buflen = self.sig_probes_buflen[period]
                A_js = self.sig_probes_A_js[period]
                X_js = self.sig_probes_X_js[period]

                bufidx = int(self.sim_step // period)
                orig_buffer = probe_out.buf
                this_buffer = probe_out.buf[bufidx * buflen:]
                try:
                    probe_out.buf = this_buffer
                    ragged_gather_gemv(
                        Ms=self.sig_probes_Ms,
                        Ns=self.sig_probes_Ns,
                        alpha=1.0,
                        A=self.sig_probes_A,
                        A_js=A_js,
                        X=self.all_data_B,
                        X_js=X_js,
                        beta=0.0,
                        Y=probe_out,
                        )
                finally:
                    probe_out.buf = orig_buffer

    def do_all(self):
        # -- step 1: sigs store all signal values of step t (sigs_t)
        self.do_encoders()          # encode sigs_t into pops
        self.do_populations()       # pop dynamics
        self.do_decoders()          # decode into sigs_ic
        self.do_filters()           # make sigs_t's contribution to t + 1
        self.do_transforms()        # add sigs_ic's contribution to t + 1
        self.do_probes()
        self.all_data_A.buf[:] = self.all_data_B.buf

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
        period = int(probe.dt // self.model.dt)
        last_elem = int(math.ceil(self.sim_step / float(period)))

        # -- figure out which signal it is among the ones with the same dt
        sps_dt = [sp for sp in self.model.probes if sp.dt == probe.dt]
        probe_idx = sps_dt.index(probe)
        all_rows = self.sig_probes_output[period].buf.reshape(
                (-1, self.sig_probes_buflen[period]))
        assert all_rows.dtype == self.sig_probes_output[period].buf.dtype
        start = self.sig_probes_output[period].starts[probe_idx]
        olen = self.sig_probes_output[period].lens[probe_idx]
        start = start % self.sig_probes_buflen[period]
        rval = all_rows[:last_elem, start:start + olen]
        return rval


