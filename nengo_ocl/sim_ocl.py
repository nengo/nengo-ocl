import time
import os
import numpy as np
import pyopencl as cl

import sim_npy
#from ocl.array import Array
from ocl.array import to_device
from ocl.plan import Plan, Prog
#from ocl.array import empty
from ocl.gemv_batched import plan_ragged_gather_gemv
from ocl.lif import plan_lif
from ocl.elemwise import plan_copy, plan_inc

class RaggedArray(object):
    # a linear buffer that is partitioned into
    # sections of various lengths.
    # 
    def __init__(self, queue, listoflists):

        starts = []
        lens = []
        buf = []

        for l in listoflists:
            starts.append(len(buf))
            lens.append(len(l))
            buf.extend(l)

        self.starts = to_device(queue, np.asarray(starts).astype('int32'))
        self.lens = to_device(queue, np.asarray(lens).astype('int32'))
        buf = np.asarray(buf)
        if 'int' in str(buf.dtype):
            buf = buf.astype('int32')
        if buf.dtype == np.dtype('float64'):
            buf = buf.astype('float32')
        self.buf = to_device(queue, buf)
        self.queue = queue

    def add_view(self, start, length):
        starts = list(self.starts.get(self.queue))
        lens = list(self.lens.get(self.queue))
        rval = len(starts)
        assert len(starts) == len(lens)
        starts.append(start)
        lens.append(length)
        self.starts = to_device(self.queue, np.asarray(starts).astype('int32'))
        self.lens = to_device(self.queue, np.asarray(lens).astype('int32'))
        return rval

    def __len__(self):
        return self.starts.shape[0]

    def __getitem__(self, item):
        #print 'OCL RaggedArray getitem horribly slow'
        # XXX only retrieve the bit we need
        starts = self.starts.get(self.queue)
        lens = self.lens.get(self.queue)
        buf = self.buf.get(self.queue)
        o = starts[item]
        n = lens[item]
        rval = buf[o: o + n]
        if len(rval) != n:
            raise ValueError('buf not long enough')
        return rval

    def __setitem__(self, item, val):
        # XXX only set the bit we need
        print 'OCL RaggedArray setitem horribly slow'
        starts = self.starts.get(self.queue)
        lens = self.lens.get(self.queue)
        buf = self.buf.get(self.queue)
        o = starts[item]
        n = lens[item]
        if len(val) != n:
            raise ValueError('wrong len')
        if len(buf[o: o + n]) != n:
            raise ValueError('buf not long enough')
        buf[o: o + n] = val
        self.buf.set(buf, queue=self.queue)


class Simulator(sim_npy.Simulator):

    def __init__(self, context, model, n_prealloc_probes=1000,
                profiling=None):
        if profiling is None:
            profiling = bool(int(os.getenv("NENGO_OCL_PROFILING", 0)))
        self.context = context
        self.profiling = profiling
        if profiling:
            self.queue = cl.CommandQueue(
                context,
                properties=cl.command_queue_properties.PROFILING_ENABLE)
        else:
            self.queue = cl.CommandQueue(context)
        sim_npy.Simulator.__init__(self, model,
                                  n_prealloc_probes=n_prealloc_probes)

    def RaggedArray(self, listoflists):
        return RaggedArray(self.queue, listoflists)

    def alloc_signal_probes(self):
        signal_probes = self.model.signal_probes
        dts = list(sorted(set([sp.dt for sp in signal_probes])))
        self.sig_probes_output = {}
        self.sig_probes_buflen = {}
        self.sig_probes_Ms = self.RaggedArray([[1]])
        self.sig_probes_Ns = self.RaggedArray([[1]])
        self.sig_probes_A = self.RaggedArray([[1.0]])
        self.sig_probes_A_js = {}
        self.sig_probes_X_js = {}
        for dt in dts:
            sp_dt = [sp for sp in signal_probes if sp.dt == dt]
            period = int(dt // self.model.dt)

            # -- allocate storage for the probe output
            sig_probes = self.RaggedArray([[0.0] for sp in sp_dt])
            buflen = len(sig_probes.buf)
            sig_probes.buf = to_device(self.queue, np.zeros(
                    buflen * self.n_prealloc_probes,
                    dtype=sig_probes.buf.dtype))
            self.sig_probes_output[period] = sig_probes
            self.sig_probes_buflen[period] = buflen
            self.sig_probes_A_js[period] = self.RaggedArray([[0] for sp in sp_dt])
            self.sig_probes_X_js[period] = self.RaggedArray(
                [[self.sidx[sp.sig]] for sp in sp_dt])

    def plan_transforms(self):
        """
        Combine the elements of input accumulator buffer (sigs_ic)
        into *add* them into sigs
        """
        # ADD linear combinations of signals to some signals
        return plan_ragged_gather_gemv(
            self.queue,
            Ms=self.tf_Ms,
            Ns=self.tf_Ns,
            alpha=1.0,
            A=self.tf_weights,
            A_js=self.tf_weights_js,
            X=self.sigs_ic,
            X_js=self.tf_signals_js,
            beta=1.0,
            Y=self.sigs,
            tag='transforms',
            )

    def plan_copy_sigs(self):
        return plan_copy(self.queue,
                         self.sigs.buf,
                         self._sigs_copy.buf,
                         tag='copy_sigs')

    def plan_filters(self):
        """
        Recombine the elements of previous signal buffer (sigs)
        and write them back to `sigs`
        """
        # ADD linear combinations of signals to some signals
        return plan_ragged_gather_gemv(
            self.queue,
            Ms=self.f_Ms,
            Ns=self.f_Ns,
            alpha=1.0,
            A=self.f_weights,
            A_js=self.f_weights_js,
            X=self._sigs_copy,
            X_js=self.f_signals_js,
            beta=0.0,
            Y=self.sigs,
            tag='filters'
            )

    def plan_custom_transforms(self):
        rval = []
        for ct in self.model.custom_transforms:
            # XXX Think of a better way...
            ipos = self.sidx[ct.insig]
            opos = self.sidx[ct.outsig]
            text =  """ __kernel void fn(__global %s *x)
                {
                x[%i] = sin(x[%i]);
                }
                """ % (self.sigs.buf.ocldtype, opos, ipos)
            print text
            kern = cl.Program(self.queue.context, text).build().fn
            kern.set_args(self.sigs.buf.data)
            plan = Plan(
                self.queue, kern, (1,), None, tag='custom sin transform')
            rval.append(plan)
        return rval

    def plan_encoders(self):
        return plan_ragged_gather_gemv(
            self.queue,
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
            tag='encoders'
            )

    def plan_populations(self):
        return plan_lif(self.queue,
            V=self.pop_voltage.buf,
            RT=self.pop_rt.buf,
            J=self.pop_ic.buf,
            OV=self.pop_voltage.buf,
            ORT=self.pop_rt.buf,
            OS=self.pop_output.buf,
            dt=self.model.dt,
            tau_rc=0.02,
            tau_ref=0.002,
            V_threshold=1.0,
            upsample=2)

    def plan_decoders(self):
        return plan_ragged_gather_gemv(
            self.queue,
            Ms=self.dec_Ms,
            Ns=self.dec_Ns,
            alpha=1.0,
            A=self.dec_weights,
            A_js=self.dec_weights_js,
            X=self.pop_output,
            X_js=self.dec_pops_js,
            beta=0.0,
            Y=self.sigs_ic,
            tag='decoders',
            )

    def plan_probes(self):
        plans = {}
        for period in self.sig_probes_output:
            probe_out = self.sig_probes_output[period]
            buflen = self.sig_probes_buflen[period]
            A_js = self.sig_probes_A_js[period]
            X_js = self.sig_probes_X_js[period]

            write_plan = plan_ragged_gather_gemv(
                self.queue,
                Ms=self.sig_probes_Ms,
                Ns=self.sig_probes_Ns,
                alpha=1.0,
                A=self.sig_probes_A,
                A_js=A_js,
                X=self.sigs,
                X_js=X_js,
                beta=0.0,
                Y=probe_out,
                tag='probe:%i' % period,
                )
            advance_plan = plan_inc(
                self.queue,
                probe_out.starts,
                amt=buflen,
                tag='probe_inc:%i' % period,
                )
            plans[period] = (write_plan, advance_plan,)
        return plans

    def plan_all(self):
        # see sim_npy for rationale of this ordering
        tick_plans = [
            self.plan_encoders(),
            self.plan_populations(),
            self.plan_decoders(),
            self.plan_copy_sigs(),
            self.plan_filters(),
            self.plan_transforms(),
        ]
        self.tick_plans = tick_plans
        tick_plans.extend(self.plan_custom_transforms())
        self.probe_plans = self.plan_probes()
        periods = [1] + self.probe_plans.keys()
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        def lcm(a, b):
            return a * b // gcd(a, b)
        period_len = reduce(lcm, periods)
        print 'creating a command sequence for', period_len,
        print 'iterations at once: periods =', periods
        self.all_plans = []
        self.all_plans_period_len = period_len
        self.all_plans_start = []
        self.all_plans_stop = []
        # XXX figure out when probe pointers go off the end
        for ii in xrange(period_len):
            self.all_plans_start.append(len(self.all_plans))
            self.all_plans.extend(tick_plans)
            for period in self.probe_plans:
                if ii % period == 0:
                    self.all_plans.extend(self.probe_plans[period])
            self.all_plans_stop.append(len(self.all_plans))
        self.all_plans_prog = Prog(self.all_plans)

    def step(self):
        try:
            self.all_plans
        except AttributeError:
            self.plan_all()
        period_pos = self.sim_step % self.all_plans_period_len
        start = self.all_plans_start[period_pos]
        stop = self.all_plans_stop[period_pos]
        plans = self.all_plans[start:stop]
        for p in plans:
            p.enqueue()
        self.queue.finish()
        self.sim_step += 1

    def run_steps(self, N):
        try:
            self.all_plans
        except AttributeError:
            self.plan_all()
        plen = self.all_plans_period_len
        period_pos = self.sim_step % plen
        if period_pos:
            steps_left = min(self.all_plans_period_len - period_pos, N)
            period_goal = period_pos + steps_left
            start = self.all_plans_start[period_pos]
            stop = self.all_plans_stop[period_goal - 1]
            plans = self.all_plans[start:stop]
            if self.profiling:
                for p in plans:
                    p(profiling=True)
            else:
                for p in plans:
                    p.enqueue()
            N -= steps_left
            self.sim_step += steps_left
        t0 = time.time()
        full_cycles = N // plen
        if self.profiling:
            for ii in xrange(full_cycles):
                for p in self.all_plans:
                    p(profiling=True)
            #for nn in xrange(N):
            #    for p in self.tick_plans:
            #        p(self.profiling)
            #for p in self.tick_plans:
            #    print p.ctime / p.n_calls, p
        else:
            self.all_plans_prog.enqueue_n_times(full_cycles)

        N -= full_cycles * plen
        self.sim_step += full_cycles * plen
        t1 = time.time()
        print "A", (t1 - t0)
        if N:
            return self.run_steps(N)
        else:

            if self.profiling:
                times = [(p.ctime, p) for p in set(self.all_plans)]
                for (ctime, p) in reversed(sorted(times)):
                    n_calls = max(1, p.n_calls)
                    print ctime, p.n_calls, (ctime / n_calls), p.atime, p.btime, p
            else:
                self.queue.finish()

    def signal(self, sig):
        probes = [sp for sp in self.model.signal_probes if sp.sig == sig]
        if len(probes) == 0:
            raise KeyError()
        elif len(probes) > 1:
            raise KeyError()
        else:
            return self.signal_probe_output(probes[0])

    def signal_probe_output(self, probe):
        period = int(probe.dt // self.model.dt)
        last_elem = int(np.ceil(self.sim_step / float(period)))

        # -- figure out which signal it is among the ones with the same dt
        sps_dt = [sp for sp in self.model.signal_probes if sp.dt == probe.dt]
        probe_idx = sps_dt.index(probe)
        all_rows = self.sig_probes_output[period].buf.get()
        starts = self.sig_probes_output[period].starts.get()
        lens = self.sig_probes_output[period].lens.get()

        all_rows = all_rows.reshape((-1, self.sig_probes_buflen[period]))
        start = starts[probe_idx] % self.sig_probes_buflen[period]
        olen = lens[probe_idx]
        return all_rows[:last_elem, start:start + olen]



