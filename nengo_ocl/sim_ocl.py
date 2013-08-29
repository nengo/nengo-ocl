#import time
import os
import numpy as np
import pyopencl as cl

import sim_npy
#from ocl.array import Array
from ocl.array import to_device
from ocl.plan import Prog
#from ocl.array import empty
from ocl.gemv_batched import plan_ragged_gather_gemv
from ocl.lif import plan_lif
from ocl.elemwise import plan_copy, plan_inc


class DevRaggedArray(object):
    # a linear buffer that is partitioned into
    # sections of various lengths.
    #

    @property
    def dtype(self):
        return self.buf.dtype

    def __init__(self, queue, np_raggedarray):
        self.queue = queue
        self.starts = np_raggedarray.starts
        self.shape0s = np_raggedarray.shape0s
        self.shape1s = np_raggedarray.shape1s
        self.ldas = np_raggedarray.ldas
        self.buf = np_raggedarray.buf
        self.names = np_raggedarray.names

    @property
    def starts(self):
        return map(int, self.cl_starts.get())

    @starts.setter
    def starts(self, starts):
        self.cl_starts = to_device(self.queue,
                                   np.asarray(starts).astype('int32'))
        self.queue.flush()

    @property
    def shape0s(self):
        return map(int, self.cl_shape0s.get())

    @shape0s.setter
    def shape0s(self, shape0s):
        self.cl_shape0s = to_device(self.queue,
                                    np.asarray(shape0s).astype('int32'))
        self.queue.flush()

    @property
    def shape1s(self):
        return map(int, self.cl_shape1s.get())

    @shape1s.setter
    def shape1s(self, shape1s):
        self.cl_shape1s = to_device(self.queue,
                                    np.asarray(shape1s).astype('int32'))
        self.queue.flush()

    @property
    def ldas(self):
        return map(int, self.cl_ldas.get())

    @ldas.setter
    def ldas(self, ldas):
        self.cl_ldas = to_device(self.queue,
                                   np.asarray(ldas).astype('int32'))
        self.queue.flush()

    @property
    def buf(self):
        return self.cl_buf.get()

    @buf.setter
    def buf(self, buf):
        buf = np.asarray(buf)
        if 'int' in str(buf.dtype):
            buf = buf.astype('int32')
        if buf.dtype == np.dtype('float64'):
            buf = buf.astype('float32')
        self.cl_buf = to_device(self.queue, buf)
        self.queue.flush()

    # def shallow_copy(self):
    #     rval = self.__class__.__new__(self.__class__)
    #     rval.cl_starts = self.cl_starts
    #     rval.cl_lens = self.cl_lens
    #     rval.cl_buf = self.cl_buf
    #     rval.queue = self.queue
    #     return rval

    def __len__(self):
        return self.cl_starts.shape[0]

    def __getitem__(self, item):
        """
        Getting one item returns a numpy array (on the host).
        Getting multiple items returns a view into the device.
        """

        starts = self.starts
        shape0s = self.shape0s
        shape1s = self.shape1s
        ldas = self.ldas

        if isinstance(item, (list, tuple)):
            items = item
            del item

            rval = self.__class__.__new__(self.__class__)
            rval.queue = self.queue
            rval.starts = [starts[i] for i in items]
            rval.shape0s = [shape0s[i] for i in items]
            rval.shape1s = [shape1s[i] for i in items]
            rval.ldas = [ldas[i] for i in items]
            rval.cl_buf = self.cl_buf
            rval.names = [self.names[i] for i in items]
            return rval
        else:
            m, n = shape0s[item], shape1s[item]
            if m * n == 0:
                return np.zeros((m,n), dtype=self.dtype)

            lda = ldas[item]
            assert lda >= 0, "lda must be non-negative"

            itemsize = self.dtype.itemsize
            bytestart = itemsize * starts[item]
            byteend = bytestart + itemsize * (m + lda * (n-1))

            temp_buf = np.zeros((byteend - bytestart), dtype=np.int8)
            cl.enqueue_copy(self.queue, temp_buf, self.cl_buf.data,
                            device_offset=bytestart, is_blocking=True)

            bytestrides = (itemsize, itemsize * lda)
            try:
                view = np.ndarray(
                    shape=(m, n),
                    dtype=self.dtype,
                    buffer=temp_buf.data,
                    offset=0,
                    strides=bytestrides)
            except:
                raise
            return view


# class Simulator(sim_npy.Simulator):

#     def __init__(self, context, model, n_prealloc_probes=1000,
#                 profiling=None):
#         if profiling is None:
#             profiling = bool(int(os.getenv("NENGO_OCL_PROFILING", 0)))
#         self.context = context
#         self.profiling = profiling
#         if profiling:
#             self.queue = cl.CommandQueue(
#                 context,
#                 properties=cl.command_queue_properties.PROFILING_ENABLE)
#         else:
#             self.queue = cl.CommandQueue(context)
#         sim_npy.Simulator.__init__(self, model,
#                                   n_prealloc_probes=n_prealloc_probes)

#     def RaggedArray(self, listoflists):
#         return RaggedArray(self.queue, listoflists)

#     def alloc_signal_probes(self):
#         signal_probes = self.model.signal_probes
#         dts = list(sorted(set([sp.dt for sp in signal_probes])))
#         self.sig_probes_output = {}
#         self.sig_probes_buflen = {}
#         self.sig_probes_Ms = [1]
#         self.sig_probes_Ns = [1]
#         self.sig_probes_A = self.RaggedArray([[1.0]])
#         self.sig_probes_A_js = {}
#         self.sig_probes_X_js = {}
#         for dt in dts:
#             sp_dt = [sp for sp in signal_probes if sp.dt == dt]
#             period = int(dt // self.model.dt)

#             # -- allocate storage for the probe output
#             sig_probes = self.RaggedArray(
#                 [[0.0] * sp.sig.n for sp in sp_dt])
#             buflen = len(sig_probes.buf)
#             sig_probes.buf = np.zeros(
#                 buflen * self.n_prealloc_probes,
#                 dtype=sig_probes.buf.dtype)
#             self.sig_probes_output[period] = sig_probes
#             self.sig_probes_buflen[period] = buflen
#             self.sig_probes_A_js[period] = self.RaggedArray([[0] for sp in sp_dt])
#             self.sig_probes_X_js[period] = self.RaggedArray(
#                 [[self.sidx[sp.sig]] for sp in sp_dt])

#     def plan_transforms(self):
#         """
#         Combine the elements of input accumulator buffer (sigs_ic)
#         into *add* them into sigs
#         """
#         # ADD linear combinations of signals to some signals
#         return [plan_ragged_gather_gemv(
#             self.queue,
#             Ms=self.tf_Ms,
#             Ns=self.tf_Ns,
#             alpha=1.0,
#             A=self.tf_weights,
#             A_js=self.tf_weights_js,
#             X=self.tf_signals,
#             X_js=self.tf_signals_js,
#             beta=1.0,
#             Y=self.sigs,
#             tag='transforms',
#             )]

#     def plan_copy_sigs(self):
#         return [plan_copy(self.queue,
#                          self.sigs.cl_buf,
#                          self._sigs_copy.cl_buf,
#                          tag='copy_sigs')]

#     def plan_filters(self):
#         """
#         Recombine the elements of previous signal buffer (sigs)
#         and write them back to `sigs`
#         """
#         # ADD linear combinations of signals to some signals
#         return [plan_ragged_gather_gemv(
#             self.queue,
#             Ms=self.f_Ms,
#             Ns=self.f_Ns,
#             alpha=1.0,
#             A=self.f_weights,
#             A_js=self.f_weights_js,
#             X=self.f_signals,
#             X_js=self.f_signals_js,
#             beta=0.0,
#             Y=self.sigs,
#             tag='filters'
#             )]

#     def plan_encoders(self):
#         return [plan_ragged_gather_gemv(
#             self.queue,
#             Ms=self.enc_Ms,
#             Ns=self.enc_Ns,
#             alpha=1.0,
#             A=self.enc_weights,
#             A_js=self.enc_weights_js,
#             X=self.sigs,
#             X_js=self.enc_signals_js,
#             beta=1.0,
#             Y_in=self.pop_bias,
#             Y=self.pop_J,
#             tag='encoders'
#             )]

#     def plan_pop_lifs(self):
#         if not self.pop_lif_idxs:
#             return []

#         lif_start = self.pop_lif_start
#         lif_end = self.pop_lif_end

#         return [plan_lif(
#             queue=self.queue,
#             V=self.pop_lif_voltage.cl_buf,
#             RT=self.pop_lif_reftime.cl_buf,
#             J=self.pop_lif_J.cl_buf[lif_start:lif_end],
#             OV=self.pop_lif_voltage.cl_buf,
#             ORT=self.pop_lif_reftime.cl_buf,
#             OS=self.pop_output.cl_buf[lif_start:lif_end],
#             dt=self.model.dt,
#             tau_ref=self.pop_lif_rep.tau_ref,
#             tau_rc=self.pop_lif_rep.tau_rc,
#             upsample=self.pop_lif_rep.upsample,
#             V_threshold=1.0,
#             )]

#     def plan_pop_directs_pre(self):
#         if self.pop_direct_idxs:
#             self.pop_direct_copy_Ms = [1]
#             self.pop_direct_copy_Ns = [1]
#             self.pop_direct_copy_A = self.RaggedArray([[1]])
#             self.pop_direct_copy_A_js = self.RaggedArray(
#                 [[0]] * len(self.pop_direct_idxs))
#             self.pop_direct_copy_X_js = self.RaggedArray(
#                 [[di] for di in self.pop_direct_idxs])

#             return [plan_ragged_gather_gemv(
#                 self.queue,
#                 Ms=self.pop_direct_copy_Ms,
#                 Ns=self.pop_direct_copy_Ns,
#                 alpha=1.0,
#                 A=self.pop_direct_copy_A,
#                 A_js=self.pop_direct_copy_A_js,
#                 X=self.pop_J,
#                 X_js=self.pop_direct_copy_X_js,
#                 beta=0.0,
#                 Y=self.pop_direct_ins,
#                 tag='pop_direct_copy_in')]
#         else:
#             return []

#     def plan_pop_directs_post(self):
#         if self.pop_direct_idxs:
#             self.pop_direct_outs_js = self.RaggedArray(
#                 [[ii] for ii, di in enumerate(self.pop_direct_idxs)])
#             self.pop_direct_output = self.pop_output[self.pop_direct_idxs]

#             return [plan_ragged_gather_gemv(
#                 self.queue,
#                 Ms=self.pop_direct_copy_Ms,
#                 Ns=self.pop_direct_copy_Ns,
#                 alpha=1.0,
#                 A=self.pop_direct_copy_A,
#                 A_js=self.pop_direct_copy_A_js,
#                 X=self.pop_direct_outs,
#                 X_js=self.pop_direct_outs_js,
#                 beta=0.0,
#                 Y=self.pop_direct_output,
#                 tag='pop_direct_copy_out')]
#         else:
#             return []

#     def enqueue_pop_directs(self):
#         if self.pop_direct_idxs:
#             self.pop_direct_ins_npy.buf[:] = self.pop_direct_ins.buf

#             for ii, di in enumerate(self.pop_direct_idxs):
#                 nl = self.nonlinearities[di]
#                 indata = self.pop_direct_ins_npy[ii]
#                 self.pop_direct_outs_npy[ii][...] = nl.fn(indata)
#             cl.enqueue_copy(self.queue,
#                             self.pop_direct_outs.cl_buf.data,
#                             self.pop_direct_outs_npy.buf,
#                            )

#     def plan_decoders(self):
#         return [plan_ragged_gather_gemv(
#             self.queue,
#             Ms=self.dec_Ms,
#             Ns=self.dec_Ns,
#             alpha=1.0,
#             A=self.dec_weights,
#             A_js=self.dec_weights_js,
#             X=self.pop_output,
#             X_js=self.dec_pops_js,
#             beta=0.0,
#             Y=self.sigs_ic,
#             tag='decoders',
#             )]

#     def plan_probes(self):
#         plans = {}
#         for period in self.sig_probes_output:
#             probe_out = self.sig_probes_output[period]
#             buflen = self.sig_probes_buflen[period]
#             A_js = self.sig_probes_A_js[period]
#             X_js = self.sig_probes_X_js[period]

#             write_plan = plan_ragged_gather_gemv(
#                 self.queue,
#                 Ms=self.sig_probes_Ms,
#                 Ns=self.sig_probes_Ns,
#                 alpha=1.0,
#                 A=self.sig_probes_A,
#                 A_js=A_js,
#                 X=self.sigs,
#                 X_js=X_js,
#                 beta=0.0,
#                 Y=probe_out,
#                 tag='probe:%i' % period,
#                 )
#             advance_plan = plan_inc(
#                 self.queue,
#                 probe_out.cl_starts,
#                 amt=buflen,
#                 tag='probe_inc:%i' % period,
#                 )
#             plans[period] = (write_plan, advance_plan,)
#         return plans

#     def plan_all(self):
#         # see sim_npy for rationale of this ordering
#         self.plans_pre_direct = (self.plan_encoders()
#                                  + self.plan_pop_lifs()
#                                  + self.plan_pop_directs_pre()
#                                 )

#         self.plans_post_direct = (self.plan_pop_directs_post()
#                                   + self.plan_decoders()
#                                   + self.plan_copy_sigs()
#                                   + self.plan_filters()
#                                   + self.plan_transforms()
#                                  )
#         self.probe_plans = self.plan_probes()
#         self._iter = self.run_iter()
#         assert 0 == self._iter.send(None)

#     def run_iter(self):
#         pre_direct = Prog(self.plans_pre_direct)
#         post_direct = Prog(self.plans_post_direct)
#         by_period = dict((k, Prog(v))
#                          for (k, v) in self.probe_plans.items())
#         while True:
#             self.queue.flush()
#             N = yield self.sim_step
#             while N:
#                 pre_direct.enqueue()
#                 self.enqueue_pop_directs()
#                 post_direct.enqueue()

#                 # -- probes
#                 for period in by_period:
#                     if self.sim_step % period == 0:
#                         by_period[period].enqueue()
#                 self.queue.flush()
#                 #print 'sigs      ', self.sigs.buf
#                 #print 'pop output', self.pop_output.buf
#                 #print 'probe buf ', self.sig_probes_output[1].buf[:10]
#                 #print ''
#                 N -= 1
#                 self.sim_step += 1

#     def step(self):
#         return self.run_steps(1)

#     def run_steps(self, N, verbose=False):
#         rval = self._iter.send(N)
#         self.queue.finish()
#         return rval

#         # -- TODO: refresh this code...
#         if self.profiling:
#             times = [(p.ctime, p) for p in set(self.all_plans)]
#             for (ctime, p) in reversed(sorted(times)):
#                 n_calls = max(1, p.n_calls)
#                 print ctime, p.n_calls, (ctime / n_calls), p.atime, p.btime, p
#         else:
#             self.queue.finish()


#     def probe_data(self, probe):
#         period = int(probe.dt // self.model.dt)
#         last_elem = int(math.ceil(self.sim_step / float(period)))

#         # -- figure out which signal it is among the ones with the same dt
#         sps_dt = [sp for sp in self.model.probes if sp.dt == probe.dt]
#         probe_idx = sps_dt.index(probe)
#         all_rows = self.sig_probes_output[period].buf.reshape(
#                 (-1, self.sig_probes_buflen[period]))
#         assert all_rows.dtype == self.sig_probes_output[period].buf.dtype
#         start = self.sig_probes_output[period].starts[probe_idx]
#         olen = self.sig_probes_output[period].lens[probe_idx]
#         start = start % self.sig_probes_buflen[period]
#         rval = all_rows[:last_elem, start:start + olen]
#         return rval
