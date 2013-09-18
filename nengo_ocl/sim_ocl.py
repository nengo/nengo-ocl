
import os
import time
import collections
import numpy as np
import pyopencl as cl

from . import sim_npy
from .raggedarray import RaggedArray
from .clraggedarray import CLRaggedArray
from .clra_gemv import plan_ragged_gather_gemv
from .clra_nonlinearities import plan_lif, plan_lif_rate, plan_direct, plan_probes
from .plan import Plan, Prog, PythonPlan, HybridProg
from .ast_conversion import OCL_Function

import logging
logger = logging.getLogger(__name__)

class Simulator(sim_npy.Simulator):

    def RaggedArray(self, *args, **kwargs):
        val = RaggedArray(*args, **kwargs)
        if len(val.buf) == 0:
            return None
        else:
            return CLRaggedArray(self.queue, val)

    def __init__(self, model, context, n_prealloc_probes=1000,
                 profiling=None):
        if profiling is None:
            profiling = int(os.getenv("NENGO_OCL_PROFILING", 0))
        self.context = context
        self.profiling = profiling
        if profiling:
            self.queue = cl.CommandQueue(
                context,
                properties=cl.command_queue_properties.PROFILING_ENABLE)
        else:
            self.queue = cl.CommandQueue(context)
        sim_npy.Simulator.__init__(self,
                                   model,
                                   )
        if all(isinstance(p, Plan) for p in self._plan):
            self._prog = Prog(self._plan)
        else:
            self._prog = None

    def _prep_all_data(self):
        # -- replace the numpy-allocated RaggedArray with OpenCL one
        self.all_data = CLRaggedArray(self.queue, self.all_data)

    def plan_ragged_gather_gemv(self, *args, **kwargs):
        return plan_ragged_gather_gemv(self.queue, *args, **kwargs)

    def plan_SimDirect(self, ops):
        ### TOOD: test with a hybrid program (Python and OCL)

        ### group nonlinearities
        unique_ops = collections.OrderedDict()
        for op in ops:
            if op.fn not in unique_ops:
                unique_ops[op.fn] = {'in': [], 'out': []}
            unique_ops[op.fn]['in'].append(op.J)
            unique_ops[op.fn]['out'].append(op.output)

        ### make plans
        plans = []
        for fn, signals in unique_ops.items():
            fn_name = fn.__name__

            # check signal input and output shape (implicitly checks
            # for indexing errors)
            vector_dims = lambda shape, dim: len(shape) == 1 and shape[0] == dim
            unit_stride = lambda es: len(es) == 1 and es[0] == 1
            in_dim = signals['in'][0].size
            out_dim = signals['out'][0].size
            for sig_in, sig_out in zip(signals['in'], signals['out']):
                assert sig_in.n == in_dim and sig_out.n == out_dim
                assert vector_dims(sig_in.shape, in_dim)
                assert vector_dims(sig_out.shape, out_dim)
                assert unit_stride(sig_in.elemstrides)
                assert unit_stride(sig_out.elemstrides)

            x = np.zeros(in_dim)
            y = np.asarray(fn(x))
            assert y.size == out_dim

            ### try to get OCL code
            try:
                ocl_fn = OCL_Function(fn)
                Xname = ocl_fn.translator.arg_names[0]
                X = self.all_data[[self.sidx[i] for i in signals['in']]]
                Y = self.all_data[[self.sidx[i] for i in signals['out']]]
                plan = plan_direct(self.queue, ocl_fn.code, ocl_fn.init,
                                   Xname, X, Y, tag=fn_name)
                plans.append(plan)
            except NotImplementedError, AssertionError:
                logger.warning("Function '%s' could not be converted to OCL"
                               % fn_name)

                def temp_fn():
                    for sin, sout in zip(signals['in'], signals['out']):
                        J = self.all_data[self.sidx[sin]]
                        self.all_data[self.sidx[sout]] = fn(J)

                plans.append(PythonPlan(temp_fn, name=fn_name, tag=fn_name))

        return plans

    def plan_SimLIF(self, ops):
        J = self.all_data[[self.sidx[op.J] for op in ops]]
        V = self.all_data[[self.sidx[op.voltage] for op in ops]]
        W = self.all_data[[self.sidx[op.refractory_time] for op in ops]]
        S = self.all_data[[self.sidx[op.output] for op in ops]]
        ref = self.RaggedArray([op.nl.tau_ref for op in ops])
        tau = self.RaggedArray([op.nl.tau_rc for op in ops])
        dt = self.model.dt
        return [plan_lif(self.queue, J, V, W, V, W, S, ref, tau, dt,
                        tag="lif", upsample=1)]

    def plan_SimLIFRate(self, nls):
        raise NotImplementedError()
        J = self.all_data[[self.sidx[nl.input_signal] for nl in nls]]
        R = self.all_data[[self.sidx[nl.output_signal] for nl in nls]]
        ref = self.RaggedArray([nl.tau_ref for nl in nls])
        tau = self.RaggedArray([nl.tau_rc for nl in nls])
        return plan_lif_rate(self.queue, J, R, ref, tau,
                             tag="lif_rate", n_elements=10)

    def plan_probes(self):
        if len(self.model.probes) > 0:
            buf_len = 1000

            probes = self.model.probes
            periods = [int(np.round(float(p.dt) / self.model.dt))
                       for p in probes]

            sim_step = self.all_data[[self.sidx[self.model.steps]]]
            P = self.RaggedArray(periods)
            X = self.all_data[[self.sidx[p.sig] for p in probes]]
            Y = self.RaggedArray(
                [np.zeros((buf_len, p.sig.shape[0])) for p in probes])

            cl_plan = plan_probes(self.queue, sim_step, P, X, Y, tag="probes")

            lengths = [period * buf_len for period in periods]
            def probe_copy_fn(profiling=False):
                copy_step = self._probe_copy_step
                for i, length in enumerate(lengths):
                    if self.sim_step % length == length - 1:
                        y = Y[i]
                        if (copy_step > 0 and
                            (self.sim_step - copy_step) < length):
                            j = (copy_step / periods[i]) % buf_len
                            y = y[j:]
                        self.probe_outputs[probes[i]].extend(y)

            self._probe_copy_step = 0
            self._probe_periods = periods
            self._probe_buffers = Y
            return [cl_plan,
                    PythonPlan(probe_copy_fn, name='probes', tag='probes')]
        else:
            return []

    def post_run(self):
        """Perform cleanup tasks after a run"""

        ### Copy any remaining probe data off device
        copy_step = self._probe_copy_step
        for i, probe in enumerate(self.model.probes):
            period = self._probe_periods[i]
            buf_len = self._probe_buffers.shape0s[i]
            pos = (self.sim_step / period) % buf_len
            old_pos = (copy_step / period) % buf_len
            if pos > 0:
                buffer = self._probe_buffers[i]
                if ((self.sim_step - copy_step) < period * buf_len
                    and old_pos < pos):
                    self.probe_outputs[probe].append(buffer[old_pos:pos])
                else:
                    self.probe_outputs[probe].append(buffer[:pos])

        self._probe_copy_step = self.sim_step

        if self.profiling > 1:
            self.print_profiling()

    def print_profiling(self):
        ### TODO: fix this to work with PythonPlan
        print '-' * 80
        print '%s\t%s\t%s\t%s' % (
            'n_calls', 'runtime', 'q-time', 'subtime')
        time_running_kernels = 0.0
        for p in self._plan:
            if isinstance(p, Plan):
                print '%i\t%2.3f\t%2.3f\t%2.3f\t<%s, tag=%s>' % (
                    p.n_calls, sum(p.ctimes), sum(p.btimes), sum(p.atimes),
                    p.name, p.tag)
                time_running_kernels += sum(p.ctimes)
            else:
                print p, getattr(p, 'cumtime', '<unknown>')
        print '-' * 80
        print 'totals:\t%2.3f\t%2.3f\t%2.3f' % (
            time_running_kernels, 0.0, 0.0)

        #import matplotlib.pyplot as plt
        #for p in self._plan:
            #if isinstance(p, Plan):
                #plt.plot(p.btimes)
        #plt.show()

    def run_steps(self, N, verbose=False):
        if self._prog is None:
            for i in xrange(N):
                self.step()
        else:
            self._prog.call_n_times(N, self.profiling)
        self.post_run()

    def probe_data(self, probe):
        return np.vstack(self.probe_outputs[probe])
