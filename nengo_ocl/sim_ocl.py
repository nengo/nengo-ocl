
import os
import pyopencl as cl

import sim_npy
from clraggedarray import CLRaggedArray

from raggedarray import RaggedArray
from clra_gemv import plan_ragged_gather_gemv
from clra_nonlinearities import plan_lif, plan_lif_rate
from plan import Plan, Prog

class Simulator(sim_npy.Simulator):

    def RaggedArray(self, *args, **kwargs):
        val = RaggedArray(*args, **kwargs)
        if len(val.buf) == 0:
            return None
        else:
            return CLRaggedArray(self.queue, val)

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
        sim_npy.Simulator.__init__(self,
                                   model,
                                   )
        if all(isinstance(p, Plan) for p in self._plan):
            self._prog = Prog(self._plan)
        else:
            self._prog = None

    def print_profiling(self):
        print '-' * 80
        print '%s\t%s\t%s\t%s' % (
            'n_calls', 'runtime', 'q-time', 'subtime')
        time_running_kernels = 0.0
        for p in self._plan:
            if isinstance(p, Plan):
                print '%i\t%2.3f\t%2.3f\t%2.3f\t%s' % (
                    p.n_calls, sum(p.ctimes), sum(p.btimes), sum(p.atimes), p)
                time_running_kernels += sum(p.ctimes)
            else:
                print p, getattr(p, 'cumtime', '<unknown>')
        print '-' * 80
        print 'totals:\t%2.3f\t%2.3f\t%2.3f' % (
            time_running_kernels, 0.0, 0.0)
        import matplotlib.pyplot as plt
        for p in self._plan:
            plt.plot(p.btimes)
            #print p.btimes
        plt.show()

    def run_steps(self, N, verbose=False):
        if self._prog is None:
            for i in xrange(N):
                self.step(self.profiling)
        else:
            self._prog.call_n_times(N, self.profiling)

    def _prep_all_data(self):
        # -- replace the numpy-allocated RaggedArray with OpenCL one
        self.all_data = CLRaggedArray(self.queue, self.all_data)

    def plan_ragged_gather_gemv(self, *args, **kwargs):
        return plan_ragged_gather_gemv(
            self.queue, *args, **kwargs)

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

