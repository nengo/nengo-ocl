import os
import pyopencl as cl

import sim_npy
from .raggedarray import RaggedArray
from .clraggedarray import CLRaggedArray
from .plan import Plan, PythonPlan

from .clra_gemv import plan_ragged_gather_gemv
from .clra_nonlinearities import plan_lif, plan_lif_rate

class Simulator(sim_npy.Simulator):

    def RaggedArray(self, *args, **kwargs):
        val = RaggedArray(*args, **kwargs)
        if len(val.buf) == 0:
            return None
        else:
            return CLRaggedArray(self.queue, val)

    def __init__(self, context, model, n_prealloc_probes=1000,
                 profiling=False):
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
                                   n_prealloc_probes=n_prealloc_probes)

        # -- replace the numpy-allocated RaggedArray with OpenCL one
        self.all_data = CLRaggedArray(self.queue, self.all_data)

    def plan_ragged_gather_gemv(self, *args, **kwargs):
        return plan_ragged_gather_gemv(self.queue, *args, **kwargs)

    def plan_direct(self, nls):
        ### TODO: this is sub-optimal, since it involves copying everything
        ### off the device, running the nonlinearity, then copying back on
        sidx = self.sidx
        def direct():
            for nl in nls:
                J = self.all_data[sidx[nl.input_signal]]
                output = nl.fn(J)
                self.all_data[sidx[nl.output_signal]] = output
        return PythonPlan(direct, name="direct", tag="direct")

    def plan_lif(self, nls):
        J = self.all_data[[self.sidx[nl.input_signal] for nl in nls]]
        V = self.all_data[[self.sidx[self.lif_voltage[nl]] for nl in nls]]
        W = self.all_data[[self.sidx[self.lif_reftime[nl]] for nl in nls]]
        S = self.all_data[[self.sidx[nl.output_signal] for nl in nls]]
        ref = self.RaggedArray([nl.tau_ref for nl in nls])
        tau = self.RaggedArray([nl.tau_rc for nl in nls])
        dt = self.model.dt
        return plan_lif(self.queue, J, V, W, V, W, S, ref, tau, dt,
                        tag="lif", upsample=1, n_elements=10)

    def plan_lif_rate(self, nls):
        J = self.all_data[[self.sidx[nl.input_signal] for nl in nls]]
        R = self.all_data[[self.sidx[nl.output_signal] for nl in nls]]
        ref = self.RaggedArray([nl.tau_ref for nl in nls])
        tau = self.RaggedArray([nl.tau_rc for nl in nls])
        return plan_lif_rate(self.queue, J, R, ref, tau,
                             tag="lif_rate", n_elements=10)

    def step(self):
        for fn in self._plan:
            fn(profiling=self.profiling)
        self.sim_step += 1

    def run_steps(self, N, verbose=False):
        for i in xrange(N):
            self.step()
