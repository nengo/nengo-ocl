import os
import pyopencl as cl

import sim_npy
from clraggedarray import CLRaggedArray

from raggedarray import RaggedArray
from clra_gemv import plan_ragged_gather_gemv
from clra_nonlinearities import plan_lif

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
                                   n_prealloc_probes=n_prealloc_probes)

        # -- replace the numpy-allocated RaggedArray with OpenCL one
        self.all_data = CLRaggedArray(self.queue, self.all_data)

    def plan_ragged_gather_gemv(self, Ms, Ns, *args, **kwargs):
        return plan_ragged_gather_gemv(
            self.queue, *args, **kwargs)

    def plan_lif(self, *args, **kwargs):
        return plan_lif(self.queue, *args, **kwargs)



