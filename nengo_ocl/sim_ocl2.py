import os
import pyopencl as cl

import sim_npy2
from sim_ocl import RaggedArray

from ocl.gemv_batched import plan_ragged_gather_gemv
#from ocl.lif import plan_lif
#from ocl.elemwise import plan_copy, plan_inc

class Simulator(sim_npy2.Simulator):

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
        sim_npy2.Simulator.__init__(self,
                                    model,
                                    n_prealloc_probes=n_prealloc_probes)

    def RaggedArray(self, listoflists):
        return RaggedArray(self.queue, listoflists)

    def plan_ragged_gather_gemv(self, *args, **kwargs):
        return plan_ragged_gather_gemv(
            self.queue, *args, **kwargs)



