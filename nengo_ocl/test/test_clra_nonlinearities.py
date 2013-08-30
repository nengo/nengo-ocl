import os
import numpy as np
import pyopencl as cl
from array import to_device
from lif import plan_lif
ctx = cl.create_some_context()

profiling = bool(int(os.getenv("NENGO_OCL_PROFILING")))

def test_lif0():
    if profiling:
        queue = cl.CommandQueue(
            ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(ctx)
    N_neurons = 50000
    N_iters = 50

    J = to_device(queue, np.random.rand(N_neurons).astype('float32'))
    V = to_device(queue, np.random.rand(N_neurons).astype('float32'))
    RT = to_device(queue, np.random.rand(N_neurons).astype('float32'))
    OS = to_device(queue, np.random.rand(N_neurons).astype('float32'))

    plan = plan_lif(queue, V, RT, J, V, RT, OS,
            .001, .02, .002, 1.0, 2)

    for ii in range(N_iters):
        plan(profiling=profiling)
    print 'time per call', plan.ctime / plan.n_calls

