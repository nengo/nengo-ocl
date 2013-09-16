
import time
import numpy as np

import nengo
import nengo.core
import nengo.simulator
from nengo.core import Signal

import nengo_ocl
from nengo_ocl.tricky_imports import unittest
from nengo_ocl import sim_ocl
from nengo_ocl import ast_conversion
from nengo_ocl.plan import Plan

import pyopencl as cl

model = nengo.Model("Dummy")
model.prep_for_simulation(model, 0.001)

ctx = cl.create_some_context()

if 1:
    sim = sim_ocl.Simulator(ctx, model, profiling=True)

    t0 = time.time()
    # sim.run(0.2)
    # sim.run(10.0)
    sim.run(9.9985)
    t1 = time.time()
    print "Done in %s seconds" % (t1 - t0)

    t = sim.probe_outputs[sim.model.probed[sim.model.t]]
    t_diff = np.diff(t, axis=0)

    print "t: %f, %f" % (len(t) * sim.model.dt, t[-1])
    print "t_diff: mean %s, std %s, min %s, max %s" % (
        t_diff.mean(), t_diff.std(), t_diff.min(), t_diff.max())

    sim.print_profiling()

else:
    sim = sim_ocl.Simulator(ctx, model, profiling=False)

    print '='*80

    import cProfile as profile
    # sim.run(1.0)
    profile.run('sim.run(10.0)', sort=1)
