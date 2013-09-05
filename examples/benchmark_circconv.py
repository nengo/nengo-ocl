
import numpy as np

import pyopencl as cl

import nengo
from nengo.objects import Constant, Signal
from nengo.templates.circularconv import DirectCircularConvolution, circconv

import nengo_ocl
from nengo_ocl import sim_ocl
from nengo_ocl.plan import Plan

dims = 1000

rng = np.random.RandomState(8238)
a = rng.randn(dims)
b = rng.randn(dims)
# c = circconv(a, b)

### model
m = nengo.Model("")
A = m.add(Constant(n=dims, value=a))
B = m.add(Constant(n=dims, value=b))
C = m.add(Signal(n=dims, name="C"))

DirectCircularConvolution(m, A, B, C)

### simulate
ctx = cl.create_some_context()
sim = sim_ocl.Simulator(ctx, m, profiling=True)
sim.plan_all()

sim.run_steps(10)
for plan in sim._plan:
    print "Plan", plan.name, plan.tag
    print "run time", plan.ctime
    print "wait time", plan.atime + plan.btime

# import cProfile as profile
# def run():
#     sim.run_steps(10)
# profile.run('run()', sort=1)
