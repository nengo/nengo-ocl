
import sys, os
import datetime
import time
import numpy as np
import pickle as pickle

import pyopencl as cl

import nengo
from nengo.templates import EnsembleArray
from nengo.networks.circularconvolution import circconv, CircularConvolution

from nengo_ocl import sim_npy, sim_ocl

def ReferenceSimulator(model):
    return nengo.simulator.Simulator(model)

def NumpySimulator(model):
    return sim_npy.Simulator(model)

def CpuOclSimulator(model):
    os.environ['PYOPENCL_CTX'] = '1'
    ctx = cl.create_some_context()
    return sim_ocl.Simulator(model, ctx)

def GpuOclSimulator(model):
    os.environ['PYOPENCL_CTX'] = '0'
    ctx = cl.create_some_context()
    return sim_ocl.Simulator(model, ctx)

sim_classes = [ReferenceSimulator, NumpySimulator,
               CpuOclSimulator, GpuOclSimulator]

# dims = [2, 5]
dims = [5, 10, 20, 50, 100]
# dims = [10, 50, 100, 500]
neurons_per_product = 50
simtime = 1.0

runtimes = []
exceptions = []

for i, dim in enumerate(dims):

    n_neurons = neurons_per_product * dim
    n_neurons_d = 2 * neurons_per_product * (
        2*dim - (2 if dim % 2 == 0 else 1))
    radius = 1

    rng = np.random.RandomState(123)
    a = rng.normal(scale=np.sqrt(1./dim), size=dim)
    b = rng.normal(scale=np.sqrt(1./dim), size=dim)
    c = circconv(a, b)

    ### model
    model = nengo.Model("circular convolution")
    inputA = model.make_node("inputA", output=a)
    inputB = model.make_node("inputB", output=b)
    A = model.add(EnsembleArray('A', nengo.LIF(n_neurons), dim, radius=radius))
    B = model.add(EnsembleArray('B', nengo.LIF(n_neurons), dim, radius=radius))
    C = model.add(EnsembleArray('C', nengo.LIF(n_neurons), dim, radius=radius))
    D = model.add(CircularConvolution('D', neurons=nengo.LIF(n_neurons_d),
                                      dimensions=A.dimensions, radius=radius))

    inputA.connect_to(A)
    inputB.connect_to(B)
    A.connect_to(D.A)
    B.connect_to(D.B)
    D.connect_to(C)

    model.probe(A, filter=0.03)
    model.probe(B, filter=0.03)
    model.probe(C, filter=0.03)
    model.probe(D, filter=0.03)

    sim_runtimes = []
    sim_exceptions = []
    for j, sim_class in enumerate(sim_classes):
        sim_name = sim_class.__name__
        try:
            ### simulation
            sim = model.simulator(sim_class=sim_class)

            timer = time.time()
            sim.run(simtime)
            timer = time.time() - timer

            sim_runtimes.append(timer)
            sim_exceptions.append(None)
            print "%s, dims=%d successful" % (sim_name, dim)
        except Exception as e:
            sim_runtimes.append(np.nan)
            sim_exceptions.append(e)
            print "%s, dims=%d exception" % (sim_name, dim)
            print e

    runtimes.append(sim_runtimes)
    exceptions.append(sim_exceptions)

filename = "benchmark_circconv_%s.pkl" % ((
    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
f = open(filename, 'w')
pickle.dump(dict(
    dims=dims, neurons_per_product=neurons_per_product, simtime=simtime,
    sim_class_names=map(lambda x: x.__name__, sim_classes),
    runtimes=runtimes, exceptions=exceptions), f)
f.close()
