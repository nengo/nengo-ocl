
import sys
import datetime
import time
import numpy as np
import pickle as pickle

import pyopencl as cl

import nengo
from nengo.templates import EnsembleArray
from nengo.networks.circularconvolution import circconv, CircularConvolution

from nengo_ocl import sim_ocl

def ReferenceSimulator(model):
    return ('ref', nengo.simulator.Simulator(model))

def OclSimulator(model):
    ctx = cl.create_some_context()
    name = ctx.devices[0].name
    return (name, sim_ocl.Simulator(model, ctx))

if sys.argv[1] == 'ref':
    sim_classes = [ReferenceSimulator]
elif sys.argv[1] == 'ocl':
    sim_classes = [OclSimulator]
else:
    raise Exception('unknown sim', sys.argv[1])

dims = map(int, sys.argv[2].split(','))

neurons_per_product = 128
simtime = 1.0

records = []

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

    try:
        ### simulation
        t_start = time.time()
        sim_name, sim = model.simulator(sim_class=sim_classes[0])

        # -- warmup
        t_sim = time.time()
        sim.run(0.01)
        t_warm = time.time()

        # -- long-term timing
        sim.run(simtime)
        t_run = time.time()

        records.append({
            'benchmark': 'circ-conv',
            'name': sim_name,
            'dim': dim,
            'simtime': simtime,
            'neurons_per_product': neurons_per_product,
            'status': 'ok',
            'profiling': getattr(sim, 'profiling', 0),
            'buildtime': t_sim - t_start,
            'warmtime': t_warm - t_sim,
            'runtime': t_run - t_warm,
            })
        print "%s, dims=%d successful" % (sim_name, dim)
    except Exception as e:
        records.append({
            'benchmark': 'circ-conv',
            'name': sim_name,
            'dim': dim,
            'simtime': simtime,
            'neurons_per_product': neurons_per_product,
            'status': 'exception',
            'exception': str(e)
            })
        print "%s, dims=%d exception" % (sim_name, dim)
        print e

filename = "records_circconv_%s.pkl" % ((
    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
f = open(filename, 'w')
pickle.dump(records, f)
f.close()

