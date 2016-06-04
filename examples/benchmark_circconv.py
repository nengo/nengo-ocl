from collections import OrderedDict
import datetime
try:
    import cPickle as pickle
except ImportError:
    import pickle
import sys
import time

import numpy as np
import pyopencl as cl

import nengo
from nengo.networks.circularconvolution import circconv
# nengo.log('debug')

import nengo_ocl


if sys.argv[1] == 'ref':
    sim_name = 'ref'
    sim_class = nengo.Simulator
elif sys.argv[1] == 'ocl':
    ctx = cl.create_some_context()
    sim_name = ctx.devices[0].name
    sim_class = nengo_ocl.Simulator
else:
    raise Exception('unknown sim', sys.argv[1])

dims = map(int, sys.argv[2].split(','))

# neurons_per_product = 128
neurons_per_product = 256
simtime = 1.0
radius = 1

records = []

for i, dim in enumerate(dims):

    rng = np.random.RandomState(123)
    a = rng.normal(scale=np.sqrt(1./dim), size=dim)
    b = rng.normal(scale=np.sqrt(1./dim), size=dim)
    c = circconv(a, b)

    # --- Model
    with nengo.Network(seed=9) as model:
        inputA = nengo.Node(a)
        inputB = nengo.Node(b)
        A = nengo.networks.EnsembleArray(
            neurons_per_product, dim, radius=radius)
        B = nengo.networks.EnsembleArray(
            neurons_per_product, dim, radius=radius)
        C = nengo.networks.EnsembleArray(
            neurons_per_product, dim, radius=radius)
        D = nengo.networks.CircularConvolution(
            neurons_per_product, dim, input_magnitude=radius)

        nengo.Connection(inputA, A.input, synapse=None)
        nengo.Connection(inputB, B.input, synapse=None)
        nengo.Connection(A.output, D.A)
        nengo.Connection(B.output, D.B)
        nengo.Connection(D.output, C.input)

        A_p = nengo.Probe(A.output, synapse=0.03)
        B_p = nengo.Probe(B.output, synapse=0.03)
        C_p = nengo.Probe(C.output, synapse=0.03)
        D_p = nengo.Probe(D.output, synapse=0.03)

    # --- Simulation
    try:
        t_start = time.time()

        # -- build
        sim = sim_class(model)
        t_sim = time.time()

        # -- warmup
        sim.run(0.01)
        t_warm = time.time()

        # -- long-term timing
        sim.run(simtime)
        t_run = time.time()

        # error for sanity checking (should be below ~0.25, definitely 0.5)
        y = sim.data[C_p]
        crms = nengo.utils.numpy.rms(c)
        rmse = (nengo.utils.numpy.rms(y - c[None, :], axis=1) / crms).mean()

        records.append(OrderedDict((
            ('benchmark', 'circ-conv'),
            ('name', sim_name),
            ('dim', dim),
            ('simtime', simtime),
            ('neurons_per_product', neurons_per_product),
            ('neurons', sum(e.n_neurons for e in model.all_ensembles)),
            ('status', 'ok'),
            ('profiling', getattr(sim, 'profiling', 0)),
            ('buildtime', t_sim - t_start),
            ('warmtime', t_warm - t_sim),
            ('runtime', t_run - t_warm),
            ('rmse', rmse),
            )))
        print(records[-1])
        print("%s, dims=%d successful" % (sim_name, dim))
    except Exception as e:
        records.append(OrderedDict((
            ('benchmark', 'circ-conv'),
            ('name', sim_name),
            ('dim', dim),
            ('simtime', simtime),
            ('neurons_per_product', neurons_per_product),
            ('status', 'exception'),
            ('exception', str(e)),
            )))
        print(records[-1])
        print("%s, dims=%d exception" % (sim_name, dim))
        raise

filename = "records_circconv_%s.pkl" % ((
    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
f = open(filename, 'wb')
pickle.dump(records, f)
f.close()
