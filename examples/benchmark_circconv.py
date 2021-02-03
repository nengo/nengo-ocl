#!/usr/bin/env python
"""usage: python benchmark_circonv.py (ref|ocl) d1,... [name]

(ref|ocl)
  Passing 'ref' will use the reference simulator, `nengo.Simulator`,
  while passing 'ocl' will use `nengo_ocl.Simulator`.

d1,...
  A comma separated list of integers referring to
  the number of dimensions in the vectors being convolved in the benchmark.
  A typical value would be 16,32,64,128,256,512.

name
  An optional name to give to the benchmark run. The name will be
  displayed in the legend plotted with `view_records.py`.
  If not given, one will be automatically generated for you.

Example usage:
  python benchmark_circconv.py ref 2,4,6 "Reference simulator"
"""

import datetime
import sys
import time

import nengo
import numpy as np
import pyopencl as cl
import yaml
from nengo.networks.circularconvolution import circconv

import nengo_ocl
from nengo_ocl.utils import SimRunner

if len(sys.argv) not in (3, 4, 5):
    print(__doc__)
    sys.exit()

sim_runner = SimRunner.get_runner(
    sys.argv[1], name=sys.argv[3] if len(sys.argv) > 3 else None
)
dims = map(int, sys.argv[2].split(","))

# neurons_per_product = 128
neurons_per_product = 256
simtime = 1.0
radius = 1

records = []

for i, dim in enumerate(dims):

    rng = np.random.RandomState(123)
    a = rng.normal(scale=np.sqrt(1.0 / dim), size=dim)
    b = rng.normal(scale=np.sqrt(1.0 / dim), size=dim)
    c = circconv(a, b)

    # --- Model
    with nengo.Network(seed=9) as net:
        sim_runner.configure_network(net)

        inputA = nengo.Node(a)
        inputB = nengo.Node(b)
        A = nengo.networks.EnsembleArray(neurons_per_product, dim, radius=radius)
        B = nengo.networks.EnsembleArray(neurons_per_product, dim, radius=radius)
        C = nengo.networks.EnsembleArray(neurons_per_product, dim, radius=radius)
        D = nengo.networks.CircularConvolution(
            neurons_per_product, dim, input_magnitude=radius
        )

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
        sim = sim_runner.make_sim(net)
        with sim:
            t_sim = time.time()

            # -- warmup
            sim_runner.run_sim(sim, simtime=0.01)
            t_warm = time.time()

            # -- long-term timing
            sim_runner.run_sim(sim, simtime=simtime)
            t_run = time.time()

            if getattr(sim, "profiling", False):
                sim.print_profiling(sort=1)

        # error for sanity checking (should be below ~0.25, definitely 0.5)
        y = sim.data[C_p]
        crms = nengo.utils.numpy.rms(c)
        rmse = (nengo.utils.numpy.rms(y - c[None, :], axis=1) / crms).mean()

        records.append(
            {
                "benchmark": "circ-conv",
                "name": sim_runner.name,
                "dim": dim,
                "simtime": simtime,
                "neurons_per_product": neurons_per_product,
                "neurons": sum(e.n_neurons for e in net.all_ensembles),
                "status": "ok",
                "profiling": getattr(sim, "profiling", 0),
                "buildtime": t_sim - t_start,
                "warmtime": t_warm - t_sim,
                "runtime": t_run - t_warm,
                "rmse": rmse,
            }
        )
        print(records[-1])
        print("%s, dims=%d successful" % (sim_runner.name, dim))

    except Exception as e:
        records.append(
            {
                "benchmark": "circ-conv",
                "name": sim_runner.name,
                "dim": dim,
                "simtime": simtime,
                "neurons_per_product": neurons_per_product,
                "status": "exception",
                "exception": str(e),
            }
        )
        print(records[-1])
        print("%s, dims=%d exception" % (sim_runner.name, dim))
        raise

if len(sys.argv) > 4:
    filename = sys.argv[4]
else:
    filename = "records_circconv_%s.yml" % (
        (datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    )
with open(filename, "w") as fh:
    yaml.dump(records, fh)
