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

from collections import OrderedDict
import datetime

import sys
import time
import yaml

import numpy as np
import pyopencl as cl

import nengo
from nengo.networks.circularconvolution import circconv

import nengo_ocl

if len(sys.argv) not in (3, 4):
    print(__doc__)
    sys.exit()

if sys.argv[1] == "ref":
    sim_name = "ref" if len(sys.argv) == 3 else sys.argv[3]
    sim_class = nengo.Simulator
    sim_kwargs = {}
elif sys.argv[1].startswith("ocl"):
    assert sys.argv[1] in ("ocl", "ocl_profile")
    profiling = sys.argv[1] == "ocl_profile"

    ctx = cl.create_some_context()
    sim_name = ctx.devices[0].name if len(sys.argv) == 3 else sys.argv[3]
    sim_class = nengo_ocl.Simulator
    sim_kwargs = dict(context=ctx, profiling=profiling)
else:
    raise Exception("unknown sim", sys.argv[1])


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
    with nengo.Network(seed=9) as model:
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
        with sim_class(model, **sim_kwargs) as sim:
            t_sim = time.time()

            # -- warmup
            sim.run(0.01)
            t_warm = time.time()

            # -- long-term timing
            sim.run(simtime)
            t_run = time.time()

            if getattr(sim, "profiling", False):
                sim.print_profiling(sort=1)

        # error for sanity checking (should be below ~0.25, definitely 0.5)
        y = sim.data[C_p]
        crms = nengo.utils.numpy.rms(c)
        rmse = (nengo.utils.numpy.rms(y - c[None, :], axis=1) / crms).mean()

        records.append(
            OrderedDict(
                (
                    ("benchmark", "circ-conv"),
                    ("name", sim_name),
                    ("dim", dim),
                    ("simtime", simtime),
                    ("neurons_per_product", neurons_per_product),
                    ("neurons", sum(e.n_neurons for e in model.all_ensembles)),
                    ("status", "ok"),
                    ("profiling", getattr(sim, "profiling", 0)),
                    ("buildtime", t_sim - t_start),
                    ("warmtime", t_warm - t_sim),
                    ("runtime", t_run - t_warm),
                    ("rmse", rmse),
                )
            )
        )
        print(records[-1])
        print("%s, dims=%d successful" % (sim_name, dim))
        del model, sim
    except Exception as e:
        records.append(
            OrderedDict(
                (
                    ("benchmark", "circ-conv"),
                    ("name", sim_name),
                    ("dim", dim),
                    ("simtime", simtime),
                    ("neurons_per_product", neurons_per_product),
                    ("status", "exception"),
                    ("exception", str(e)),
                )
            )
        )
        print(records[-1])
        print("%s, dims=%d exception" % (sim_name, dim))
        raise

filename = "records_circconv_%s.yml" % (
    (datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
)
with open(filename, "w") as fh:
    yaml.dump(records, fh)
