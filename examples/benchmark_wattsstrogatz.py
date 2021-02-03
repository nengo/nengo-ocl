#!/usr/bin/env python
"""usage: python benchmark_wattsstrogatz.py (ref|ocl) n1,... [name]

(ref|ocl|dl)
  Passing 'ref' will use the reference simulator, `nengo.Simulator`,
  while passing 'ocl' will use `nengo_ocl.Simulator`.

n1,...
  A comma separated list of integers referring to the number of neurons
  in the Ensemble being recurrently connected in the benchmark.
  A typical value would be 16,32,64,128,256,512.

name
  An optional name to give to the benchmark run. The name will be
  displayed in the legend plotted with `view_records.py`.
  If not given, one will be automatically generated for you.

Example usage:
  python benchmark_wattsstrogatz.py ref 128,256,512 "Reference simulator"
"""

import datetime
import sys
import time

import nengo
import numpy as np
import pyopencl as cl
import yaml
from nengo.utils.numpy import scipy_sparse

import nengo_ocl
from nengo_ocl.utils import SimRunner

# --- User options
simtime = 1.0
fan_outs = 100
rewire_frac = 0.2
directed = True
# n_prealloc_probes = 32  # this applies to OCL and DL
sparse = True  # setting to False will probably overwhelm your GPU memory

# --- Command line options
if len(sys.argv) not in (3, 4, 5):
    print(__doc__)
    sys.exit()

sim_runner = SimRunner.get_runner(
    sys.argv[1], name=sys.argv[3] if len(sys.argv) > 3 else None
)
ns_neurons = map(int, sys.argv[2].split(","))


def wattsstrogatz_adjacencies(
    n, fan_outs=fan_outs, rewire_frac=rewire_frac, directed=directed
):
    # the ring network
    offsets = list(range(1, fan_outs + 1))
    offsets.extend([offs - n for offs in offsets])
    if not directed:
        offsets.extend([-offs for offs in offsets])
    swmat = scipy_sparse.diags(
        np.ones((len(offsets), n), dtype=np.float32),
        offsets=offsets,
        shape=(n, n),
        format="csr",
    )
    # random rewires
    n_edges = len(swmat.data)
    rewires_bool = np.random.random(n_edges) < rewire_frac
    rewire_inds = rewires_bool.nonzero()[0]
    rewire_to = np.random.randint(n, size=rewire_inds.size)
    swmat.indices[rewire_inds] = rewire_to
    return swmat


records = []

for i, n_neurons in enumerate(ns_neurons):

    rng = np.random.RandomState(123)
    # a = rng.normal(scale=np.sqrt(1./dim), size=n_neurons)

    # --- Model
    with nengo.Network(seed=9) as net:
        sim_runner.configure_network(net)

        # inputA = nengo.Node(a)
        neuron_type = nengo.neurons.LIF(tau_ref=0.001)
        ens = nengo.Ensemble(n_neurons, 1, neuron_type=neuron_type)
        # nengo.Connection(inputA, ens.neurons, synapse=0.03)

        weimat = wattsstrogatz_adjacencies(n_neurons)
        if sparse:
            transform = nengo.transforms.Sparse((n_neurons, n_neurons), init=weimat)
        else:
            transform = weimat.toarray()
        nengo.Connection(ens.neurons, ens.neurons, transform=transform, synapse=0.1)

        # A_p = nengo.Probe(A.output, synapse=0.03)
        E_p = nengo.Probe(ens.neurons, synapse=0.03)

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

        records.append(
            {
                "benchmark": "watts-strogatz",
                "name": sim_runner.name,
                "neurons": n_neurons,
                "synapses": n_neurons * fan_outs,
                "simtime": simtime,
                "status": "ok",
                "profiling": getattr(sim, "profiling", 0),
                "buildtime": t_sim - t_start,
                "warmtime": t_warm - t_sim,
                "runtime": t_run - t_warm,
            }
        )
        print(records[-1])
        print("%s, n_neurons=%d successful" % (sim_runner.name, n_neurons))

    except Exception as e:
        records.append(
            {
                "benchmark": "watts-strogatz",
                "name": sim_runner.name,
                "n_neurons": n_neurons,
                "simtime": simtime,
                "status": "exception",
                "exception": str(e),
            }
        )
        print(records[-1])
        print("%s, n_neurons=%d exception" % (sim_runner.name, n_neurons))
        raise

if len(sys.argv) > 4:
    filename = sys.argv[4]
else:
    filename = "records_wattsstrogatz_%s.yml" % (
        (datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    )
with open(filename, "w") as fh:
    yaml.dump(records, fh)
