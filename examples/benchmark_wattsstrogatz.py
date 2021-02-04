#!/usr/bin/env python

import datetime
import pathlib
import time

import click
import nengo
import numpy as np
import yaml
from nengo.utils.numpy import scipy_sparse

from nengo_ocl.utils import SimRunner


@click.command()
@click.argument("backend", default="ocl")
@click.option(
    "--name",
    default=None,
    help="Name of the backend being benchmarked (defaults to the backend key)",
)
@click.option("--save-dir", default=None, help="Directory in which to save the results")
@click.option("--save-name", default=None, help="Filename to save the results")
@click.option(
    "--neurons",
    default="128,256,512",
    help="Comma-separated list of number of neurons to benchmark",
)
@click.option("--simtime", default=1.0, type=float, help="Amount of simulation time")
@click.option(
    "--fan-outs", default=100, type=int, help="Number of fan-out connections per neuron"
)
@click.option("--rewire-frac", default=0.2, type=float, help="")
@click.option(
    "--directed", default=True, type=bool, help="Whether the network is directed"
)
@click.option(
    "--sparse",
    default=True,
    type=bool,
    help="Whether the connections are represented sparsely "
    "(setting to False may overwhelm system/GPU memory)",
)
def main(
    backend,
    name,
    save_dir,
    save_name,
    neurons,
    simtime,
    fan_outs,
    rewire_frac,
    directed,
    sparse,
):
    """Run the Watts-Strogatz benchmark.

    BACKEND is the backend key to benchmark.
    """

    sim_runner = SimRunner.get_runner(backend, name=name)
    ns_neurons = [int(n) for n in neurons.split(",")]

    records = []

    for i, n_neurons in enumerate(ns_neurons):
        # --- Model
        with nengo.Network(seed=9) as net:
            sim_runner.configure_network(net)

            # inputA = nengo.Node(a)
            neuron_type = nengo.neurons.LIF(tau_ref=0.001)
            ens = nengo.Ensemble(n_neurons, 1, neuron_type=neuron_type)
            # nengo.Connection(inputA, ens.neurons, synapse=0.03)

            weimat = wattsstrogatz_adjacencies(
                n_neurons, fan_outs=fan_outs, rewire_frac=rewire_frac, directed=directed
            )
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

    save_dir = pathlib.Path("." if save_dir is None else save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=False)

    if save_name is None:
        save_name = "records_wattsstrogatz_%s.yml" % (
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )

    with (save_dir / save_name).open("w") as fh:
        yaml.dump(records, fh)


def wattsstrogatz_adjacencies(n, fan_outs, rewire_frac, directed):
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


if __name__ == "__main__":
    main()
