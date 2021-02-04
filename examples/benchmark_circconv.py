#!/usr/bin/env python

import datetime
import pathlib
import time

import click
import nengo
import numpy as np
import yaml
from nengo.networks.circularconvolution import circconv

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
    "--dims",
    default="2,5,10",
    help="Comma-separated list of dimensions to benchmark",
)
@click.option("--simtime", default=1.0, type=float, help="Amount of simulation time")
@click.option(
    "--neurons-per-product", default=256, type=int, help="Number of neurons per product"
)
def main(backend, name, save_dir, save_name, dims, neurons_per_product, simtime):
    """Run the circular-convolution benchmark.

    BACKEND is the backend key to benchmark.
    """

    sim_runner = SimRunner.get_runner(backend, name=name)
    dims = [int(dim) for dim in dims.split(",")]

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

    save_dir = pathlib.Path("." if save_dir is None else save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=False)

    if save_name is None:
        save_name = "records_circconv_%s.yml" % (
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )

    with (save_dir / save_name).open("w") as fh:
        yaml.dump(records, fh)


if __name__ == "__main__":
    main()
