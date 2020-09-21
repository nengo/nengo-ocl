# pylint: disable=missing-module-docstring,missing-function-docstring

import nengo
import numpy as np

import nengo_ocl.operators as ocl_ops


def test_remove_unmodified_resets_functional(Simulator, monkeypatch, seed, allclose):
    assert ocl_ops.remove_unmodified_resets in ocl_ops.simplifications

    simtime = 0.1

    with nengo.Network(seed=seed) as net:
        ens = nengo.Ensemble(10, 2, noise=nengo.processes.WhiteNoise())
        ens_p = nengo.Probe(ens)

    monkeypatch.setattr(ocl_ops, "simplifications", [ocl_ops.remove_unmodified_resets])

    with Simulator(net) as sim_rem:
        ops_rem = sim_rem.operators
        sim_rem.run(simtime)

    monkeypatch.setattr(ocl_ops, "simplifications", [])

    with Simulator(net) as sim:
        ops = sim.operators
        sim.run(simtime)

    assert len(ops_rem) == len(ops) - 1
    assert not np.allclose(sim.data[ens_p], 0, atol=1e-2)
    assert allclose(sim_rem.data[ens_p], sim.data[ens_p])


def test_remove_zero_incs_functional(Simulator, monkeypatch, seed, allclose):
    assert ocl_ops.remove_zero_incs in ocl_ops.simplifications

    simtime = 0.1

    with nengo.Network(seed=seed) as net:
        ens = nengo.Ensemble(10, 2, noise=nengo.processes.WhiteNoise())
        ens_p = nengo.Probe(ens)

    monkeypatch.setattr(ocl_ops, "simplifications", [ocl_ops.remove_zero_incs])
    with Simulator(net) as sim_rem:
        ops_rem = sim_rem.operators
        sim_rem.run(simtime)

    monkeypatch.setattr(ocl_ops, "simplifications", [])
    with Simulator(net) as sim:
        ops = sim.operators
        sim.run(simtime)

    assert len(ops_rem) == len(ops) - 1
    assert sum("encod" in op.tag for op in ops if op.tag) == 1  # encoding op
    assert sum("encod" in op.tag for op in ops_rem if op.tag) == 0  # no encoding op
    assert not np.allclose(sim.data[ens_p], 0, atol=1e-2)
    assert allclose(sim_rem.data[ens_p], sim.data[ens_p])


def test_all_simps_functional(Simulator, monkeypatch, seed, allclose):
    simtime = 0.1

    with nengo.Network(seed=seed) as net:
        ens = nengo.Ensemble(
            10,
            1,
            noise=nengo.processes.WhiteNoise(),
            label="ens",
        )
        p = nengo.Probe(ens)

    with Simulator(net) as sim_rem:
        ops_rem = sim_rem.operators
        sim_rem.run(simtime)

    monkeypatch.setattr(ocl_ops, "simplifications", [])
    with Simulator(net) as sim:
        ops = sim.operators
        sim.run(simtime)

    assert len(ops_rem) == len(ops) - 2  # encoding op and reset removed
    assert sum("encod" in op.tag for op in ops if op.tag) == 1  # encoding op
    assert sum("encod" in op.tag for op in ops_rem if op.tag) == 0  # no encoding op
    assert not np.allclose(sim.data[p], 0, atol=1e-2)
    assert allclose(sim_rem.data[p], sim.data[p])
