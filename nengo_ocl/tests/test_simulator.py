import numpy as np
import pytest

import nengo
from nengo.builder import Model
from nengo.builder.operator import DotInc, Reset
from nengo.builder.signal import Signal

import nengo_ocl
from nengo_ocl.version import latest_nengo_version_info


def test_multidotinc_compress(monkeypatch):
    if nengo.version.version_info < (2, 3, 1):  # LEGACY
        # Nengo versions <= 2.3.0 have more stringent op validation which
        # required PreserveValue. That's been removed, so the strict
        # validation causes this test to fail despite it working.
        monkeypatch.setattr(nengo.utils.simulator, "validate_ops", lambda *args: None)

    a = Signal([0, 0])
    b = Signal([0, 0])
    A = Signal([[1, 2], [0, 1]])
    B = Signal([[2, 1], [-1, 1]])
    x = Signal([1, 1])
    y = Signal([1, -1])

    m = Model(dt=0)
    m.operators += [Reset(a), DotInc(A, x, a), DotInc(B, y, a)]
    m.operators += [DotInc(A, y, b), DotInc(B, x, b)]

    with nengo_ocl.Simulator(None, model=m) as sim:
        sim.step()
        assert np.allclose(sim.signals[a], [4, -1])
        assert np.allclose(sim.signals[b], [2, -1])
        sim.step()
        assert np.allclose(sim.signals[a], [4, -1])
        assert np.allclose(sim.signals[b], [4, -2])


def test_error_on_version_in_blacklist(monkeypatch):
    with nengo.Network() as model:
        nengo.Ensemble(10, 1)

    monkeypatch.setattr(nengo.version, "version_info", (2, 1, 1))
    with pytest.raises(ValueError):
        with nengo_ocl.Simulator(model):
            pass


def test_warn_on_future_version(monkeypatch):
    with nengo.Network() as model:
        nengo.Ensemble(10, 1)

    future_version = tuple(v + 1 for v in latest_nengo_version_info)
    monkeypatch.setattr(nengo.version, "version_info", future_version)
    with pytest.warns(UserWarning):
        with nengo_ocl.Simulator(model):
            pass


def test_reset():
    seed = 3

    class CustomProcess(nengo.Process):
        def make_state(self, shape_in, shape_out, dt, dtype=None):
            return {}

        def make_step(self, shape_in, shape_out, dt, rng, state):
            def step(t):
                return rng.uniform(size=shape_out).ravel()

            return step

    with nengo.Network() as model:
        u = nengo.Node(CustomProcess())
        up = nengo.Probe(u)

    with nengo_ocl.Simulator(model, seed=seed) as sim:
        sim.run_steps(10)
        ua = np.array(sim.data[up])

        sim.reset()
        sim.run_steps(10)
        ub = np.array(sim.data[up])

    assert np.allclose(ua, ub)


def test_clear_probes(Simulator, allclose):
    fn = lambda t: np.sin(10 * t)

    with nengo.Network() as net:
        u = nengo.Node(fn)
        up = nengo.Probe(u)

    with Simulator(net, seed=0) as sim:
        sim.run_steps(10)
        assert len(sim.data[up]) == 10
        assert allclose(sim.data[up], fn(sim.trange())[:, None])

        sim.clear_probes()
        assert len(sim.data[up]) == 0

        sim.run_steps(12)
        assert len(sim.data[up]) == 12
        assert allclose(sim.data[up], fn(sim.trange()[10:, None]))
