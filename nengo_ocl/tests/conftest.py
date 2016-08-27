import nengo
import nengo.tests.test_synapses
from nengo.utils.testing import allclose
import pyopencl as cl
import pytest

from nengo.conftest import plt, rng  # noqa: F401


ctx = cl.create_some_context()
@pytest.fixture(scope="session")


# --- Change allclose tolerences for some Nengo tests
def allclose_tol(*args, **kwargs):
    """Use looser tolerance"""
    kwargs.setdefault('atol', 2e-7)
    return allclose(*args, **kwargs)


nengo.tests.test_synapses.allclose = allclose_tol  # looser tolerances
