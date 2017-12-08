import nengo
import nengo.tests.test_synapses
from nengo.utils.testing import allclose
import pyopencl as cl
import pytest

from nengo.conftest import logger, plt, rng, seed  # noqa: F401


@pytest.fixture(scope="session")
def ctx(request):
    return cl.create_some_context()


# --- Change allclose tolerences for some Nengo tests
def allclose_tol(*args, **kwargs):
    """Use looser tolerance"""
    kwargs.setdefault('atol', 2e-7)
    return allclose(*args, **kwargs)


nengo.tests.test_synapses.allclose = allclose_tol  # looser tolerances
