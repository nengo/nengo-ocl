import nengo
import nengo.tests.test_synapses
from nengo.utils.testing import allclose
import pyopencl as cl
import pytest

import nengo_ocl

from nengo.conftest import plt, rng  # noqa: F401


# --- Ensure all Simulators have same context
ctx = cl.create_some_context()


class OclSimulator(nengo_ocl.Simulator):
    def __init__(self, *args, **kwargs):
        super(OclSimulator, self).__init__(*args, context=ctx, **kwargs)


@pytest.fixture(scope="session")
def Simulator(request):
    """A nengo_ocl.Simulator"""
    return OclSimulator


# --- Change allclose tolerences for some Nengo tests
def allclose_tol(*args, **kwargs):
    """Use looser tolerance"""
    kwargs.setdefault('atol', 2e-7)
    return allclose(*args, **kwargs)


nengo.tests.test_synapses.allclose = allclose_tol  # looser tolerances
