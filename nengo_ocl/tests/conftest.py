import nengo
import nengo.tests.test_synapses
from nengo.utils.testing import signals_allclose
import pyopencl as cl
import pytest

from pytest_plt.plugin import plt
from pytest_rng.plugin import rng


# copied out of nengo 2.8.0. Might have a pytest builtin version...
@pytest.fixture
def logger(request):
    """A logging.Logger object.

    Please use this if your test emits log messages.

    This will keep saved logs organized in a simulator-specific folder,
    with an automatically generated name.
    """
    dirname = recorder_dirname(request, 'logs')
    logger = Logger(
        dirname, request.module.__name__,
        parametrize_function_name(request, request.function.__name__))
    request.addfinalizer(lambda: logger.__exit__(None, None, None))
    return logger.__enter__()
##


@pytest.fixture(scope="session")
def ctx(request):
    return cl.create_some_context()


# --- Change allclose tolerences for some Nengo tests
def allclose_tol(*args, **kwargs):
    """Use looser tolerance"""
    kwargs.setdefault('atol', 2e-7)
    return signals_allclose(*args, **kwargs)


nengo.tests.test_synapses.signals_allclose = allclose_tol  # looser tolerances
