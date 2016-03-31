"""
Black-box testing of the nengo_ocl.Simulator using Nengo tests.

TestCase classes are added automatically from nengo.tests,
but you can still run individual test files like this:

$ py.test nengo_ocl/tests/test_simulator.py -k test_ensemble.test_scalar

See http://pytest.org/latest/usage.html for more invocations.

"""
import sys
import os

import pyopencl as cl
import pytest

import nengo
import nengo.tests.test_synapses
from nengo.utils.testing import find_modules, allclose

import nengo_ocl
from nengo_ocl.tests.utils import load_functions

ctx = cl.create_some_context()


def OclSimulator(*args, **kwargs):
    return nengo_ocl.Simulator(*args, context=ctx, **kwargs)


def pytest_funcarg__Simulator(request):
    """The Simulator class being tested."""
    return OclSimulator


def allclose_tol(*args, **kwargs):
    """Use looser tolerance"""
    kwargs.setdefault('atol', 2e-7)
    return allclose(*args, **kwargs)


nengo_dir = os.path.dirname(nengo.__file__)
modules = find_modules(nengo_dir, prefix='nengo')
tests = load_functions(modules, arg_pattern='^Simulator$')

nengo.tests.test_synapses.allclose = allclose_tol  # looser tolerances

locals().update(tests)

# --- nengo_deeplearning
try:
    import nengo_deeplearning
except ImportError:
    pass
else:
    nengo_deeplearning_dir = os.path.dirname(nengo_deeplearning.__file__)
    modules = find_modules(nengo_deeplearning_dir, prefix='nengo_deeplearning')
    tests = load_functions(modules, arg_pattern='^Simulator$')
    locals().update(tests)


if __name__ == '__main__':
    # To profile, run `python -m cProfile -o test_sim.log test_nengo.py`.
    # Appending the argument `-k <filter>` allows you to control which tests
    # are run (e.g. `-k "test_ensemble."` runs all tests in test_ensemble.py).
    pytest.main(sys.argv)
