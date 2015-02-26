"""
Black-box testing of the sim_ocl Simulator.

TestCase classes are added automatically from
nengo.tests, but you can still run individual
test files like this:

$ py.test test/test_sim_ocl.py -k test_ensemble.test_scalar

See http://pytest.org/latest/usage.html for more invocations.

"""
import sys
import os
import pyopencl as cl
import pytest

import nengo
from nengo.utils.testing import find_modules
from nengo_ocl.tests.utils import load_functions

from nengo_ocl import sim_ocl

ctx = cl.create_some_context()


def OclSimulator(*args, **kwargs):
    return sim_ocl.Simulator(*args, context=ctx, **kwargs)


def pytest_funcarg__Simulator(request):
    """The Simulator class being tested."""
    return OclSimulator


def pytest_funcarg__RefSimulator(request):
    """The Simulator class being tested."""
    return OclSimulator


nengo_dir = os.path.dirname(nengo.__file__)
modules = find_modules(nengo_dir, prefix='nengo')
tests = load_functions(modules, arg_pattern='^(Ref)?Simulator$')
locals().update(tests)


if __name__ == '__main__':
    # To profile, run `python -m cProfile -o test_sim_ocl.log test_sim_ocl.py`.
    # Appending the argument `-k <filter>` allows you to control which tests
    # are run (e.g. `-k "test_ensemble."` runs all tests in test_ensemble.py).
    pytest.main(sys.argv)
