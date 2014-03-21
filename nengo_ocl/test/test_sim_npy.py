"""
Black-box testing of the sim_npy Simulator.

TestCase classes are added automatically from
nengo.tests, but you can still run individual
test files like this:

$ py.test test/test_sim_npy.py -k test_ensemble.test_scalar

See http://pytest.org/latest/usage.html for more invocations.

"""
import os
import pytest

import nengo
from nengo.utils.testing import find_modules, load_functions

from nengo_ocl import sim_npy


def pytest_funcarg__Simulator(request):
    """The Simulator class being tested."""
    return sim_npy.Simulator


def pytest_funcarg__RefSimulator(request):
    """The Simulator class being tested."""
    return sim_npy.Simulator


nengo_dir = os.path.dirname(nengo.__file__)
modules = find_modules(nengo_dir, prefix='nengo')
tests = load_functions(modules, arg_pattern='^(Ref)?Simulator$')
locals().update(tests)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
