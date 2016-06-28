"""
Black-box testing of the nengo_ocl.Simulator using Nengo tests.

TestCase classes are added automatically from nengo.tests,
but you can still run individual test files like this:

$ py.test nengo_ocl/tests/test_simulator.py -k test_ensemble.test_scalar

See http://pytest.org/latest/usage.html for more invocations.

"""
import sys
import os

import nengo
import nengo.tests.test_synapses
from nengo.utils.testing import allclose, find_modules, load_functions
import pytest


def allclose_tol(*args, **kwargs):
    """Use looser tolerance"""
    kwargs.setdefault('atol', 2e-7)
    return allclose(*args, **kwargs)


nengo_dir = os.path.dirname(nengo.__file__)
modules = find_modules(nengo_dir, prefix='nengo')
tests = load_functions(modules, arg_pattern='^(Ref)?Simulator$')

nengo.tests.test_synapses.allclose = allclose_tol  # looser tolerances

locals().update(tests)

# --- nengo_extras
try:
    import nengo_extras
except ImportError:
    pass
else:
    nengo_extras_dir = os.path.dirname(nengo_extras.__file__)
    modules = find_modules(nengo_extras_dir, prefix='nengo_extras')
    tests = load_functions(modules, arg_pattern='^(Ref)?Simulator$')
    locals().update(tests)


if __name__ == '__main__':
    # To profile, run `python -m cProfile -o test_sim.log test_nengo.py`.
    # Appending the argument `-k <filter>` allows you to control which tests
    # are run (e.g. `-k "test_ensemble."` runs all tests in test_ensemble.py).
    pytest.main(sys.argv)
