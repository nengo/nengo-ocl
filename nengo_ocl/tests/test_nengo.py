"""
Black-box testing of the nengo_ocl.Simulator using Nengo tests.

TestCase classes are added automatically from nengo.tests,
but you can still run individual test files like this:

$ py.test nengo_ocl/tests/test_simulator.py -k test_ensemble.test_scalar

See http://pytest.org/latest/usage.html for more invocations.

"""
import sys

import nengo
import nengo.tests.test_synapses
from nengo.utils.testing import allclose
import pyopencl as cl
import pytest

import nengo_ocl

ctx = cl.create_some_context()


class Simulator(nengo_ocl.Simulator):
    def __init__(self, *args, **kwargs):
        super(Simulator, self).__init__(*args, context=ctx, **kwargs)


def allclose_tol(*args, **kwargs):
    """Use looser tolerance"""
    kwargs.setdefault('atol', 2e-7)
    return allclose(*args, **kwargs)


nengo.tests.test_synapses.allclose = allclose_tol  # looser tolerances


if __name__ == '__main__':
    # To profile, run `python -m cProfile -o test_sim.log test_nengo.py`.
    # Appending the argument `-k <filter>` allows you to control which tests
    # are run (e.g. `-k "test_ensemble."` runs all tests in test_ensemble.py).
    pytest.main(sys.argv)
