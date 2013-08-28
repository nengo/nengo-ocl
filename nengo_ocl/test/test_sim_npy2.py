"""
Black-box testing of the sim_npy Simulator.

TestCase classes are added automatically from
nengo.tests.helpers.simulator_test_cases, but
you can still run individual test files like this:

    nosetests -sv test/test_sim_npy.py:TestSimulator.test_simple_direct_mode

"""

try:
    # For Python <=2.6
    import unittest2 as unittest
except ImportError:
    import unittest

import nengo.tests.helpers

from nengo_ocl import sim_npy2

def simulator_allocator(model):
    rval = sim_npy2.Simulator(model)
    rval.alloc_all()
    rval.plan_all()
    return rval

load_tests = nengo.tests.helpers.load_nengo_tests(simulator_allocator)


if __name__ == "__main__":
    unittest.main(
        testLoader=nengo.tests.helpers.NengoTestLoader(simulator_allocator))
