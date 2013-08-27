"""
Black-box testing of the sim_npy Simulator.

TestCase classes are added automatically from
nengo.tests.helpers.simulator_test_cases, but
you can still run individual test files like this:

    nosetests -sv test/test_sim_npy.py:TestSimulator.test_simple_direct_mode

"""

from nengo.tests.helpers import simulator_suite
from nengo_ocl import sim_npy2

def simulator_allocator(model):
    rval = sim_npy2.Simulator(model)
    rval.alloc_all()
    rval.plan_all()
    return rval

globals().update(simulator_suite(simulator_allocator))

