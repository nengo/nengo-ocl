"""
Black-box testing of the sim_npy Simulator.

TestCase classes are added automatically from
nengo.tests.helpers.simulator_test_cases, but
you can still run individual test files like this:

    nosetests -sv test/test_sim_npy.py:TestSimulator.test_simple_direct_mode

"""

import pyopencl as cl
from nengo.tests.helpers import simulator_test_cases
from nengo_ocl import sim_npy

for TestCase in simulator_test_cases:
    class MyTestCase(TestCase):
        simulator_test_case_ignore = True
        def Simulator(self, model):
            rval = sim_npy.Simulator(model)
            rval.alloc_all()
            rval.plan_all()
            return rval
    MyTestCase.__name__ = TestCase.__name__
    globals()[TestCase.__name__] = MyTestCase
    # -- delete these symbols so that nose will not
    #    detect and run them as extra unit tests.
    del MyTestCase
    del TestCase


