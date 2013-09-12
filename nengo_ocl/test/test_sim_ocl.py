"""
Black-box testing of the sim_ocl Simulator.

TestCase classes are added automatically from
nengo.tests.helpers.simulator_test_cases, but
you can still run individual test files like this:

    nosetests -sv test/test_sim_ocl.py:TestSimulator.test_simple_direct_mode

"""

from nengo_ocl.tricky_imports import unittest
from nengo.tests.helpers import NengoTestLoader
from nengo.tests.helpers import load_nengo_tests
from nengo_ocl import sim_ocl

import pyopencl as cl

ctx = cl.create_some_context()

def Ocl2Simulator(model):
    return sim_ocl.Simulator(ctx, model)

load_tests = load_nengo_tests(Ocl2Simulator)

if __name__ == '__main__':
   unittest.main(testLoader=NengoTestLoader(Ocl2Simulator))

