"""
Black-box testing of the sim_ocl Simulator.

TestCase classes are added automatically from
nengo.tests.helpers.simulator_test_cases, but
you can still run individual test files like this:

$ python test/test_sim_ocl.py test_ensemble.TestEnsemble.test_lifrate

"""

from nengo_ocl.tricky_imports import unittest
from nengo.tests.helpers import NengoTestLoader
from nengo.tests.helpers import load_nengo_tests
from nengo_ocl import sim_ocl

import pyopencl as cl

ctx = cl.create_some_context()

def Ocl2Simulator(*args, **kwargs):
    kwargs['context'] = ctx
    return sim_ocl.Simulator(*args, **kwargs)

# -- see comments in test_sim_npy.py for why these two
#    classes are treated differently.
from nengo.tests.test_simulator import TestSimulator, TestNonlinear
TestSimulator.Simulator = staticmethod(Ocl2Simulator)
TestNonlinear.Simulator = staticmethod(Ocl2Simulator)

load_tests = load_nengo_tests(Ocl2Simulator)


if __name__ == '__main__':
   unittest.main(testLoader=NengoTestLoader(Ocl2Simulator))
