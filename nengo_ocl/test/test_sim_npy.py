"""
Black-box testing of the sim_npy Simulator.

TestCase classes are added automatically from
nengo.tests.helpers.simulator_test_cases, but
you can still run individual test files like this:

    nosetests -sv test/test_sim_npy.py:TestSimulator.test_simple_direct_mode

"""

from nengo_ocl.tricky_imports import unittest
from nengo.tests.helpers import NengoTestLoader
from nengo.tests.helpers import load_nengo_tests
from nengo_ocl import sim_npy

from nengo.tests.test_simulator import TestSimulator, TestNonlinear
TestSimulator.Simulator = sim_npy.Simulator
TestNonlinear.Simulator = sim_npy.Simulator

load_tests = load_nengo_tests(sim_npy.Simulator)

if __name__ == '__main__':
   unittest.main(testLoader=NengoTestLoader(sim_npy.Simulator))
