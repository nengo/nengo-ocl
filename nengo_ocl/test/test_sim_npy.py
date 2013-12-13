"""
Black-box testing of the sim_npy Simulator.

TestCase classes are added automatically from
nengo.tests.helpers.simulator_test_cases, but
you can still run individual test files like this:

$ python test/test_sim_npy.py test_ensemble.TestEnsemble.test_lifrate

"""

from nengo_ocl.tricky_imports import unittest
from nengo.tests.helpers import NengoTestLoader
from nengo.tests.helpers import load_nengo_tests
import nengo.tests.test_simulator
from nengo_ocl import sim_npy

# -- these TestSimulator and TestNonlinear are handled differently because
# NengoTestLoader only picks up subclasses of SimulatorTestCase. TestSimulat
# and TestNonlinear in nengo do not inherit from SimulatorTestCase.  The
# semantics of SimulatorTestCase can be understood as being "These should pass
# for ANY Simulator". TestSimulator and TestNonlinear are stricter tests of
# the reference simulator's API, which we also attempt to pass, because we can
# so why not.

class TestSimulator(nengo.tests.test_simulator.TestSimulator):
    Simulator = sim_npy.Simulator


class TestNonlinear(nengo.tests.test_simulator.TestNonlinear):
    Simulator = sim_npy.Simulator


load_tests = load_nengo_tests(sim_npy.Simulator)

if __name__ == '__main__':
   unittest.main(testLoader=NengoTestLoader(sim_npy.Simulator))
