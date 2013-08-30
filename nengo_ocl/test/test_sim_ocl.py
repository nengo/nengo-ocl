"""
Black-box testing of the sim_npy Simulator.

TestCase classes are added automatically from
nengo.tests.helpers.simulator_test_cases, but
you can still run individual test files like this:

    nosetests -sv test/test_sim_npy.py:TestSimulator.test_simple_direct_mode

"""

from nengo_ocl.tricky_imports import unittest

import nengo.tests.helpers

from nengo_ocl import sim_ocl

import pyopencl as cl

ctx = cl.create_some_context()

def simulator_allocator(*args, **kwargs):
    rval = sim_ocl.Simulator(ctx, *args, **kwargs)
    rval.plan_all()
    return rval

load_tests = nengo.tests.helpers.load_nengo_tests(simulator_allocator)

for foo in load_tests(None, None, None):
    class MyCLS(foo.__class__):
        def Simulator(self, model):
            return simulator_allocator(model)
    globals()[foo.__class__.__name__] = MyCLS
    MyCLS.__name__ = foo.__class__.__name__
    del MyCLS
    del foo
del load_tests


if __name__ == "__main__":
    unittest.main()
