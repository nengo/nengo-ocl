try:
    # For Python <=2.6
    import unittest2 as unittest
except ImportError:
    import unittest

import pyopencl as cl
from nengo.tests.helpers import load_nengo_tests, simulator_test_suite

from nengo_ocl import sim_ocl


ctx = cl.create_some_context(interactive=False)


def simulator_allocator(*args, **kwargs):
    rval = sim_ocl.Simulator(ctx, *args, **kwargs)
    rval.alloc_all()
    rval.plan_all()
    return rval

load_tests = load_nengo_tests(simulator_allocator)

try:
    import nose  # If we have nose, maybe we're doing nosetests.

    # Stop some functions from running automatically
    load_nengo_tests.__test__ = False
    load_tests.__test__ = False
    simulator_test_suite.__test__ = False

    # unittest won't try to run this, but nose will
    def test_nengo():
        nengo_suite = simulator_test_suite(simulator_allocator)

        # Unfortunately, this makes us run nose twice,
        # which gives two sets of results.
        # I don't know a way around this.
        assert nose.run(suite=nengo_suite, exit=False)

except ImportError:
    pass


if __name__ == "__main__":
    unittest.main()
