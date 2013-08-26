import pyopencl as cl

#from nengo_ocl import sim_ocl
#import nengo.test.test_old_api
from nengo.tests.helpers import simulator_test_cases
import nengo.simulator



for TestCase in simulator_test_cases:
    class MyTestCase(TestCase):
        simulator_test_case_ignore = True
        def Simulator(self, model):
            return nengo.simulator.Simulator(model)
    MyTestCase.__name__ = TestCase.__name__
    globals()[TestCase.__name__] = MyTestCase
    del MyTestCase
    del TestCase


if 0:
    ctx = cl.create_some_context()


    class UseSimOcl(object):
        def Simulator(self, *args, **kwargs):
            rval = sim_ocl.Simulator(ctx, *args, **kwargs)
            rval.alloc_all()
            rval.plan_all()
            return rval


    class TestOldAPI(UseSimOcl, nengo.test.test_old_api.TestOldAPI):
        show = False

