import pyopencl as cl

from nengo_ocl import sim_ocl
import nengo.test.test_old_api

ctx = cl.create_some_context()


class UseSimOcl(object):
    def Simulator(self, *args, **kwargs):
        rval = sim_ocl.Simulator(ctx, *args, **kwargs)
        rval.alloc_all()
        rval.plan_all()
        return rval


class TestOldAPI(UseSimOcl, nengo.test.test_old_api.TestOldAPI):
    show = True

