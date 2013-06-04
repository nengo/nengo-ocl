from nengo_ocl import sim_npy
import nengo.test.test_old_api

class UseSimNpy(object):
    def Simulator(self, *args, **kwargs):
        rval = sim_npy.Simulator(*args, **kwargs)
        rval.alloc_all()
        return rval


class TestOldAPI(UseSimNpy, nengo.test.test_old_api.TestOldAPI):
    show = False

