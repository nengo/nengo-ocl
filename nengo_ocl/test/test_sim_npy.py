from nengo_ocl import sim_npy
import nengo.test.test_old_api
import nengo.test.test_circularconv
NT = nengo.test

class UseSimNpy(object):
    def Simulator(self, *args, **kwargs):
        rval = sim_npy.Simulator(*args, **kwargs)
        rval.alloc_all()
        return rval


class TestOldAPI(UseSimNpy, NT.test_old_api.TestOldAPI):
    show = False


class TestCircularConv(UseSimNpy, NT.test_circularconv.TestCircularConv):
    show = False
