import nengo
from nengo.conftest import *  # noqa: F403
import pyopencl as cl

import nengo_ocl


ctx = cl.create_some_context()


class OclSimulator(nengo_ocl.Simulator):
    def __init__(self, *args, **kwargs):
        super(OclSimulator, self).__init__(*args, context=ctx, **kwargs)


TestConfig.Simulator = OclSimulator
TestConfig.neuron_types = [nengo.Direct, nengo.LIF, nengo.LIFRate]
