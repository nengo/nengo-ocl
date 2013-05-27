import sim_ocl
import pyopencl as cl

import test_sim_npy
import test_matmul

ctx = cl.create_some_context()

def test_probe_with_base(show=False):
    def Simulator(*args, **kwargs):
        return sim_ocl.Simulator(ctx, *args, **kwargs)
    return test_sim_npy.test_probe_with_base(show=show, Simulator=Simulator)


def test_matrix_mult(show=False):
    def Simulator(*args, **kwargs):
        return sim_ocl.Simulator(ctx, *args, **kwargs)
    return test_matmul.test_matrix_mult_example(
            D1=5, D2=5, D3=5, N=200,
            show=show, Simulator=Simulator)

