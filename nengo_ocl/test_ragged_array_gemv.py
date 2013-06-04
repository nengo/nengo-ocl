import copy
import numpy as np
from sim_npy import RaggedArray
from sim_npy import ragged_gather_gemv

def make_ragged_array(starts, lens, buf):
    rval = RaggedArray.__new__(RaggedArray)
    rval.starts = starts
    rval.lens = lens
    rval.buf = buf
    return rval

def test_problem_case_1():
    Ms = [1, 1]
    Ns = [1, 1000]
    dec_weights = make_ragged_array(
        starts=[0, 1],
        lens=[1, 1000],
        buf=np.random.randn(1001))

    dec_weights_js = make_ragged_array(
        starts=[0, 0, 0, 0, 1, 1, 2, 2],
        lens=[0, 0, 0, 1, 0, 1, 0, 0],
        buf=[0, 1])

    pop_output = make_ragged_array(
        starts=[0, 1],
        lens=[1, 1000],
        buf=np.random.randn(1001))

    pops_js = make_ragged_array(
        starts=[0, 0, 0, 0, 1, 1, 2, 2],
        lens=[0, 0, 0, 1, 0, 1, 0, 0],
        buf=[0, 1],
        )

    sigs_ic = make_ragged_array(
        starts=[0, 1, 2, 3, 4, 5, 6, 7],
        lens=[1, 1, 1, 1, 1, 1, 1, 1],
        buf=np.random.randn(8))

    sigs_ic_copy = copy.deepcopy(sigs_ic)


    ragged_gather_gemv(Ms, Ns,
                           1.0, dec_weights, dec_weights_js,
                           pop_output, pops_js,
                           0.0, sigs_ic, use_raw_fn=True)

    ragged_gather_gemv(Ms, Ns,
                           1.0, dec_weights, dec_weights_js,
                           pop_output, pops_js,
                           0.0, sigs_ic_copy, use_raw_fn=False)


    assert np.allclose(sigs_ic.buf, sigs_ic_copy.buf)



