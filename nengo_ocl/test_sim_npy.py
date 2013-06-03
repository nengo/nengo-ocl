
from sim_npy import Simulator
import nengo.test.test_simulator as nengo_tsim

def test_signal_indexing_1():
    nengo_tsim.test_signal_indexing_1(sim_cls=Simulator)


def test_simple_direct_mode():
    nengo_tsim.test_simple_direct_mode(sim_cls=Simulator)

