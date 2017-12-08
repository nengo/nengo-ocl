import numpy as np

import nengo
import nengo_ocl
from nengo.utils.numpy import rms


def test_LstsqL2_solver(seed, plt):
    nengo.rc.set('decoder_cache', 'enabled', 'False')

    function = lambda x: [(x[0] + x[1])**2, (x[0] - x[1])**2]
    # solver = nengo.solvers.LstsqL2(reg=0.1)
    solver = nengo.solvers.LstsqL2(reg=0.001)

    with nengo.Network(seed=seed) as model:
        a = nengo.Ensemble(1000, 2)
        b = nengo.Node(size_in=2)
        c = nengo.Connection(
            a, b, solver=solver, function=function)

    # with nengo.Simulator(model) as sim:
    with nengo_ocl.Simulator(model) as sim:
        X = sim.data[c].eval_points
        _, A = nengo.utils.ensemble.tuning_curves(a, sim, inputs=X)
        Y = np.dot(A, sim.data[c].weights.T)

    xa = X[:, 0] + X[:, 1]
    xb = X[:, 0] - X[:, 1]
    ya, yb = Y.T
    plt.plot(xa, ya, '.')
    plt.plot(xb, yb, '.')

    da = xa**2 - ya
    db = xb**2 - yb
    assert rms(da) < 0.002  # based on nengo solver
    assert rms(db) < 0.002  # based on nengo solver
