import numpy as np

import nengo
import nengo_ocl


def test_lstsql2_solver(seed, plt):
    with nengo.Network(seed=seed) as model:
        a = nengo.Ensemble(1000, 2)
        b = nengo.Node(size_in=2)
        c = nengo.Connection(a, b, function=lambda x: [(x[0] + x[1])**2,
                                                       (x[0] - x[1])**2])

    with nengo_ocl.Simulator(model) as sim:
        X = sim.data[c].eval_points
        _, A = nengo.utils.ensemble.tuning_curves(a, sim, inputs=X)
        Y = np.dot(A, sim.data[c].weights.T)

    xa = X[:, 0] + X[:, 1]
    xb = X[:, 0] - X[:, 1]
    plt.plot(xa, Y[:, 0], '.')
    plt.plot(xb, Y[:, 1], '.')
