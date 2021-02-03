import nengo
import numpy as np
import pyopencl as cl

import nengo_ocl

ctx = cl.create_some_context()

dims = 16
# neurons_per_dim = 64
neurons_per_dim = 256
# neurons_per_dim = 1024
n_neurons = dims * neurons_per_dim

r = 1e-4
W0 = np.random.uniform(-r, r, size=(n_neurons, n_neurons))

runtime = 1.0

# --- model
target_fn = lambda x: x ** 2
pes = nengo.PES()

with nengo.Network(label="ProfilePES", seed=3) as model:
    x = nengo.Node(nengo.processes.WhiteSignal(runtime, high=1.0), size_out=dims)
    y = nengo.Node(size_in=dims)
    nengo.Connection(x, y, function=target_fn)

    a = nengo.Ensemble(n_neurons, dims)
    b = nengo.Ensemble(n_neurons, dims)
    nengo.Connection(x, a, synapse=None)

    e = nengo.Ensemble(dims * neurons_per_dim, dims)
    # nengo.Connection(b, e)  # time-consuming decoder finding involved
    nengo.Connection(y, e, transform=-1)

    c = nengo.Connection(a.neurons, b.neurons, transform=W0, learning_rule_type=pes)
    nengo.Connection(e, c.learning_rule, synapse=None)

# --- simulation
with nengo_ocl.Simulator(model, context=ctx, profiling=True) as sim:
    sim.run(runtime)
    sim.print_profiling(sort=1)
