import matplotlib.pyplot as plt
import nengo
import numpy as np
import pyopencl as cl
from nengo.networks.circularconvolution import circconv, transform_in

import nengo_ocl

ctx = cl.create_some_context()

dim = 16
# dim = 256
neurons_per_product = 64
# neurons_per_product = 256
radius = 1

rng = np.random.RandomState(129)
a = rng.normal(scale=np.sqrt(1.0 / dim), size=dim)
b = rng.normal(scale=np.sqrt(1.0 / dim), size=dim)
c = circconv(a, b)
assert np.abs(a).max() < radius
assert np.abs(b).max() < radius
assert np.abs(c).max() < radius

# check FFT magnitude
tr_A = transform_in(dim, "A", invert=False)
tr_B = transform_in(dim, "B", invert=False)
d = np.dot(tr_A, a) * np.dot(tr_B, b)
assert np.abs(d).max() < (4 * radius)
# ^ TODO: 4 * radius just seems to work from looking at Nengo code. Why?

# --- model
with nengo.Network(label="ProfileConv", seed=3) as model:
    inputA = nengo.Node(a, label="inputA")
    inputB = nengo.Node(b, label="inputB")
    A = nengo.networks.EnsembleArray(neurons_per_product, dim, radius=radius, label="A")
    B = nengo.networks.EnsembleArray(neurons_per_product, dim, radius=radius, label="B")
    C = nengo.networks.EnsembleArray(neurons_per_product, dim, radius=radius, label="C")
    D = nengo.networks.CircularConvolution(
        neurons_per_product, dim, input_magnitude=radius
    )

    model.config[nengo.Connection].synapse = nengo.Alpha(0.005)
    nengo.Connection(inputA, A.input, synapse=None)
    nengo.Connection(inputB, B.input, synapse=None)
    nengo.Connection(A.output, D.A)
    nengo.Connection(B.output, D.B)
    nengo.Connection(D.output, C.input)

    model.config[nengo.Probe].synapse = nengo.Alpha(0.03)
    A_p = nengo.Probe(A.output)
    B_p = nengo.Probe(B.output)
    C_p = nengo.Probe(C.output)
    D_p = nengo.Probe(D.product.output)

# --- simulation
with nengo_ocl.Simulator(model, context=ctx, profiling=True) as sim:
    sim.run(1.0)
    sim.print_profiling(sort=1)

# --- results
t = sim.trange()


def plot(sim, a, A, title=""):
    a_ref = np.tile(a, (len(t), 1))
    a_sim = sim.data[A]
    colors = ["b", "g", "r", "c", "m", "y"]
    for i in range(min(dim, len(colors))):
        plt.plot(t, a_ref[:, i], "--", color=colors[i])
        plt.plot(t, a_sim[:, i], "-", color=colors[i])
        plt.title(title)


plt.subplot(221)
plot(sim, a, A_p, title="A")
plt.subplot(222)
plot(sim, b, B_p, title="B")
plt.subplot(223)
plot(sim, c, C_p, title="C")
plt.subplot(224)
plot(sim, d, D_p, title="D")
plt.show()
