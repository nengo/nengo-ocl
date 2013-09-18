
import time
import numpy as np

import pyopencl as cl

import nengo
from nengo.templates import EnsembleArray
from nengo.networks.circularconvolution import circconv, CircularConvolution

# import nengo_ocl
from nengo_ocl import sim_ocl

dims = 10
# dims = 128
neurons_per_product = 50

n_neurons = neurons_per_product * dims
n_neurons_d = 2 * neurons_per_product * (
    2*dims - (2 if dims % 2 == 0 else 1))
radius = 1

rng = np.random.RandomState(123)
a = rng.normal(scale=np.sqrt(1./dims), size=dims)
b = rng.normal(scale=np.sqrt(1./dims), size=dims)
c = circconv(a, b)
assert np.abs(a).max() < radius
assert np.abs(b).max() < radius
assert np.abs(c).max() < radius

### model
model = nengo.Model("circular convolution")
inputA = model.make_node("inputA", output=a)
inputB = model.make_node("inputB", output=b)
A = model.add(EnsembleArray('A', nengo.LIF(n_neurons), dims, radius=radius))
B = model.add(EnsembleArray('B', nengo.LIF(n_neurons), dims, radius=radius))
C = model.add(EnsembleArray('C', nengo.LIF(n_neurons), dims, radius=radius))
D = model.add(CircularConvolution('D', neurons=nengo.LIF(n_neurons_d),
                                  dimensions=A.dimensions, radius=radius))

inputA.connect_to(A)
inputB.connect_to(B)
A.connect_to(D.A)
B.connect_to(D.B)
D.connect_to(C)

model.probe(A, filter=0.03)
model.probe(B, filter=0.03)
model.probe(C, filter=0.03)
model.probe(D, filter=0.03)

# check FFT magnitude
d = np.dot(D.transformA, a) + np.dot(D.transformB, b)
assert np.abs(d).max() < radius

### simulation
def OclSimulator(model):
    ctx = cl.create_some_context()
    return sim_ocl.Simulator(model, ctx, profiling=True)

# sim = model.simulator()
sim = model.simulator(sim_class=OclSimulator)

print "Starting simulation..."
timer = time.time()
sim.run(1.0)
print "Done in %s seconds" % (time.time() - timer)

sim.print_profiling(sort=1)

### results
import matplotlib.pyplot as plt

t = sim.data(model.t).flatten()

def plot(sim, a, A, title=""):
    a_ref = np.tile(a, (len(t), 1))
    a_sim = sim.data(A)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for i in xrange(min(dims, len(colors))):
        plt.plot(t, a_ref[:,i], '--', color=colors[i])
        plt.plot(t, a_sim[:,i], '-', color=colors[i])
        plt.title(title)

plt.subplot(221)
plot(sim, a, A, title="A")
plt.subplot(222)
plot(sim, b, B, title="B")
plt.subplot(223)
plot(sim, c, C, title="C")
plt.subplot(224)
plot(sim, d, D, title="D")
# plt.show()
