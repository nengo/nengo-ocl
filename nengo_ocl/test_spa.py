import numpy as np
import pyopencl as cl
import time
from base import Model
from sim_ocl import Simulator

ctx = cl.create_some_context()

D = 500  # dimensionality of SPA

neurons_per_input = 64
neurons_per_product = 128 # -- way faster than 150

m = Model(.001)

dummy = np.random.randn(D, D)

A = [m.signal() for ii in range(D)]
B = [m.signal() for ii in range(D)]
AF = [m.signal() for ii in range(D)]
BF = [m.signal() for ii in range(D)]
CF = [m.signal() for ii in range(D)]
C = [m.signal() for ii in range(D)]


for ii in range(D):
    m.filter(1.0, A[ii], A[ii])
    m.filter(1.0, B[ii], B[ii])

for ii in range(D):
    for jj in range(D):
        m.filter(dummy[ii, jj], A[ii], AF[jj])
        m.filter(dummy[ii, jj], B[ii], BF[jj])

for jj in range(4):
    for ii in range(D):
        pop = m.population(neurons_per_product)
        m.encoder(AF[ii], pop)
        m.encoder(BF[ii], pop)
        m.decoder(pop, CF[ii])

if 0:
    # second circular convolution
    for jj in range(4):
        for ii in range(D):
            pop = m.population(neurons_per_product)
            m.encoder(AF[ii], pop)
            m.encoder(BF[ii], pop)
            m.decoder(pop, C[ii])

    # third circular convolution
    for jj in range(4):
        for ii in range(D):
            pop = m.population(neurons_per_product)
            m.encoder(AF[ii], pop)
            m.encoder(BF[ii], pop)
            m.decoder(pop, A[ii])

    # fourth circular convolution
    for jj in range(4):
        for ii in range(D):
            pop = m.population(neurons_per_product)
            m.encoder(AF[ii], pop)
            m.encoder(BF[ii], pop)
            m.decoder(pop, B[ii])


for ii in range(D):
    for jj in range(D):
        m.transform(dummy[ii, jj], CF[ii], C[jj])

sim = Simulator(ctx, m)
sim.alloc_all()
sim.plan_all()

t0 = time.time()
sim.run_steps(1000)
t1 = time.time()

print 'time', (t1 - t0)



