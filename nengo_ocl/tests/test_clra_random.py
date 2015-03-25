import logging
import numpy as np
import pyopencl as cl
import pytest

import nengo
from nengo.neurons import LIF, LIFRate, Direct
from nengo.utils.testing import Timer

from nengo_ocl.ra_gemv import ragged_gather_gemv
from nengo_ocl import raggedarray as ra
from nengo_ocl.raggedarray import RaggedArray as RA
from nengo_ocl.clraggedarray import CLRaggedArray as CLRA

from nengo_ocl.clra_random import *


ctx = cl.create_some_context()
logger = logging.getLogger(__name__)
# nengo.log(True)


def not_close(a, b, rtol=1e-3, atol=1e-3):
    return np.abs(a - b) > atol + rtol * np.abs(b)


def test_rand(plt, rng):

    n_streams = 100
    n_samples = [10000000, 1000000]

    state = rng.randint(0, 2**31, size=(n_streams, 6))
    state = RA([state])
    samples = RA([np.zeros(i) for i in n_samples])

    queue = cl.CommandQueue(ctx)
    cl_state = CLRA(queue, state)
    cl_samples = CLRA(queue, samples)

    plan = plan_rand(queue, cl_state, cl_samples)

    with Timer() as t:
        plan()
    print(t.duration)

    samples = cl_samples.to_host()
    samples = [samples[i] for i in range(len(samples))]
    samples = np.vstack(samples)

    plt.hist(samples, bins=30)


def test_randn(plt, rng):
    n_streams = 100
    n_samples = [10000000, 1000000]

    state = rng.randint(0, 2**31, size=(n_streams, 6))
    state = RA([state])
    samples = RA([np.zeros(i) for i in n_samples])

    queue = cl.CommandQueue(ctx)
    cl_state = CLRA(queue, state)
    cl_samples = CLRA(queue, samples)

    plan = plan_randn(queue, cl_state, cl_samples)
    with Timer() as t:
        plan()
    print(t.duration)

    samples = cl_samples.to_host()
    samples = [samples[i] for i in range(len(samples))]
    samples = np.vstack(samples)

    plt.hist(samples, bins=30)
