
import nose
import numpy as np
from ..ra_gemv import ragged_gather_gemv
from  .. import raggedarray as ra
RA = ra.RaggedArray
from ..clraggedarray import CLRaggedArray as CLRA

import pyopencl as cl
ctx = cl.create_some_context()

def test_unit():
    val = np.random.randn()
    A = RA([val])

    queue = cl.CommandQueue(ctx)
    dA = CLRA(queue, A)

    assert np.allclose(val, dA[0])

def test_random_vectors():
    n = np.random.randint(low=5, high=10)
    sizes = np.random.randint(low=3000, high=4000, size=n)

    vals = [np.random.normal(size=size) for size in sizes]
    A = RA(vals)

    queue = cl.CommandQueue(ctx)
    dA = CLRA(queue, A)

    B = dA.to_host()

    assert ra.allclose(A, B)

def test_random_matrices():
    n = 3
    shapes = zip(np.random.randint(low=20, high=40, size=n),
                 np.random.randint(low=20, high=40, size=n))

    vals = [np.random.normal(size=shape) for shape in shapes]
    A = RA(vals)

    queue = cl.CommandQueue(ctx)
    dA = CLRA(queue, A)

    B = dA.to_host()

    assert ra.allclose(A, B)
