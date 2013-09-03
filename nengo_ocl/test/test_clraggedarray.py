
import nose
import numpy as np
from ..ra_gemv import ragged_gather_gemv
from  .. import raggedarray as ra
RA = ra.RaggedArray
from ..clraggedarray import CLRaggedArray as CLRA

import pyopencl as cl
ctx = cl.create_some_context()

def make_random_pair(n, d=1, low=20, high=40):
    """Helper to make a pair of RaggedArrays, one host and one device"""
    shapes = zip(*(np.random.randint(low=low, high=high, size=n).tolist()
                 for dd in xrange(d)))
    vals = [np.random.normal(size=shape) for shape in shapes]
    A = RA(vals)

    queue = cl.CommandQueue(ctx)
    clA = CLRA(queue, A)
    return A, clA

def test_unit():
    val = np.random.randn()
    A = RA([val])

    queue = cl.CommandQueue(ctx)
    clA = CLRA(queue, A)
    assert np.allclose(val, clA[0])

def test_small():
    n = 3
    sizes = [3] * 3
    vals = [np.random.normal(size=size) for size in sizes]
    A = RA(vals)

    queue = cl.CommandQueue(ctx)
    clA = CLRA(queue, A)
    assert ra.allclose(A, clA.to_host())

def test_random_vectors():
    n = np.random.randint(low=5, high=10)
    A, clA = make_random_pair(n, 1, low=3000, high=4000)
    assert ra.allclose(A, clA.to_host())

def test_random_matrices():
    n = np.random.randint(low=5, high=10)
    A, clA = make_random_pair(n, 2, low=20, high=40)
    assert ra.allclose(A, clA.to_host())

def test_getitem():
    """Try getting a single item with a single index"""
    A, clA = make_random_pair(5, 2)
    s = 3
    assert np.allclose(A[s], clA[s])

def test_getitems():
    """Try getting multiple items using a list of indices"""
    A, clA = make_random_pair(10, 2)
    s = [1,3,7,8]
    assert ra.allclose(A[s], clA[s].to_host())
