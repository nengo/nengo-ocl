import numpy as np
import pyopencl as cl
import pytest

from nengo.utils.compat import range

from nengo_ocl import raggedarray as ra
from nengo_ocl.raggedarray import RaggedArray as RA
from nengo_ocl.clraggedarray import CLRaggedArray as CLRA

from .conftest import ctx


def make_random_ra(n, d, low=20, high=40, rng=None):
    """Helper to make a random RaggedArray on the host"""
    shapes = zip(*(rng.randint(low=low, high=high+1, size=n).tolist()
                   for dd in range(d)))
    vals = [rng.normal(size=shape).astype(np.float32) for shape in shapes]
    return RA(vals)


def make_random_pair(n, d, **kwargs):
    """Helper to make a pair of RaggedArrays, one host and one device"""
    A = make_random_ra(n, d, **kwargs)
    queue = cl.CommandQueue(ctx)
    clA = CLRA(queue, A)
    return A, clA


def test_shape_zeros():
    A = RA([[]], dtype=np.float32)
    assert A[0].shape == (0, 1)


def test_unit(rng):
    val = np.float32(rng.randn())
    A = RA([val])

    queue = cl.CommandQueue(ctx)
    clA = CLRA(queue, A)
    assert np.allclose(val, clA[0])


def test_small(rng):
    sizes = [3] * 3
    vals = [rng.normal(size=size).astype(np.float32) for size in sizes]
    A = RA(vals)

    queue = cl.CommandQueue(ctx)
    clA = CLRA(queue, A)
    assert ra.allclose(A, clA.to_host())


def test_random_vectors(rng):
    n = np.int32(rng.randint(low=5, high=10))
    A, clA = make_random_pair(n, 1, low=3000, high=4000, rng=rng)
    assert ra.allclose(A, clA.to_host())


def test_random_matrices(rng):
    n = rng.randint(low=5, high=10)
    A, clA = make_random_pair(n, 2, low=20, high=40, rng=rng)
    assert ra.allclose(A, clA.to_host())


def test_getitem(rng):
    """Try getting a single item with a single index"""
    A, clA = make_random_pair(5, 2, rng=rng)
    s = 3
    assert np.allclose(A[s], clA[s])


def test_getitems(rng):
    """Try getting multiple items using a list of indices"""
    A, clA = make_random_pair(10, 2, rng=rng)
    s = [1, 3, 7, 8]
    assert ra.allclose(A[s], clA[s].to_host())


def test_setitem(rng):
    A, clA = make_random_pair(3, 2, rng=rng)

    v = rng.uniform(0, 1, size=A[0].shape)
    A[0] = v.astype(A.dtype)
    clA[0] = v.astype(clA.dtype)
    A[1] = 3
    clA[1] = 3
    assert ra.allclose(A, clA.to_host())


def test_discontiguous_setitem(rng):
    A = make_random_ra(3, 2, rng=rng)
    A0 = np.array(A[0])
    a = A0[::3, ::2]
    v = rng.uniform(-1, 1, size=a.shape)
    assert a.size > 0

    A.add_views([A.starts[0]], [a.shape[0]], [a.shape[1]],
                [a.strides[0] / a.itemsize], [a.strides[1] / a.itemsize])

    queue = cl.CommandQueue(ctx)
    clA = CLRA(queue, A)

    a[...] = v
    A[-1] = v
    assert np.allclose(A[0], A0)
    assert np.allclose(A[-1], v)

    print(clA[0].shape)
    print(clA[-1].shape)

    clA[-1] = v
    assert ra.allclose(A, clA.to_host())
    assert np.allclose(clA[-1], v)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
