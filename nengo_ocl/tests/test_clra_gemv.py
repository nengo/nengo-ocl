import logging
import numpy as np
import pyopencl as cl
import pyopencl.array  # noqa: F401
import pytest

from nengo.utils.stdlib import Timer

from nengo_ocl.raggedarray import RaggedArray
from nengo_ocl.clraggedarray import CLRaggedArray as CLRA, to_device
from nengo_ocl.clra_gemv import (
    plan_reduce_gemv,
    plan_many_dots_gemv,
    plan_block_gemv,
    plan_ragged_gather_gemv,
    plan_sparse_dot_inc,
)

logger = logging.getLogger(__name__)
RA = lambda arrays, dtype=np.float32: RaggedArray(arrays, dtype=dtype)


def pytest_generate_tests(metafunc):
    if "planner" in metafunc.fixturenames:
        metafunc.parametrize(
            "planner",
            [
                plan_reduce_gemv,
                plan_many_dots_gemv,
                plan_block_gemv,
                plan_ragged_gather_gemv,
            ],
        )


def allclose(raA, raB):
    assert len(raA) == len(raB)
    for i in range(len(raA)):
        if not np.allclose(raA[i], raB[i]):
            return False
    return True


def test_basic(ctx):
    # -- prepare initial conditions on host
    A = RA([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6]]])
    X = RA([[3, 5]])
    Y = RA([[0.0], [2, 3]])
    A_js = RA([[1], [0]], dtype=np.int32)
    X_js = RA([[0], [0]], dtype=np.int32)
    # alpha = 0.5
    alpha = 1.0
    # beta = 0.1
    beta = 1.0

    # -- prepare initial conditions on device
    queue = cl.CommandQueue(ctx)
    clA = CLRA(queue, A)
    clX = CLRA(queue, X)
    clY = CLRA(queue, Y)
    assert allclose(A, clA)
    assert allclose(X, clX)
    assert allclose(Y, clY)

    # -- run cl computation
    prog = plan_ragged_gather_gemv(queue, alpha, clA, A_js, clX, X_js, beta, clY)
    # plans = prog.choose_plans()
    # assert len(plans) == 1
    for plan in prog.plans:
        plan()

    # -- ensure they match
    for i in range(len(A_js)):
        aj, xj = int(A_js[i]), int(X_js[i])
        ref = alpha * np.dot(A[aj], X[xj]) + beta * Y[i]
        sim = clY[i]
        assert np.allclose(ref, sim)


def _test_random(ctx, k=4, p=1, m=10, n=10):
    """
    Parameters
    ----------
    k : number of operations (length of A_js)
    p : number of dots per operation (width of A_js)
    m : output dimensions
    n : input dimensions
    """

    rng = np.random.RandomState(3294)

    aa = [rng.normal(size=(m, n)) for i in range(k)]
    xx = [rng.normal(size=n) for i in range(k)]
    yy = [rng.normal(size=m) for i in range(k)]
    ajs = [rng.randint(k, size=p) for i in range(k)]
    xjs = [rng.randint(k, size=p) for i in range(k)]

    A = RA(aa)
    X = RA(xx)
    Y = RA(yy)
    A_js = RA(ajs, dtype=np.int32)
    X_js = RA(xjs, dtype=np.int32)
    alpha = 0.5
    beta = 0.1

    # -- prepare initial conditions on device
    queue = cl.CommandQueue(ctx)
    clA = CLRA(queue, A)
    clX = CLRA(queue, X)
    clY = CLRA(queue, Y)
    assert allclose(A, clA)
    assert allclose(X, clX)
    assert allclose(Y, clY)

    # -- run cl computation
    prog = plan_ragged_gather_gemv(queue, alpha, clA, A_js, clX, X_js, beta, clY)

    print("-" * 5 + " Plans " + "-" * 45)
    for plan in prog.plans:
        print(plan)
        plan()

    # -- ensure they match
    for i in range(k):
        ref = beta * Y[i]
        for aj, xj in zip(A_js[i].ravel(), X_js[i].ravel()):
            ref += alpha * np.dot(A[aj], X[xj])
        sim = clY[i]
        assert np.allclose(ref, sim, atol=1e-3, rtol=1e-3)


def test_random_small(ctx):
    _test_random(ctx, k=4, m=10, n=10)


def test_random_large(ctx):
    _test_random(ctx, k=10, m=550, n=550)


def test_many_dots_small(ctx):
    _test_random(ctx, k=4, p=4, m=10, n=10)


def test_many_dots_large(ctx):
    # _test_random(k=4, p=4, m=550, n=550)
    _test_random(ctx, k=4, p=4, m=2000, n=1000)


def check_from_shapes(ctx, planner, alpha, beta, gamma, A_shapes, X_shapes, A_js, X_js):
    rng = np.random.RandomState(1234)
    A = RA([0.1 + rng.rand(*shp) for shp in A_shapes])
    X = RA([0.1 + rng.rand(*shp) for shp in X_shapes])
    Y = RA(
        [
            0.1 + rng.rand(A_shapes[A_js[ii][0]][0], X_shapes[X_js[ii][0]][1])
            for ii in range(len(A_js))
        ]
    )
    A_js = RA(A_js, dtype=np.int32)
    X_js = RA(X_js, dtype=np.int32)
    # -- prepare initial conditions on device
    queue = cl.CommandQueue(ctx)
    clA = CLRA(queue, A)
    clX = CLRA(queue, X)
    clY = CLRA(queue, Y)
    assert allclose(A, clA)
    assert allclose(X, clX)
    assert allclose(Y, clY)

    # -- run cl computation
    prog = planner(queue, alpha, clA, A_js, clX, X_js, beta, clY, gamma=gamma)

    plans = prog.plans
    # assert len(plans) == 1
    for plan in plans:
        plan()

    # -- ensure they match
    for i in range(len(A_js)):
        ref = (
            gamma
            + beta * Y[i]
            + alpha
            * sum(
                [
                    np.dot(A[aj], X[xj])
                    for aj, xj in zip(A_js[i].ravel(), X_js[i].ravel())
                ]
            )
        )
        sim = clY[i]
        if not np.allclose(ref, sim, atol=1e-3, rtol=1e-3):
            print("A_shapes", A_shapes)
            print("X_shapes", X_shapes)
            if len(ref) > 20:
                print("ref", ref[:10], "...", ref[-10:])
                print("sim", sim[:10], "...", sim[-10:])
            else:
                print("ref", ref)
                print("sim", sim)
            assert 0


def test_one_element(ctx, planner):
    check_from_shapes(
        ctx,
        planner,
        0.5,
        0.6,
        0.7,
        A_shapes=[(1, 1)],
        X_shapes=[(1, 1)],
        A_js=[[0]],
        X_js=[[0]],
    )


def test_one_short_segment(ctx, planner):
    check_from_shapes(
        ctx,
        planner,
        0.5,
        0.6,
        0.7,
        A_shapes=[(10, 1)],
        X_shapes=[(1, 1)],
        A_js=[[0]],
        X_js=[[0]],
    )


def test_one_long_segment(ctx, planner):
    check_from_shapes(
        ctx,
        planner,
        0.5,
        0.6,
        0.7,
        A_shapes=[(2001, 1)],
        X_shapes=[(1, 1)],
        A_js=[[0]],
        X_js=[[0]],
    )


def test_one_short_segment_many_dots(ctx, planner):
    for ND in 2, 20, 100:
        check_from_shapes(
            ctx,
            planner,
            0.5,
            0.6,
            0.7,
            A_shapes=[(10, 1 + ii % 2) for ii in range(ND)],
            X_shapes=[(1 + ii % 2, 1) for ii in range(ND)],
            A_js=[range(ND)],
            X_js=[range(ND)],
        )


def test_one_short_segment_many_longer_dots(ctx, planner):
    for ND in 2, 20, 100:
        check_from_shapes(
            ctx,
            planner,
            0.5,
            0.6,
            0.7,
            A_shapes=[(2000, ii + 1) for ii in range(ND)],
            X_shapes=[(ii + 1, 1) for ii in range(ND)],
            A_js=[range(ND)],
            X_js=[range(ND)],
        )


def test_speed(ctx, rng):
    try:
        import pyopencl_blas
    except ImportError:
        pyopencl_blas = None

    # enable_out_of_order = (
    #     cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)

    k = 300
    # k = 100
    # k = 32
    # k = 16
    ms = [rng.randint(100, 1000) for i in range(k)]
    ns = [rng.randint(100, 1000) for i in range(k)]
    # ms = [4096 for i in range(k)]
    # ns = [4096 for i in range(k)]

    aa = [rng.uniform(-1, 1, size=(m, n)).astype("float32") for m, n in zip(ms, ns)]
    xx = [rng.uniform(-1, 1, size=n).astype("float32") for n in ns]
    yy = [rng.uniform(-1, 1, size=m).astype("float32") for m in ms]
    ajs = [np.int32(i) for i in range(k)]
    xjs = [np.int32(i) for i in range(k)]
    # ajs = [rng.randint(k, size=p) for i in range(k)]
    # xjs = [rng.randint(k, size=p) for i in range(k)]

    # alpha = 0.5
    # beta = 0.1
    alpha = 1.0
    beta = 1.0

    # -- prepare initial conditions on device
    queue = cl.CommandQueue(ctx)
    # queue = cl.CommandQueue(ctx, properties=enable_out_of_order)
    clA = CLRA.from_arrays(queue, aa)
    clX = CLRA.from_arrays(queue, xx)
    clY = CLRA.from_arrays(queue, yy)
    A_js = RA(ajs, dtype=np.int32)
    X_js = RA(xjs, dtype=np.int32)

    # -- run cl computation
    prog = plan_ragged_gather_gemv(queue, alpha, clA, A_js, clX, X_js, beta, clY)
    plans = prog.choose_plans()

    print("")
    print("-" * 5 + " Plans " + "-" * 45)
    for plan in plans:
        print(plan)

    with Timer() as timer:
        for plan in plans:
            plan()
    print("nengo_ocl: %0.3f" % timer.duration)

    # -- speed test in ocl blas
    if pyopencl_blas:
        pyopencl_blas.setup()

        def array(a):
            cla = cl.array.Array(queue, a.shape, a.dtype)
            cla.set(a)
            return cla

        clAs = [array(a) for a in aa]
        clXs = [array(x.ravel()) for x in xx]
        clYs = [array(y.ravel()) for y in yy]

        queues = [cl.CommandQueue(ctx) for _ in range(k)]
        # queues = [cl.CommandQueue(ctx, properties=enable_out_of_order)
        #           for _ in range(k)]

        queue.finish()
        with Timer() as timer:
            if 0:
                # use a single queue
                for A, X, Y in zip(clAs, clXs, clYs):
                    pyopencl_blas.gemv(queue, A, X, Y)
                queue.finish()
            else:
                # use multiple parallel queues
                events = []
                for i, [A, X, Y] in enumerate(zip(clAs, clXs, clYs)):
                    q = queues[i % len(queues)]
                    e = pyopencl_blas.gemv(q, A, X, Y)
                    events.append(e)
                for q in queues:
                    q.flush()
                cl.wait_for_events(events)
        print("clBLAS: %0.3f" % timer.duration)


@pytest.mark.parametrize("inc", [False, True])
def test_sparse(ctx, inc, rng, allclose):
    scipy_sparse = pytest.importorskip("scipy.sparse")

    # -- prepare initial conditions on host
    if 0:
        # diagonal matrix
        shape = (32, 32)
        s = min(shape[0], shape[1])
        data = list(range(s))
        ii = list(range(s))
        jj = list(range(s))[::-1]
        A = scipy_sparse.coo_matrix((data, (ii, jj)), shape=shape).tocsr()
        X = RA([np.arange(1, shape[1] + 1)])
        Y = RA([np.arange(1, shape[0] + 1)])
    else:
        # random sparse matrix
        shape = (500, 500)
        sparsity = 0.002
        mask = rng.uniform(size=shape) < sparsity
        ii, jj = mask.nonzero()
        assert len(ii) > 0
        data = rng.uniform(-1, 1, size=len(ii))
        A = scipy_sparse.coo_matrix((data, (ii, jj)), shape=shape).tocsr()
        X = RA([rng.uniform(-1, 1, size=shape[1])])
        Y = RA([rng.uniform(-1, 1, size=shape[0])])

    # -- prepare initial conditions on device
    queue = cl.CommandQueue(ctx)
    A_data = to_device(queue, A.data.astype(np.float32))
    A_indices = to_device(queue, A.indices.astype(np.int32))
    A_indptr = to_device(queue, A.indptr.astype(np.int32))
    clX = CLRA(queue, X)
    clY = CLRA(queue, Y)
    assert allclose(X, clX)
    assert allclose(Y, clY)

    # -- run cl computation
    plan = plan_sparse_dot_inc(queue, A_indices, A_indptr, A_data, clX, clY, inc=inc)
    plan()

    # -- ensure they match
    ref = (Y[0] if inc else 0) + A.dot(X[0])
    sim = clY[0]
    assert allclose(ref, sim, atol=1e-7)
