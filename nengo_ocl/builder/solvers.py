import warnings
import time

import numpy as np
import pyopencl as cl
import pyopencl.array
import pyopencl.reduction

import nengo
import nengo.utils.numpy as npext
from nengo.exceptions import BuildError
from nengo.utils.least_squares_solvers import rmses

from nengo_ocl.builder import Builder
from nengo_ocl.clraggedarray import CLRaggedArray
from nengo_ocl.clra_nonlinearities import plan_lif_rate

try:
    import scipy_optimize
except ImportError:
    scipy_optimize = None

try:
    import pyopencl_blas
except ImportError:
    pyopencl_blas = None

try:
    from nengo_extras.convnet import softmax
    from nengo_extras.solvers import (
        SoftmaxClassifier, HingeClassifier, LstsqClassifier)
except ImportError:
    SoftmaxClassifier = None
    HingeClassifier = None
    LstsqClassifier = None


def cho_solve(A, y, overwrite=False):
    # Solve A x = y for x
    try:
        import scipy.linalg
        factor = scipy.linalg.cho_factor(A, overwrite_a=overwrite)
        x = scipy.linalg.cho_solve(factor, y)
    except ImportError:
        L = np.linalg.cholesky(A)
        L = np.linalg.inv(L.T)
        x = np.dot(L, np.dot(L.T, y))

    return x


@Builder.register(nengo.solvers.Solver)
def build_solver(model, solver, conn, rng, transform):
    """Build a solver

    Same role as ``nengo.builder.connection.build_decoders``. Differences:
      - wraps this file's implementation of ``solve_for_decoders``
      - passes ``queue`` to wrapped_solver.
    """
    from nengo.builder.connection import multiply
    eval_points, gain, bias, x, targets, E = get_solve_params(
        model, solver, conn, rng, transform)
    wrapped_solver = wrap_solver(model, conn, solve_for_decoders)
    decoders, solver_info = wrapped_solver(
        conn, gain, bias, x, targets, rng=rng, E=E,
        queue=model.builder.queue)
    weights = (decoders.T if conn.solver.weights else
               multiply(transform, decoders.T))
    return eval_points, weights, solver_info


def get_solve_params(model, solver, conn, rng, transform):
    from nengo.builder.connection import get_eval_points, get_targets, multiply
    encoders = model.params[conn.pre_obj].encoders
    gain = model.params[conn.pre_obj].gain
    bias = model.params[conn.pre_obj].bias

    eval_points = get_eval_points(model, conn, rng)
    targets = get_targets(conn, eval_points)

    x = np.dot(eval_points, encoders.T / conn.pre_obj.radius)
    E = None
    if conn.solver.weights:
        E = model.params[conn.post_obj].scaled_encoders.T[conn.post_slice]
        # include transform in solved weights
        targets = multiply(targets, transform.T)

    return eval_points, gain, bias, x, targets, E


def wrap_solver(model, conn, solve_fn):
    return (model.decoder_cache.wrap_solver(solve_fn)
            if model.seeded[conn] else solve_fn)


def solve_for_decoders(conn, gain, bias, x, targets, rng, E=None, queue=None):
    ocl_activity_min_size = 1e6  # TODO: find best value

    neuron_type = conn.pre_obj.neuron_type
    solver = conn.solver
    X = x
    Y = targets

    A, clA = None, None
    if queue is not None and X.size > ocl_activity_min_size:
        clA = neuron_rates(neuron_type, X, gain, bias, queue=queue)
    else:
        A = neuron_type.rates(X, gain, bias)

    if queue is not None and Builder.has_ocl_solver(solver):
        try:
            # --- use OCL device
            if clA is None:
                clA = cl.array.to_device(queue, A.astype(np.float32))
            decoders, solver_info = Builder.solve(
                solver, queue, clA, Y, rng=rng, E=E)
            return decoders, solver_info
        except Exception as e:
            warnings.warn("OCL solver failed: %s" % e)
            raise

    # --- use Numpy
    if clA is not None:
        A = clA.get().astype(np.float64)

    if np.count_nonzero(A) == 0:
        raise BuildError(
            "Building %s: 'activities' matrix is all zero for %s. "
            "This is because no evaluation points fall in the firing "
            "ranges of any neurons." % (conn, conn.pre_obj))

    decoders, solver_info = solver(A, Y, rng=rng, E=E)
    return decoders, solver_info


def neuron_rates(neuron_type, X, gain, bias, queue=None):
    m, n = X.shape

    # Compute rates in-place in X
    clX = cl.array.to_device(queue, X.astype(np.float32))
    clgain = cl.array.to_device(queue, gain.astype(np.float32))
    clbias = cl.array.to_device(queue, bias.astype(np.float32))
    compute_J(queue, clX, clgain, clbias)

    raX = CLRaggedArray.from_clarray_rows(queue, clX)
    if type(neuron_type) in (nengo.neurons.LIF, nengo.neurons.LIFRate,
                             nengo.neurons.AdaptiveLIF,
                             nengo.neurons.AdaptiveLIFRate):
        zm = [0] * m  # use same gains and biases for all eval points
        tau = CLRaggedArray.from_arrays(
            queue, [neuron_type.tau_rc * np.ones(n)], dtype=np.float32)[zm]
        ref = CLRaggedArray.from_arrays(
            queue, [neuron_type.tau_ref * np.ones(n)], dtype=np.float32)[zm]
        dt = 1.  # never actually gets used
        plan = plan_lif_rate(queue, dt, raX, raX, ref, tau, blockify=False)
        plan()
    else:
        raise NotImplementedError()

    return clX


def compute_J(queue, X, gain, bias):
    from mako.template import Template
    from nengo_ocl.utils import as_ascii

    m, n = X.shape

    assert X.elemstrides[1] == 1
    assert gain.ndim == 1 and gain.elemstrides[0] == 1
    assert bias.ndim == 1 and bias.elemstrides[0] == 1

    text = """
        __kernel void fn(
            __global ${Xtype} *X,
            __global const ${gaintype} *gain,
            __global const ${biastype} *bias
        )
        {
            const int j = get_global_id(0);
            const int i = get_global_id(1);
            if (i < ${m} && j < ${n})
                X[i*${Xstride0} + j] =
                    X[i*${Xstride0} + j] * gain[j] + bias[j];
        }
        """
    textconf = dict(Xtype=X.ctype, gaintype=gain.ctype, biastype=bias.ctype,
                    m=m, n=n, Xstride0=X.elemstrides[0])
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))

    fn = cl.Program(queue.context, text).build().fn
    gsize, lsize = (n, m), None
    return fn(queue, gsize, lsize, X.data, gain.data, bias.data)


def plan_softmax(queue, X, Y):
    from mako.template import Template
    from nengo_ocl.utils import as_ascii
    from nengo_ocl.plan import Plan

    m, n = X.shape
    assert n <= 32
    assert Y.shape == X.shape
    assert X.elemstrides[1] == 1
    assert Y.elemstrides[1] == 1

    text = """
        __kernel void fn(
            __global const ${Xtype} *X,
            __global ${Ytype} *Y
        )
        {
            const int i = get_global_id(0);

            ${Xtype} ex[${n}];
            __global const ${Xtype} *x = X + i*${Xstride0};
            __global ${Ytype} *y = Y + i*${Ystride0};

            ${Xtype} maxx = -INFINITY;
            for (int j = 0; j < ${n}; j++)
                if (x[j] > maxx)
                    maxx = x[j];

            ${Xtype} sumex = 0;
            for (int j = 0; j < ${n}; j++) {
                ex[j] = exp(x[j] - maxx);
                sumex += ex[j];
            }

            for (int j = 0; j < ${n}; j++)
                y[j] = ex[j] / sumex;
        }
        """
    textconf = dict(Xtype=X.ctype, Ytype=Y.ctype, m=m, n=n,
                    Xstride0=X.elemstrides[0], Ystride0=Y.elemstrides[0])
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))

    fn = cl.Program(queue.context, text).build().fn
    fn.set_args(*[arr.data for arr in (X, Y)])
    plan = Plan(queue, fn, gsize=(m,))
    return plan


if scipy_optimize and pyopencl_blas and SoftmaxClassifier:
    @Builder.register_ocl_solver(SoftmaxClassifier)
    def solve_SoftmaxClassifier(solver, queue, clA, Y, rng=None, E=None):
        pyopencl_blas.setup()

        tstart = time.time()

        assert clA.shape[0] == Y.shape[0]
        m, n = clA.shape
        _, d = Y.shape
        Xshape = (n, d)

        # regularization
        sigma = solver.reg * cl.array.max(clA).get()
        lamb = m * sigma**2

        # --- initialization
        # X0 = np.zeros(Xshape, dtype=np.float32)
        X0 = np.zeros(Xshape, dtype=np.float64)

        # --- solve with L-BFGS
        clY = cl.array.to_device(queue, Y.astype(np.float32))
        clyi = cl.array.to_device(queue, np.argmax(Y, axis=1).astype(np.int32))
        clX = cl.array.Array(queue, (n, d), dtype=np.float32)
        clE = cl.array.Array(queue, (m, d), dtype=np.float32)
        clG = cl.array.Array(queue, (n, d), dtype=np.float32)

        softmax_plan = plan_softmax(queue, clE, clE)

        # sum_square = cl.reduction.ReductionKernel(
        #     queue.context, np.float32, neutral="0",
        #     reduce_expr="a+b", map_expr="x[i]*x[i]",
        #     arguments="__global float *x")

        sum_logloss = cl.reduction.ReductionKernel(
            queue.context, np.float32, neutral="0", reduce_expr="a+b",
            map_expr="-log(max(Y[i*%(d)d + yi[i]], 1e-16f))" % dict(d=d),
            arguments="__global const int *yi, __global const float *Y")
        assert clE.elemstrides[0] == d

        def f_df(x):
            clX.set(x.astype(np.float32).reshape(Xshape))
            pyopencl_blas.gemm(queue, clA, clX, clE)
            softmax_plan()
            cost = sum_logloss(clyi, clE).get()
            clE[:] -= clY
            pyopencl_blas.gemm(queue, clA, clE, clG, transA=True)
            if lamb > 0:
                cost += 0.5 * lamb * pyopencl.array.sum(clX**2).get()
                # cost += 0.5 * lamb * sum_square(clX).get()
                clG[:] += lamb * clX

            G = clG.get().astype(np.float64)
            return cost, G.ravel()

        x0 = X0.ravel()
        x, mincost, info = scipy_optimize.fmin_l_bfgs_b(
            f_df, x0, maxfun=solver.n_epochs, iprint=solver.verbose)

        tend = time.time()

        A = clA.get()
        X = x.reshape(Xshape)
        return solver.mul_encoders(X, E), {
            'rmses': npext.rms(softmax(np.dot(A, X), axis=1) - Y, axis=1),
            'time': tend - tstart}


def plan_hingeloss(queue, yinds, Z, c, E):
    from mako.template import Template
    from nengo_ocl.utils import as_ascii
    from nengo_ocl.plan import Plan

    m, n = Z.shape
    assert n <= 32
    assert Z.shape == E.shape
    assert Z.elemstrides[1] == 1
    assert E.elemstrides[1] == 1
    assert yinds.shape == (m,)
    assert yinds.elemstrides[0] == 1
    assert c.shape == (m,)
    assert c.elemstrides[0] == 1

    text = """
        __kernel void fn(
            __global const ${yindstype} *yinds,
            __global const ${Ztype} *Z,
            __global ${ctype} *c,
            __global ${Etype} *E
        )
        {
            const int i = get_global_id(0);

            const ${yindstype} yi = yinds[i];
            __global const ${Ztype} *z = Z + i*${Zstride0};
            __global ${Etype} *e = E + i*${Estride0};

            ${yindstype} ti;
            ${Ztype} zj, zy, zt = -INFINITY;
            zt = -INFINITY;
            for (int j = 0; j < ${n}; j++) {
                e[j] = 0;
                zj = z[j];
                if (j == yi) {
                    zy = zj;
                } else if (zj > zt) {
                    zt = zj;
                    ti = j;
                }
            }

            ${Ztype} margin = zy - zt;
            if (margin < 1) {
                e[yi] = -1;
                e[ti] = 1;
            }
            c[i] = max(1 - margin, 0.0f);
        }
        """
    textconf = dict(yindstype=yinds.ctype, Ztype=Z.ctype, ctype=c.ctype,
                    Etype=E.ctype, m=m, n=n,
                    Zstride0=Z.elemstrides[0], Estride0=E.elemstrides[0])
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))

    fn = cl.Program(queue.context, text).build().fn
    fn.set_args(*[arr.data for arr in (yinds, Z, c, E)])
    plan = Plan(queue, fn, gsize=(m,))
    return plan


if scipy_optimize and pyopencl_blas and HingeClassifier:
    @Builder.register_ocl_solver(HingeClassifier)
    def solve_HingeClassifier(solver, queue, clA, Y, rng=None, E=None):
        pyopencl_blas.setup()

        tstart = time.time()

        assert clA.shape[0] == Y.shape[0]
        m, n = clA.shape
        _, d = Y.shape
        Xshape = (n, d)

        # regularization
        sigma = solver.reg * cl.array.max(clA).get()
        lamb = m * sigma**2

        # --- initialization
        X0 = rng.uniform(-1./n, 1./n, size=Xshape)

        # --- solve with L-BFGS
        yinds = Y.argmax(axis=1)

        clX = cl.array.Array(queue, (n, d), dtype=np.float32)
        clyinds = cl.array.to_device(queue, yinds.astype(np.int32))
        clZ = cl.array.Array(queue, (m, d), dtype=np.float32)
        clc = cl.array.Array(queue, (m,), dtype=np.float32)
        clE = cl.array.Array(queue, (m, d), dtype=np.float32)
        clG = cl.array.Array(queue, (n, d), dtype=np.float32)

        hingeloss_plan = plan_hingeloss(queue, clyinds, clZ, clc, clE)

        def f_df(x):
            clX.set(x.astype(np.float32).reshape(Xshape))
            pyopencl_blas.gemm(queue, clA, clX, clZ)
            hingeloss_plan()

            cost = pyopencl.array.sum(clc).get()
            pyopencl_blas.gemm(queue, clA, clE, clG, transA=True)
            if lamb > 0:
                cost += 0.5 * lamb * pyopencl.array.sum(clX**2).get()
                # cost += 0.5 * lamb * sum_square(clX).get()
                clG[:] += lamb * clX

            G = clG.get().astype(np.float64)
            return cost, G.ravel()

        x0 = X0.ravel()
        x, mincost, info = scipy_optimize.fmin_l_bfgs_b(
            f_df, x0, maxfun=solver.n_epochs, iprint=solver.verbose)

        tend = time.time()

        A = clA.get()
        X = x.reshape(Xshape)
        return solver.mul_encoders(X, E), {
            'rmses': npext.rms(np.dot(A, X) - Y, axis=1),
            'time': tend - tstart}


if pyopencl_blas and LstsqClassifier:
    @Builder.register(LstsqClassifier)
    def build_lstsqclassifier(model, solver, conn, rng, transform):
        from nengo.builder.connection import multiply
        eval_points, gain, bias, X, Y, E = get_solve_params(
            model, solver, conn, rng, transform)

        # sort eval points by category
        assert Y.ndim == 2
        Yi = np.argmax(Y, axis=1)
        i = np.argsort(Yi)
        X[:] = X[i]
        Y[:] = Y[i]

        wrapped_solver = wrap_solver(model, conn, solve_for_decoders)
        decoders, solver_info = wrapped_solver(
            conn, gain, bias, X, Y, rng=rng, E=E,
            queue=model.builder.queue)
        weights = (decoders.T if conn.solver.weights else
                   multiply(transform, decoders.T))
        return eval_points, weights, solver_info

    @Builder.register_ocl_solver(LstsqClassifier)
    def solve_lstsqclassifier(solver, queue, clA, Y, rng=None, E=None):
        pyopencl_blas.setup()

        m, n = clA.shape
        _, d = Y.shape
        precompute_ai = solver.precompute_ai

        def XTdotX(clX):
            clXX = cl.array.Array(queue, (n, n), dtype=np.float32)
            pyopencl_blas.gemm(queue, clX, clX, clXX, transA=True)
            return clXX.get()

        def ATdotx(x):
            clx = cl.array.to_device(queue, x.astype(np.float32))
            cly = cl.array.Array(queue, (n,), dtype=np.float32)
            pyopencl_blas.gemv(queue, clA, clx, cly, transA=True)
            return cly.get()

        def AdotX(X):
            clX = cl.array.to_device(queue, X.astype(np.float32))
            clAX = cl.array.Array(queue, (m, clX.shape[1]), dtype=np.float32)
            pyopencl_blas.gemm(queue, clA, clX, clAX)
            return clAX.get()

        def getAi(i, cache={}):
            if i in cache:
                return cache[i]

            clAi = clAis[i]
            AAi = XTdotX(clAi)
            if precompute_ai:
                cache[i] = AAi
            return AAi

        tstart = time.time()

        sigma = solver.reg * cl.array.max(clA).get()

        # Get Y inds
        Yi = np.argmax(Y, axis=1)
        Yd = np.diff(Yi)
        assert set(np.unique(Yd)) == set((0, 1)), (
            "Y not sorted, or missing some classes")

        clAis = []
        for i in range(d):
            inds, = (Yi == i).nonzero()
            a, b = inds.min(), inds.max()+1
            clAis.append(clA[a:b])

        if not precompute_ai:
            AA = XTdotX(clA)
        else:
            AA = np.zeros((n, n))
            for i in range(d):
                AA += getAi(i)

        X = np.zeros((n, d))
        for i in range(d):
            y = Y[:, i]

            # weight for classification
            p = y.mean()
            q = solver.weight_power
            wr = p*(1-p)**q + (1-p)*p**q
            w0 = p**q / wr
            w1 = (1-p)**q / wr
            dw = w1 - w0
            w = w0 + dw*y

            # form Gram matrix G = A.T W A + m * sigma**2
            G = w0*AA + dw*getAi(i)
            np.fill_diagonal(G, G.diagonal() + m * sigma**2)
            b = ATdotx(w * y)

            # X[:, i] = cho_solve(G, b, overwrite=True)
            X[:, i] = np.linalg.solve(G, b)

        tend = time.time()

        AX = AdotX(X)
        return solver.mul_encoders(X, E), {
            'rmses': npext.rms(AX - Y, axis=1),
            'time': tend - tstart}
