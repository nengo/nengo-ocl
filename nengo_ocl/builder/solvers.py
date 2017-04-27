import warnings
import time

import numpy as np
import pyopencl as cl
import pyopencl.array

import nengo
from nengo.exceptions import BuildError
from nengo.utils.least_squares_solvers import rmses

from nengo_ocl.builder import Builder
from nengo_ocl.clraggedarray import CLRaggedArray
from nengo_ocl.clra_nonlinearities import plan_lif_rate


@Builder.register(nengo.solvers.Solver)
def build_solver(model, solver, conn, rng, transform):
    from nengo.builder.connection import multiply, get_eval_points, get_targets

    encoders = model.params[conn.pre_obj].encoders
    gain = model.params[conn.pre_obj].gain
    bias = model.params[conn.pre_obj].bias

    eval_points = get_eval_points(model, conn, rng)
    targets = get_targets(conn, eval_points)

    x = np.dot(eval_points, encoders.T / conn.pre_obj.radius)
    E = None
    if solver.weights:
        E = model.params[conn.post_obj].scaled_encoders.T[conn.post_slice]
        # include transform in solved weights
        targets = multiply(targets, transform.T)

    wrapped_solver = (model.decoder_cache.wrap_solver(solve_for_decoders)
                      if model.seeded[conn] else solve_for_decoders)
    decoders, solver_info = wrapped_solver(
        solver, conn.pre_obj.neuron_type, gain, bias, x, targets,
        rng=rng, E=E, conn=conn, queue=model.builder.queue)

    weights = (decoders.T if solver.weights else
               multiply(transform, decoders.T))
    return eval_points, weights, solver_info


def solve_for_decoders(
        solver, neuron_type, gain, bias, X, Y, rng, E=None, conn=None,
        queue=None):
    ocl_activity_min_size = 1e6

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


def neuron_rates(neuron_type, x, gain, bias, queue=None):
    import pyopencl.elementwise

    m, n = x.shape

    X = cl.array.to_device(queue, x.astype(np.float32))
    gain = cl.array.to_device(queue, gain.astype(np.float32))
    bias = cl.array.to_device(queue, bias.astype(np.float32))
    J = cl.array.Array(queue, X.shape, dtype=np.float32)
    R = cl.array.Array(queue, J.shape, dtype=np.float32)
    compute_J(queue, X, gain, bias, J)

    raJ = CLRaggedArray.from_clarray_rows(queue, J)
    raR = CLRaggedArray.from_clarray_rows(queue, R)

    if type(neuron_type) in (nengo.neurons.LIF, nengo.neurons.LIFRate,
                             nengo.neurons.AdaptiveLIF,
                             nengo.neurons.AdaptiveLIFRate):
        zm = [0] * m  # use same gains and biases for all eval points
        tau = CLRaggedArray.from_arrays(
            queue, [neuron_type.tau_rc * np.ones(n)], dtype=np.float32)[zm]
        ref = CLRaggedArray.from_arrays(
            queue, [neuron_type.tau_ref * np.ones(n)], dtype=np.float32)[zm]
        dt = 1.  # never actually gets used
        plan = plan_lif_rate(queue, dt, raJ, raR, ref, tau, blockify=False)
        plan()
    else:
        raise NotImplementedError()

    return R


def compute_J(queue, X, gain, bias, J):
    from mako.template import Template
    from nengo_ocl.utils import as_ascii

    m, n = X.shape

    assert X.elemstrides[1] == 1
    assert J.elemstrides[1] == 1
    assert gain.ndim == 1 and gain.elemstrides[0] == 1
    assert bias.ndim == 1 and bias.elemstrides[0] == 1

    text = """
        __kernel void fn(
            __global const ${Xtype} *X,
            __global const ${gaintype} *gain,
            __global const ${biastype} *bias,
            __global ${Jtype} *J
        )
        {
            const int j = get_global_id(0);
            const int i = get_global_id(1);
            if (i < ${m} && j < ${n})
                J[i*${Jstride0} + j] = X[i*${Xstride0} + j] * gain[j] + bias[j];
        }
        """
    textconf = dict(Xtype=X.ctype, gaintype=gain.ctype, biastype=bias.ctype,
                    Jtype=J.ctype, m=m, n=n,
                    Jstride0=J.elemstrides[0], Xstride0=X.elemstrides[0])
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))

    fn = cl.Program(queue.context, text).build().fn
    gsize, lsize = (n, m), None
    return fn(queue, gsize, lsize, X.data, gain.data, bias.data, J.data)


# @Builder.register_ocl_solver(nengo.solvers.LstsqL2)
# def solve_LstsqL2(solver, queue, clA, Y, rng=None, E=None):
#     from hunse_tools.timing import tic, toc

#     dtype = np.float32
#     # dtype = np.float64

#     import pyopencl_blas
#     pyopencl_blas.setup()

#     tstart = time.time()

#     A = clA.get()
#     clA = clA.astype(dtype)
#     clY = cl.array.to_device(queue, Y.astype(dtype))

#     sigma = solver.reg * cl.array.max(clA).get()

#     m, n = clA.shape
#     _, d = Y.shape

#     transpose = solver.solver.transpose
#     if transpose is None:
#         transpose = m < n  # transpose if matrix is fat

#     if transpose:
#         # substitution: x = A'*xbar, G*xbar = b where G = A*A' + lambda*I
#         clG = cl.array.Array(queue, (m, m), dtype=dtype)
#         pyopencl_blas.gemm(queue, clA, clA, clG, transB=True)
#         G, B = clG.get(), Y
#     else:
#         # multiplication by A': G*x = A'*b where G = A'*A + lambda*I
#         clG = cl.array.Array(queue, (n, n), dtype=dtype)
#         clB = cl.array.Array(queue, (n, d), dtype=dtype)
#         pyopencl_blas.gemm(queue, clA, clA, clG, transA=True)
#         pyopencl_blas.gemm(queue, clA, clY, clB, transA=True)
#         G, B = clG.get().astype(np.float64), clB.get().astype(np.float64)

#     # pyopencl_blas.teardown()

#     np.fill_diagonal(G, G.diagonal() + 2*m * sigma**2)

#     try:
#         import scipy.linalg
#         factor = scipy.linalg.cho_factor(G, overwrite_a=True)
#         X = scipy.linalg.cho_solve(factor, B)
#     except ImportError:
#         L = np.linalg.cholesky(G)
#         L = np.linalg.inv(L.T)
#         X = np.dot(L, np.dot(L.T, B))

#     # U, s, V = np.linalg.svd(G, full_matrices=False)
#     # si = 1. / s
#     # si[s < 1e-8*s.max()] = 0
#     # X = np.dot(V.T, si[:, None] * np.dot(U.T, B))

#     X = np.dot(A.T, X) if transpose else X
#     tend = time.time()

#     info = {'rmses': rmses(A, X, Y), 'time': tend - tstart}
#     return solver.mul_encoders(X, E), info
