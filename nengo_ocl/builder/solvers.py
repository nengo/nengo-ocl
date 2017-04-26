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


def solve_for_decoders(
        solver, neuron_type, gain, bias, x, targets, rng, E=None, conn=None,
        queue=None):
    if queue is not None and Builder.has_ocl_solver(solver):
        try:
            # --- use OCL device
            clA = neuron_rates(neuron_type, x, gain, bias, queue=queue)
            Y = targets
            decoders, solver_info = Builder.solve(solver, queue, clA, Y, E=E)
            return decoders, solver_info
        except Exception as e:
            warnings.warn("OCL solver failed: %s" % e)

    # --- use Numpy
    activities = neuron_type.rates(x, gain, bias)
    if np.count_nonzero(activities) == 0:
        raise BuildError(
            "Building %s: 'activities' matrix is all zero for %s. "
            "This is because no evaluation points fall in the firing "
            "ranges of any neurons." % (conn, conn.pre_obj))

    decoders, solver_info = solver(activities, targets, rng=rng, E=E)
    return decoders, solver_info


def neuron_rates(neuron_type, x, gain, bias, queue=None):
    m, n = x.shape
    J = gain * x + bias
    J = cl.array.to_device(queue, J.astype(np.float32))
    R = cl.array.Array(queue, J.shape, dtype=np.float32)

    Jra = CLRaggedArray.from_clarray_rows(queue, J)
    Rra = CLRaggedArray.from_clarray_rows(queue, R)

    if type(neuron_type) in (nengo.neurons.LIF, nengo.neurons.LIFRate,
                             nengo.neurons.AdaptiveLIF,
                             nengo.neurons.AdaptiveLIFRate):
        tau = CLRaggedArray.from_arrays(
            queue, [neuron_type.tau_rc * np.ones(n)], dtype=np.float32)
        ref = CLRaggedArray.from_arrays(
            queue, [neuron_type.tau_ref * np.ones(n)], dtype=np.float32)
        zm = [0] * m  # use same gains and biases for all eval points
        dt = 1.  # never actually gets used
        plan = plan_lif_rate(queue, dt, Jra, Rra, ref[zm], tau[zm])
        plan()
    else:
        raise NotImplementedError()

    return R


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


@Builder.register_ocl_solver(nengo.solvers.LstsqL2)
def solve_LstsqL2(solver, queue, clA, Y, E=None):
    import pyopencl_blas
    pyopencl_blas.setup()

    tstart = time.time()

    # sigma = solver.reg * A.max()
    sigma = solver.reg * pyopencl.array.max(clA).get()

    m, n = clA.shape
    _, d = Y.shape

    transpose = solver.solver.transpose
    if transpose is None:
        transpose = m < n  # transpose if matrix is fat

    A = clA.get()
    clY = cl.array.to_device(queue, Y.astype(np.float32))

    if transpose:
        # substitution: x = A'*xbar, G*xbar = b where G = A*A' + lambda*I
        clG = cl.array.Array(queue, (m, m), dtype=np.float32)
        pyopencl_blas.gemm(queue, clA, clA, clG, transB=True)
        G, B = clG.get(), Y
    else:
        # multiplication by A': G*x = A'*b where G = A'*A + lambda*I
        clG = cl.array.Array(queue, (n, n), dtype=np.float32)
        clB = cl.array.Array(queue, (n, d), dtype=np.float32)
        pyopencl_blas.gemm(queue, clA, clA, clG, transA=True)
        pyopencl_blas.gemm(queue, clA, clY, clB, transA=True)
        G, B = clG.get(), clB.get()

    # pyopencl_blas.teardown()

    np.fill_diagonal(G, G.diagonal() + m * sigma**2)

    try:
        import scipy.linalg
        factor = scipy.linalg.cho_factor(G, overwrite_a=True)
        X = scipy.linalg.cho_solve(factor, B)
    except ImportError:
        L = np.linalg.cholesky(G)
        L = np.linalg.inv(L.T)
        X = np.dot(L, np.dot(L.T, B))

    X = np.dot(A.T, X) if transpose else X
    tend = time.time()

    info = {'rmses': rmses(A, X, Y), 'time': tend - tstart}
    return solver.mul_encoders(X, E), info
