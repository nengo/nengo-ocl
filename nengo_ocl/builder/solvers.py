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
