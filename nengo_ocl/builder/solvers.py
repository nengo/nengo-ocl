import numpy as np

from nengo.exceptions import BuildError
from nengo.solvers import Solver

from nengo_ocl.builder import Builder


@Builder.register(Solver)
def build_solver(model, solver, conn, rng, transform):
    from nengo.builder.connection import solve_for_decoders, multiply, get_eval_points, get_targets

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

    try:
        wrapped_solver = (model.decoder_cache.wrap_solver(solve_for_decoders)
                          if model.seeded[conn] else solve_for_decoders)
        decoders, solver_info = wrapped_solver(
            solver, conn.pre_obj.neuron_type, gain, bias, x, targets,
            rng=rng, E=E)
    except BuildError:
        raise BuildError(
            "Building %s: 'activities' matrix is all zero for %s. "
            "This is because no evaluation points fall in the firing "
            "ranges of any neurons." % (conn, conn.pre_obj))

    weights = (decoders.T if solver.weights else
               multiply(transform, decoders.T))
    return eval_points, weights, solver_info
