import nengo.builder
from nengo.builder import Model  # noqa: F401
from nengo.exceptions import BuildError


# Allow us to register build functions specific to this Simulator
class Builder(nengo.builder.Builder):

    solvers = {}

    def __init__(self, queue=None):
        super(Builder, self).__init__()
        self.queue = queue

    @classmethod
    def has_ocl_solver(cls, solver):
        for obj_cls in type(solver).__mro__:
            if obj_cls in cls.solvers:
                return True

        return False

    @classmethod
    def register_ocl_solver(cls, solver_class):
        def register_solver(solve_fn):
            cls.solvers[solver_class] = solve_fn
            return solve_fn
        return register_solver

    @classmethod
    def solve(cls, solver, queue, clA, Y, E=None):
        for obj_cls in type(solver).__mro__:
            if obj_cls in cls.solvers:
                break
        else:
            raise BuildError(
                "No OCL solver for type %r" % type(solver).__name__)

        return cls.solvers[obj_cls](solver, queue, clA, Y, E=E)
