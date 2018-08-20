import nengo.builder
from nengo.builder import Model  # noqa: F401
from nengo.exceptions import BuildError


# Allow us to register build functions specific to this Simulator
class Builder(nengo.builder.Builder):

    ocl_builders = {}
    ocl_solvers = {}

    def __init__(self, queue=None):
        super(Builder, self).__init__()
        self.queue = queue

    def build(cls, model, obj, *args, **kwargs):
        if model.has_built(obj):
            warnings.warn("Object %s has already been built." % obj)
            return None

        for obj_cls in type(obj).__mro__:
            if obj_cls in cls.ocl_builders:
                return cls.ocl_builders[obj_cls](model, obj, *args, **kwargs)

        return super(Builder, cls).build(model, obj, *args, **kwargs)

    @classmethod
    def register(cls, nengo_class):
        def register_builder(build_fn):
            if nengo_class in cls.ocl_builders:
                warnings.warn("Type '%s' already has a builder. Overwriting."
                              % nengo_class)
            cls.ocl_builders[nengo_class] = build_fn
            return build_fn
        return register_builder

    @classmethod
    def has_ocl_solver(cls, solver):
        for obj_cls in type(solver).__mro__:
            if obj_cls in cls.ocl_solvers:
                return True

        return False

    @classmethod
    def register_ocl_solver(cls, solver_class):
        def register_solver(solve_fn):
            cls.ocl_solvers[solver_class] = solve_fn
            return solve_fn
        return register_solver

    @classmethod
    def solve(cls, solver, queue, clA, Y, rng=None, E=None):
        for obj_cls in type(solver).__mro__:
            if obj_cls in cls.ocl_solvers:
                break
        else:
            raise BuildError(
                "No OCL solver for type %r" % type(solver).__name__)

        return cls.ocl_solvers[obj_cls](solver, queue, clA, Y, rng=rng, E=E)
