import nengo.builder
from nengo.builder import Model  # noqa: F401


# Allow us to register build functions specific to this Simulator
class Builder(nengo.builder.Builder):
    pass
