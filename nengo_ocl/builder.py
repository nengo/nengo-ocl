import nengo.builder
from nengo.builder import Signal
from nengo.builder.learning_rules import build_or_passthrough, get_post_ens, SimPES
from nengo.builder.operator import Copy, DotInc, Reset
from nengo.exceptions import BuildError
from nengo.learning_rules import PES


class Builder(nengo.builder.Builder):
    """
    Copy of the default Nengo builder.

    This class is here so that we can register new build functions for
    NengoOCL without affecting the default Nengo build process.
    """

    builders = {}  # this will be different than nengo.builder.Builder.builders

    @classmethod
    def build(cls, model, obj, *args, **kwargs):
        """
        Build ``obj`` into ``model``.

        This method looks up the appropriate build function for ``obj`` and
        calls it with the model and other arguments provided.

        In addition to the parameters listed below, further positional and
        keyword arguments will be passed unchanged into the build function.

        Parameters
        ----------
        model : Model
            The `~nengo.builder.Model` instance in which to store build
            artifacts.
        obj : object
            The object to build into the model.
        """

        try:
            # first try building obj using any custom build functions that have
            # been registered by NengoOCL
            return nengo.builder.Builder.build.__func__(
                Builder, model, obj, *args, **kwargs
            )
        except BuildError:
            # fallback on normal nengo builder
            return nengo.builder.Builder.build.__func__(
                nengo.builder.Builder, model, obj, *args, **kwargs
            )


@Builder.register(PES)
def build_pes(model, pes, rule):
    """Builds a `.PES` object into a model.

    Calls synapse build functions to filter the pre activities,
    and adds a `.SimPES` operator to the model to calculate the delta.

    Parameters
    ----------
    model : Model
        The model to build into.
    pes : PES
        Learning rule type to build.
    rule : LearningRule
        The learning rule object corresponding to the neuron type.

    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.PES` instance.
    """

    conn = rule.connection

    # Create input error signal
    error = Signal(shape=(rule.size_in,), name="PES:error")
    model.add_op(Reset(error))
    model.sig[rule]["in"] = error  # error connection will attach here

    acts = build_or_passthrough(model, pes.pre_synapse, model.sig[conn.pre_obj]["out"])

    if not conn.is_decoded:
        # multiply error by post encoders to get a per-neuron error

        post = get_post_ens(conn)
        encoders = model.sig[post]["encoders"]

        if conn.post_obj is not conn.post:
            # in order to avoid slicing encoders along an axis > 0, we pad
            # `error` out to the full base dimensionality and then do the
            # dotinc with the full encoder matrix
            padded_error = Signal(shape=(encoders.shape[1],))
            model.add_op(Copy(error, padded_error, dst_slice=conn.post_slice))
        else:
            padded_error = error

        # error = dot(encoders, error)
        local_error = Signal(shape=(post.n_neurons,))
        model.add_op(Reset(local_error))
        model.add_op(DotInc(encoders, padded_error, local_error, tag="PES:encode"))
    else:
        local_error = error

    model.operators.append(
        SimPES(acts, local_error, model.sig[rule]["delta"], pes.learning_rate)
    )

    # expose these for probes
    model.sig[rule]["error"] = error
    model.sig[rule]["activities"] = acts
