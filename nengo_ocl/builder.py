"""NengoOCL-specific build functionality.

NengoOCL primarily uses the Nengo core builder, but may override some build functions,
which is done in this file.
"""

import nengo.builder
from nengo.exceptions import BuildError


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
