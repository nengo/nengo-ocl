# pylint: disable=missing-module-docstring

import nengo_ocl


def make_test_sim(request):
    """A Simulator factory to be used in tests.

    This is passed to the ``nengo_simloader`` option when running tests, so that Nengo
    knows to use it in the ``Simulator`` fixture.
    """

    def TestSimulator(net, *args, **kwargs):
        kwargs.setdefault("progress_bar", False)
        return nengo_ocl.Simulator(net, *args, **kwargs)

    return TestSimulator
