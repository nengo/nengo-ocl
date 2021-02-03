"""Utility functions and compatibility imports."""

# pylint: disable=missing-class-docstring,missing-function-docstring

import warnings

import nengo
import numpy as np
import pyopencl as cl

import nengo_ocl


def as_ascii(string):
    if isinstance(string, bytes):
        return string.decode("ascii")
    elif not isinstance(string, str):
        return str(string)
    else:
        return string


def equal_strides(strides1, strides2, shape):
    """Check whether two arrays have equal strides.

    Code from https://github.com/inducer/compyte
    """
    if len(strides1) != len(strides2) or len(strides2) != len(shape):
        return False

    for s, st1, st2 in zip(shape, strides1, strides2):
        if s != 1 and st1 != st2:
            return False

    return True


def get_closures(f):
    return dict(zip(f.__code__.co_freevars, (c.cell_contents for c in f.__closure__)))


def indent(s, i):
    return "\n".join([(" " * i) + line for line in s.split("\n")])


def nonelist(*args):
    return [arg for arg in args if arg is not None]


def round_up(x, n):
    return int(np.ceil(float(x) / n)) * n


def split(iterator, criterion):
    """Returns a list of objects that match criterion and those that do not."""
    a = []
    b = []
    for x in iterator:
        if criterion(x):
            a.append(x)
        else:
            b.append(x)

    return a, b


def stable_unique(seq):
    seen = set()
    rval = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            rval.append(item)
    return rval


class SimRunner:
    """Run benchmarks with different simulators/backends.

    This allows network configuration, simulator creation, and simulator running to
    all be customized for different simulators/backends.
    """

    _runners = {}

    @classmethod
    def get_runner(cls, key, **new_kwargs):
        runner, kwargs = cls._runners[key]
        kwargs.update(new_kwargs)
        return runner(**kwargs)

    @classmethod
    def register_runner(cls, key, runner, **kwargs):
        if key in cls._runners:
            warnings.warn(f"Runner for '{key}' already registered. Overwriting.")

        cls._runners[key] = (runner, kwargs)

    def __init__(self, name=None, **kwargs):
        self.name = "sim" if name is None else name
        self.kwargs = kwargs

    def configure_network(self, network):
        pass

    def make_sim(self, network):
        raise NotImplementedError("subclass must implement")

    def run_sim(self, sim, simtime):
        sim.run(simtime)


class RefRunner(SimRunner):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.name = "ref" if name is None else name

    def make_sim(self, network):
        return nengo.Simulator(network, progress_bar=False, **self.kwargs)


class OCLRunner(SimRunner):
    def __init__(self, name=None, n_prealloc_probes=32, **kwargs):
        super().__init__(name=name, **kwargs)

        context = self.kwargs.get("context", None)
        if context is None:
            context = cl.create_some_context()
            self.kwargs["context"] = context

        self.name = context.devices[0].name if name is None else name
        self.n_prealloc_probes = n_prealloc_probes

    def make_sim(self, network):
        return nengo_ocl.Simulator(
            network,
            n_prealloc_probes=self.n_prealloc_probes,
            progress_bar=False,
            **self.kwargs,
        )


class DLRunner(SimRunner):
    def __init__(self, name=None, n_prealloc_probes=32, inference_only=True, **kwargs):
        super().__init__(name=name, **kwargs)
        self.name = "dl" if name is None else name
        self.n_prealloc_probes = n_prealloc_probes
        self.inference_only = inference_only

    def configure_network(self, network):
        import nengo_dl  # pylint: disable=import-outside-toplevel

        if self.inference_only:
            nengo_dl.configure_settings(inference_only=True)

    def make_sim(self, network):
        import nengo_dl  # pylint: disable=import-outside-toplevel

        return nengo_dl.Simulator(network, progress_bar=False, **self.kwargs)

    def run_sim(self, sim, simtime):
        max_steps = self.kwargs.get("n_prealloc_probes", None)

        if max_steps is None:
            sim.run(simtime)
        else:
            n_steps = int(np.round(simtime / sim.dt))
            while n_steps > 0:
                steps = min(n_steps, max_steps)
                sim.run_steps(steps)
                n_steps -= steps


SimRunner.register_runner("ref", RefRunner)
SimRunner.register_runner("ocl", OCLRunner)
SimRunner.register_runner("ocl-profile", OCLRunner, profiling=True)
SimRunner.register_runner("dl", DLRunner)
