"""Utility functions and compatibility imports."""

# pylint: disable=missing-class-docstring,missing-function-docstring

from collections import OrderedDict

import numpy as np


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
    return OrderedDict(
        zip(f.__code__.co_freevars, (c.cell_contents for c in f.__closure__))
    )


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
