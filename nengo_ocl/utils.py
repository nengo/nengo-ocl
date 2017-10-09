import collections
from collections import OrderedDict

import numpy as np

from nengo.utils.compat import PY2
try:
    from nengo.utils.stdlib import Timer
except ImportError:
    from nengo.utils.testing import Timer  # noqa: F401


class OrderedSet(collections.MutableSet):
    """An ordered version of set.

    Courtesy Raymond Hettinger (2009) under the MIT licence.
    http://code.activestate.com/recipes/576694/
    """
    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)


def as_ascii(string):
    if not PY2 and isinstance(string, bytes):  # Python 3
        return string.decode('ascii')
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
    return OrderedDict(zip(
        f.__code__.co_freevars, (c.cell_contents for c in f.__closure__)))


def indent(s, i):
    return '\n'.join([(' ' * i) + line for line in s.split('\n')])


def round_up(x, n):
    return int(np.ceil(float(x) / n)) * n


def split(iterator, criterion):
    """Returns a list of objects that match criterion and those that do not.
    """
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
