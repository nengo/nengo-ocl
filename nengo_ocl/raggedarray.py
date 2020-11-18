"""
Numpy implementation of RaggedArray data structure.
"""

# pylint: disable=missing-function-docstring

from io import StringIO

import numpy as np
from nengo.utils.numpy import is_iterable

from nengo_ocl.utils import round_up


def allclose(a, b, atol=1e-3, rtol=1e-3):
    if not np.allclose(a.starts, b.starts):
        return False
    if not np.allclose(a.shape0s, b.shape0s):
        return False
    if not np.allclose(a.shape1s, b.shape1s):
        return False
    if not np.allclose(a.stride0s, b.stride0s):
        return False
    if not np.allclose(a.stride1s, b.stride1s):
        return False
    if not np.allclose(a.buf, b.buf, atol=atol, rtol=rtol):
        return False
    return True


class RaggedArray:
    """A linear buffer partitioned into sections of various lengths.

    Can also be viewed as an efficient way of storing a list of arrays,
    in the same underlying buffer.
    """

    def __init__(self, arrays, names=None, dtype=None, align=False):
        arrays = [np.asarray(a) for a in arrays]
        assert len(arrays) > 0
        assert all(a.ndim <= 2 for a in arrays)

        dtype = arrays[0].dtype if dtype is None else dtype
        assert dtype in [np.float32, np.int32]

        if not align:
            starts = np.cumsum([0] + [a.size for a in arrays[:-1]]).tolist()
        else:
            starts = [0]
            sizes = [arrays[0].size]
            for a in arrays[1:]:
                size = a.size
                power = 16 if size >= 16 else 4 if size >= 8 else 1
                starts.append(round_up(starts[-1] + sizes[-1], power))
                sizes.append(size)

        self.starts = starts
        self.shape0s = [a.shape[0] if a.ndim > 0 else 1 for a in arrays]
        self.shape1s = [a.shape[1] if a.ndim > 1 else 1 for a in arrays]
        self.stride0s = [a.shape[1] if a.ndim == 2 else 1 for a in arrays]
        self.stride1s = [1 for a in arrays]

        buf = np.zeros(starts[-1] + arrays[-1].size, dtype=dtype)
        for a, s in zip(arrays, starts):
            buf[s : s + a.size] = a.ravel()

        self.buf = buf
        self.names = names

        self._sizes = None

    @classmethod
    def from_buffer(cls, buf, starts, shape0s, shape1s, stride0s, stride1s, names=None):
        rval = cls.__new__(cls)
        rval.starts = starts
        rval.shape0s = shape0s
        rval.shape1s = shape1s
        rval.stride0s = stride0s
        rval.stride1s = stride1s
        rval.buf = buf
        rval.names = names
        return rval

    @property
    def dtype(self):
        return self.buf.dtype

    @property
    def names(self):
        return self._names

    @names.setter
    def names(self, names):
        if names is None:
            names = [""] * len(self.starts)
        self._names = tuple(names)
        assert len(self.names) == self.starts.size

    @property
    def nbytes(self):
        return self.sizes.sum() * self.dtype.itemsize

    @property
    def starts(self):
        return self._starts

    @starts.setter
    def starts(self, starts):
        self._starts = np.array(starts, dtype="int32")
        self._starts.setflags(write=False)
        assert np.all(self.starts >= 0)

    @property
    def shape0s(self):
        return self._shape0s

    @shape0s.setter
    def shape0s(self, shape0s):
        self._shape0s = np.array(shape0s, dtype="int32")
        self._shape0s.setflags(write=False)
        self._sizes = None
        assert np.all(self.shape0s >= 0)
        assert self.shape0s.size == self.starts.size

    @property
    def shape1s(self):
        return self._shape1s

    @shape1s.setter
    def shape1s(self, shape1s):
        self._shape1s = np.array(shape1s, dtype="int32")
        self._shape1s.setflags(write=False)
        self._sizes = None
        assert np.all(self.shape1s > 0)
        assert self.shape1s.size == self.starts.size

    @property
    def sizes(self):
        if self._sizes is None:
            self._sizes = self.shape0s * self.shape1s
        return self._sizes

    @property
    def stride0s(self):
        return self._stride0s

    @stride0s.setter
    def stride0s(self, stride0s):
        self._stride0s = np.array(stride0s, dtype="int32")
        self._stride0s.setflags(write=False)
        assert np.all(self.stride0s != 0)

    @property
    def stride1s(self):
        return self._stride1s

    @stride1s.setter
    def stride1s(self, stride1s):
        self._stride1s = np.array(stride1s, dtype="int32")
        self._stride1s.setflags(write=False)
        assert np.all(self.stride1s != 0)

    @property
    def buf(self):
        return self._buf

    @buf.setter
    def buf(self, buf):
        buf = np.asarray(buf)
        assert buf.dtype in [np.float32, np.int32]
        self._buf = buf

    def __str__(self):
        sio = StringIO()
        namelen = max(len(n) for n in self.names)
        fmt = "%%%is" % namelen
        for ii, nn in enumerate(self.names):
            print((fmt % nn), self[ii], file=sio)
        return sio.getvalue()

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, item):
        if isinstance(item, slice):
            item = np.arange(len(self))[item]

        if is_iterable(item):
            rval = self.__class__.__new__(self.__class__)
            rval.starts = [self.starts[i] for i in item]
            rval.shape0s = [self.shape0s[i] for i in item]
            rval.shape1s = [self.shape1s[i] for i in item]
            rval.stride0s = [self.stride0s[i] for i in item]
            rval.stride1s = [self.stride1s[i] for i in item]
            rval.buf = self.buf
            rval.names = [self.names[i] for i in item]
            return rval
        else:
            if isinstance(item, np.ndarray):
                item.shape = ()  # avoid numpy DeprecationWarning

            itemsize = self.dtype.itemsize
            shape = (self.shape0s[item], self.shape1s[item])
            byteoffset = itemsize * self.starts[item]
            bytestrides = (
                itemsize * self.stride0s[item],
                itemsize * self.stride1s[item],
            )
            return np.ndarray(
                shape=shape,
                dtype=self.dtype,
                buffer=self.buf.data,
                offset=byteoffset,
                strides=bytestrides,
            )

    def __setitem__(self, item, val):
        try:
            item = int(item)
        except TypeError as e:
            raise NotImplementedError("`item` must be castable to int") from e
        self[item][...] = val

    def add_views(self, starts, shape0s, shape1s, stride0s, stride1s, names=None):
        assert all(
            s + s0 * s1 <= self.buf.size for s, s0, s1 in zip(starts, shape0s, shape1s)
        )
        self.starts = list(self.starts) + list(starts)
        self.shape0s = list(self.shape0s) + list(shape0s)
        self.shape1s = list(self.shape1s) + list(shape1s)
        self.stride0s = list(self.stride0s) + list(stride0s)
        self.stride1s = list(self.stride1s) + list(stride1s)
        if names:
            self.names = self.names + tuple(names)
        else:
            self.names = self.names + tuple([""] * len(starts))
