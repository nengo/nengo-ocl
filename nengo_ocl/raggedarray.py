"""
Numpy implementation of RaggedArray data structure.

"""
from __future__ import print_function

from nengo.utils.compat import is_iterable, StringIO
import numpy as np

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


class RaggedArray(object):
    """A linear buffer partitioned into sections of various lengths.

    Can also be viewed as an efficient way of storing a list of arrays,
    in the same underlying buffer.
    """

    def __init__(self, arrays, names=None, dtype=None, align=False):
        arrays = [np.asarray(a) for a in arrays]
        assert len(arrays) > 0
        assert all(a.ndim <= 2 for a in arrays)

        self.names = [''] * len(arrays) if names is None else list(names)
        assert len(self.names) == len(arrays)

        self.shape0s = [a.shape[0] if a.ndim > 0 else 1 for a in arrays]
        self.shape1s = [a.shape[1] if a.ndim > 1 else 1 for a in arrays]
        assert 0 not in self.shape1s
        self.stride0s = [a.shape[1] if a.ndim == 2 else 1 for a in arrays]
        self.stride1s = [1 for a in arrays]

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

        buf = np.zeros(starts[-1] + arrays[-1].size, dtype=dtype)
        for a, s in zip(arrays, starts):
            buf[s:s+a.size] = a.ravel()

        self.starts = starts
        self.buf = buf

    @property
    def dtype(self):
        return self.buf.dtype

    @property
    def nbytes(self):
        return (self.shape0s * self.shape1s).sum() * self.dtype.itemsize

    def __str__(self):
        sio = StringIO()
        namelen = max(len(n) for n in self.names)
        fmt = '%%%is' % namelen
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
            bytestrides = (itemsize * self.stride0s[item],
                           itemsize * self.stride1s[item])
            return np.ndarray(
                shape=shape, dtype=self.dtype, buffer=self.buf.data,
                offset=byteoffset, strides=bytestrides)

    def __setitem__(self, item, val):
        try:
            item = int(item)
        except TypeError:
            raise NotImplementedError()
        self[item][...] = val

    def add_views(self, starts, shape0s, shape1s, stride0s, stride1s,
                  names=None):
        assert all(s >= 0 for s in starts)
        assert all(s + s0*s1 <= self.buf.size
                   for s, s0, s1 in zip(starts, shape0s, shape1s))
        assert 0 not in shape0s
        assert 0 not in shape1s
        self.starts = self.starts + starts
        self.shape0s = self.shape0s + shape0s
        self.shape1s = self.shape1s + shape1s
        self.stride0s = self.stride0s + stride0s
        self.stride1s = self.stride1s + stride1s
        if names:
            self.names = self.names + names
        else:
            self.names = self.names + [''] * len(starts)
