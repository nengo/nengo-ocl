"""
Numpy implementation of RaggedArray data structure.

"""
from __future__ import print_function

from nengo.utils.compat import StringIO
import numpy as np


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
    # a linear buffer that is partitioned into
    # sections of various lengths.
    #

    @property
    def dtype(self):
        return self.buf.dtype

    def __init__(self, arrays, names=None):
        arrays = [np.asarray(a) for a in arrays]
        assert len(arrays) > 0
        assert all(a.ndim <= 2 for a in arrays)

        self.names = [''] * len(arrays) if names is None else list(names)
        assert len(self.names) == len(arrays)

        sizesum = np.cumsum([0] + [a.size for a in arrays])
        self.starts = sizesum[:-1].tolist()
        self.shape0s = [a.shape[0] if a.ndim > 0 else 1 for a in arrays]
        self.shape1s = [a.shape[1] if a.ndim > 1 else 1 for a in arrays]
        assert 0 not in self.shape1s
        self.stride0s = [a.shape[1] if a.ndim == 2 else 1 for a in arrays]
        self.stride1s = [1 for a in arrays]
        self.buf = np.concatenate([a.ravel() for a in arrays])

    def __str__(self):
        sio = StringIO()
        namelen = max(len(n) for n in self.names)
        fmt = '%%%is' % namelen
        for ii, nn in enumerate(self.names):
            print((fmt % nn), self[ii], file=sio)
        return sio.getvalue()

    def shallow_copy(self):
        rval = self.__class__.__new__(self.__class__)
        rval.starts = self.starts
        rval.shape0s = self.shape0s
        rval.shape1s = self.shape1s
        rval.stride0s = self.stride0s
        rval.stride1s = self.stride1s
        rval.buf = self.buf
        rval.names = self.names
        return rval

    def add_views(self, starts, shape0s, shape1s, stride0s, stride1s,
                  names=None):
        # assert start >= 0
        # assert start + length <= len(self.buf)
        # -- creates copies, same semantics
        #    as OCL version
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

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, item):
        if isinstance(item, (list, tuple)):
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
            byteoffset = itemsize * self.starts[item]
            bytestrides = (
                itemsize * self.stride0s[item],
                itemsize * self.stride1s[item])
            shape = self.shape0s[item], self.shape1s[item]
            if shape[0] == 0:
                # -- The list of A_js for example can be
                #    an empty list.
                return []
            elif shape[1] == 0:
                raise ValueError('empty array', item)
            try:
                view = np.ndarray(
                    shape=shape,
                    dtype=self.dtype,
                    buffer=self.buf.data,
                    offset=byteoffset,
                    strides=bytestrides)
            except:
                print(self.names[item])
                print(shape)
                print(self.dtype)
                print(self.buf.size)
                print(byteoffset)
                print(bytestrides)
                raise
            return view

    def view1d(self, idxs):
        raise NotImplementedError('since cutting ldas')
        start = idxs[0]
        if idxs != range(start, start + len(idxs)):
            raise NotImplementedError('non-contiguous indexes')
        start_offset = self.starts[start]
        stop_offset = (self.starts[idxs[-1]]
                       + self.shape0s[idxs[-1]] * self.shape1s[idxs[-1]])
        for ii in idxs:
            if self.ldas[ii] != self.shape0s[ii]:
                raise NotImplementedError('non-contiguous element',
                                          (ii, self.ldas[ii], self.shape0s[ii],
                                           self.shape1s[ii]))
            if ii != idxs[-1]:
                if self.starts[ii + 1] != (
                        self.starts[ii] + self.shape0s[ii] * self.shape1s[ii]):
                    raise NotImplementedError('gap between elements', ii)
        itemsize = self.dtype.itemsize
        byteoffset = itemsize * start_offset
        shape = (stop_offset - start_offset), 1
        bytestrides = (itemsize, itemsize * shape[0])
        try:
            view = np.ndarray(
                shape=shape,
                dtype=self.dtype,
                buffer=self.buf.data,
                offset=byteoffset,
                strides=bytestrides)
        except:
            print(shape)
            print(self.dtype)
            print(self.buf.size)
            print(byteoffset)
            print(bytestrides)

            raise
        return view

    def __setitem__(self, item, val):
        try:
            item = int(item)
        except TypeError:
            raise NotImplementedError()
        self[item][...] = val
