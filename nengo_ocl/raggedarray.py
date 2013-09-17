"""
Numpy implementation of RaggedArray data structure.

"""
import StringIO
import numpy as np

def shape0(obj):
    try:
        return obj.shape[0]
    except IndexError:
        return 1


def shape1(obj):
    try:
        return obj.shape[1]
    except IndexError:
        return 1


def allclose(a, b, atol=1e-3, rtol=1e-3):
    if not np.allclose(a.starts, b.starts): return False
    if not np.allclose(a.shape0s, b.shape0s): return False
    if not np.allclose(a.shape1s, b.shape1s): return False
    if not np.allclose(a.stride0s, b.stride1s): return False
    if not np.allclose(a.buf, b.buf, atol=atol, rtol=rtol): return False
    return True


class RaggedArray(object):
    # a linear buffer that is partitioned into
    # sections of various lengths.
    # 
    @property
    def dtype(self):
        return self.buf.dtype

    def __init__(self, listofarrays, names=None):
        starts = []
        shape0s = []
        shape1s = []
        stride0s = []
        stride1s = []
        buf = []

        for l in listofarrays:
            obj = np.asarray(l)
            starts.append(len(buf))
            shape0s.append(shape0(obj))
            shape1s.append(shape1(obj))
            if obj.ndim == 0:
                stride0s.append(1)
                stride1s.append(1)
            elif obj.ndim == 1:
                stride0s.append(1)
                stride1s.append(1)
            elif obj.ndim == 2:
                stride0s.append(obj.shape[1])
                stride1s.append(1)
            else:
                raise NotImplementedError()
            buf.extend(obj.ravel())

        self.starts = starts
        self.shape0s = shape0s
        self.shape1s = shape1s
        self.stride0s = stride0s
        self.stride1s = stride1s
        self.buf = np.asarray(buf)
        if names is None:
            self.names = [''] * len(self)
        else:
            assert len(names) == len(stride0s)
            self.names = names

    def __str__(self):
        sio = StringIO.StringIO()
        namelen = max(len(n) for n in self.names)
        fmt = '%%%is' % namelen
        for ii, nn in enumerate(self.names):
            print >> sio, (fmt % nn), self[ii]
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
        #assert start >= 0
        #assert start + length <= len(self.buf)
        # -- creates copies, same semantics
        #    as OCL version
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
            itemsize = self.dtype.itemsize
            byteoffset = itemsize * self.starts[item]
            bytestrides = (
                itemsize * self.stride0s[item],
                itemsize * self.stride1s[item])
            shape = self.shape0s[item], self.shape1s[item]
            if shape[0] * shape[1] == 0:
                return []
            try:
                view = np.ndarray(
                    shape=shape,
                    dtype=self.dtype,
                    buffer=self.buf.data,
                    offset=byteoffset,
                    strides=bytestrides)
            except:
                print self.names[item]
                print shape
                print self.dtype
                print self.buf.size
                print byteoffset
                print bytestrides
                raise
            return view

    def view1d(self, idxs):
        raise NotImplementedError('since cutting ldas')
        start = idxs[0]
        if idxs != range(start, start + len(idxs)):
            raise NotImplementedError('non-contiguous indexes')
        total_len = 0
        start_offset = self.starts[start]
        stop_offset = (self.starts[idxs[-1]]
            + self.shape0s[idxs[-1]] * self.shape1s[idxs[-1]])
        for ii in idxs:
            if self.ldas[ii] != self.shape0s[ii]:
                raise NotImplementedError('non-contiguous element',
                        (ii, self.ldas[ii], self.shape0s[ii],
                            self.shape1s[ii]))
            if ii != idxs[-1]:
                if self.starts[ii + 1] != (self.starts[ii] +
                        self.shape0s[ii] * self.shape1s[ii]):
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
            print shape
            print self.dtype
            print self.buf.size
            print byteoffset
            print bytestrides

            raise
        return view


    def __setitem__(self, item, val):
        try:
            item = int(item)
        except TypeError:
            raise NotImplementedError()
        self[item][...] = val

