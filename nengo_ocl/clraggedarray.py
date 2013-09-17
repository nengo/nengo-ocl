"""
OpenCL-based implementation of RaggedArray data structure.

"""

import StringIO
import numpy as np
import pyopencl as cl
from .clarray import to_device
from .raggedarray import RaggedArray

def to_host(queue, data, dtype, start, shape, elemstrides):
    """Copy memory off the device, into a Numpy array"""

    m, n = shape
    Sm, Sn = elemstrides
    if m * n == 0:
        return np.zeros(shape, dtype=dtype)

    if min(elemstrides) < 0:
        raise NotImplementedError()

    itemsize = dtype.itemsize
    bytestart = itemsize * start
    # -- TODO: is there an extra element transferred here?
    byteend = bytestart + itemsize * ((m-1) * Sm + (n-1) * Sn + 1)

    temp_buf = np.zeros((byteend - bytestart), dtype=np.int8)
    cl.enqueue_copy(queue, temp_buf, data,
                    device_offset=bytestart, is_blocking=True)

    bytestrides = (itemsize * Sm, itemsize * Sn)
    try:
        view = np.ndarray(
            shape=(m, n),
            dtype=dtype,
            buffer=temp_buf.data,
            offset=0,
            strides=bytestrides)
    except:
        raise
    return view

class CLRaggedArray(object):
    # a linear buffer that is partitioned into
    # sections of various lengths.
    #

    @property
    def dtype(self):
        return self.buf.dtype

    def __init__(self, queue, np_raggedarray):
        self.queue = queue
        self.starts = np_raggedarray.starts
        self.shape0s = np_raggedarray.shape0s
        self.shape1s = np_raggedarray.shape1s
        self.stride0s = np_raggedarray.stride0s
        self.stride1s = np_raggedarray.stride1s
        self.buf = np_raggedarray.buf
        self.names = np_raggedarray.names

    def __str__(self):
        sio = StringIO.StringIO()
        print 'names', self.names
        namelen = max(len(n) for n in self.names)
        fmt = '%%%is' % namelen
        for ii, nn in enumerate(self.names):
            print '->', self[ii]
            print >> sio, (fmt % nn), self[ii]
        return sio.getvalue()

    @property
    def starts(self):
        return self._starts.tolist()

    @starts.setter
    def starts(self, starts):
        self._starts = np.array(starts, dtype='int32')
        self.cl_starts = to_device(self.queue, self._starts)
        self.queue.finish()

    @property
    def shape0s(self):
        return self._shape0s.tolist()

    @shape0s.setter
    def shape0s(self, shape0s):
        self._shape0s = np.array(shape0s, dtype='int32')
        self.cl_shape0s = to_device(self.queue, self._shape0s)
        self.queue.finish()

    @property
    def shape1s(self):
        return self._shape1s.tolist()

    @shape1s.setter
    def shape1s(self, shape1s):
        self._shape1s = np.array(shape1s, dtype='int32')
        self.cl_shape1s = to_device(self.queue, self._shape1s)
        self.queue.finish()

    @property
    def stride0s(self):
        return self._stride0s.tolist()

    @stride0s.setter
    def stride0s(self, stride0s):
        self._stride0s = np.array(stride0s, dtype='int32')
        self.cl_stride0s = to_device(self.queue, self._stride0s)
        self.queue.finish()

    @property
    def stride1s(self):
        return self._stride1s.tolist()

    @stride1s.setter
    def stride1s(self, stride1s):
        self._stride1s = np.array(stride1s, dtype='int32')
        self.cl_stride1s = to_device(self.queue, self._stride1s)
        self.queue.finish()

    @property
    def buf(self):
        buf = self.cl_buf.get()
        buf.setflags(write=False)
        return buf

    @buf.setter
    def buf(self, buf):
        buf = np.asarray(buf)
        if 'int' in str(buf.dtype):
            buf = buf.astype('int32')
        if buf.dtype == np.dtype('float64'):
            buf = buf.astype('float32')
        self.cl_buf = to_device(self.queue, buf)
        self.queue.finish()

    # def shallow_copy(self):
    #     rval = self.__class__.__new__(self.__class__)
    #     rval.cl_starts = self.cl_starts
    #     rval.cl_lens = self.cl_lens
    #     rval.cl_buf = self.cl_buf
    #     rval.queue = self.queue
    #     return rval

    def __len__(self):
        return self.cl_starts.shape[0]

    def __getitem__(self, item):
        """
        Getting one item returns a numpy array (on the host).
        Getting multiple items returns a view into the device.
        """

        # -- these are each OCL fetch operations (could be sped up)
        starts = self.starts
        shape0s = self.shape0s
        shape1s = self.shape1s
        stride0s = self.stride0s
        stride1s = self.stride1s

        if isinstance(item, (list, tuple)):
            items = item
            del item

            rval = self.__class__.__new__(self.__class__)
            rval.queue = self.queue
            rval.starts = [starts[i] for i in items]
            rval.shape0s = [shape0s[i] for i in items]
            rval.shape1s = [shape1s[i] for i in items]
            rval.stride0s = [stride0s[i] for i in items]
            rval.stride1s = [stride1s[i] for i in items]
            rval.cl_buf = self.cl_buf
            rval.names = [self.names[i] for i in items]
            return rval
        else:
            buf = to_host(
                self.queue, self.cl_buf.data, self.dtype, self.starts[item],
                (self.shape0s[item], self.shape1s[item]),
                (self.stride0s[item], self.stride1s[item]),
                )
            buf.setflags(write=False)
            return buf

    def __setitem__(self, item, new_value):
        starts = self.starts
        shape0s = self.shape0s
        shape1s = self.shape1s
        stride0s = self.stride0s
        stride1s = self.stride1s
        if isinstance(item, (list, tuple)):
            raise NotImplementedError('TODO')
        else:
            m, n = shape0s[item], shape1s[item]
            sM, sN = stride0s[item], stride1s[item]

            if sM < 0 or sN < 0:
                raise NotImplementedError()
            if not (sM, sN) in [(1, m), (n, 1)]:
                raise NotImplementedError('discontiguous setitem')

            itemsize = self.dtype.itemsize
            bytestart = itemsize * starts[item]
            # -- N.B. match to getitem
            byteend = bytestart + itemsize * ((m-1) * sM + (n-1) * sN + 1)

            temp_buf = np.zeros((byteend - bytestart), dtype=np.int8)
            # -- TODO: if copying into a contiguous region, this
            #          first copy from the device is unnecessary
            cl.enqueue_copy(self.queue, temp_buf, self.cl_buf.data,
                            device_offset=bytestart, is_blocking=True)

            bytestrides = (itemsize * sM, itemsize * sN)
            view = np.ndarray(
                shape=(m, n),
                dtype=self.dtype,
                buffer=temp_buf.data,
                offset=0,
                strides=bytestrides)
            view[...] = new_value
            # print temp_buf.view('float32')
            cl.enqueue_copy(self.queue, self.cl_buf.data, temp_buf,
                            device_offset=bytestart, is_blocking=True)

    def to_host(self):
        """Copy the whole object to a host RaggedArray"""
        rval = RaggedArray.__new__(RaggedArray)
        rval.starts = self.starts
        rval.shape0s = self.shape0s
        rval.shape1s = self.shape1s
        rval.stride0s = self.stride0s
        rval.stride1s = self.stride1s
        rval.buf = self.buf
        rval.names = self.names[:]
        return rval
