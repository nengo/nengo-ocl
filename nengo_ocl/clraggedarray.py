"""
OpenCL-based implementation of RaggedArray data structure.
"""

# pylint: disable=missing-function-docstring

from io import StringIO

import numpy as np
import pyopencl as cl
from nengo.utils.numpy import is_iterable
from pyopencl.array import Array, to_device

from nengo_ocl.raggedarray import RaggedArray
from nengo_ocl.utils import equal_strides

# add 'ctype' property to Array (returned by 'to_device')
Array.ctype = property(lambda self: cl.tools.dtype_to_ctype(self.dtype))
Array.start = property(lambda self: self.offset / self.dtype.itemsize)


def data_ptr(array):
    """Given an Array, get a Buffer that starts at the right offset

    This fails unless ``array.offset`` is a multiple of
    ``queue.device.mem_base_addr_align``, which is rare, so this isn't really
    a good function.
    """
    if array.offset:
        # ignore buffer size, since we don't use it
        return array.base_data.get_sub_region(array.offset, 1)
    else:
        return array.base_data


def to_host(queue, data, dtype, start, shape, elemstrides, is_blocking=True):
    """Copy memory off the device, into a Numpy array.

    If the requested array is discontiguous, the whole block is copied off
    the device, and a view is created to show the appropriate part.
    """
    if min(elemstrides) < 0:
        raise NotImplementedError()

    m, n = shape
    sm, sn = elemstrides
    if m * n == 0:
        return np.zeros(shape, dtype=dtype)

    itemsize = dtype.itemsize
    bytestart = itemsize * start
    bytelen = itemsize * ((m - 1) * sm + (n - 1) * sn + 1)

    temp_buf = np.zeros(bytelen, dtype=np.int8)
    cl.enqueue_copy(
        queue, temp_buf, data, device_offset=bytestart, is_blocking=is_blocking
    )

    bytestrides = (itemsize * sm, itemsize * sn)
    return np.ndarray(
        shape=(m, n), dtype=dtype, buffer=temp_buf.data, offset=0, strides=bytestrides
    )


class CLRaggedArray:
    """A linear device buffer partitioned into sections of various lengths.

    Can also be viewed as an efficient way of storing a list of arrays on
    the device, in the same underlying buffer.
    """

    def __init__(self, queue, np_raggedarray):
        self.queue = queue
        self.starts = np_raggedarray.starts
        self.shape0s = np_raggedarray.shape0s
        self.shape1s = np_raggedarray.shape1s
        self.stride0s = np_raggedarray.stride0s
        self.stride1s = np_raggedarray.stride1s
        self.buf = np_raggedarray.buf
        self.names = np_raggedarray.names

    @classmethod
    def from_arrays(cls, queue, arrays, names=None, dtype=None, align=False):
        return cls(queue, RaggedArray(arrays, names=names, dtype=dtype, align=align))

    @classmethod
    def from_buffer(
        cls, queue, cl_buf, starts, shape0s, shape1s, stride0s, stride1s, names=None
    ):
        rval = cls.__new__(cls)
        rval.queue = queue
        rval.starts = starts
        rval.shape0s = shape0s
        rval.shape1s = shape1s
        rval.stride0s = stride0s
        rval.stride1s = stride1s
        rval.cl_buf = cl_buf
        rval.names = names
        return rval

    @property
    def ctype(self):
        return self.cl_buf.ctype

    @property
    def dtype(self):
        return self.cl_buf.dtype

    @property
    def names(self):
        return self._names

    @names.setter
    def names(self, names):
        if names is None:
            names = [""] * len(self.starts)
        self._names = tuple(names)

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
        self.cl_starts = to_device(self.queue, self._starts)
        self.queue.finish()

    @property
    def shape0s(self):
        return self._shape0s

    @shape0s.setter
    def shape0s(self, shape0s):
        self._shape0s = np.array(shape0s, dtype="int32")
        self._shape0s.setflags(write=False)
        self.cl_shape0s = to_device(self.queue, self._shape0s)
        self.queue.finish()
        self._sizes = None

    @property
    def shape1s(self):
        return self._shape1s

    @shape1s.setter
    def shape1s(self, shape1s):
        self._shape1s = np.array(shape1s, dtype="int32")
        self._shape1s.setflags(write=False)
        self.cl_shape1s = to_device(self.queue, self._shape1s)
        self.queue.finish()
        self._sizes = None

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
        self.cl_stride0s = to_device(self.queue, self._stride0s)
        self.queue.finish()

    @property
    def stride1s(self):
        return self._stride1s

    @stride1s.setter
    def stride1s(self, stride1s):
        self._stride1s = np.array(stride1s, dtype="int32")
        self._stride1s.setflags(write=False)
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
        assert buf.dtype in [np.float32, np.int32]
        self.cl_buf = to_device(self.queue, buf)
        self.queue.finish()

    def __str__(self):
        sio = StringIO()
        namelen = max([0] + [len(n) for n in self.names])
        fmt = "%%%is" % namelen
        for ii, nn in enumerate(self.names):
            print("->", self[ii])
            print((fmt % nn), self[ii], file=sio)
        return sio.getvalue()

    def __len__(self):
        return self.cl_starts.shape[0]

    def __getitem__(self, item):
        """
        Getting one item returns a numpy array (on the host).
        Getting multiple items returns a view into the device.
        """
        if is_iterable(item):
            return self.getitem_device(item)
        else:
            buf = to_host(
                self.queue,
                self.cl_buf.data,
                self.dtype,
                self.starts[item],
                (self.shape0s[item], self.shape1s[item]),
                (self.stride0s[item], self.stride1s[item]),
            )
            buf.setflags(write=False)
            return buf

    def getitem_device(self, item):
        if isinstance(item, slice):
            item = np.arange(len(self))[item]

        if is_iterable(item):
            return CLRaggedArray.from_buffer(
                self.queue,
                self.cl_buf,
                self.starts[item],
                self.shape0s[item],
                self.shape1s[item],
                self.stride0s[item],
                self.stride1s[item],
                names=[self.names[i] for i in item],
            )
        else:
            s = self.dtype.itemsize
            return Array(
                self.queue,
                (self.shape0s[item], self.shape1s[item]),
                self.dtype,
                strides=(self.stride0s[item] * s, self.stride1s[item] * s),
                data=self.cl_buf.data,
                offset=self.starts[item] * s,
            )

    def __setitem__(self, item, new_value):
        if isinstance(item, slice) or is_iterable(item):
            raise NotImplementedError("TODO")
        else:
            m, n = self.shape0s[item], self.shape1s[item]
            sm, sn = self.stride0s[item], self.stride1s[item]

            if (sm, sn) in [(1, m), (n, 1)]:
                # contiguous
                clarray = self.getitem_device(item)
                if isinstance(new_value, np.ndarray):
                    array = np.asarray(new_value, order="C", dtype=self.dtype)
                else:
                    array = np.zeros(clarray.shape, dtype=clarray.dtype)
                    array[...] = new_value

                array.shape = clarray.shape  # reshape to avoid warning
                assert equal_strides(array.strides, clarray.strides, clarray.shape)
                clarray.set(array)
            else:
                # discontiguous
                #   Copy a contiguous region off the device that surrounds the
                #   discontiguous, set the appropriate values, and copy back
                s = self.starts[item]
                array = to_host(
                    self.queue,
                    self.cl_buf.data,
                    self.dtype,
                    s,
                    (m, n),
                    (sm, sn),
                    is_blocking=True,
                )
                array[...] = new_value

                buf = array.base if array.base is not None else array
                bytestart = self.dtype.itemsize * s
                cl.enqueue_copy(
                    self.queue,
                    self.cl_buf.data,
                    buf,
                    device_offset=bytestart,
                    is_blocking=True,
                )

    def to_host(self):
        """Copy the whole object to a host RaggedArray"""
        return RaggedArray.from_buffer(
            self.buf,
            self.starts,
            self.shape0s,
            self.shape1s,
            self.stride0s,
            self.stride1s,
            names=self.names,
        )
