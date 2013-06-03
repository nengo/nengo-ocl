import numpy as np
import pyopencl as cl
import pyopencl.array

# XXX: use cl.array.Array, which has strides, flags

class Array(cl.array.Array):
    def __init__(self, queue, shape, dtype, order='C', allocator=None,
                 data=None, strides=None, offset=0):
        try:
            np.dtype(dtype)
        except Exception, e:
            e.args = e.args + (dtype,)
            raise
        cl.array.Array.__init__(self, queue,
                                shape=tuple(map(int, shape)),
                                allocator=allocator,
                                dtype=np.dtype(dtype),
                                order=order,
                                data=data,
                                strides=tuple(map(int, strides)))
        if 0 in self.strides:
            print self.strides
        self.offset = offset
        assert self.data.size >= self.size * self.dtype.itemsize

    def __getitem__(self, item):
        shape = list(self.shape)
        strides = list(self.strides)
        offset = self.offset
        dims_to_del = []
        if isinstance(item, (list, tuple)):
            for ii, idx in enumerate(item):
                if isinstance(idx, int):
                    dims_to_del.append(ii)
                    offset += idx * strides[ii]
                elif isinstance(idx, slice):
                    start, stop, stride = idx.indices(shape[ii])
                    offset += start * strides[ii]
                    if stride != 1:
                        raise NotImplementedError()
                    shape[ii] = stop - start
            for dim in reversed(dims_to_del):
                shape.pop(dim)
                strides.pop(dim)
            return self.__class__(
                    self.queue, shape, self.dtype,
                    data=self.data,
                    strides=strides,
                    offset=offset)
        elif isinstance(item, (int, np.integer)):
            if len(self.shape) == 0:
                raise IndexError()
            if not (0 <= item < self.shape[0]):
                raise NotImplementedError()
            shape = self.shape[1:]
            strides = self.strides[1:]
            offset = self.offset + item * self.strides[0]
            return self.__class__(
                    self.queue, shape, self.dtype,
                    data=self.data,
                    strides=strides,
                    offset=offset)
        else:
            raise NotImplementedError(item)

    def __int__(self):
        if self.size == 1:
            return int(self.get())
        else:
            raise TypeError()

    def __float__(self):
        if self.size == 1:
            return float(self.get())
        else:
            raise TypeError()

    def same_view_as(self, other):
        return ( self.data is other.offset
                and self.offset == other.offset
                and self.strides == other.strides
                and self.shape == other.shape
                and self.dtype == other.dtype)

    @property
    def itemstrides(self):
        rval = []
        for si in self.strides:
            assert 0 == si % self.dtype.itemsize
            rval.append(si // self.dtype.itemsize)
        return tuple(rval)

    def empty_like(self):
        # XXX consider whether to allocate just enough for data
        # or keep using the full buffer.
        data = cl.Buffer(self.data.context,
            flags=cl.mem_flags.READ_WRITE,
            size=self.data.size)
        return Array(self.queue,
                     shape=self.shape,
                     dtype=self.dtype,
                     data=data,
                     strides=self.strides,
                     offset=self.offset,
                    )

    def transpose(self, *new_order):
        assert set(new_order) == set(range(self.ndim))
        new_shape = [self.shape[i] for i in new_order]
        new_strides = [self.strides[i] for i in new_order]
        return Array(self.queue,
                     shape=new_shape,
                     dtype=self.dtype,
                     data=self.data,
                     strides=new_strides,
                     offset=self.offset,
                    )


    # TODO: call this ctype, use cl.tools.dtype_to_ctype
    @property
    def ocldtype(self):
        return ocldtype(self.dtype)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def structure(self):
        return (self.dtype, self.shape, self.strides, self.offset)

    def get(self, queue=None):
        if queue is None:
            queue = self.queue
        hostbuf = np.empty(shape=(self.data.size,), dtype='int8')
        #print self.data.size
        #print hostbuf.shape
        assert self.data.size >= self.size * self.dtype.itemsize
        #print self.structure, hostbuf.size
        cl.enqueue_copy(queue, hostbuf, self.data)
        try:
            rval = np.ndarray(buffer=hostbuf, strides=self.strides,
                          offset=self.offset, shape=self.shape, dtype=self.dtype)
        except:
            import pdb; pdb.set_trace()
        return rval


def ocldtype(obj):
    obj = str(obj)
    if isinstance(obj, basestring):
        return {
            'float32': 'float',
            'float64': 'double',
            'int64': 'long',
            'int32': 'int',
        }[obj]
    else:
        raise NotImplementedError('ocldtype', obj)

# consider array.to_device
def to_device(queue, arr, flags=cl.mem_flags.READ_WRITE):
    arr = np.asarray(arr)
    buf = cl.Buffer(queue.context, flags, size=len(arr.data))
    cl.enqueue_copy(queue, buf, arr.data).wait()
    rval = Array(queue,
                 data=buf,
                 dtype=arr.dtype,
                 shape=arr.shape,
                 strides=arr.strides)
    debugval = rval.get(queue)
    assert np.all(arr == debugval)
    return rval


# consider array.empty
def empty(queue, shape, dtype, flags=cl.mem_flags.READ_WRITE,
        strides=None, order='C'):

    dtype = np.dtype(dtype)
    
    if strides is None:
        strides = [int(dtype.itemsize)]
        if order.upper() == 'C':
            for shp in reversed(shape):
                strides = [strides[0] * int(shp)] + strides
            bufsize = strides[0]
            strides = strides[1:]
        elif order.upper() == 'F':
            raise NotImplementedError()
        else:
            raise ValueError(order)
    try:
        buf = cl.Buffer(queue.context, flags, size=bufsize)
    except:
        print queue, flags, type(flags), bufsize, type(bufsize)
        raise
    return Array(queue, data=buf, dtype=dtype, shape=shape, strides=strides)


def zeros(queue, *args, **kwargs):
    rval = empty(queue, *args, **kwargs)
    rval.fill(0, queue)
    queue.flush()
    return rval

