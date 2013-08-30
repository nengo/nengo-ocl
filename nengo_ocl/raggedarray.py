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
        ldas = []
        buf = []

        for l in listofarrays:
            obj = np.asarray(l)
            starts.append(len(buf))
            shape0s.append(shape0(obj))
            shape1s.append(shape1(obj))
            if obj.ndim == 0:
                ldas.append(0)
            elif obj.ndim == 1:
                ldas.append(obj.shape[0])
            elif obj.ndim == 2:
                # -- N.B. the original indexing was
                #    based on ROW-MAJOR storage, and
                #    this simulator uses COL-MAJOR storage
                ldas.append(obj.shape[0])
            else:
                raise NotImplementedError()
            buf.extend(obj.ravel('F'))

        self.starts = starts
        self.shape0s = shape0s
        self.shape1s = shape1s
        self.ldas = ldas
        self.buf = np.asarray(buf)
        if names is None:
            self.names = [''] * len(self)
        else:
            assert len(names) == len(ldas)
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
        rval.ldas = self.ldas
        rval.buf = self.buf
        rval.names = self.names
        return rval

    def add_views(self, starts, shape0s, shape1s, ldas, names=None):
        #assert start >= 0
        #assert start + length <= len(self.buf)
        # -- creates copies, same semantics
        #    as OCL version
        self.starts = self.starts + starts
        self.shape0s = self.shape0s + shape0s
        self.shape1s = self.shape1s + shape1s
        self.ldas = self.ldas + ldas
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
            rval.ldas = [self.ldas[i] for i in item]
            rval.buf = self.buf
            rval.names = [self.names[i] for i in item]
            return rval
        else:
            itemsize = self.dtype.itemsize
            byteoffset = itemsize * self.starts[item]
            bytestrides = (itemsize, itemsize * self.ldas[item])
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


def raw_ragged_gather_gemv(BB,
        Ns, alphas,
        A_starts, A_data,
        A_js_starts,
        A_js_lens,
        A_js_data,
        X_starts,
        X_data,
        X_js_starts,
        X_js_data,
        betas,
        Y_in_starts,
        Y_in_data,
        Y_starts,
        Y_lens,
        Y_data):
    for bb in xrange(BB):
        alpha = alphas[bb]
        beta = betas[bb]
        n_dot_products = A_js_lens[bb]
        y_offset = Y_starts[bb]
        y_in_offset = Y_in_starts[bb]
        M = Y_lens[bb]
        for mm in xrange(M):
            Y_data[y_offset + mm] = beta * Y_in_data[y_in_offset + mm]

        for ii in xrange(n_dot_products):
            x_i = X_js_data[X_js_starts[bb] + ii]
            a_i = A_js_data[A_js_starts[bb] + ii]
            N_i = Ns[a_i]
            x_offset = X_starts[x_i]
            a_offset = A_starts[a_i]
            for mm in xrange(M):
                y_sum = 0.0
                for nn in xrange(N_i):
                    y_sum += X_data[x_offset + nn] * A_data[a_offset + nn * M + mm]
                Y_data[y_offset + mm] += alpha * y_sum


def ragged_gather_gemv(Ms, Ns, alpha, A, A_js, X, X_js,
                       beta, Y, Y_in=None,
                       use_raw_fn=False,
                      ):
    """
    """
    del Ms
    try:
        float(alpha)
        alpha = [alpha] * len(Y)
    except TypeError:
        pass

    try:
        float(beta)
        beta = [beta] * len(Y)
    except TypeError:
        pass

    if Y_in is None:
        Y_in = Y

    if use_raw_fn:
        # This is close to the OpenCL reference impl
        return raw_ragged_gather_gemv(
            len(Y),
            Ns,
            alpha,
            A.starts,
            A.buf,
            A_js.starts,
            A_js.shape0s,
            A_js.buf,
            X.starts,
            X.buf,
            X_js.starts,
            X_js.buf,
            beta,
            Y_in.starts,
            Y_in.buf,
            Y.starts,
            Y.shape0s,
            Y.buf)
    else:
        # -- less-close to the OpenCL impl
        #print alpha
        #print A.buf, 'A'
        #print X.buf, 'X'
        #print Y_in.buf, 'in'

        for i in xrange(len(Y)):
            try:
                y_i = beta[i] * Y_in[i]  # -- ragged getitem
            except:
                print i, beta, Y_in
                raise
            alpha_i = alpha[i]

            x_js_i = X_js[i] # -- ragged getitem
            A_js_i = A_js[i] # -- ragged getitem
            assert len(x_js_i) == len(A_js_i)
            for xi, ai in zip(x_js_i, A_js_i):
                x_ij = X[xi]  # -- ragged getitem
                A_ij = A[ai]  # -- ragged getitem
                try:
                    y_i += alpha_i * np.dot(A_ij, x_ij)
                except:
                    print i, xi, ai, A_ij, x_ij
                    print y_i.shape, x_ij.shape, A_ij.shape
                    raise
            Y[i] = y_i

