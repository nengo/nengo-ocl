from __future__ import division

import numpy as np
import pyopencl as cl
from mako.template import Template
import nengo.dists as nengod
from nengo.utils.compat import itervalues, range

from nengo_ocl.raggedarray import RaggedArray
from nengo_ocl.clraggedarray import CLRaggedArray, to_device
from nengo_ocl.plan import Plan
from nengo_ocl.utils import as_ascii, indent, round_up


def blockify_matrices(max_size, ras):
    # NOTE: must be contiguous
    ras = list(ras)
    ra0 = ras[0]
    N = len(ra0)
    for ra in ras:
        assert len(ra) == N
        assert np.all(ra.shape1s == ra0.shape1s)
        assert np.all(ra.shape0s == ra0.shape0s)
        assert np.all(ra.shape1s == ra.stride0s), "not contiguous"

    sizes = []
    inds = []
    starts = [[] for _ in ras]
    for i in range(N):
        size = ra0.sizes[i]
        startsi = [ra.starts[i] for ra in ras]
        while size > 0:
            sizes.append(min(size, max_size))
            size -= max_size
            inds.append(i)
            for k, ra in enumerate(ras):
                starts[k].append(startsi[k])
                startsi[k] += max_size

    return (np.array(sizes, dtype=np.int32),
            np.array(inds, dtype=np.int32),
            np.array(starts, dtype=np.int32))


def blockify_matrix(max_size, ra):
    sizes, inds, starts = blockify_matrices(max_size, [ra])
    return sizes, inds, starts[0]


def blockify_vectors(max_size, ras):
    ras = list(ras)
    ra0 = ras[0]
    N = len(ra0)
    for ra in ras:
        assert len(ra) == N
        assert np.all(ra.shape1s == 1)
        assert np.all(ra.shape0s == ra0.shape0s)

    sizes = []
    inds = []
    starts = [[] for _ in ras]
    for i in range(N):
        size = ra0.shape0s[i]
        startsi = [ra.starts[i] for ra in ras]
        while size > 0:
            sizes.append(min(size, max_size))
            size -= max_size
            inds.append(i)
            for k, ra in enumerate(ras):
                starts[k].append(startsi[k])
                startsi[k] += max_size * ra.stride0s[i]

    return (np.array(sizes, dtype=np.int32),
            np.array(inds, dtype=np.int32),
            np.array(starts, dtype=np.int32))


def blockify_vector(max_size, ra):
    sizes, inds, starts = blockify_vectors(max_size, [ra])
    return sizes, inds, starts[0]


def plan_timeupdate(queue, step, time, dt):
    assert len(step) == len(time) == 1
    assert step.ctype == time.ctype == 'float'
    assert step.shape0s[0] == step.shape1s[0] == 1
    assert time.shape0s[0] == time.shape1s[0] == 1

    text = """
        ////////// MAIN FUNCTION //////////
        __kernel void timeupdate(
            __global const int *step_starts,
            __global float *step_data,
            __global const int *time_starts,
            __global float *time_data
        )
        {
            __global float *step = step_data + step_starts[0];
            __global float *time = time_data + time_starts[0];
            step[0] += 1;
            time[0] = ${dt} * step[0];
        }
        """

    text = as_ascii(Template(text, output_encoding='ascii').render(dt=dt))
    full_args = (step.cl_starts, step.cl_buf, time.cl_starts, time.cl_buf)
    _fn = cl.Program(queue.context, text).build().timeupdate
    _fn.set_args(*[arr.data for arr in full_args])

    gsize = (1,)
    lsize = None
    rval = Plan(queue, _fn, gsize, lsize=lsize, name="cl_timeupdate")
    rval.full_args = full_args     # prevent garbage-collection
    return rval


def plan_reset(queue, Y, values, tag=None):
    assert len(Y) == len(values)

    assert np.all(Y.stride0s == Y.shape1s)
    assert np.all(Y.stride1s == 1)
    assert Y.ctype == values.ctype

    text = """
        ////////// MAIN FUNCTION //////////
        __kernel void reset(
            __global const ${Ytype} *values,
            __global const int *Ysizes,
            __global const int *Ystarts,
            __global ${Ytype} *Ydata
        )
        {
            const int i = get_global_id(0);
            int n = get_global_id(1);

    % for k in range(n_per_item):
            if (n < ${N} && i < Ysizes[n])
                (Ydata + Ystarts[n])[i] = values[n];
            n += get_global_size(1);
    % endfor
        }
        """

    max_group = min(queue.device.max_work_group_size, 256)
    n_per_item = 1
    local_i = 16
    local_j = max_group // local_i
    # local_i = min(256, Y.sizes.max())

    Ysizes, Yinds, Ystarts = blockify_matrix(local_i, Y)
    clYsizes = to_device(queue, Ysizes)
    clYstarts = to_device(queue, Ystarts)
    values = values.get()
    clvalues = to_device(queue, values[Yinds])

    N = len(Ysizes)
    NN = -(-N // n_per_item)  # ceiling division
    lsize = (local_i, local_j)
    gsize = (local_i, round_up(NN, local_j))
    # lsize = None
    # gsize = (local_i, NN)

    textconf = dict(Ytype=Y.ctype, N=N, n_per_item=n_per_item)
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))

    full_args = (
        clvalues,
        clYsizes,
        clYstarts,
        Y.cl_buf,
    )
    _fn = cl.Program(queue.context, text).build().reset
    _fn.set_args(*[arr.data for arr in full_args])

    rval = Plan(queue, _fn, gsize, lsize=lsize, name="cl_reset", tag=tag)
    rval.full_args = full_args     # prevent garbage-collection
    rval.bw_per_call = (
        Y.nbytes + values.nbytes + clYsizes.nbytes + clYstarts.nbytes)
    rval.description = (
        "groups: %d; items: %d; items/group: %0.1f [%d, %d]" %
        (len(Y), Y.sizes.sum(), Y.sizes.mean(), Y.sizes.min(), Y.sizes.max()))
    return rval


def plan_copy(queue, A, B, incs, tag=None):
    N = len(A)
    assert len(A) == len(B)

    for arr in [A, B]:
        assert (arr.shape1s == 1).all()
    assert (A.shape0s == B.shape0s).all()

    assert A.ctype == B.ctype

    text = """
        ////////// MAIN FUNCTION //////////
        __kernel void copy(
    % if inc is None:
            __global const int *incdata,
    % endif
            __global const int *Astrides,
            __global const int *Astarts,
            __global const ${Atype} *Adata,
            __global const int *Bstrides,
            __global const int *Bstarts,
            __global ${Btype} *Bdata,
            __global const int *sizes
        )
        {
            const int i = get_global_id(0);
            const int n = get_global_id(1);
            if (n >= ${N} || i >= sizes[n])
                return;

            __global const ${Atype} *a = Adata + Astarts[n];
            __global ${Btype} *b = Bdata + Bstarts[n];

    % if inc is True:
            b[i*Bstrides[n]] += a[i*Astrides[n]];
    % elif inc is False:
            b[i*Bstrides[n]] = a[i*Astrides[n]];
    % else:
            if (incdata[n])  b[i*Bstrides[n]] += a[i*Astrides[n]];
            else             b[i*Bstrides[n]] = a[i*Astrides[n]];
    % endif
        }
        """

    max_group = min(queue.device.max_work_group_size, 256)
    # n_per_item = 1
    local_i = 16
    local_j = max_group // local_i
    # local_i = min(256, A.sizes.max())

    sizes, inds, [Astarts, Bstarts] = blockify_vectors(local_i, [A, B])

    N = len(sizes)
    # NN = -(-N // n_per_item)  # ceiling division
    lsize = (local_i, local_j)
    gsize = (local_i, round_up(N, local_j))
    # lsize = None
    # gsize = (local_i, N)

    textconf = dict(Atype=A.ctype, Btype=B.ctype, N=N, inc=None)

    full_args = [
        to_device(queue, A.stride0s[inds]),
        to_device(queue, Astarts),
        A.cl_buf,
        to_device(queue, B.stride0s[inds]),
        to_device(queue, Bstarts),
        B.cl_buf,
        to_device(queue, sizes),
    ]
    if (incs == 0).all():
        textconf['inc'] = False
    elif (incs == 1).all():
        textconf['inc'] = True
    else:
        full_args.insert(0, to_device(queue, incs[inds].astype(np.int32)))

    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))
    _fn = cl.Program(queue.context, text).build().copy
    _fn.set_args(*[arr.data for arr in full_args])

    rval = Plan(queue, _fn, gsize, lsize=lsize, name="cl_copy", tag=tag)
    rval.full_args = tuple(full_args)  # prevent garbage-collection
    rval.bw_per_call = A.nbytes + B.nbytes
    rval.description = (
        "groups: %d; items: %d; items/group: %0.1f [%d, %d]" %
        (len(A), A.sizes.sum(), A.sizes.mean(), A.sizes.min(), A.sizes.max()))
    return rval


def plan_slicedcopy(queue, A, B, Ainds, Binds, incs, tag=None):
    N = len(A)
    assert len(A) == len(B) == len(Ainds) == len(Binds)

    for arr in [A, B, Ainds, Binds]:
        assert (arr.shape1s == 1).all()
        assert (arr.stride1s == 1).all()
    for arr in [Ainds, Binds]:
        assert (arr.stride0s == 1).all()
    assert (Ainds.shape0s == Binds.shape0s).all()

    assert A.ctype == B.ctype
    assert Ainds.ctype == Binds.ctype == 'int'

    text = """
        ////////// MAIN FUNCTION //////////
        __kernel void slicedcopy(
    % if inc is None:
            __global const int *incdata,
    % endif
            __global const int *Astride0s,
            __global const int *Astarts,
            __global const ${Atype} *Adata,
            __global const int *Bstride0s,
            __global const int *Bstarts,
            __global ${Btype} *Bdata,
            __global const int *Isizes,
            __global const int *AIstarts,
            __global const int *AIdata,
            __global const int *BIstarts,
            __global const int *BIdata
        )
        {
            const int i = get_global_id(0);
            const int n = get_global_id(1);
            if (n >= ${N})
                return;

            __global const ${Atype} *a = Adata + Astarts[n];
            __global ${Btype} *b = Bdata + Bstarts[n];
            __global const int *aind = AIdata + AIstarts[n];
            __global const int *bind = BIdata + BIstarts[n];
            const int Astride0 = Astride0s[n], Bstride0 = Bstride0s[n];

            if (i < Isizes[n]) {
    % if inc is True:
                b[bind[i]*Bstride0] += a[aind[i]*Astride0];
    % elif inc is False:
                b[bind[i]*Bstride0] = a[aind[i]*Astride0];
    % else:
                if (incdata[n])
                    b[bind[i]*Bstride0] += a[aind[i]*Astride0];
                else
                    b[bind[i]*Bstride0] = a[aind[i]*Astride0];
    % endif
            }
        }
        """

    max_group = min(queue.device.max_work_group_size, 256)
    # n_per_item = 1
    local_i = 16
    local_j = max_group // local_i

    sizes, inds, [AIstarts, BIstarts] = blockify_vectors(
        local_i, [Ainds, Binds])

    N = len(sizes)
    # NN = -(-N // n_per_item)  # ceiling division
    lsize = (local_i, local_j)
    gsize = (local_i, round_up(N, local_j))

    textconf = dict(Atype=A.ctype, Btype=B.ctype, N=N, inc=None)

    full_args = [
        to_device(queue, A.stride0s[inds]),
        to_device(queue, A.starts[inds]),
        A.cl_buf,
        to_device(queue, B.stride0s[inds]),
        to_device(queue, B.starts[inds]),
        B.cl_buf,
        to_device(queue, sizes),
        to_device(queue, AIstarts),
        Ainds.cl_buf,
        to_device(queue, BIstarts),
        Binds.cl_buf,
    ]
    if (incs == 0).all():
        textconf['inc'] = False
    elif (incs == 1).all():
        textconf['inc'] = True
    else:
        full_args.insert(0, to_device(queue, incs[inds].astype(np.int32)))

    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))
    _fn = cl.Program(queue.context, text).build().slicedcopy
    _fn.set_args(*[arr.data for arr in full_args])

    rval = Plan(queue, _fn, gsize, lsize=lsize, name="cl_slicedcopy", tag=tag)
    rval.full_args = tuple(full_args)  # prevent garbage-collection
    rval.bw_per_call = 2 * (Ainds.nbytes + Ainds.sizes.sum()*A.dtype.itemsize)
    rval.description = (
        "groups: %d; items: %d; items/group: %0.1f [%d, %d]" %
        (len(Ainds), Ainds.sizes.sum(),
         Ainds.sizes.mean(), Ainds.sizes.min(), Ainds.sizes.max()))
    return rval


def plan_elementwise_inc(queue, A, X, Y, tag=None):
    """Implements an element-wise increment Y += A * X"""
    N = len(X)
    assert len(Y) == N and len(A) == N

    for arr in [A, X, Y]:
        assert (arr.stride1s == 1).all()
    assert ((X.shape0s == 1) | (X.shape0s == Y.shape0s)).all()
    assert ((X.shape1s == 1) | (X.shape1s == Y.shape1s)).all()
    assert ((A.shape0s == 1) | (A.shape0s == Y.shape0s)).all()
    assert ((A.shape1s == 1) | (A.shape1s == Y.shape1s)).all()
    assert (X.stride1s == 1).all()
    assert (Y.stride1s == 1).all()
    assert (A.stride1s == 1).all()

    assert X.ctype == Y.ctype
    assert A.ctype == Y.ctype

    text = """
        inline ${Ytype} get_element(
            __global const ${Ytype} *data,
            const int shape0, const int shape1, const int stride0,
            const int i, const int j
        )
        {
            if (shape0 == 1 && shape1 == 1)
                return data[0];
            else if (shape0 == 1)
                return data[j];
            else if (shape1 == 1)
                return data[i * stride0];
            else
                return data[i * stride0 + j];
        }

        ////////// MAIN FUNCTION //////////
        __kernel void elementwise_inc(
            __global const int *Ashape0s,
            __global const int *Ashape1s,
            __global const int *Astride0s,
            __global const int *Astarts,
            __global const ${Atype} *Adata,
            __global const int *Xshape0s,
            __global const int *Xshape1s,
            __global const int *Xstride0s,
            __global const int *Xstarts,
            __global const ${Xtype} *Xdata,
            __global const int *Yshape0s,
            __global const int *Yshape1s,
            __global const int *Ystride0s,
            __global const int *Ystarts,
            __global ${Ytype} *Ydata
        )
        {
            const int n = get_global_id(1);
            __global const ${Atype} *a = Adata + Astarts[n];
            __global const ${Xtype} *x = Xdata + Xstarts[n];
            __global ${Ytype} *y = Ydata + Ystarts[n];

            const int Ashape0 = Ashape0s[n];
            const int Ashape1 = Ashape1s[n];
            const int Astride0 = Astride0s[n];
            const int Xshape0 = Xshape0s[n];
            const int Xshape1 = Xshape1s[n];
            const int Xstride0 = Xstride0s[n];
            const int Yshape1 = Yshape1s[n];
            const int Ystride0 = Ystride0s[n];
            const int Ysize = Yshape0s[n] * Yshape1;

            for (int ij = get_global_id(0);
                 ij < Ysize;
                 ij += get_global_size(0))
            {
                int i = ij / Yshape1;
                int j = ij % Yshape1;

                ${Atype} aa = get_element(a, Ashape0, Ashape1, Astride0, i, j);
                ${Xtype} xx = get_element(x, Xshape0, Xshape1, Xstride0, i, j);
                y[i * Ystride0 + j] += aa * xx;
            }
        }
        """

    textconf = dict(Atype=A.ctype, Xtype=X.ctype, Ytype=Y.ctype)
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))

    full_args = (
        A.cl_shape0s,
        A.cl_shape1s,
        A.cl_stride0s,
        A.cl_starts,
        A.cl_buf,
        X.cl_shape0s,
        X.cl_shape1s,
        X.cl_stride0s,
        X.cl_starts,
        X.cl_buf,
        Y.cl_shape0s,
        Y.cl_shape1s,
        Y.cl_stride0s,
        Y.cl_starts,
        Y.cl_buf,
    )
    _fn = cl.Program(queue.context, text).build().elementwise_inc
    _fn.set_args(*[arr.data for arr in full_args])

    max_group = queue.device.max_work_group_size
    mn = min(Y.sizes.max(), max_group)
    gsize = (mn, N)
    # lsize = (mn, 1)
    lsize = None
    rval = Plan(
        queue, _fn, gsize, lsize=lsize, name="cl_elementwise_inc", tag=tag)
    rval.full_args = full_args     # prevent garbage-collection
    rval.bw_per_call = A.nbytes + X.nbytes + Y.nbytes
    rval.description = (
        "groups: %d; items: %d; items/group: %0.1f [%d, %d]" %
        (len(Y), Y.sizes.sum(), Y.sizes.mean(), Y.sizes.min(), Y.sizes.max()))
    return rval


def plan_linear_synapse(queue, X, Y, A, B, Xbuf, Ybuf, tag=None):
    """
    Implements a filter of the form

        y[n+1] + a[0] y[n] + ... + a[i] y[n-i] = b[0] x[n] + ... + b[j] x[n-j]
    """
    N = len(X)
    assert len(Y) == N and len(A) == N and len(B) == N

    for arr in [X, Y, A, B, Xbuf, Ybuf]:
        assert (arr.shape1s == arr.stride0s).all()
        assert (arr.stride1s == 1).all()
    for arr in [X, Y, A, B]:  # vectors
        assert (arr.shape1s == 1).all()
    assert (X.shape0s == Y.shape0s).all()

    assert (B.shape0s >= 1).all()
    assert ((B.shape0s == 1) | (Xbuf.shape0s == B.shape0s)).all()
    assert (Xbuf.shape1s == X.shape0s).all()
    assert ((A.shape0s == 1) | (Ybuf.shape0s == A.shape0s)).all()
    assert (Ybuf.shape1s == Y.shape0s).all()

    assert X.ctype == Xbuf.ctype
    assert Y.ctype == Ybuf.ctype

    Xbufpos = to_device(queue, np.zeros(N, dtype='int32'))
    Ybufpos = to_device(queue, np.zeros(N, dtype='int32'))

    text = """
        ////////// MAIN FUNCTION //////////
        __kernel void linear_synapse(
            __global const int *shape0s,
            __global const int *Xstarts,
            __global const ${Xtype} *Xdata,
            __global const int *Ystarts,
            __global ${Ytype} *Ydata,
            __global const int *Ashape0s,
            __global const int *Astarts,
            __global const ${Atype} *Adata,
            __global const int *Bshape0s,
            __global const int *Bstarts,
            __global const ${Btype} *Bdata,
            __global const int *Xbufstarts,
            __global ${Xtype} *Xbufdata,
            __global const int *Ybufstarts,
            __global ${Ytype} *Ybufdata,
            __global int *Xbufpos,
            __global int *Ybufpos
        )
        {
            int i = get_global_id(0);
            const int k = get_global_id(1);
            __global const ${Xtype} *x = Xdata + Xstarts[k];
            __global ${Ytype} *y = Ydata + Ystarts[k];
            __global const ${Atype} *a = Adata + Astarts[k];
            __global const ${Btype} *b = Bdata + Bstarts[k];

            const int n = shape0s[k];
            const int na = Ashape0s[k];
            const int nb = Bshape0s[k];
            if (na == 0 && nb == 1) {
                for (; i < n; i += get_global_size(0))
                    y[i] = b[0] * x[i];
            } else if (na == 1 && nb == 1) {
                for (; i < n; i += get_global_size(0)) {
                    y[i] *= -a[0];
                    y[i] += b[0] * x[i];
                }
            } else {  // general filtering
                __global ${Xtype} *xbuf = Xbufdata + Xbufstarts[k];
                __global ${Ytype} *ybuf = Ybufdata + Ybufstarts[k];
                const int ix = Xbufpos[k];
                const int iy = Ybufpos[k];
                const int ix1 = (ix > 0) ? ix - 1 : nb - 1;
                const int iy1 = (iy > 0) ? iy - 1 : na - 1;

                ${Ytype} yi;
                int j, jj;
                for (; i < n; i += get_global_size(0)) {
                    yi = b[0] * x[i];
                    if (nb > 1) {
                        xbuf[ix*n + i] = x[i];  // copy input to buffer
                        for (j = 1; j < nb; j++) {
                            jj = (ix + j) % nb;
                            yi += b[j] * xbuf[jj*n + i];
                        }
                    }

                    if (na > 0) {
                        yi -= a[0] * y[i];
                        if (na > 1) {
                            for (j = 1; j < na; j++) {
                                jj = (iy + j) % na;
                                yi -= a[j] * ybuf[jj*n + i];
                            }
                            ybuf[iy1*n + i] = yi;  // copy output to buffer
                        }
                    }

                    y[i] = yi;
                }

                Xbufpos[k] = ix1;
                Ybufpos[k] = iy1;
            }
        }
        """

    textconf = dict(
        Xtype=X.ctype, Ytype=Y.ctype,
        Atype=A.ctype, Btype=B.ctype
    )
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))

    full_args = (
        X.cl_shape0s,
        X.cl_starts,
        X.cl_buf,
        Y.cl_starts,
        Y.cl_buf,
        A.cl_shape0s,
        A.cl_starts,
        A.cl_buf,
        B.cl_shape0s,
        B.cl_starts,
        B.cl_buf,
        Xbuf.cl_starts,
        Xbuf.cl_buf,
        Ybuf.cl_starts,
        Ybuf.cl_buf,
        Xbufpos,
        Ybufpos,
    )
    _fn = cl.Program(queue.context, text).build().linear_synapse
    _fn.set_args(*[arr.data for arr in full_args])

    max_len = min(max(X.shape0s), queue.device.max_work_group_size)
    gsize = (max_len, N)
    lsize = (max_len, 1)
    rval = Plan(
        queue, _fn, gsize, lsize=lsize, name="cl_linear_synapse", tag=tag)
    rval.full_args = full_args     # prevent garbage-collection
    rval.bw_per_call = (
        X.nbytes + Y.nbytes + A.nbytes + B.nbytes + Xbuf.nbytes + Ybuf.nbytes)
    rval.description = (
        "groups: %d; items: %d; items/group: %0.1f [%d, %d]" %
        (len(Y), Y.sizes.sum(), Y.sizes.mean(), Y.sizes.min(), Y.sizes.max()))
    return rval


def plan_probes(queue, periods, X, Y, tag=None):
    """
    Parameters
    ----------
    P : raggedarray of ints
        The period (in time-steps) of each probe
    """
    assert len(X) == len(Y)
    assert len(X) == len(periods)
    assert X.ctype == Y.ctype
    N = len(X)

    # N.B.  X[i].shape = (M, N)
    #       Y[i].shape = (buf_len, M * N)
    for arr in [X, Y]:
        assert (arr.stride1s == 1).all()
    assert (X.shape0s * X.shape1s == Y.shape1s).all()
    assert (X.stride0s == X.shape1s).all()
    assert (X.stride1s == 1).all()
    assert (Y.stride0s == Y.shape1s).all()
    assert (Y.stride1s == 1).all()

    periods = np.asarray(periods, dtype='float32')
    cl_periods = to_device(queue, periods)
    cl_countdowns = to_device(queue, periods - 1)
    cl_bufpositions = to_device(queue, np.zeros(N, dtype='int32'))

    text = """
        ////////// MAIN FUNCTION //////////
        __kernel void probes(
            __global ${Ctype} *countdowns,
            __global int *bufpositions,
            __global const ${Ptype} *periods,
            __global const int *Xstarts,
            __global const int *Xshape0s,
            __global const int *Xshape1s,
            __global const ${Xtype} *Xdata,
            __global const int *Ystarts,
            __global ${Ytype} *Ydata
        )
        {
            const int n = get_global_id(1);
            const ${Ctype} countdown = countdowns[n];

            if (countdown <= 0) {
                const int n_dims = Xshape0s[n] * Xshape1s[n];
                __global const ${Xtype} *x = Xdata + Xstarts[n];
                const int bufpos = bufpositions[n];

                __global ${Ytype} *y = Ydata + Ystarts[n] + bufpos * n_dims;

                for (int ii = get_global_id(0);
                         ii < n_dims;
                         ii += get_global_size(0))
                {
                    y[ii] = x[ii];
                }
                // This should *not* cause deadlock because
                // all local threads guaranteed to be
                // in this branch together.
                barrier(CLK_LOCAL_MEM_FENCE);
                if (get_global_id(0) == 0)
                {
                    countdowns[n] = countdown + periods[n] - 1;
                    bufpositions[n] = bufpos + 1;
                }
            }
            else
            {
                barrier(CLK_LOCAL_MEM_FENCE);
                if (get_global_id(0) == 0)
                {
                    countdowns[n] = countdown - 1;
                }
            }
        }
        """

    textconf = dict(N=N,
                    Xtype=X.ctype,
                    Ytype=Y.ctype,
                    Ctype=cl_countdowns.ctype,
                    Ptype=cl_periods.ctype)
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))

    full_args = (
        cl_countdowns,
        cl_bufpositions,
        cl_periods,
        X.cl_starts,
        X.cl_shape0s,
        X.cl_shape1s,
        X.cl_buf,
        Y.cl_starts,
        Y.cl_buf,
    )
    _fn = cl.Program(queue.context, text).build().probes
    _fn.set_args(*[arr.data for arr in full_args])

    max_len = min(queue.device.max_work_group_size, max(X.shape0s))
    gsize = (max_len, N,)
    lsize = (max_len, 1)
    rval = Plan(queue, _fn, gsize, lsize=lsize, name="cl_probes", tag=tag)
    rval.full_args = full_args     # prevent garbage-collection
    rval.cl_bufpositions = cl_bufpositions
    rval.Y = Y
    rval.bw_per_call = (2*X.nbytes + cl_periods.nbytes +
                        cl_countdowns.nbytes + cl_bufpositions.nbytes)
    rval.description = (
        "groups: %d; items: %d; items/group: %0.1f [%d, %d]" %
        (len(X), X.sizes.sum(), X.sizes.mean(), X.sizes.min(), X.sizes.max()))
    return rval


def plan_direct(queue, code, init, input_names, inputs, output, tag=None):
    from . import ast_conversion

    assert len(input_names) == len(inputs)

    N = len(inputs[0])
    for x in inputs:
        assert len(x) == len(output)
    for x in inputs + [output]:
        assert (x.shape1s == 1).all() and (x.stride1s == 1).all()
        assert (x.stride0s == 1).all()

    input_types = [x.ctype for x in inputs]
    output_type = output.ctype

    text = """
        ////////// MAIN FUNCTION //////////
        __kernel void direct(
% for iname, itype in zip(input_names, input_types):
            __global const int *${iname}_starts__,
            __global const ${itype} *${iname}_data__,
% endfor
            __global const int *${oname}_starts__,
            __global ${otype} *${oname}_data__
        )
        {
            const int n = get_global_id(0);
            if (n >= ${N}) return;

% for iname, itype in zip(input_names, input_types):
            __global const ${itype} *${iname} =
                ${iname}_data__ + ${iname}_starts__[n];
% endfor
            __global ${otype} *${oname} =
                ${oname}_data__ + ${oname}_starts__[n];

            /////vvvvv USER DECLARATIONS BELOW vvvvv
${init}

            /////vvvvv USER COMPUTATIONS BELOW vvvvv
${code}
            // END OF FUNC: put nothing after user code, since it can return
        }
        """

    textconf = dict(init=indent(init, 12),
                    code=indent(code, 12),
                    N=N, input_names=input_names, input_types=input_types,
                    oname=ast_conversion.OUTPUT_NAME, otype=output_type,
                    )
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))

    full_args = []
    for x in inputs:
        full_args.extend([x.cl_starts, x.cl_buf])
    full_args.extend([output.cl_starts, output.cl_buf])
    _fn = cl.Program(queue.context, text).build().direct
    _fn.set_args(*[arr.data for arr in full_args])

    gsize = (N,)
    rval = Plan(queue, _fn, gsize, lsize=None, name="cl_direct", tag=tag)
    rval.full_args = tuple(full_args)  # prevent garbage-collection
    rval.description = (
        "groups: %d; items: %d; items/group: %0.1f [%d, %d]" %
        (len(output), output.sizes.sum(),
         output.sizes.mean(), output.sizes.min(), output.sizes.max()))
    return rval


def plan_lif(queue, dt, J, V, W, outS, ref, tau, N=None, tau_n=None,
             inc_n=None, upsample=1, **kwargs):
    adaptive = N is not None
    assert J.ctype == 'float'
    for array in [V, W, outS]:
        assert V.ctype == J.ctype

    inputs = dict(J=J, V=V, W=W)
    outputs = dict(outV=V, outW=W, outS=outS)
    parameters = dict(tau=tau, ref=ref)
    if adaptive:
        assert all(ary is not None for ary in [N, tau_n, inc_n])
        assert N.ctype == J.ctype
        inputs.update(dict(N=N))
        outputs.update(dict(outN=N))
        parameters.update(dict(tau_n=tau_n, inc_n=inc_n))

    dt = float(dt)
    textconf = dict(type=J.ctype, dt=dt, upsample=upsample, adaptive=adaptive,
                    dtu=dt/upsample, dtu_inv=upsample/dt, dt_inv=1/dt)
    decs = """
        char spiked;
        ${type} dV, overshoot;
        const ${type} dt = ${dt}, dtu = ${dtu}, dtu_inv = ${dtu_inv},
                      dt_inv = ${dt_inv};
        const ${type} V_threshold = 1;
        """
    # TODO: could precompute -expm1(-dtu / tau)
    text = """
        spiked = 0;

% for ii in range(upsample):
% if adaptive:
        dV = -expm1(-dtu / tau) * (J - N - V);
% else:
        dV = -expm1(-dtu / tau) * (J - V);
% endif
        V += dV;
        W -= dtu;

        if (V < 0 || W > dtu)
            V = 0;
        else if (W >= 0)
            V *= 1 - W * dtu_inv;

        if (V > V_threshold) {
            overshoot = dtu * (V - V_threshold) / dV;
            W = ref - overshoot + dtu;
            V = 0;
            spiked = 1;
        }
% endfor
        outV = V;
        outW = W;
        outS = (spiked) ? dt_inv : 0;
% if adaptive:
        outN = N + (dt / tau_n) * (inc_n * outS - N);
% endif
        """
    decs = as_ascii(Template(decs, output_encoding='ascii').render(**textconf))
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))
    cl_name = "cl_alif" if adaptive else "cl_lif"
    return _plan_template(
        queue, cl_name, text, declares=decs,
        inputs=inputs, outputs=outputs, parameters=parameters, **kwargs)


def plan_lif_rate(queue, dt, J, R, ref, tau, N=None, tau_n=None, inc_n=None,
                  **kwargs):
    assert J.ctype == 'float'
    assert R.ctype == J.ctype
    adaptive = N is not None

    inputs = dict(J=J)
    outputs = dict(R=R)
    parameters = dict(tau=tau, ref=ref)
    textconf = dict(type=J.ctype, dt=dt, adaptive=adaptive)
    if adaptive:
        assert all(ary is not None for ary in [N, tau_n, inc_n])
        assert N.ctype == J.ctype
        inputs.update(dict(N=N))
        outputs.update(dict(outN=N))
        parameters.update(dict(tau_n=tau_n, inc_n=inc_n))

    decs = """
        const ${type} c0 = 0, c1 = 1;
        const ${type} dt = ${dt};
        """
    text = """
    % if adaptive:
        J = max(J - N - 1, c0);
    % else:
        J = max(J - 1, c0);
    % endif
        R = c1 / (ref + tau * log1p(c1/J));
    % if adaptive:
        outN = N + (dt / tau_n) * (inc_n*R - N);
    % endif
        """
    decs = as_ascii(Template(decs, output_encoding='ascii').render(**textconf))
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))
    cl_name = "cl_alif_rate" if adaptive else "cl_lif_rate"
    return _plan_template(
        queue, cl_name, text, declares=decs,
        inputs=inputs, outputs=outputs, parameters=parameters, **kwargs)


def _plan_template(queue, name, core_text, declares="", tag=None,
                   blockify=True, inputs={}, outputs={}, parameters={}):
    """Template for making a plan for vector nonlinearities.

    This template assumes that all inputs and outputs are vectors.

    Parameters
    ----------
    blockify : bool
        If true, divide the inputs up into blocks with a maximum size.

    inputs: dictionary of CLRaggedArrays
        Inputs to the function. RaggedArrays must be a list of vectors.

    outputs: dictionary of CLRaggedArrays
        Outputs of the function. RaggedArrays must be a list of vectors.

    parameters: dictionary of CLRaggedArrays
        Parameters to the function. Each RaggedArray element must be a vector
        of the same length of the inputs, or a scalar (to be broadcasted).
        Providing a float instead of a RaggedArray makes that parameter
        constant.

    """
    input0 = list(inputs.values())[0]   # input to use as reference for lengths

    # split parameters into static and updated params
    static_params = {}  # static params (hard-coded)
    params = {}  # variable params (updated)
    for k, v in parameters.items():
        if isinstance(v, CLRaggedArray):
            params[k] = v
        else:
            try:
                static_params[k] = ('float', float(v))
            except TypeError:
                raise

    avars = {}
    bw_per_call = 0
    for vname, v in (list(inputs.items()) + list(outputs.items()) +
                     list(params.items())):
        assert vname not in avars, "Name clash"
        assert len(v) == len(input0)
        assert (v.shape0s == input0.shape0s).all()
        assert (v.stride0s == v.shape1s).all()  # rows contiguous
        assert (v.stride1s == 1).all()  # columns contiguous
        assert (v.shape1s == 1).all()  # vectors only

        offset = '%(name)s_starts[gind1]' % {'name': vname}
        avars[vname] = (v.ctype, offset)
        bw_per_call += v.nbytes

    ivars = dict((k, avars[k]) for k in inputs.keys())
    ovars = dict((k, avars[k]) for k in outputs.keys())
    pvars = dict((k, avars[k]) for k in params.keys())

    fn_name = str(name)
    textconf = dict(fn_name=fn_name, declares=declares, core_text=core_text,
                    ivars=ivars, ovars=ovars, pvars=pvars,
                    static_params=static_params)

    text = """
    ////////// MAIN FUNCTION //////////
    __kernel void ${fn_name}(
% for name, [type, offset] in ivars.items():
        __global const int *${name}_starts,
        __global const ${type} *in_${name},
% endfor
% for name, [type, offset] in ovars.items():
        __global const int *${name}_starts,
        __global ${type} *in_${name},
% endfor
% for name, [type, offset] in pvars.items():
        __global const int *${name}_starts,
        __global const int *${name}_shape0s,
        __global const ${type} *in_${name},
% endfor
        __global const int *sizes
    )
    {
        const int gind0 = get_global_id(0);
        const int gind1 = get_global_id(1);
        if (gind1 >= ${N} || gind0 >= sizes[gind1])
            return;

% for name, [type, offset] in ivars.items():
        ${type} ${name} = in_${name}[${offset} + gind0];
% endfor
% for name, [type, offset] in ovars.items():
        ${type} ${name};
% endfor
% for name, [type, offset] in pvars.items():
        const ${type} ${name} = in_${name}[${offset} + gind0];
% endfor
% for name, [type, value] in static_params.items():
        const ${type} ${name} = ${value};
% endfor
        //////////////////////////////////////////////////
        //vvvvv USER DECLARATIONS BELOW vvvvv
        ${declares}
        //^^^^^ USER DECLARATIONS ABOVE ^^^^^
        //////////////////////////////////////////////////

        /////vvvvv USER COMPUTATIONS BELOW vvvvv
        ${core_text}
        /////^^^^^ USER COMPUTATIONS ABOVE ^^^^^

% for name, [type, offset] in ovars.items():
        in_${name}[${offset} + gind0] = ${name};
% endfor
    }
    """

    if blockify:
        # blockify to help with heterogeneous sizes

        # find best block size
        block_sizes = [32, 64, 128, 256, 512, 1024]
        N = np.inf
        for block_size_i in block_sizes:
            sizes_i, inds_i, _ = blockify_vector(block_size_i, input0)
            if len(sizes_i) < N:
                block_size = block_size_i
                sizes = sizes_i
                inds = inds_i

        clsizes = to_device(queue, sizes)
        get_starts = lambda ras: [to_device(queue, starts) for starts in
                                  blockify_vectors(block_size, ras)[2]]
        Istarts = get_starts(itervalues(inputs))
        Ostarts = get_starts(itervalues(outputs))
        Pstarts = get_starts(itervalues(params))
        Pshape0s = [
            to_device(queue, x.shape0s[inds]) for x in itervalues(params)]

        lsize = None
        gsize = (block_size, len(sizes))

        full_args = []
        for vstarts, v in zip(Istarts, itervalues(inputs)):
            full_args.extend([vstarts, v.cl_buf])
        for vstarts, v in zip(Ostarts, itervalues(outputs)):
            full_args.extend([vstarts, v.cl_buf])
        for vstarts, vshape0s, v in zip(Pstarts, Pshape0s, itervalues(params)):
            full_args.extend([vstarts, vshape0s, v.cl_buf])
        full_args.append(clsizes)
    else:
        # Allocate more than enough kernels in a matrix
        lsize = None
        gsize = (input0.shape0s.max(), len(input0))

        full_args = []
        for v in itervalues(inputs):
            full_args.extend([v.cl_starts, v.cl_buf])
        for v in itervalues(outputs):
            full_args.extend([v.cl_starts, v.cl_buf])
        for vname, v in params.items():
            full_args.extend([v.cl_starts, v.cl_shape0s, v.cl_buf])
        full_args.append(input0.cl_shape0s)

    textconf['N'] = gsize[1]
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))
    fns = cl.Program(queue.context, text).build()
    _fn = getattr(fns, fn_name)
    _fn.set_args(*[arr.data for arr in full_args])

    rval = Plan(queue, _fn, gsize, lsize=lsize, name=name, tag=tag)
    rval.full_args = tuple(full_args)  # prevent garbage-collection
    rval.bw_per_call = bw_per_call
    rval.description = ("groups: %d; items: %d; items/group: %0.1f [%d, %d]" %
                        (gsize[1], input0.sizes.sum(), input0.sizes.mean(),
                         input0.sizes.min(), input0.sizes.max()))
    return rval


def init_rng(queue, seed):

    work_items = queue.device.max_work_group_size
    ranluxcltab = to_device(queue, np.zeros(28 * work_items, dtype='int32'))

    text = """
        #include "pyopencl-ranluxcl.cl"

        ////////// MAIN FUNCTION //////////
        __kernel void init_rng(
            uint ins,
            __global ranluxcl_state_t *ranluxcltab
        )
        {
            ranluxcl_initialization(ins, ranluxcltab);
        }
        """

    textconf = dict()
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))

    kernel = cl.Program(queue.context, text).build().init_rng
    gsize = (work_items,)
    lsize = None
    kernel(queue, gsize, lsize, np.uint32(seed), ranluxcltab.data)
    queue.finish()

    return ranluxcltab


_dist_enums = {nengod.Uniform: 0, nengod.Gaussian: 1}
_dist_params = {
    nengod.Uniform: lambda d: np.array([d.low, d.high], dtype=np.float32),
    nengod.Gaussian: lambda d: np.array([d.mean, d.std], dtype=np.float32),
    }
dist_header = """
#include "pyopencl-ranluxcl.cl"

inline float4 sample_dist(
    int dist, __global const float *params, ranluxcl_state_t *state)
{
    switch (dist) {
        case 0:  // Uniform (params: low, high)
            //return ranluxcl32(state);
            return params[0] + (params[1] - params[0]) * ranluxcl32(state);
        case 1:  // Gaussian (params: mean, std)
            //return 0.0f;
            return params[0] + params[1] * ranluxcl32norm(state);
        default:
            return 0.0f;
    }
}

inline float getfloat4(float4 a, int i) {
    switch (i) {
        case 0: return a.s0;
        case 1: return a.s1;
        case 2: return a.s2;
        case 3: return a.s3;
    }
}
"""


def get_dist_enums_params(dists):
    enums = [_dist_enums[d.__class__] for d in dists]
    params = [_dist_params[d.__class__](d) for d in dists]
    return (RaggedArray(enums, dtype=np.int32),
            RaggedArray(params, dtype=np.float32))


def plan_whitenoise(queue, Y, dist_enums, dist_params, scale, dt, ranluxcltab,
                    tag=None):
    N = len(Y)
    assert len(Y) == len(dist_enums) == len(dist_params) == len(scale)

    assert dist_enums.ctype == 'int'
    assert scale.ctype == 'int'

    for i in range(N):
        for arr in [Y, dist_enums, dist_params, scale]:
            assert arr.stride1s[i] == 1

        assert Y.shape1s[i] == 1
        assert Y.stride0s[i] == 1
        assert Y.stride1s[i] == 1

        assert dist_enums.shape0s[i] == dist_enums.shape1s[i] == 1
        assert dist_params.shape1s[i] == 1

        assert scale.shape0s[i] == scale.shape1s[i] == 1
        assert scale.stride0s[i] == scale.stride1s[i] == 1

    text = """
        ${dist_header}

        ////////// MAIN FUNCTION //////////
        __kernel void whitenoise(
            __global const int *shape0s,
            __global const int *Ystarts,
            __global ${Ytype} *Ydata,
            __global const int *Estarts,
            __global const int *Edata,
            __global const int *Pstarts,
            __global const ${Ptype} *Pdata,
            __global const int *scalestarts,
            __global const int *scaledata,
            __global ranluxcl_state_t *ranluxcltab
        )
        {
            const int i0 = get_global_id(0);
            const int k = get_global_id(1);
            const int m = shape0s[k];
            if (i0 >= m)
                return;

            __global ${Ytype} *y = Ydata + Ystarts[k];

            ranluxcl_state_t state;
            ranluxcl_download_seed(&state, ranluxcltab);

            const int scale = *(scaledata + scalestarts[k]);
            const int dist_enum = *(Edata + Estarts[k]);
            __global const float *dist_params = Pdata + Pstarts[k];

            float4 samples;
            float sample;
            int samplei = 4;
            for (int i = i0; i < m; i += get_global_size(0))
            {
                if (samplei >= 4) {
                    samples = sample_dist(dist_enum, dist_params, &state);
                    samplei = 0;
                }

                sample = getfloat4(samples, samplei);
                y[i] = (scale) ? ${sqrt_dt_inv} * sample : sample;
                samplei++;
            }

            ranluxcl_upload_seed(&state, ranluxcltab);
        }
        """

    textconf = dict(Ytype=Y.ctype, Ptype=dist_params.ctype,
                    sqrt_dt_inv=1 / np.sqrt(dt), dist_header=dist_header)
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))

    full_args = (
        Y.cl_shape0s,
        Y.cl_starts,
        Y.cl_buf,
        dist_enums.cl_starts,
        dist_enums.cl_buf,
        dist_params.cl_starts,
        dist_params.cl_buf,
        scale.cl_starts,
        scale.cl_buf,
        ranluxcltab,
    )
    _fn = cl.Program(queue.context, text).build().whitenoise
    _fn.set_args(*[arr.data for arr in full_args])

    max_len = min(queue.device.max_work_group_size, max(Y.shape0s))
    gsize = (max_len, N)
    lsize = (max_len, 1)
    rval = Plan(queue, _fn, gsize, lsize=lsize, name="cl_whitenoise", tag=tag)
    rval.full_args = full_args     # prevent garbage-collection
    return rval


def plan_presentinput(queue, Y, t, signals, dt, pres_t=None, tag=None):
    N = len(Y)
    assert len(Y) == len(t) == len(signals)
    assert pres_t is None or pres_t.shape == (N,)

    for i in range(N):
        for arr in [Y, t, signals]:
            assert arr.stride1s[i] == 1

        assert Y.shape1s[i] == 1
        assert Y.stride0s[i] == Y.stride1s[i] == 1

        assert t.shape0s[i] == t.shape1s[i] == 1

        assert Y.shape0s[i] == signals.shape1s[i]
        assert signals.stride1s[i] == 1

    text = """
        ////////// MAIN FUNCTION //////////
        __kernel void presentinput(
    % if Ptype is not None:
            __global ${Ptype} *Pdata,
    % endif
            __global const int *Yshape0s,
            __global const int *Ystarts,
            __global ${Ytype} *Ydata,
            __global const int *Tstarts,
            __global ${Ttype} *Tdata,
            __global const int *Sshape0s,
            __global const int *Sstarts,
            __global ${Stype} *Sdata
        )
        {
            int i = get_global_id(0);
            const int k = get_global_id(1);
            const int m = Yshape0s[k];
            if (i >= m)
                return;

            __global ${Ytype} *y = Ydata + Ystarts[k];
            __global ${Ytype} *s = Sdata + Sstarts[k];
            const int it = *(Tdata + Tstarts[k]);
            const int nt = Sshape0s[k];
    % if Ptype is not None:
            const float pt = Pdata[k];
            const int ti = (int)((it + 0.5f) * (${dt}f / pt)) % nt;
    % else:
            const int ti = (int)it % nt;
    % endif

            for (; i < m; i += get_global_size(0))
                y[i] = s[m*ti + i];
        }
        """

    textconf = dict(Ytype=Y.ctype, Ttype=t.ctype, Stype=signals.ctype,
                    Ptype=pres_t.ctype if pres_t is not None else None,
                    dt=dt)
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))

    full_args = ((pres_t,) if pres_t is not None else ()) + (
        Y.cl_shape0s,
        Y.cl_starts,
        Y.cl_buf,
        t.cl_starts,
        t.cl_buf,
        signals.cl_shape0s,
        signals.cl_starts,
        signals.cl_buf,
    )
    _fn = cl.Program(queue.context, text).build().presentinput
    _fn.set_args(*[arr.data for arr in full_args])

    max_len = min(queue.device.max_work_group_size, max(Y.shape0s))
    gsize = (max_len, N)
    lsize = (max_len, 1)
    rval = Plan(
        queue, _fn, gsize, lsize=lsize, name="cl_presentinput", tag=tag)
    rval.full_args = full_args     # prevent garbage-collection
    return rval


def plan_conv2d(queue, X, Y, filters, biases, shape_in, shape_out,
                kernel_shape, conv, padding, strides,
                tag=None, transposed=False):
    """
    Parameters
    ----------
        filters = ch x size_i x size_j x nf             # conv transposed
        filters = ch x size_i x size_j x nf x ni x nj   # local transposed
        biases = nf x ni x nj

        conv : whether this is a convolution (true) or local filtering (false)
    """
    for ary in [X, Y, filters, biases]:
        # assert that arrays are contiguous
        assert len(ary.shape) in [1, 2]
        assert ary.strides[-1] == ary.dtype.itemsize
        if len(ary.shape) == 2:
            assert ary.strides[0] == ary.dtype.itemsize * ary.shape[1]

    assert filters.start == biases.start == 0
    assert X.ctype == Y.ctype == filters.ctype == biases.ctype

    text = """
    __kernel void conv2d(
        __global const ${type} *x,
        __global const ${type} *f,
        __global const ${type} *b,
        __global ${type} *y
    )
    {
        const int j = get_global_id(0);
        const int i = get_global_id(1);
        const int ij = i*${nyj} + j;

        const int tj = get_local_id(0);
        const int ti = get_local_id(1);
        const int lsizej = get_local_size(0);
        const int lsizei = get_local_size(1);
        const int lsize = lsizei * lsizej;
        const int tij = ti*lsizej + tj;
        const int j0 = (j - tj)*${stj} - ${pj};
        const int i0 = (i - ti)*${sti} - ${pi};
        __local ${type} patch[${nipatch}][${njpatch}];
    % if conv:
        __local ${type} filters[${nf*si*sj}];
    % else:
        f += ij;
    % endif
        x += ${xstart};
        y += ${ystart};

    % if conv or nf_per >= nf:
        ${type} outs[${nf}];
        for (int k = 0; k < ${nf}; k++)
            outs[k] = b[k*${nyi*nyj} + ij];
    % else:
        const int k0 = get_global_id(2);
        const int ksize = get_global_size(2);

        ${type} outs[${nf_per}];
        for (int k = 0, kk = k0; k < ${nf_per}; k++, kk += ksize)
            outs[k] = (kk < ${nf}) ? b[kk*${nyi*nyj} + ij] : 0;
    % endif

        for (int c = 0; c < ${nc}; c++) {

            // load image section
            __global const ${type} *xc = &x[c * ${nxi * nxj}];
            for (int k = tij; k < ${npatch}; k += lsize) {
                const int ki = k / ${njpatch};
                const int kj = k % ${njpatch};
                const int ii = i0 + ki;
                const int jj = j0 + kj;
                if (ii >= 0 && ii < ${nxi} && jj >= 0 && jj < ${nxj})
                    patch[ki][kj] = xc[ii*${nxj} + jj];
                else
                    patch[ki][kj] = 0;
            }

    % if conv:
            // load filters
            __global const ${type} *fc = f + c*${si*sj}*${nf};
            for (int k = tij; k < ${si*sj} * ${nf}; k += lsize) {
                filters[k] = fc[k];
            }
    % endif

            barrier(CLK_LOCAL_MEM_FENCE);

            for (int ii = 0; ii < ${si}; ii++) {
            for (int jj = 0; jj < ${sj}; jj++) {
                const ${type} xcij = patch[${sti}*ti+ii][${stj}*tj+jj];
    % if conv:
                __local const ${type} *fcij = &filters[(ii*${sj} + jj)*${nf}];
    % else:
                __global const ${type} *fcij =
                    &f[(c*${si*sj} + ii*${sj} + jj)*${nf}*${nyi*nyj}];
    % endif

    % if conv:
                for (int k = 0; k < ${nf}; k++)
                    outs[k] += fcij[k] * xcij;
    % elif nf_per >= nf:
                for (int k = 0; k < ${nf}; k++)
                    outs[k] += fcij[k*${nyi*nyj}] * xcij;
    % else:
                for (int k = 0, kk = k0; k < ${nf_per}; k++, kk += ksize)
                    outs[k] += (kk < ${nf}) ? fcij[kk*${nyi*nyj}] * xcij : 0;
    % endif
            }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (i < ${nyi} && j < ${nyj}) {
    % if conv or nf_per >= nf:
            for (int k = 0; k < ${nf}; k++)
                y[k*${nyi*nyj} + ij] = outs[k];
    % else:
            for (int k = 0, kk = k0; k < ${nf_per}; k++, kk += ksize)
                if (kk < ${nf})  y[kk*${nyi*nyj} + ij] = outs[k];
    % endif
        }
    }
    """

    nc, nxi, nxj = shape_in
    nf, nyi, nyj = shape_out
    si, sj = kernel_shape
    pi, pj = padding
    sti, stj = strides
    nf_per = 16

    max_group = min(queue.device.max_work_group_size, 128)
    assert max_group >= 32
    lsize0 = min(nyj, 32)
    lsize1 = min(max_group // lsize0, nyi)
    lsize = (lsize0, lsize1)
    gsize = (round_up(nyj, lsize[0]), round_up(nyi, lsize[1]))
    if not conv and nf_per < nf:
        lsize = lsize + (1,)
        gsize = gsize + (int(np.ceil(nf / nf_per)),)

    njpatch = (lsize[0] - 1) * stj + sj
    nipatch = (lsize[1] - 1) * sti + si
    npatch = nipatch * njpatch

    assert np.prod(lsize) <= queue.device.max_work_group_size
    assert (npatch*X.dtype.itemsize + conv*nf*si*sj*filters.dtype.itemsize
            <= queue.device.local_mem_size)

    textconf = dict(
        type=X.ctype, conv=conv, nf=nf, nxi=nxi, nxj=nxj, nyi=nyi, nyj=nyj,
        nc=nc, si=si, sj=sj, pi=pi, pj=pj, sti=sti, stj=stj, nf_per=nf_per,
        nipatch=nipatch, njpatch=njpatch, npatch=npatch,
        xstart=X.start, ystart=Y.start)
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))

    full_args = (X.base_data, filters.data, biases.data, Y.base_data)
    _fn = cl.Program(queue.context, text).build().conv2d
    _fn.set_args(*full_args)

    rval = Plan(queue, _fn, gsize, lsize=lsize, name="cl_conv2d", tag=tag)
    rval.full_args = full_args     # prevent garbage-collection
    rval.flops_per_call = 2 * nyi * nyj * nf * nc * si * sj
    rval.bw_per_call = X.nbytes + filters.nbytes + biases.nbytes + Y.nbytes
    return rval


def plan_pool2d(queue, X, Y, shape, size, stride, tag=None):
    for ary in [X, Y]:
        # assert that arrays are contiguous
        assert len(ary.shape) in [1, 2]
        assert ary.strides[-1] == ary.dtype.itemsize
        if len(ary.shape) == 2:
            assert ary.strides[0] == ary.dtype.itemsize * ary.shape[1]

    assert X.ctype == Y.ctype

    text = """
    ////////// MAIN FUNCTION //////////
    __kernel void pool2d(
        __global const ${type} *x,
        __global ${type} *y
    )
    {
        const int j = get_global_id(0);
        const int i = get_global_id(1);
        const int c = get_global_id(2);

        const int tj = get_local_id(0);
        const int ti = get_local_id(1);
        const int lsizej = get_local_size(0);
        const int lsizei = get_local_size(1);
        const int lsize = lsizei * lsizej;
        const int tij = ti*lsizej + tj;
        const int i0 = i - ti;
        const int j0 = j - tj;
        __local ${type} patch[${nipatch}][${njpatch}];

        x += ${Xstart};
        y += ${Ystart};

        // load image patch
        __global const ${type} *xc = &x[c * ${nxi * nxj}];
        for (int k = tij; k < ${nipatch * njpatch}; k += lsize) {
            const int ki = k / ${njpatch};
            const int kj = k % ${njpatch};
            const int ii = i0*${st} + ki;
            const int jj = j0*${st} + kj;
            if (ii >= 0 && ii < ${nxi} && jj >= 0 && jj < ${nxj})
                patch[ki][kj] = xc[ii*${nxj} + jj];
            else
                patch[ki][kj] = NAN;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        ${type} out = 0;
        int n = 0;
        ${type} xij;
        for (int ii = 0; ii < ${s}; ii++) {
        for (int jj = 0; jj < ${s}; jj++) {
            xij = patch[ti*${st} + ii][tj*${st} + jj];
            if (!isnan(xij)) {
                out += xij;
                n++;
            }
        }
        }

        if (i < ${nyi} && j < ${nyj})
            y[c*${nyi * nyj} + i*${nyj} + j] = out / n;
    }
    """
    nc, nyi, nyj, nxi, nxj = shape

    max_group = min(queue.device.max_work_group_size, 64)
    assert max_group >= 32
    lsize0 = min(nyj, 8)
    lsize1 = min(nyi, max_group // lsize0)
    lsize = (lsize0, lsize1, 1)
    gsize = (round_up(nyj, lsize[0]), round_up(nyi, lsize[1]), nc)

    njpatch = lsize[0]*stride + size - 1
    nipatch = lsize[1]*stride + size - 1
    assert nipatch*njpatch <= queue.device.local_mem_size / X.dtype.itemsize

    textconf = dict(
        type=X.ctype, Xstart=X.start, Ystart=Y.start,
        nxi=nxi, nxj=nxj, nyi=nyi, nyj=nyj, s=size, st=stride,
        nipatch=nipatch, njpatch=njpatch)

    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))

    full_args = (X.base_data, Y.base_data)
    _fn = cl.Program(queue.context, text).build().pool2d
    _fn.set_args(*full_args)

    rval = Plan(queue, _fn, gsize, lsize=lsize, name="cl_pool2d", tag=tag)
    rval.full_args = full_args     # prevent garbage-collection
    rval.flops_per_call = X.size
    rval.bw_per_call = X.nbytes + Y.nbytes
    return rval
