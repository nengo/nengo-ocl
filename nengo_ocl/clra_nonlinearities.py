import numpy as np
import pyopencl as cl
from mako.template import Template
import nengo.dists as nengod
from nengo.utils.compat import range

from nengo_ocl.raggedarray import RaggedArray
from nengo_ocl.clraggedarray import CLRaggedArray, as_ascii, to_device
from nengo_ocl.plan import Plan


def all_equal(a, b):
    return (np.asarray(a) == np.asarray(b)).all()


def _indent(s, i):
    return '\n'.join([(' ' * i) + line for line in s.split('\n')])


def plan_slicedcopy(queue, A, B, Ainds, Binds, incs, tag=None):
    N = len(A)
    assert len(A) == len(B) == len(Ainds) == len(Binds)

    for i in range(N):
        for arr in [A, B, Ainds, Binds]:
            assert arr.stride1s[i] == 1
        assert A.stride0s[i] == 1
        assert A.stride1s[i] == 1
        assert B.stride0s[i] == 1
        assert B.stride1s[i] == 1
        assert Ainds.stride0s[i] == 1
        assert Ainds.stride1s[i] == 1
        assert Binds.stride0s[i] == 1
        assert Binds.stride1s[i] == 1
        assert Ainds.shape0s[i] == Binds.shape0s[i]

    assert A.cl_buf.ctype == B.cl_buf.ctype
    assert Ainds.cl_buf.ctype == Binds.cl_buf.ctype == 'int'
    assert incs.cl_buf.ctype == 'int'

    text = """
        ////////// MAIN FUNCTION //////////
        __kernel void slicedcopy(
            __global const int *Astarts,
            __global const ${Atype} *Adata,
            __global const int *Bstarts,
            __global ${Btype} *Bdata,
            __global const int *Ishape0s,
            __global const int *AIstarts,
            __global const int *AIdata,
            __global const int *BIstarts,
            __global const int *BIdata,
            __global const int *incdata
        )
        {
            const int n = get_global_id(1);
            __global const ${Atype} *a = Adata + Astarts[n];
            __global ${Btype} *b = Bdata + Bstarts[n];
            __global const int *aind = AIdata + AIstarts[n];
            __global const int *bind = BIdata + BIstarts[n];
            const int inc = incdata[n];

            int i = get_global_id(0);
            if (inc)
                for (; i < Ishape0s[n]; i += get_global_size(0))
                    b[bind[i]] += a[aind[i]];
            else
                for (; i < Ishape0s[n]; i += get_global_size(0))
                    b[bind[i]] = a[aind[i]];
        }
        """

    textconf = dict(Atype=A.cl_buf.ctype, Btype=B.cl_buf.ctype)
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))

    full_args = (
        A.cl_starts,
        A.cl_buf,
        B.cl_starts,
        B.cl_buf,
        Ainds.cl_shape0s,
        Ainds.cl_starts,
        Ainds.cl_buf,
        Binds.cl_starts,
        Binds.cl_buf,
        incs.cl_buf,
    )
    _fn = cl.Program(queue.context, text).build().slicedcopy
    _fn.set_args(*[arr.data for arr in full_args])

    max_group = queue.device.max_work_group_size
    n = min(max(Ainds.shape0s), max_group)
    gsize = (n, N)
    lsize = (n, 1)
    rval = Plan(
        queue, _fn, gsize, lsize=lsize, name="cl_slicedcopy", tag=tag)
    rval.full_args = full_args     # prevent garbage-collection
    return rval


def plan_elementwise_inc(queue, A, X, Y, tag=None):
    """Implements an element-wise increment Y += A * X"""
    N = len(X)
    assert len(Y) == N and len(A) == N

    for i in range(N):
        for arr in [A, X, Y]:
            assert arr.stride1s[i] == 1
        assert X.shape0s[i] in [1, Y.shape0s[i]]
        assert X.shape1s[i] in [1, Y.shape1s[i]]
        assert A.shape0s[i] in [1, Y.shape0s[i]]
        assert A.shape1s[i] in [1, Y.shape1s[i]]
        assert X.stride1s[i] == 1
        assert Y.stride1s[i] == 1
        assert A.stride1s[i] == 1

    assert X.cl_buf.ctype == Y.cl_buf.ctype
    assert A.cl_buf.ctype == Y.cl_buf.ctype

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

            const int Ysize = Yshape0s[n] * Yshape1s[n];
            for (int ij = get_global_id(0);
                 ij < Ysize;
                 ij += get_global_size(0))
            {
                int i = ij / Yshape1s[n];
                int j = ij - i * Yshape1s[n];

                ${Atype} aa = get_element(
                    a, Ashape0s[n], Ashape1s[n], Astride0s[n], i, j);
                ${Xtype} xx = get_element(
                    x, Xshape0s[n], Xshape1s[n], Xstride0s[n], i, j);

                y[i * Ystride0s[n] + j] += aa * xx;
            }
        }
        """

    textconf = dict(Atype=A.cl_buf.ctype, Xtype=X.cl_buf.ctype,
                    Ytype=Y.cl_buf.ctype)
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
    mn = min(max(max(Y.shape0s), max(Y.shape1s)), max_group)
    gsize = (mn, N)
    lsize = (mn, 1)
    rval = Plan(
        queue, _fn, gsize, lsize=lsize, name="cl_elementwise_inc", tag=tag)
    rval.full_args = full_args     # prevent garbage-collection
    return rval


def plan_linear_synapse(queue, X, Y, A, B, Xbuf, Ybuf, tag=None):
    """
    Implements a filter of the form

        y[n+1] + a[0] y[n] + ... + a[i] y[n-i] = b[0] x[n] + ... + b[j] x[n-j]
    """
    N = len(X)
    assert len(Y) == N and len(A) == N and len(B) == N

    for i in range(N):
        for arr in [X, Y, A, B, Xbuf, Ybuf]:
            assert arr.shape1s[i] == arr.stride0s[i]
            assert arr.stride1s[i] == 1
        for arr in [X, Y, A, B]:  # vectors
            assert arr.shape1s[i] == 1
        assert X.shape0s[i] == Y.shape0s[i]

        assert B.shape0s[i] >= 1
        if B.shape0s[i] > 1:
            assert Xbuf.shape0s[i] == B.shape0s[i]
        assert Xbuf.shape1s[i] == X.shape0s[i]
        if A.shape0s[i] > 1:
            assert Ybuf.shape0s[i] == A.shape0s[i]
        assert Ybuf.shape1s[i] == Y.shape0s[i]

    assert X.cl_buf.ctype == Xbuf.cl_buf.ctype
    assert Y.cl_buf.ctype == Ybuf.cl_buf.ctype

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
        Xtype=X.cl_buf.ctype, Ytype=Y.cl_buf.ctype,
        Atype=A.cl_buf.ctype, Btype=B.cl_buf.ctype
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

    max_len = min(queue.device.max_work_group_size, max(X.shape0s))
    gsize = (max_len, N)
    lsize = (max_len, 1)
    rval = Plan(
        queue, _fn, gsize, lsize=lsize, name="cl_linear_synapse", tag=tag)
    rval.full_args = full_args     # prevent garbage-collection
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
    assert X.cl_buf.ctype == Y.cl_buf.ctype
    N = len(X)

    # N.B.  X[i].shape = (M, N)
    #       Y[i].shape = (buf_len, M * N)
    for i in range(N):
        for arr in [X, Y]:
            assert arr.stride1s[i] == 1
        assert X.shape0s[i] * X.shape1s[i] == Y.shape1s[i]
        assert X.stride0s[i] == X.shape1s[i]
        assert X.stride1s[i] == 1
        assert Y.stride0s[i] == Y.shape1s[i]
        assert Y.stride1s[i] == 1

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
                    Xtype=X.cl_buf.ctype,
                    Ytype=Y.cl_buf.ctype,
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
    return rval


def plan_direct(queue, code, init, input_names, inputs, output, tag=None):
    from . import ast_conversion

    assert len(input_names) == len(inputs)
    for x in inputs:
        assert len(x) == len(output)
    N = len(inputs[0])
    input_types = [x.cl_buf.ctype for x in inputs]
    output_type = output.cl_buf.ctype

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

    textconf = dict(init=_indent(init, 12),
                    code=_indent(code, 12),
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
    rval.full_args = full_args     # prevent garbage-collection
    return rval


def plan_lif(queue, J, V, W, outV, outW, outS, ref, tau, dt,
             tag=None, n_elements=0, upsample=1):
    for array in [V, W, outV, outW, outS]:
        assert V.cl_buf.ctype == J.cl_buf.ctype

    inputs = dict(j=J, v=V, w=W)
    outputs = dict(ov=outV, ow=outW, os=outS)
    parameters = dict(tau=tau, ref=ref)

    dt = float(dt)
    textconf = dict(Vtype=V.cl_buf.ctype,
                    upsample=upsample,
                    dtu=dt / upsample,
                    dtu_inv=upsample / dt,
                    dt_inv=1 / dt,
                    V_threshold=1.)
    declares = """
        char spiked;
        ${Vtype} dV, overshoot;

        const ${Vtype} dtu = ${dtu},
                       dtu_inv = ${dtu_inv},
                       dt_inv = ${dt_inv},
                       V_threshold = ${V_threshold};
        """
    # TODO: could precompute -expm1(-dtu / tau)
    text = """
        spiked = 0;

% for ii in range(upsample):
        dV = -expm1(-dtu / tau) * (j - v);
        v += dV;
        w -= dtu;

        if (v < 0 || w > dtu)
            v = 0;
        else if (w >= 0)
            v *= 1 - w * dtu_inv;

        if (v > V_threshold) {
            overshoot = dtu * (v - V_threshold) / dV;
            w = ref - overshoot + dtu;
            v = 0;
            spiked = 1;
        }
% endfor
        ov = v;
        ow = w;
        os = (spiked) ? dt_inv : 0;
        """
    declares = as_ascii(
        Template(declares, output_encoding='ascii').render(**textconf))
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))
    return _plan_template(
        queue, "cl_lif", text, declares=declares,
        tag=tag, n_elements=n_elements,
        inputs=inputs, outputs=outputs, parameters=parameters)


def plan_lif_rate(queue, J, R, ref, tau, dt, tag=None, n_elements=0):
    assert R.cl_buf.ctype == J.cl_buf.ctype

    inputs = dict(j=J)
    outputs = dict(r=R)
    parameters = dict(tau=tau, ref=ref)
    textconf = dict(Rtype=R.cl_buf.ctype)
    declares = """
        const ${Rtype} c0 = 0, c1 = 1;
        """
    text = """
        j = max(j - 1, c0);
        r = c1 / (ref + tau * log1p(c1/j));
        """
    declares = as_ascii(
        Template(declares, output_encoding='ascii').render(**textconf))
    return _plan_template(
        queue, "cl_lif_rate", text, declares=declares,
        tag=tag, n_elements=n_elements,
        inputs=inputs, outputs=outputs, parameters=parameters)


def _plan_template(queue, name, core_text, declares="", tag=None, n_elements=0,
                   inputs={}, outputs={}, parameters={}):
    """Template for making a plan for vector nonlinearities.

    This template assumes that all inputs and outputs are vectors.

    Parameters
    ----------
    n_elements: int
        If n_elements == 0, then the kernels are allocated as a block. This is
        simple, but can be slow for large computations where input vector sizes
        are not uniform (e.g. one large population and many small ones).
        If n_elements >= 1, then all the vectors in the RaggedArray are
        flattened so that the exact number of required kernels is allocated.
        Each kernel performs computations for `n_elements` elements.

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
    base = list(inputs.values())[0]   # input to use as reference (for lengths)
    N = len(base)

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
    for vname, v in list(inputs.items()) + list(outputs.items()):
        assert vname not in avars, "Name clash"
        assert len(v) == N
        assert all_equal(v.shape0s, base.shape0s)

        # N.B. - we should be able to ignore ldas as long as all vectors
        assert all_equal(v.shape1s, 1)

        dtype = v.cl_buf.ctype
        offset = '%(name)s_starts[n]' % {'name': vname}
        avars[vname] = (dtype, offset)

    for vname, v in params.items():
        assert vname not in avars, "Name clash"
        assert len(v) == N
        for i in range(N):
            assert v.shape0s[i] == base.shape0s[i] or v.shape0s[i] == 1, \
                "%s.shape0s[%d] must be 1 or %d (not %d)" % \
                (vname, i, base.shape0s[i], v.shape0s[i])
            assert v.shape1s[i] == 1

        dtype = v.cl_buf.ctype
        offset = '%(name)s_starts[n]' % {'name': vname}
        avars[vname] = (dtype, offset)

    ivars = dict((k, avars[k]) for k in inputs.keys())
    ovars = dict((k, avars[k]) for k in outputs.keys())
    pvars = dict((k, avars[k]) for k in params.keys())

    fn_name = "%s_%d" % (name, n_elements)
    textconf = dict(fn_name=fn_name, N=N, n_elements=n_elements,
                    declares=declares, core_text=core_text,
                    ivars=ivars, ovars=ovars, pvars=pvars,
                    static_params=static_params)

    if n_elements > 0:
        # Allocate the exact number of required kernels in a vector
        gsize = (int(np.ceil(np.sum(base.shape0s) / float(n_elements))),)
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
            __global const int *lengths
        )
        {
            const int gid = get_global_id(0);
            int m = gid * ${n_elements}, n = 0;
            while (m >= lengths[n]) {
                m -= lengths[n];
                n++;
            }
            if (n >= ${N}) return;

% for name, [type, offset] in ivars.items():
            __global const ${type} *cur_${name} = in_${name} + ${offset} + m;
% endfor
% for name, [type, offset] in ovars.items():
            __global ${type} *cur_${name} = in_${name} + ${offset} + m;
% endfor
% for name, [type, offset] in pvars.items():
            __global const ${type} *cur_${name} = in_${name} + ${offset};
            int ${name}_isvector = ${name}_shape0s[n] > 1;
            if (${name}_isvector) cur_${name} += m;
% endfor
% for name, [type, offset] in \
        list(ivars.items()) + list(ovars.items()) + list(pvars.items()):
            ${type} ${name};
% endfor
% for name, [type, value] in static_params.items():
            const ${type} ${name} = ${value};
% endfor
            //////////////////////////////////////////////////
            //vvvvv USER DECLARATIONS BELOW vvvvv
            ${declares}
            //^^^^^ USER DECLARATIONS ABOVE ^^^^^
            //////////////////////////////////////////////////

% for ii in range(n_elements):
            //////////////////////////////////////////////////
            ////////// LOOP ITERATION ${ii}
  % for name, [type, offset] in ivars.items():
            ${name} = *cur_${name};
  % endfor
  % for name, [type, offset] in pvars.items():
            if ((${ii} == 0) || ${name}_isvector) ${name} = *cur_${name};
  % endfor

            /////vvvvv USER COMPUTATIONS BELOW vvvvv
            ${core_text}
            /////^^^^^ USER COMPUTATIONS ABOVE ^^^^^

  % for name, [type, offset] in ovars.items():
            *cur_${name} = ${name};
  % endfor

  % if ii + 1 < n_elements:
            m++;
            if (m >= lengths[n]) {
                n++;
                m = 0;
                if (n >= ${N}) return;

    % for name, [_, offset] in \
        list(ivars.items()) + list(ovars.items()) + list(pvars.items()):
                cur_${name} = in_${name} + ${offset};
    % endfor
    % for name, _ in pvars.items():
                ${name}_isvector = ${name}_shape0s[n] > 1;
                if (!${name}_isvector) ${name} = *cur_${name};
    % endfor
            } else {
    % for name, _ in list(ivars.items()) + list(ovars.items()):
                cur_${name}++;
    % endfor
    % for name, _ in pvars.items():
                if (${name}_isvector) cur_${name}++;
    % endfor
            }
  % endif
% endfor
        }
        """
    else:
        # Allocate more than enough kernels in a matrix
        gsize = (int(np.max(base.shape0s)), int(N))
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
            __global const int *lengths
        )
        {
            const int m = get_global_id(0);
            const int n = get_global_id(1);
            const int M = lengths[n];
            if (m >= M) return;

% for name, [type, offset] in ivars.items():
            ${type} ${name} = in_${name}[${offset} + m];
% endfor
% for name, [type, offset] in ovars.items():
            ${type} ${name};
% endfor
% for name, [type, offset] in pvars.items():
            const ${type} ${name} = (${name}_shape0s[n] > 1) ?
                in_${name}[${offset} + m] : in_${name}[${offset}];
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
            in_${name}[${offset} + m] = ${name};
% endfor
        }
        """

    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))
    if 0:
        for i, line in enumerate(text.split('\n')):
            print("%3d %s" % (i + 1, line))

    full_args = []
    for vname, v in list(inputs.items()) + list(outputs.items()):
        full_args.extend([v.cl_starts, v.cl_buf])
    for vname, v in params.items():
        full_args.extend([v.cl_starts, v.cl_shape0s, v.cl_buf])
    full_args.append(base.cl_shape0s)
    full_args = tuple(full_args)

    fns = cl.Program(queue.context, text).build()
    _fn = getattr(fns, fn_name)
    _fn.set_args(*[arr.data for arr in full_args])

    rval = Plan(queue, _fn, gsize, lsize=None, name=name, tag=tag)
    rval.full_args = full_args     # prevent garbage-collection
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
    enums = [np.array(_dist_enums[d.__class__], dtype=int) for d in dists]
    params = [_dist_params[d.__class__](d) for d in dists]
    return RaggedArray(enums), RaggedArray(params)


def plan_whitenoise(queue, Y, dist_enums, dist_params, scale, dt, ranluxcltab,
                    tag=None):
    N = len(Y)
    assert len(Y) == len(dist_enums) == len(dist_params) == len(scale)

    assert dist_enums.cl_buf.ctype == 'int'
    assert scale.cl_buf.ctype == 'int'

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

    textconf = dict(Ytype=Y.cl_buf.ctype, Ptype=dist_params.cl_buf.ctype,
                    sqrt_dt_inv=1. / np.sqrt(dt), dist_header=dist_header)
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


def plan_whitesignal(queue, Y, t, signals, dt, tag=None):
    N = len(Y)
    assert len(Y) == len(t) == len(signals)

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
        __kernel void whitesignal(
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
            const float t = *(Tdata + Tstarts[k]);
            const int nt = Sshape0s[k];
            const int ti = (int)round(t / ${dt}) % nt;

            for (; i < m; i += get_global_size(0))
                y[i] = s[m*ti + i];
        }
        """

    textconf = dict(Ytype=Y.cl_buf.ctype, Ttype=t.cl_buf.ctype,
                    Stype=signals.cl_buf.ctype, dt=dt)
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))

    full_args = (
        Y.cl_shape0s,
        Y.cl_starts,
        Y.cl_buf,
        t.cl_starts,
        t.cl_buf,
        signals.cl_shape0s,
        signals.cl_starts,
        signals.cl_buf,
    )
    _fn = cl.Program(queue.context, text).build().whitesignal
    _fn.set_args(*[arr.data for arr in full_args])

    max_len = min(queue.device.max_work_group_size, max(Y.shape0s))
    gsize = (max_len, N)
    lsize = (max_len, 1)
    rval = Plan(queue, _fn, gsize, lsize=lsize, name="cl_whitesignal", tag=tag)
    rval.full_args = full_args     # prevent garbage-collection
    return rval
