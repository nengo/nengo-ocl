import numpy as np
import pyopencl as cl
from plan import Plan
from mako.template import Template
from array import to_device

def gemv_batched_ref(context, B, M, N, alpha,
                             Aoffset, AsB, AsM, AsN,
                             XsN,
                             beta,
                             YsM,
                            ):
    return cl.Program(context, """
        __kernel void fn(__global const float *A_data,
                         __global const float *X_data,
                         __global const int *X_offsets,
                         __global float *Y_data,
                         __global const int *Y_offsets
                         )
        {
            const int bb = get_global_id(0);

            A_data += %(Aoffset)s + bb * %(AsB)s;
            X_data += X_offsets[bb];
            Y_data += Y_offsets[bb];

            for (int mm = 0; mm < %(M)s; ++mm)
            {
                float ksum = 0.0;
                for (int nn = 0; nn < %(N)s; ++nn)
                {
                    ksum += A_data[nn * %(AsN)s  + mm * %(AsM)s] * X_data[nn * %(XsN)s];
                }

                if (%(beta)s == 0)
                {
                    Y_data[%(YsM)s * mm] = %(alpha)s * ksum;
                }
                else
                {
                    Y_data[%(YsM)s * mm] = %(beta)s * Y_data[%(YsM)s * mm] + %(alpha)s * ksum;
                }
            }
        }
        """ % locals()).build().fn


def gemv_batched_parout_nolocal(context, B, M, N, alpha,
                             Aoffset, AsB, AsM, AsN,
                             XsN,
                             beta,
                             YsM,
                            ):
    return cl.Program(context, """
        __kernel void fn(__global const float *A_data,
                         __global const float *X_data,
                         __global const int *X_offsets,
                         __global float *Y_data,
                         __global const int *Y_offsets
                         )
        {
            const int mm0 = get_global_id(0);
            const int bb = get_global_id(1);

            A_data += %(Aoffset)s + bb * %(AsB)s;
            X_data += X_offsets[bb];
            Y_data += Y_offsets[bb];

            for (int mm = mm0; mm < %(M)s; mm += get_local_size(0))
            {
                float ksum = 0.0;
                for (int nn = 0; nn < %(N)s; ++nn)
                {
                    ksum += A_data[nn * %(AsN)s  + mm * %(AsM)s] * X_data[nn * %(XsN)s];
                }

                if (%(beta)s == 0)
                {
                    Y_data[%(YsM)s * mm] = %(alpha)s * ksum;
                }
                else
                {
                    Y_data[%(YsM)s * mm] = %(beta)s * Y_data[%(YsM)s * mm] + %(alpha)s * ksum;
                }
            }
        }
        """ % locals()).build().fn


def gemv_batched_parout_local(context, B, M, N, alpha,
                             Aoffset, AsB, AsM, AsN,
                             XsN,
                             beta,
                             YsM,
                            ):
    return cl.Program(context, """
        __kernel void fn(__global const float *A_data,
                         __global const float *X_data,
                         __global const int *X_offsets,
                         __global float *Y_data,
                         __global const int *Y_offsets,
                         __local float * Xbuf
                         )
        {
            const int bb = get_global_id(1);

            A_data += %(Aoffset)s + bb * %(AsB)s;
            X_data += X_offsets[bb];
            Y_data += Y_offsets[bb];
            __local float * Ybuf = Xbuf + %(N)s;

            for(int nn = get_local_id(0); nn < %(N)s; nn += get_local_size(0))
            {
                Xbuf[nn] = X_data[nn * %(XsN)s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            for (int mm = get_local_id(0); mm < %(M)s; mm += get_local_size(0))
            {
                float tmp = 0.0;
                for (int nn = 0; nn < %(N)s; nn += 1)
                {
                    tmp += A_data[nn * %(AsN)s + mm * %(AsM)s] * Xbuf[nn];
                }

                if (%(beta)s != 0)
                {
                    Y_data[mm * %(YsM)s] = Y_data[mm * %(YsM)s] * %(beta)s
                        + %(alpha)s * tmp;
                }
                else
                {
                    Y_data[mm * %(YsM)s] = %(alpha)s * tmp;
                }
            }
        }
        """ % locals()).build().fn


def choose_gemv_batched_plan(queue,
    BMN, alpha, Aparams, Xparams, beta, Yparams, Zparams=None,
    ):
    """
    For each i, compute

    Zi <- alpha * dot(Ai, Xi) + beta * Yi

    Where
    Yi = Y[Y_offsets[i]: Y_offsets[i] + YsM * M: YsM]
    Xi = X[X_offsets[i]: X_offsets[i] + XsN * N: XsN]
    Ai is an M x N matrix whose first element is at
        A[A_offsets[i]] and is strided on the dimension
        of size M by AsM, and is strided on the dimension
        of size N by AsN.
    Zi defaults to Yi, but it can also be specified for an out-of-place gemv

    """
    B, M, N = BMN
    A_buf, Aoffset, AsB, AsM, AsN = Aparams
    X_buf, X_offsets, XsN = Xparams
    Y_buf, Y_offsets, YsM = Yparams
    if Zparams is None:
        Z_buf, Z_offsets, ZsM = Yparams
    else:
        Z_buf, Z_offsets, ZsM = Zparams

    if np.float32 != A_buf.dtype:
        raise NotImplementedError('A dtype', A_buf.dtype)
    if np.float32 != X_buf.dtype:
        raise NotImplementedError('X dtype', X_buf.dtype)
    if np.float32 != Y_buf.dtype:
        raise NotImplementedError('Y dtype', Y_buf.dtype)

    # TODO: autotune decision/regression tree
    if M == 1:
        _fn = gemv_batched_ref(queue.context,
            B, M, N,
            alpha,
            Aoffset, AsB, AsM, AsN,
            XsN,
            beta,
            YsM)
        global_shape = (B,)
        local_shape = None
        _fn_args = (queue, global_shape, local_shape, A_buf.data,
                    X_buf.data, X_offsets.data,
                    Y_buf.data, Y_offsets.data)
    elif 0: # this is a good idea for A in Fortran order
        _fn = gemv_batched_parout_nolocal(queue.context,
            B, M, N,
            alpha,
            Aoffset, AsB, AsM, AsN,
            XsN,
            beta,
            YsM)
        mpergroup = min(queue.context.devices[0].max_work_group_size, M)
        global_shape = (mpergroup, B,)
        local_shape = (mpergroup, 1)
        _fn_args = (queue, global_shape, local_shape, A_buf.data,
                    X_buf.data, X_offsets.data,
                    Y_buf.data, Y_offsets.data)
    else: # this is a good idea for A in C order
        _fn = gemv_batched_parout_local(queue.context,
            B, M, N,
            alpha,
            Aoffset, AsB, AsM, AsN,
            XsN,
            beta,
            YsM)
        mpergroup = min(queue.context.devices[0].max_work_group_size, M)
        global_shape = (mpergroup, B,)
        local_shape = (mpergroup, 1)
        local_mem = cl.LocalMemory(4 * N)
        _fn_args = (queue, global_shape, local_shape, A_buf.data,
                    X_buf.data, X_offsets.data,
                    Y_buf.data, Y_offsets.data, local_mem)


    return Plan(locals())



def plan_map_gemv(queue, alpha, A, X, beta, Y, Y_in=None):
    if Y_in is None:
        Y_in = Y
    assert Y.dtype == Y_in.dtype

    B, M, N = A.shape
    assert B, N == X.shape
    assert B, M == Y.shape
    assert B, M == Y_in.shape

    As0, As1, As2 = A.itemstrides
    Xs0, Xs1, = X.itemstrides
    Ys0, Ys1, = Y.itemstrides
    Ys0_in, Ys1_in, = Y_in.itemstrides

    Atype = A.ocldtype
    Xtype = X.ocldtype
    Ytype = Y.ocldtype

    # TODO: is there another way to do this other than retrieving all these
    # constants?
    Aoffset = A.offset
    Xoffset = X.offset
    Yoffset = Y.offset
    Y_in_offset = Y_in.offset

    _fn = cl.Program(queue.context, """
        __kernel void fn(__global const %(Atype)s *A_data,
                         __global const %(Xtype)s *X_data,
                         __global const %(Ytype)s *Y_in_data,
                         __global %(Ytype)s *Y_data)
        {
            const int bb = get_global_id(0);

            A_data += %(Aoffset)s + bb * %(As0)s;
            X_data += %(Xoffset)s + bb * %(Xs0)s;
            Y_data += %(Yoffset)s + bb * %(Ys0)s;
            Y_in_data += %(Y_in_offset)s + bb * %(Ys0_in)s;

            for (int mm = 0; mm < %(M)s; ++mm)
            {
                %(Ytype)s ksum = 0.0;
                for (int nn = 0; nn < %(N)s; ++nn)
                {
                    ksum += A_data[nn * %(As2)s  + mm * %(As1)s] * X_data[nn * %(Xs1)s];
                }

                if (%(beta)s == 0)
                {
                    Y_data[%(Ys1)s * mm] = %(alpha)s * ksum;
                }
                else
                {
                    Y_data[%(Ys1)s * mm] = %(beta)s * Y_in_data[%(Ys1_in)s * mm]
                        + %(alpha)s * ksum;
                }
            }
        }
        """ % locals()).build().fn

    _fn_args = (queue, (B,), None, A.data, X.data, Y_in.data, Y.data)

    return Plan(locals())


def _misc_gemv_ref(kwargs):
    text = """
        __kernel void fn(__global const %(Atype)s *A_data,
                         __global const %(Xtype)s *X_data,
                         __global const %(Xitype)s *Xi_data,
                         __global const %(Ytype)s *Y_in_data,
                         __global %(Ytype)s *Y_data)
        {
            const int bb = get_global_id(0);

            A_data += %(Aoffset)s + bb * %(As0)s;
            X_data += %(Xoffset)s + Xi_data[%(Xioffset)s + bb * %(Xis0)s] * %(Xs0)s;
            Y_data += %(Yoffset)s + bb * %(Ys0)s;
            Y_in_data += %(Y_in_offset)s + bb * %(Ys0_in)s;

            for (int mm = 0; mm < %(M)s; ++mm)
            {
                %(Ytype)s ksum = 0.0;
                for (int nn = 0; nn < %(N)s; ++nn)
                {
                    ksum += A_data[nn * %(As2)s  + mm * %(As1)s] * X_data[nn * %(Xs1)s];
                }

                if (%(beta)s == 0)
                {
                    Y_data[%(Ys1)s * mm] = %(alpha)s * ksum;
                }
                else
                {
                    Y_data[%(Ys1)s * mm] = %(beta)s * Y_in_data[%(Ys1_in)s * mm]
                        + %(alpha)s * ksum;
                }
            }
        }
    """ % kwargs
    return text, (kwargs['B'],), None


def _misc_gemv_1(kwargs):
    text = """
        __kernel void fn(__global const ${Atype} *A_data,
                         __global const ${Xtype} *X_data,
                         __global const ${Xitype} *Xi_data,
                         __global const ${Ytype} *Y_in_data,
                         __global ${Ytype} *Y_data)
        {
            const int mm = get_global_id(0);
            const int bb = get_global_id(1);

            A_data += ${Aoffset} + bb * ${As0} + mm * ${As1};
            X_data += ${Xoffset} + Xi_data[${Xioffset} + bb * ${Xis0}] * ${Xs0};
            Y_data += ${Yoffset} + bb * ${Ys0} + ${Ys1} * mm;
            Y_in_data += ${Y_in_offset} + bb * ${Ys0_in} + ${Ys1_in} * mm;

            ${Ytype} ksum = 0.0;

            for (int nn = 0; nn < ${N}; ++nn)
            {
                ksum += A_data[nn * ${As2}  ] * X_data[nn * ${Xs1}];
            }

        % if beta == 0:
            % if alpha == 1:
                Y_data[0] = ksum;
            % else:
                Y_data[0] = ${alpha} * ksum;
            % endif
        % else:
            Y_data[0] = ${beta} * Y_in_data[0] + ${alpha} * ksum;
        % endif
        }
    """
    text = Template(text).render(**kwargs)
    return text, (kwargs['M'], kwargs['B'],), None


def plan_misc_gemv(queue, alpha, A, X, Xi, beta, Y, Y_in=None, tag=None):
    if Y_in is None:
        Y_in = Y
    assert Y.dtype == Y_in.dtype

    B, M, N = A.shape
    assert N == X.shape[1]
    assert (B,) == Xi.shape
    assert (B, M) == Y.shape
    assert (B, M) == Y_in.shape

    As0, As1, As2 = A.itemstrides
    Xs0, Xs1, = X.itemstrides
    Xis0, = Xi.itemstrides
    Ys0, Ys1, = Y.itemstrides
    Ys0_in, Ys1_in, = Y_in.itemstrides

    Atype = A.ocldtype
    Xtype = X.ocldtype
    Xitype = Xi.ocldtype
    Ytype = Y.ocldtype

    # TODO: is there another way to do this other than retrieving all these
    # constants?
    Aoffset = A.offset
    Xoffset = X.offset
    Xioffset = Xi.offset
    Yoffset = Y.offset
    Y_in_offset = Y_in.offset

    if As1 == 1:
        name = '_misc_gemv_1'
    else:
        name = '_misc_gemv_ref'

    text, gsize, lsize = globals()[name](locals())

    _fn = cl.Program(queue.context, text).build().fn
    _fn.set_args(A.data, X.data, Xi.data, Y_in.data, Y.data)

    return Plan(queue, _fn, gsize, lsize,
                name=name,
                shape=(B, M, N),
                tag=tag,
               )


def plan_ragged_gather_gemv(queue, Ms, Ns, alpha, A, A_js, X, X_js,
                       beta, Y, Y_in=None, tag=None):
    """
    """
    # TODO: if alpha or beta is a float
    #       then render it into the kernel text.
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

    cl_alpha = to_device(queue, np.asarray(alpha, Y.buf.dtype))
    cl_beta = to_device(queue, np.asarray(beta, Y.buf.dtype))

    if Y_in is None:
        Y_in = Y

    # XXX check for e.g. all Ns being the same thing
    #     especially all Ns == 1
    cl_Ns = to_device(queue, np.asarray(Ns, 'int32'))

    # XXX check that all the ints are ints not longs
    textconf = {
        'type_alpha': cl_alpha.ocldtype,
        'type_beta': cl_beta.ocldtype,
        'type_A': A.buf.ocldtype,
        'type_X': X.buf.ocldtype,
        'type_Y': Y.buf.ocldtype,
    }

    text = """
        __kernel void fn(
            __global int *Ns,
            __global ${type_alpha} * alphas,
            __global int *A_starts,
            __global ${type_A} *A_data,
            __global int *A_js_starts,
            __global int *A_js_lens,
            __global int *A_js_data,
            __global int *X_starts,
            __global ${type_X} *X_data,
            __global int *X_js_starts,
            __global int *X_js_data,
            __global ${type_beta} * betas,
            __global int *Y_in_starts,
            __global ${type_Y} *Y_in_data,
            __global int *Y_starts,
            __global int *Y_lens,
            __global ${type_Y} *Y_data)
        {
            const int mm = get_global_id(0);
            const int bb = get_global_id(1);
            const int M = Y_lens[bb];
            if (mm < M)
            {
                const ${type_alpha} alpha = alphas[bb];
                const ${type_beta} beta = betas[bb];

                int n_dot_products = A_js_lens[bb];
                int y_offset = Y_starts[bb];
                int y_in_offset = Y_in_starts[bb];

                X_js_data += X_js_starts[bb];
                A_js_data += A_js_starts[bb];

                Y_data[y_offset + mm] = beta * Y_in_data[y_in_offset + mm];

                for (int ii = 0; ii < n_dot_products; ++ii)
                {
                    int x_ji = X_js_data[ii];
                    int a_ji = A_js_data[ii];
                    int N_i = Ns[a_ji];
                    int x_offset = X_starts[x_ji];
                    int a_offset = A_starts[a_ji];

                    // compute the matrix-vector product
                    // dot(X[x_ji], A[a_ji])
                    ${type_Y} y_sum = 0;
                    for (int nn = 0; nn < N_i; ++nn)
                    {
                        y_sum += X_data[x_offset + nn]
                        * A_data[a_offset + nn * M + mm];
                    }
                    Y_data[y_offset + mm] += alpha * y_sum;
                }
            }
        }
    """

    text = Template(text, output_encoding='ascii').render(**textconf)
    gsize = (int(max(Ms)), int(len(Y)),)
    lsize = None
    _fn = cl.Program(queue.context, text).build().fn
    full_args = (cl_Ns,
                 cl_alpha,
                 A.starts,
                 A.buf,
                 A_js.starts,
                 A_js.lens,
                 A_js.buf,
                 X.starts,
                 X.buf,
                 X_js.starts,
                 X_js.buf,
                 cl_beta,
                 Y_in.starts,
                 Y_in.buf,
                 Y.starts,
                 Y.lens,
                 Y.buf,
                )
    #print [str(arr.dtype)[0] for arr in full_args]
    _fn.set_args(*[arr.data for arr in full_args])
    rval = Plan(queue, _fn, gsize, lsize,
                name='ref_ragged_gather_gemv',
                tag=tag,
               )
    # prevent garbage-collection
    rval.alpha = cl_alpha
    rval.beta = cl_beta
    rval.Ns = cl_Ns
    return rval

