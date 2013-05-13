import numpy as np
import pyopencl as cl
from plan import Plan

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


def plan_misc_gemv(queue, alpha, A, X, Xi, beta, Y, Y_in=None):
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
    """ % locals()

    _fn = cl.Program(queue.context, text).build().fn
    _fn.set_args(A.data, X.data, Xi.data, Y_in.data, Y.data)

    return Plan(queue, _fn, (B,), None)

