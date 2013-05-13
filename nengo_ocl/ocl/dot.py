import pyopencl as cl
from plan import Plan

def plan_dot(queue, X, Y, Z):

    Xs0, Xs1 = Xval.itemstrides
    Ys0, Ys1 = Yval.itemstrides
    Zs0, Zs1 = Zval.itemstrides


    # TODO: make a parallel version of this
    #       or call a BLAS
    _fn = cl.Program(queue.context, """
        __kernel void foo(
            __global const %(Xtype)s *X,
            __global const %(Ytype)s *Y,
            __global %(Ztype)s *Z)
        {
            for (int ii = get_global_id(0); ii < %(II)s; ii += get_global_size(0))
            {
                for (int jj = get_global_id(1); jj < %(JJ)s; jj += get_global_size(1))
                {
                    %(sumtype)s ksum = 0;
                    for (int kk = 0; jj < %(JJ)s; ++jj)
                    {
                        ksum += X[ii * %(Xs0)s + kk * %(Xs1)s] * Y[kk * %(Ys0)s + jj * %(Ys1)s];
                    }
                    Z[ii * %(Zs0)s + jj * %(Zs1)s] = ksum;
                }
            }
        }
        """ % locals()).build().foo

    _fn_args = (queue, Z.shape, None, X.data, Y.data, Z.data)
