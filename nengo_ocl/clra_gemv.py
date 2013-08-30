import numpy as np
import pyopencl as cl
from plan import Plan
from mako.template import Template
from clarray import to_device


def plan_ragged_gather_gemv(queue, alpha, A, A_js, X, X_js,
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
    # cl_Ns = to_device(queue, np.asarray(Ns, 'int32'))

    # XXX check that all the ints are ints not longs
    textconf = {
        'type_alpha': cl_alpha.ocldtype,
        'type_beta': cl_beta.ocldtype,
        'type_A': A.cl_buf.ocldtype,
        'type_X': X.cl_buf.ocldtype,
        'type_Y': Y.cl_buf.ocldtype,
        'tag': str(tag),
        'do_inner_products': (A_js is not None),
    }

    text = """
        __kernel void fn(
            __global int *A_shape1s,
            __global ${type_alpha} * alphas,
            __global int *A_starts,
            __global int *A_ldas,
            __global ${type_A} *A_data,
            __global int *A_js_starts,
            __global int *A_js_shape0s,
            __global int *A_js_data,
            __global int *X_starts,
            __global ${type_X} *X_data,
            __global int *X_js_starts,
            __global int *X_js_data,
            __global ${type_beta} * betas,
            __global int *Y_in_starts,
            __global ${type_Y} *Y_in_data,
            __global int *Y_starts,
            __global int *Y_shape0s,
            __global ${type_Y} *Y_data)
        {
            const int mm = get_global_id(0);
            const int bb = get_global_id(1);
            const int M = Y_shape0s[bb];
            if (mm < M)
            {
                // printf("${tag}: mm=%i bb=%i\\n", mm, bb);

                const ${type_alpha} alpha = alphas[bb];
                const ${type_beta} beta = betas[bb];

                const int y_offset = Y_starts[bb];
                const int y_in_offset = Y_in_starts[bb];

                Y_data[y_offset + mm] = beta * Y_in_data[y_in_offset + mm];

                % if do_inner_products :
                const int n_dot_products = A_js_shape0s[bb];
                X_js_data += X_js_starts[bb];
                A_js_data += A_js_starts[bb];

                for (int ii = 0; ii < n_dot_products; ++ii)
                {
                    //printf("${tag}: ii=%i / %i\\n", ii, n_dot_products);
                    const int x_ji = X_js_data[ii];
                    const int a_ji = A_js_data[ii];
                    //printf("x_ji=%i a_ji=%i\\n", x_ji, a_ji);
                    const int N_i = A_shape1s[a_ji];
                    const int x_offset = X_starts[x_ji];
                    const int a_offset = A_starts[a_ji];
                    //printf("x_offset=%i a_offset=%i\\n", x_offset, a_offset);
                    const int lda_i = A_ldas[a_ji];


                    ${type_Y} y_sum = 0;
                    for (int nn = 0; nn < N_i; ++nn)
                    {
                        y_sum += X_data[x_offset + nn]
                                 * A_data[a_offset + nn * lda_i + mm];
                    }
                    Y_data[y_offset + mm] += alpha * y_sum;
                }
                % endif
            }
        }
    """

    text = Template(text, output_encoding='ascii').render(**textconf)
    if False: #tag == 'transforms':
        print text
        print A_js

    ### TODO: use the maximum of A.shape0s that is actually used in this op
    gsize = (int(max(A.shape0s)), int(len(Y)),)
    lsize = None
    _fn = cl.Program(queue.context, text).build().fn
    dummy = A.cl_buf
    full_args = (
        A.cl_shape1s,
        cl_alpha,
        A.cl_starts,
        A.cl_ldas,
        A.cl_buf,
        A_js.cl_starts if A_js is not None else dummy,
        A_js.cl_shape0s if A_js is not None else dummy,
        A_js.cl_buf if A_js is not None else dummy,
        X.cl_starts,
        X.cl_buf,
        X_js.cl_starts if A_js is not None else dummy,
        X_js.cl_buf if A_js is not None else dummy,
        cl_beta,
        Y_in.cl_starts,
        Y_in.cl_buf,
        Y.cl_starts,
        Y.cl_shape0s,
        Y.cl_buf)
    if 0:
        full_args = (
                 cl_alpha,
                 A.cl_starts,
                 A.cl_shape1s,
                 A.cl_ldas,
                 A.cl_buf,
                 A_js.cl_starts if A_js is not None else dummy,
                 A_js.cl_shape0s if A_js is not None else dummy,
                 A_js.cl_buf if A_js is not None else dummy,
                 X.cl_starts,
                 X.cl_buf,
                 X_js.cl_starts if X_js is not None else dummy,
                 X_js.cl_buf if X_js is not None else dummy,
                 cl_beta,
                 Y_in.cl_starts,
                 Y_in.cl_buf,
                 Y.cl_starts,
                 Y.cl_shape0s,
                 Y.cl_buf,
                )

    #print [str(arr.dtype)[0] for arr in full_args]
    _fn.set_args(*[arr.data for arr in full_args])
    rval = Plan(queue, _fn, gsize, lsize,
                name='ref_ragged_gather_gemv',
                tag=tag,
               )
    # prevent garbage-collection
    rval.full_args = full_args
    return rval

