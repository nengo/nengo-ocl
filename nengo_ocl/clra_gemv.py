import numpy as np
import pyopencl as cl
from plan import Plan
from mako.template import Template
from clarray import to_device
from clraggedarray import CLRaggedArray

def plan_parallel_ragged_gather_gemv2(queue,
    alpha, A, A_js, X, X_js,
    beta, Y, Y_in=None, tag=None,
    group_size=32,
    ):
    """
    """

    # TODO: if alpha or beta is a float
    #       then render it into the kernel text.
    try:
        alpha = float(alpha)
        cl_alpha = None
    except TypeError:
        cl_alpha = to_device(queue,
            np.asarray([alpha] * len(Y), Y.buf.dtype))

    try:
        beta = float(beta)
        cl_beta = None
    except TypeError:
        cl_beta = to_device(queue,
            np.asarray([beta] * len(Y), Y.buf.dtype))

    if cl_alpha or cl_beta:
        raise NotImplementedError('alpha or beta non-homogeneous')

    if Y_in is None:
        Y_in = Y

    # Make the global size the closest multiple of the group size (ceiling)
    gsize = (group_size, 512)
    lsize = (group_size, 1)

    #print gsize
    #print lsize

    if not np.all(shp0 == 1 for shp0 in Y.shape0s):
        raise NotImplementedError(Y.shape0s)

    # XXX check that all the ints are ints not longs
    textconf = {
        'alpha' : str(alpha),
        'beta' : str(beta),
        'type_alpha': None, #cl_alpha.ocldtype,
        'type_beta': None, #cl_beta.ocldtype,
        'type_A': A.cl_buf.ocldtype,
        'type_X': X.cl_buf.ocldtype,
        'type_Y': Y.cl_buf.ocldtype,
        'y_len': len(Y),
        'lsize': group_size,
        'MAX_N_DOT_PRODUCTS': max(A_js.shape0s),
    }

    text = """
        __kernel void fn(
            //const __global ${type_alpha} * alphas,
            const __global int *A_starts,
            const __global int *A_shape1s,
            const __global int *A_ldas,
            const __global ${type_A} *A_data,
            const __global int *A_js_starts,
            const __global int *A_js_lens,
            const __global int *A_js_data,
            const __global int *X_starts,
            const __global ${type_X} *X_data,
            const __global int *X_js_starts,
            const __global int *X_js_data,
            //const __global ${type_beta} * betas,
            const __global int *Y_in_starts,
            const __global ${type_Y} *Y_in_data,
            const __global int *Y_starts,
            const __global int *Y_lens,
            __global ${type_Y} *Y_data)
    {
        __local ${type_Y} partialDotProduct[${lsize}]; //Scratch space for the dot products
        __local int y_offset;
        __local int n_dot_products;
        __local int A_js_offset;
        __local int X_js_offset;
        __local int A_js_row[${MAX_N_DOT_PRODUCTS}];
        __local int X_js_row[${MAX_N_DOT_PRODUCTS}];
        __local ${type_Y} y_sum_pre;
        ${type_Y} y_sum_post;

        for (int bb = get_global_id(1);
                 bb < ${y_len};
                 bb += get_global_size(1))
        {
            const int mm = 0; // TODO: SUPPORT M > 1

            if (get_local_id(0) < n_dot_products)
            {
                if (get_local_id(0) == 0)
                {
                    y_offset = Y_starts[bb];
                    n_dot_products = A_js_lens[bb];
                    A_js_offset = A_js_starts[bb];
                    X_js_offset = X_js_starts[bb];

                    % if beta != 0 :
                        y_sum_pre = ${beta} * Y_in_data[Y_in_starts[bb]];
                    % else :
                        y_sum_pre = 0;
                    % endif
                }

                // a barrier here risks *deadlock*
                // instead we assume that n_dot_products < warp size
                // barrier(CLK_LOCAL_MEM_FENCE); // need A_js_offset to be loaded
                A_js_row[get_local_id(0)] = A_js_data[A_js_offset + get_local_id(0)];
                X_js_row[get_local_id(0)] = X_js_data[X_js_offset + get_local_id(0)];
            }
            // This barrier cannot cause deadlock because
            // the whole local work group is devoted to the same value of bb
            // so they are all here, or none here.
            barrier(CLK_LOCAL_MEM_FENCE); // need A_js_row to be loaded
            y_sum_post = 0;

            for(int ii = 0; ii < n_dot_products; ii++) {
                const int a_ji = A_js_row[ii];
                const int x_ji = X_js_row[ii];
                const int N_i = A_shape1s[a_ji];
                const int lda_i = A_ldas[a_ji];

                const __global ${type_A}* A_row = A_data + A_starts[a_ji];
                const __global ${type_X}* X_row = X_data + X_starts[x_ji];

                for (int nn = get_local_id(0); nn < N_i; nn += get_local_size(0)) {
                    y_sum_post += A_row[nn * lda_i + mm] * X_row[nn];
                }
            }

            // -- Parallel reduction within local group sum registers
            barrier(CLK_LOCAL_MEM_FENCE);
            partialDotProduct[get_local_id(0)] = y_sum_post;

            for (uint stride = 1; stride < get_local_size(0); stride *= 2) {
                // XXX not necessary IFF local_size(0) < warp size
                barrier(CLK_LOCAL_MEM_FENCE);

                uint index = 2 * stride * get_local_id(0);
                if (index + stride < get_local_size(0)) {
                    partialDotProduct[index] += partialDotProduct[index + stride];
                }
            }
            if (get_local_id(0) == 0) {
                Y_data[y_offset] = y_sum_pre + ${alpha} * partialDotProduct[0];
            }
        }
    }
    """

    text = Template(text, output_encoding='ascii').render(**textconf)

    _fn = cl.Program(queue.context, text).build().fn

    full_args = (
                 #cl_alpha,
                 A.cl_starts,
                 A.cl_shape1s,
                 A.cl_ldas,
                 A.cl_buf,
                 A_js.cl_starts,
                 A_js.cl_shape0s,
                 A_js.cl_buf,
                 X.cl_starts,
                 X.cl_buf,
                 X_js.cl_starts,
                 X_js.cl_buf,
                 #cl_beta,
                 Y_in.cl_starts,
                 Y_in.cl_buf,
                 Y.cl_starts,
                 Y.cl_shape0s,
                 Y.cl_buf,
                )

    _fn.set_args(*[arr.data for arr in full_args])

    rval = Plan(queue, _fn, gsize, lsize,
                name='ref_parallel_ragged_gather_gemv',
                tag=tag,
               )
    # prevent garbage-collection
    rval.full_args = full_args
    return rval


def plan_ragged_gather_gemv(queue, alpha, A, A_js, X, X_js,
                            beta, Y, Y_in=None, tag=None, seq=None, gamma=0.0):
    """
    """

    # TODO: if alpha or beta is a float
    #       then render it into the kernel text.
    try:
        float(alpha)
        alpha = [alpha] * len(Y)
    except TypeError:
        pass
    cl_alpha = to_device(queue, np.asarray(alpha, Y.buf.dtype))

    clra_beta = None
    float_beta = None
    cl_beta = None
    if isinstance(beta, CLRaggedArray):
        clra_beta = beta
        type_beta = beta.cl_buf.ocldtype
    elif isinstance(beta, float):
        float_beta = beta
        type_beta = Y.cl_buf.ocldtype
    elif len(set(beta)) == 1:
        float_beta = beta[0]
        type_beta = Y.cl_buf.ocldtype
    else:
        cl_beta = to_device(queue, np.asarray(beta, Y.dtype))
        type_beta = Y.cl_buf.ocldtype

    clra_gamma = None
    float_gamma = None
    cl_gamma = None
    if isinstance(gamma, CLRaggedArray):
        clra_gamma = gamma
        type_gamma = gamma.cl_buf.ocldtype
    elif isinstance(gamma, float):
        float_gamma = gamma
        type_gamma = Y.cl_buf.ocldtype
    elif len(set(gamma)) == 1:
        float_gamma = gamma[0]
        type_gamma = Y.cl_buf.ocldtype
    else:
        cl_gamma = to_device(queue, np.asarray(gamma, Y.dtype))
        type_gamma = Y.cl_buf.ocldtype

   # if float_gamma == 0 and float_beta is not None:
   #     if len(alpha) == 1:
   #         alpha = alpha[0]
   #     return plan_parallel_ragged_gather_gemv3(queue, alpha, A, A_js, X, X_js,
   #             float_beta, Y, Y_in, tag, seq=seq)
   # elif float_gamma == 0 and cl_beta is not None:
   #     if len(alpha) == 1:
   #         alpha = alpha[0]
   #     return plan_parallel_ragged_gather_gemv3(queue, alpha, A, A_js, X, X_js,
   #             cl_beta.get(), Y, Y_in, tag, seq=seq)
   # else:
   #     print 'not gemv3', float_gamma, float_beta

    if Y_in is None:
        Y_in = Y

    # XXX check for e.g. all Ns being the same thing
    #     especially all Ns == 1
    # cl_Ns = to_device(queue, np.asarray(Ns, 'int32'))

    # XXX check that all the ints are ints not longs
    textconf = {
        'type_alpha': cl_alpha.ocldtype,
        'type_beta': type_beta,
        'type_gamma': type_gamma,
        'type_A': A.cl_buf.ocldtype,
        'type_X': X.cl_buf.ocldtype,
        'type_Y': Y.cl_buf.ocldtype,
        'tag': str(tag),
        'do_inner_products': (A_js is not None),
        'clra_beta': clra_beta,
        'float_beta': float_beta,
        'cl_beta': cl_beta,
        'clra_gamma': clra_gamma,
        'float_gamma': float_gamma,
        'cl_gamma': cl_gamma,
    }

    text = """
        __kernel void fn(
            __global ${type_alpha} * alphas,
            __global int *A_starts,
            __global int *A_shape1s,
            __global int *A_ldas,
            __global ${type_A} *A_data,
            __global int *A_js_starts,
            __global int *A_js_shape0s,
            __global int *A_js_data,
            __global int *X_starts,
            __global ${type_X} *X_data,
            __global int *X_js_starts,
            __global int *X_js_data,
            % if cl_beta is not None:
            __global ${type_beta} * betas,
            % endif
            % if clra_beta is not None:
            __global int *beta_starts,
            __global int *beta_data,
            % endif
            % if cl_gamma is not None:
            __global ${type_gamma} * gammas,
            % endif
            % if clra_gamma is not None:
            __global int *gamma_starts,
            __global int *gamma_data,
            % endif
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
                const int y_offset = Y_starts[bb];
                const int y_in_offset = Y_in_starts[bb];

                % if float_beta is not None:
                const ${type_beta} beta = ${float_beta};
                % endif
                % if cl_beta is not None:
                const ${type_beta} beta = betas[bb];
                % endif
                % if clra_beta is not None:
                const int beta_offset = beta_starts[bb];
                const ${type_beta} beta = beta_data[beta_offset + mm];
                % endif

                % if float_gamma is not None:
                const ${type_gamma} gamma = ${float_gamma};
                % endif
                % if cl_gamma is not None:
                const ${type_gamma} gamma = gammas[bb];
                % endif
                % if clra_gamma is not None:
                const int gamma_offset = gamma_starts[bb];
                const ${type_gamma} gamma = gamma_data[gamma_offset + mm];
                % endif

                Y_data[y_offset + mm] = gamma + beta * Y_in_data[y_in_offset + mm];

    % if do_inner_products :

                const int n_dot_products = A_js_shape0s[bb];
                X_js_data += X_js_starts[bb];
                A_js_data += A_js_starts[bb];

                ${type_Y} y_sum = 0;
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

                    for (int nn = 0; nn < N_i; ++nn)
                        y_sum += X_data[x_offset + nn]
                                 * A_data[a_offset + nn * lda_i + mm];
                }
                const ${type_alpha} alpha = alphas[bb];
                Y_data[y_offset + mm] += alpha * y_sum;
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
    full_args = [
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
        ]
    if cl_beta is not None:
        full_args += [cl_beta]
    elif clra_beta is not None:
        full_args += [clra_beta.cl_starts, clra_beta.cl_buf]
    if cl_gamma is not None:
        full_args += [cl_gamma]
    elif clra_gamma is not None:
        full_args += [clra_gamma.cl_starts, clra_gamma.cl_buf]
    full_args += [
        Y_in.cl_starts,
        Y_in.cl_buf,
        Y.cl_starts,
        Y.cl_shape0s,
        Y.cl_buf]

    #print [str(arr.dtype)[0] for arr in full_args]
    _fn.set_args(*[arr.data for arr in full_args])
    rval = Plan(queue, _fn, gsize, lsize,
                name="ref_ragged_gather_gemv",
                tag=tag,
               )
    # prevent garbage-collection
    rval.full_args = full_args
    return rval


def plan_parallel_ragged_gather_gemv3(queue,
    alpha, A, A_js, X, X_js,
    beta,
    Y, Y_in=None, tag=None,
    group_size=32,
    seq=None,
    ):
    """
    """

    print alpha
    try:
        alpha = float(alpha)
        cl_alpha = None
    except TypeError:
        cl_alpha = to_device(queue,
            np.asarray([alpha] * len(Y), Y.buf.dtype))

    try:
        float_beta = float(beta)
        cl_beta = None
    except TypeError:
        float_beta = None
        cl_beta = to_device(queue,
            np.asarray(beta, Y.buf.dtype))

    if cl_alpha:
        raise NotImplementedError('alpha non-homogeneous')

    if Y_in is None:
        Y_in = Y

    # Make the global size the closest multiple of the group size (ceiling)
    gsize = (group_size, 1024)
    lsize = (group_size, 1)

    if not np.all(shp0 == 1 for shp0 in Y.shape0s):
        raise NotImplementedError(Y.shape0s)

    max_n_dots = max(A_js.shape0s)
    n_structure_vars = 4 * max_n_dots + 3;

    gstructure = np.zeros((len(Y), n_structure_vars), dtype='int32')
    A_starts = A.starts
    A_ldas = A.ldas
    A_shape1s = A.shape1s
    X_starts = X.starts
    Y_in_starts = Y_in.starts
    Y_starts = Y.starts

    for bb in range(len(Y)):
        x_js_i = X_js[bb]
        A_js_i = A_js[bb]
        assert len(x_js_i) == len(A_js_i)
        gstructure[bb, 4 * max_n_dots + 0] = Y_in.starts[bb]
        gstructure[bb, 4 * max_n_dots + 1] = Y_starts[bb]
        gstructure[bb, 4 * max_n_dots + 2] = len(A_js_i)
        for ii, (xi, ai) in enumerate(zip(x_js_i, A_js_i)):
            gstructure[bb, 0 * max_n_dots + ii] = X_starts[xi]
            gstructure[bb, 1 * max_n_dots + ii] = A_starts[ai]
            gstructure[bb, 2 * max_n_dots + ii] = A_ldas[ai]
            gstructure[bb, 3 * max_n_dots + ii] = A_shape1s[ai]
    cl_gstructure = to_device(queue, gstructure)

    textconf = {
        'alpha' : str(alpha),
        'float_beta' : float_beta,
        'type_alpha': None, #cl_alpha.ocldtype,
        'type_beta': None if cl_beta is None else cl_beta.ocldtype,
        'cl_beta': cl_beta,
        'type_A': A.cl_buf.ocldtype,
        'type_X': X.cl_buf.ocldtype,
        'type_Y': Y.cl_buf.ocldtype,
        'y_len': len(Y),
        'lsize': group_size,
        'n_structure_vars': n_structure_vars,
        'structure_stride': n_structure_vars,
        'MAX_N_DOT_PRODUCTS': max_n_dots,
        'x_starts': 'lstructure[0 * %s + ii]' % max_n_dots,
        'a_starts': 'lstructure[1 * %s + ii]' % max_n_dots,
        'a_lda'   : 'lstructure[2 * %s + ii]' % max_n_dots,
        'N_i'     : 'lstructure[3 * %s + ii]' % max_n_dots,
        'y_in_starts': 'lstructure[4 * %s + 0]' % max_n_dots,
        'y_offset': 'lstructure[4 * %s + 1]' % max_n_dots,
        'n_dot_products': 'lstructure[4 * %s + 2]' % max_n_dots,
    }

    text = """
        __kernel void fn(
            const __global int *gstructure,
            const __global ${type_A} *A_data,
            const __global ${type_X} *X_data,
            % if cl_beta is not None:
            const __global ${type_beta} * betas,
            % endif
            const __global ${type_Y} *Y_in_data,
            __global ${type_Y} *Y_data)
    {
        __local int lstructure[${n_structure_vars}];
        __local ${type_Y} partialDotProduct[${lsize}]; //Scratch space for the dot products
        __local ${type_Y} y_sum_pre;
        ${type_Y} y_sum_post;

        for (int bb = get_global_id(1);
                 bb < ${y_len};
                 bb += get_global_size(1))
        {
            const int mm = 0; // TODO: SUPPORT M > 1

            barrier(CLK_LOCAL_MEM_FENCE);
            if (get_local_id(0) < ${n_structure_vars})
            {
                lstructure[get_local_id(0)] = gstructure[
                    bb * ${structure_stride} + get_local_id(0)];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if (get_local_id(0) == 0)
            {
                % if float_beta is not None and float_beta != 0 :
                    y_sum_pre = ${beta} * Y_in_data[${y_in_starts}];
                % elif cl_beta is not None:
                    y_sum_pre = betas[bb] * Y_in_data[${y_in_starts}];
                % else :
                    y_sum_pre = 0;
                % endif
            }
            y_sum_post = 0;

            for(int ii = 0; ii < ${n_dot_products}; ii++) {

                for (int nn = get_local_id(0); nn < ${N_i}; nn += get_local_size(0)) {
                    y_sum_post +=
                        A_data[${a_starts} + nn * ${a_lda} + mm]
                      * X_data[${x_starts} + nn];
                }
            }

            // -- Parallel reduction within local group sum registers
            partialDotProduct[get_local_id(0)] = y_sum_post;

            for (uint stride = 1; stride < get_local_size(0); stride *= 2) {
                // XXX not necessary IFF local_size(0) < warp size
                barrier(CLK_LOCAL_MEM_FENCE);

                uint index = 2 * stride * get_local_id(0);
                if (index + stride < get_local_size(0)) {
                    partialDotProduct[index] += partialDotProduct[index + stride];
                }
            }
            if (get_local_id(0) == 0) {
                Y_data[${y_offset}] = y_sum_pre + ${alpha} * partialDotProduct[0];
            }
        }
    }
        """

    text = Template(text, output_encoding='ascii').render(**textconf)

    _fn = cl.Program(queue.context, text).build().fn

    full_args = [
                 cl_gstructure,
                 A.cl_buf,
                 X.cl_buf,
                 ]
    if cl_beta is not None:
        full_args += [cl_beta]
    full_args += [
                 Y_in.cl_buf,
                 Y.cl_buf,
                ]

    _fn.set_args(*[arr.data for arr in full_args])

    rval = Plan(queue, _fn, gsize, lsize,
                name='parallel_ragged_gather_gemv3',
                tag=tag,
               )
    # prevent garbage-collection
    rval.full_args = full_args
    return rval
