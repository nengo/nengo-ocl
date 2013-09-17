from collections import defaultdict
import numpy as np
import pyopencl as cl
from plan import Plan, Prog
from mako.template import Template
from clarray import to_device
from clraggedarray import CLRaggedArray

def dhist(seq):
    """Return the counts of entries in `seq` in a defaultdict.
    """
    d = defaultdict(int)
    for obj in seq:
        d[obj] += 1
    return d


def float_cl_clra(queue, arg, cl_dtype, N):
    float_arg = None
    cl_arg = None
    clra_arg = None
    if isinstance(arg, CLRaggedArray):
        clra_arg = arg
        assert arg.dtype == cl_dtype
    elif isinstance(arg, float):
        float_arg = arg
    elif len(set(arg)) == 1:
        float_arg = arg[0]
    else:
        host_arg = np.asarray(arg, cl_dtype)
        assert host_arg.shape == (N,)
        cl_arg = to_device(queue, host_arg)
    return float_arg, cl_arg, clra_arg


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


class plan_ragged_gather_gemv(Prog):
    def __init__(self,
            queue, alpha, A, A_js, X, X_js,
            beta, Y, Y_in=None, tag=None, seq=None, gamma=0.0):
        """
        """

        self.float_alpha, self.cl_alpha, self.clra_alpha = \
                float_cl_clra(queue, alpha, Y.dtype, len(Y))
        self.float_beta, self.cl_beta, self.clra_beta = \
                float_cl_clra(queue, beta, Y.dtype, len(Y))
        self.float_gamma, self.cl_gamma, self.clra_gamma = \
                float_cl_clra(queue, gamma, Y.dtype, len(Y))

        if Y_in is None:
            self.Y_in = Y
        else:
            self.Y_in = Y_in

        self.queue = queue
        self.A = A
        self.A_js = A_js
        self.X = X
        self.X_js = X_js
        self.Y = Y
        self.tag = tag
        self.seq = seq

        self.quick_geometry_summary()
        A_shape1s = self.A.shape1s
        reduce_lens = dhist(A_shape1s[j] for j in self.A_js.buf)

        ref_plan = ref_impl(self)
        Prog.__init__(self, [ref_plan])

    def quick_geometry_summary(self):
        print 'quick_geometry_summary: tag=%s' % self.tag
        gg = self.geometry()
        print 'Y lens', dhist(ggi['y_len'] for ggi in gg)
        print 'ndots', dhist(len(ggi['dots']) for ggi in gg)
        Ks = []
        for ggi in gg:
            Ks.extend(ddb['a_shape1'] for ddb in ggi['dots'])
        print 'Ks', dhist(Ks)

    def geometry(self):
        A_starts = self.A.starts
        X_starts = self.X.starts
        Y_starts = self.Y.starts
        Y_in_starts = self.Y_in.starts
        A_stride0s = self.A.stride0s
        A_shape1s = self.A.shape1s
        Y_shape0s = self.Y.shape0s

        rval = []
        for bb in range(len(Y_shape0s)):
            x_js_i = self.X_js[bb]
            A_js_i = self.A_js[bb]
            dbb = {
                'y_len': Y_shape0s[bb],
                'dots': [],
                'y_start': Y_starts[bb],
                'y_in_start': Y_in_starts[bb],
                    }
            assert len(x_js_i) == len(A_js_i)
            for jj, (xj, aj) in enumerate(zip(x_js_i, A_js_i)):
                dbb['dots'].append({
                    'j': jj,
                    'x_j': xj,
                    'a_j': aj,
                    'x_start': X_starts[xj],
                    'a_start': A_starts[aj],
                    'a_stride0': A_stride0s[aj],
                    'a_shape1': A_shape1s[aj],
                })
            rval.append(dbb)
        return rval


def ref_impl(p):
    if p.clra_alpha is not None:
        raise NotImplementedError()
    if p.clra_gamma is not None:
        raise NotImplementedError()

    assert all(s == 1 for s in p.A.stride1s)

    if 0:
        print 'Y: ' + ' '.join(['%2s' % s for s in p.Y.shape0s])
        #print 'Y: ' + ' '.join(['%2s' % s for s in p.Y.shape1s])
        #print 'Y: ' + ' '.join(['%2s' % s for s in p.Y.stride0s])
        #print 'Y: ' + ' '.join(['%2s' % s for s in p.Y.stride1s])

    if 0:
        print 'Y_in: ' + ''
        #print 'Y_in: ' + ' '.join(['%2s' % s for s in p.Y_in.shape0s])
        #print 'Y_in: ' + ' '.join(['%2s' % s for s in p.Y_in.shape1s])
        #print 'Y_in: ' + ' '.join(['%2s' % s for s in p.Y_in.stride0s])
        #print 'Y_in: ' + ' '.join(['%2s' % s for s in p.Y_in.stride1s])

        print 'A: ' + ''
        print 'A: ' + ' '.join(['%2s' % s for s in p.A.shape0s])
        print 'A: ' + ' '.join(['%2s' % s for s in p.A.shape1s])
        print 'A: ' + ' '.join(['%2s' % s for s in p.A.stride0s])
        print 'A: ' + ' '.join(['%2s' % s for s in p.A.stride1s])

        print 'X: ' + ''
        print 'X: ' + ' '.join(['%2s' % s for s in p.X.shape0s])
        print 'X: ' + ' '.join(['%2s' % s for s in p.X.shape1s])
        print 'X: ' + ' '.join(['%2s' % s for s in p.X.stride0s])
        print 'X: ' + ' '.join(['%2s' % s for s in p.X.stride1s])
    assert all(s == 1 for s in p.X.stride1s)

    assert all(s == 1 for s in p.Y.stride0s)
    assert all(s == 1 for s in p.Y.stride1s)

    assert all(s == 1 for s in p.Y_in.stride0s)
    assert all(s == 1 for s in p.Y_in.stride1s)

    text = """
        __kernel void fn(
    % if cl_alpha is not None:
            __global ${cl_alpha.ocldtype} * alphas,
    % endif
    % if (A_js is not None):
            __global int *A_starts,
            __global int *A_shape1s,
            __global int *A_stride0s,
            __global ${A.cl_buf.ocldtype} *A_data,
            __global int *A_js_starts,
            __global int *A_js_shape0s,
            __global int *A_js_data,
            __global int *X_starts,
            __global int *X_stride0s,
            __global ${X.cl_buf.ocldtype} *X_data,
            __global int *X_js_starts,
            __global int *X_js_data,
    % endif
    % if cl_beta is not None:
            __global ${cl_beta.ocldtype} * betas,
    % endif
    % if clra_beta is not None:
            __global int *beta_starts,
            __global int *beta_data,
    % endif
    % if cl_gamma is not None:
            __global ${cl_gamma.ocldtype} * gammas,
    % endif
            __global int *Y_in_starts,
            __global ${Y_in.cl_buf.ocldtype} *Y_in_data,
            __global int *Y_starts,
            __global int *Y_shape0s,
            __global ${Y.cl_buf.ocldtype} *Y_data)
        {
            const int mm = get_global_id(0);
            const int bb = get_global_id(1);
            const int M = Y_shape0s[bb];
            if (mm < M)
            {
                const int y_offset = Y_starts[bb];
                const int y_in_offset = Y_in_starts[bb];

    % if float_beta is not None:
                const ${Y.cl_buf.ocldtype} beta = ${float_beta};
    % elif cl_beta is not None:
                const ${cl_beta.ocldtype} beta = betas[bb];
    % elif clra_beta is not None:
                const int beta_offset = beta_starts[bb];
                const ${clra_beta.cl_buf.ocldtype} beta
                    = beta_data[beta_offset + mm];
    % endif

    % if float_gamma is not None:
                const ${Y.cl_buf.ocldtype} gamma = ${float_gamma};
    % elif cl_gamma is not None:
                const ${cl_gamma.ocldtype} gamma = gammas[bb];
    % endif

                Y_data[y_offset + mm] = gamma + beta * Y_in_data[y_in_offset + mm];

    % if (A_js is not None) :

                const int n_dot_products = A_js_shape0s[bb];
                X_js_data += X_js_starts[bb];
                A_js_data += A_js_starts[bb];

                ${Y.cl_buf.ocldtype} y_sum = 0;
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
                    const int AsM = A_stride0s[a_ji];
                    const int XsM = X_stride0s[x_ji];

                    for (int nn = 0; nn < N_i; ++nn)
                        y_sum += X_data[x_offset + nn * XsM]
                                 * A_data[a_offset + mm * AsM + nn];
                }
        % if float_alpha is not None:
        % if 0:
                printf("float_alpha ysum=%f %i %i %i %f\\n",
                    y_sum,
                    mm, bb, y_offset,
                    Y_data[y_offset + mm]);
                // return;
        % endif
                Y_data[y_offset + mm] += ${float_alpha} * y_sum;
        % elif cl_alpha is not None:
        % if 0:
                printf("cl_alpha ysum=%f %f %i %i %i %f\\n",
                    y_sum,
                    alphas[bb],
                    mm, bb, y_offset,
                    Y_data[y_offset + mm]
                    );
                //return;
        % endif
                Y_data[y_offset + mm] += alphas[bb] * y_sum;
        % endif
    % endif
            }

        }
    """

    text = Template(text, output_encoding='ascii').render(**p.__dict__)
    #print text

    ### TODO: use the maximum of A.shape0s that is actually used in this op
    gsize = (int(max(p.A.shape0s)), int(len(p.Y)),)
    lsize = None
    fn = cl.Program(p.queue.context, text).build().fn
    full_args = []
    if p.cl_alpha is not None:
        full_args += [p.cl_alpha]
    if p.A_js is not None:
        full_args += [
            p.A.cl_starts,
            p.A.cl_shape1s,
            p.A.cl_stride0s,
            p.A.cl_buf,
            p.A_js.cl_starts,
            p.A_js.cl_shape0s,
            p.A_js.cl_buf,
            p.X.cl_starts,
            p.X.cl_stride0s,
            p.X.cl_buf,
            p.X_js.cl_starts,
            p.X_js.cl_buf,
            ]
    if p.cl_beta is not None:
        full_args += [p.cl_beta]
    elif p.clra_beta is not None:
        full_args += [p.clra_beta.cl_starts, clra_beta.cl_buf]

    if p.cl_gamma is not None:
        full_args += [p.cl_gamma]
    elif p.clra_gamma is not None:
        full_args += [p.clra_gamma.cl_starts, clra_gamma.cl_buf]

    full_args += [
        p.Y_in.cl_starts,
        p.Y_in.cl_buf,
        p.Y.cl_starts,
        p.Y.cl_shape0s,
        p.Y.cl_buf]

    #print [str(arr.dtype)[0] for arr in full_args]
    fn.set_args(*[arr.data for arr in full_args])
    rval = Plan(p.queue, fn, gsize, lsize, name="ref_ragged_gather_gemv",
            tag=p.tag)
    rval.full_args = full_args  # prevent GC the args
    return rval


def reduce_impl(p, group_size):
    if p.clra_alpha is not None:
        raise NotImplementedError()
    if p.clra_gamma is not None:
        raise NotImplementedError()
    if p.clra_beta is not None:
        raise NotImplementedError()
    if p.cl_alpha is not None:
        raise NotImplementedError()
    if p.cl_gamma is not None:
        raise NotImplementedError()
    if not all(s == 1 for s in p.A.stride1s):
        raise NotImplementedError()

    assert p.float_alpha is not None
    assert p.float_gamma is not None

    # -- this is a print
    p.quick_geometry_summary()

    A_js_shape0s = p.A_js.shape0s
    max_n_dots = max(A_js_shape0s)
    n_structure_vars = 4 * max_n_dots + 4
    BB = len(p.Y)

    gstructure = np.zeros((BB, n_structure_vars), dtype='int32')
    A_starts = p.A.starts
    X_starts = p.X.starts
    Y_starts = p.Y.starts
    Y_in_starts = p.Y_in.starts
    A_stride0s = p.A.stride0s
    A_shape1s = p.A.shape1s
    Y_shape0s = p.Y.shape0s

    max_reduce_len = 0
    for bb in range(BB):
        x_js_i = p.X_js[bb]
        A_js_i = p.A_js[bb]
        assert len(x_js_i) == len(A_js_i)
        for ii, (xi, ai) in enumerate(zip(x_js_i, A_js_i)):
            gstructure[bb, 0 * max_n_dots + ii] = X_starts[xi]
            gstructure[bb, 1 * max_n_dots + ii] = A_starts[ai]
            gstructure[bb, 2 * max_n_dots + ii] = A_stride0s[ai]
            gstructure[bb, 3 * max_n_dots + ii] = A_shape1s[ai]
            max_reduce_len = max(max_reduce_len, A_shape1s[ai])
        # -- offset of output and input buffers
        gstructure[bb, 4 * max_n_dots + 0] = Y_in_starts[bb]
        gstructure[bb, 4 * max_n_dots + 1] = Y_starts[bb]
        # -- number of dots for bb
        gstructure[bb, 4 * max_n_dots + 2] = len(A_js_i)
        # -- length of Y[bb]
        gstructure[bb, 4 * max_n_dots + 3] = Y_shape0s[bb]
    cl_gstructure = to_device(p.queue, gstructure)

    # Make the global size the closest multiple of the group size (ceiling)
    max_y_len = max(gstructure[:, -1])
    reductions_per_block = min(max_y_len, 2)
    gsize = (group_size, reductions_per_block, min(BB, 512))
    lsize = (group_size, reductions_per_block, 1)

    textconf = {
        'RPB'     : reductions_per_block,
        'group_size' : group_size,
        'local_count': group_size * reductions_per_block,
        'max_reduce_len': max_reduce_len,
        'n_structure_vars': n_structure_vars,
        'structure_stride': n_structure_vars,
        'MAX_N_DOT_PRODUCTS': max_n_dots,
        'x_starts': 'lstructure[0 * %s + ii]' % max_n_dots,
        'a_starts': 'lstructure[1 * %s + ii]' % max_n_dots,
        'a_s0'    : 'lstructure[2 * %s + ii]' % max_n_dots,
        'N_i'     : 'lstructure[3 * %s + ii]' % max_n_dots,
        'y_in_starts': 'lstructure[4 * %s + 0]' % max_n_dots,
        'y_offset': 'lstructure[4 * %s + 1]' % max_n_dots,
        'n_dot_products': 'lstructure[4 * %s + 2]' % max_n_dots,
        'y_len'   : 'lstructure[4 * %s + 3]' % max_n_dots,
    }
    textconf.update(p.__dict__)

    assert n_structure_vars <= group_size

    text = """
        __kernel void fn(
            const __global int *gstructure,
            const __global ${A.cl_buf.ocldtype} *A_data,
            const __global ${X.cl_buf.ocldtype} *X_data,
            % if cl_beta is not None:
            const __global ${cl_beta.ocldtype} * betas,
            % endif
            const __global ${Y_in.cl_buf.ocldtype} *Y_in_data,
            __global ${Y.cl_buf.ocldtype} *Y_data)
    {
        __local int lstructure[${n_structure_vars}];
        __local ${X.cl_buf.ocldtype} lX[${max_reduce_len}];
        //Scratch space for the dot products
        __local ${Y.cl_buf.ocldtype} partialDotProduct[${local_count}];
        __local ${Y.cl_buf.ocldtype} y_sum_pre[${RPB}];
        int local_idx = get_local_id(0) + ${group_size} * get_local_id(1);

        for (int bb = get_global_id(2);
                 bb < ${len(Y)};
                 bb += get_global_size(2))
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            if (local_idx < ${n_structure_vars})
            {
                lstructure[local_idx] = gstructure[
                    bb * ${structure_stride} + local_idx];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if (get_local_id(0) == 0)
            {
                % if float_beta is not None and float_beta != 0 :
                    y_sum_pre[get_local_id(1)] = ${float_beta} * Y_in_data[${y_in_starts}];
                % elif cl_beta is not None:
                    y_sum_pre[get_local_id(1)] = betas[bb] * Y_in_data[${y_in_starts}];
                % else :
                    y_sum_pre[get_local_id(1)] = 0;
                % endif

                % if float_gamma is not None and float_gamma != 0:
                y_sum_pre[get_local_id(1)] += ${float_gamma};
                % endif
            }

            for(int ii = 0; ii < ${n_dot_products}; ii++) {
                // load X into shared memory
                barrier(CLK_LOCAL_MEM_FENCE);
                for (int nn = local_idx; nn < ${N_i}; nn += ${local_count})
                {
                    lX[nn] = X_data[${x_starts} + nn];
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                for (int mm = get_global_id(1);
                         mm < ${y_len};
                         mm += get_global_size(1))
                {
                    partialDotProduct[local_idx] = 0;
                    for (int nn = get_local_id(0);
                             nn < ${N_i};
                             nn += get_local_size(0)) {
                        partialDotProduct[local_idx] +=
                            A_data[${a_starts} + mm * ${a_s0} + nn] * lX[nn];
                    }
                    // -- Parallel reduction within local group sum registers
                    for (uint stride = 1;
                              stride < get_local_size(0);
                              stride *= 2)
                    {
                        //barrier(CLK_LOCAL_MEM_FENCE);

                        uint index = 2 * stride * get_local_id(0)
                            + ${group_size} * get_local_id(1);
                        if (index + stride < get_local_size(0)) {
                            partialDotProduct[index] +=
                                partialDotProduct[index + stride];
                        }
                    }
                    if (get_local_id(0) == 0) {
                        Y_data[${y_offset}] = y_sum_pre[get_local_id(1)]
                            + ${float_alpha} * partialDotProduct[
                                ${group_size} * get_local_id(1)];
                    }
                }
            }
        }
    }
        """

    text = Template(text, output_encoding='ascii').render(**textconf)
    print text
    print gsize

    fn = cl.Program(p.queue.context, text).build().fn

    full_args = [
                 cl_gstructure,
                 p.A.cl_buf,
                 p.X.cl_buf,
                 ]
    if p.cl_beta is not None:
        full_args += [p.cl_beta]
    full_args += [
                 p.Y_in.cl_buf,
                 p.Y.cl_buf,
                ]

    fn.set_args(*[arr.data for arr in full_args])
    return fn, gsize, lsize, full_args, 'parallel_ragged_gather_gemv3'



