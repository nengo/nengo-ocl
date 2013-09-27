import math
from collections import defaultdict
import numpy as np
import pyopencl as cl
from plan import Plan
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


def flops_from_geometry(geometry, items):
    flops = 0
    for ii in items:
        gi = geometry[ii]
        for dotinfo in gi['dots']:
            # -- for every value of A, we
            #    (1) mult with some x
            #    (2) add to a resulting inner-product
            flops += dotinfo['a_shape1'] * gi['y_len'] * 2
        # XXX Generously assuming alpha & beta in use
        flops += gi['y_len'] * 3
    return flops

def bw_from_geometry(geometry, items):
    n_bytes = 0
    elemsize = 4
    for ii in items:
        gi = geometry[ii]
        for dotinfo in gi['dots']:
            # -- load A
            n_bytes += elemsize * dotinfo['a_shape1'] * gi['y_len']
            # -- load X
            n_bytes += elemsize * dotinfo['a_shape1']

        # -- load alpha scalar, beta scalar
        #    XXX: Account for a possible full vector read
        #    XXX: Account for a possible alpha vector read
        n_bytes += 2 * elemsize

        # -- load Y_in
        n_bytes += elemsize * gi['y_len']

        # -- write Y_out
        n_bytes += elemsize * gi['y_len']
    return n_bytes

class DotSignature(object):
    def __init__(self, dct):
        self.y_len = dct['y_len']
        self.Ax_dims = tuple([(d['a_shape1'], d['a_stride0']) for d in dct['dots']])

    def __eq__(self, other):
        return type(self) == type(other) \
                and self.y_len == other.y_len \
                and self.Ax_dims == other.Ax_dims

    def __hash__(self):
        return hash((self.y_len, self.Ax_dims))

    def __str__(self):
        counts = defaultdict(lambda: 0)
        for dim_stride in self.Ax_dims:
            counts[dim_stride] += 1
        return 'yd=%s <- %s' % (
            self.y_len,
            ', '.join(('(%s x d=%s,s=%s)' % (counts[(d, s)], d, s))
                      for (d, s) in counts))


class gemv_prog(object):
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
        self.tag = str(tag)
        self.seq = seq

        self.geometry = self._geometry()
        self.plans = self.choose_plans()

    def print_geometry_summary(self, items=None, full=False):
        print 'geometry_summary: tag=%s' % self.tag
        if items is None:
            gg = self.geometry
        else:
            gg = map(self.geometry.__getitem__, items)

        ds = map(DotSignature, gg)
        counts = defaultdict(lambda: 0)
        for dsi in ds:
            counts[dsi] += 1
        for dsi in sorted(counts):
            print '  %6s\t%s' % (counts[dsi], dsi)

    def _geometry(self):
        A_starts = self.A.starts
        X_starts = self.X.starts
        Y_starts = self.Y.starts
        Y_in_starts = self.Y_in.starts
        A_stride0s = self.A.stride0s
        A_shape1s = self.A.shape1s
        Y_shape0s = self.Y.shape0s

        rval = []
        for bb in range(len(Y_shape0s)):
            dbb = {
                'y_len': Y_shape0s[bb],
                'dots': [],
                'y_start': Y_starts[bb],
                'y_in_start': Y_in_starts[bb],
                    }
            if self.X_js:
                x_js_i = self.X_js[bb]
                A_js_i = self.A_js[bb]
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

    def cl_geometry_and_textconf(self, items, padding=4):
        p = self
        max_n_dots = max(len(p.geometry[ii]['dots']) for ii in items)
        n_structure_vars = 4 * max_n_dots + 5
        structure_vars_stride = int(
            padding * math.ceil(float(n_structure_vars) / padding))
        gstructure = np.zeros((len(items), structure_vars_stride), dtype='int32')
        A_starts = p.A.starts
        X_starts = p.X.starts
        Y_starts = p.Y.starts
        Y_in_starts = p.Y_in.starts
        A_stride0s = p.A.stride0s
        A_shape1s = p.A.shape1s
        Y_shape0s = p.Y.shape0s

        for bbi, bb in enumerate(items):
            x_js_i = p.X_js[bb]
            A_js_i = p.A_js[bb]
            assert len(x_js_i) == len(A_js_i)
            for ii, (xi, ai) in enumerate(zip(x_js_i, A_js_i)):
                gstructure[bbi, 0 * max_n_dots + ii] = X_starts[xi]
                gstructure[bbi, 1 * max_n_dots + ii] = A_starts[ai]
                gstructure[bbi, 2 * max_n_dots + ii] = A_stride0s[ai]
                gstructure[bbi, 3 * max_n_dots + ii] = A_shape1s[ai]
            # -- offset of output and input buffers
            gstructure[bbi, 4 * max_n_dots + 0] = Y_in_starts[bb]
            gstructure[bbi, 4 * max_n_dots + 1] = Y_starts[bb]
            # -- number of dots for bb
            gstructure[bbi, 4 * max_n_dots + 2] = len(A_js_i)
            # -- length of Y[bb]
            gstructure[bbi, 4 * max_n_dots + 3] = Y_shape0s[bb]
            gstructure[bbi, 4 * max_n_dots + 4] = bb
        cl_gstructure = to_device(p.queue, gstructure)

        textconf = {
            'n_structure_vars': n_structure_vars,
            'structure_vars_stride': structure_vars_stride,
            'x_starts': 'lstructure[0 * %s + ii]' % max_n_dots,
            'a_starts': 'lstructure[1 * %s + ii]' % max_n_dots,
            'a_s0'    : 'lstructure[2 * %s + ii]' % max_n_dots,
            'N_i'     : 'lstructure[3 * %s + ii]' % max_n_dots,
            'y_in_starts': 'lstructure[4 * %s + 0]' % max_n_dots,
            'y_offset': 'lstructure[4 * %s + 1]' % max_n_dots,
            'n_dot_products': 'lstructure[4 * %s + 2]' % max_n_dots,
            'y_len'   : 'lstructure[4 * %s + 3]' % max_n_dots,
            'bb'   : 'lstructure[4 * %s + 4]' % max_n_dots,
            }
        return cl_gstructure, textconf


def ref_impl(p, items):
    """
    Return an OpenCL function to calculate elements `items` of
    gemv operation `p`.

    In this reference implementation, we create a work item
    per output number, or more specifically, a work grid
    of (max_y_len, len(items)).  Each work item loops over the
    dot products and the elements within each dot product to
    compute the output value Y[global_id(1)][global_id(0)].

    """

    if p.clra_alpha is not None:
        raise NotImplementedError()
    if p.clra_gamma is not None:
        raise NotImplementedError()
    cl_items = to_device(p.queue,
        np.asarray(items, dtype='int32'))
    if 0:
        if len(items) < 10:
            print 'Falling back on reference implementation'
            p.print_geometry_summary(items, full=True)
        else:
            print 'Falling back on reference implementation'
            p.print_geometry_summary(items)

    assert all(s == 1 for s in p.A.stride1s)
    assert all(s == 1 for s in p.X.stride1s)
    assert all(s == 1 for s in p.Y.stride0s)
    assert all(s == 1 for s in p.Y.stride1s)
    assert all(s == 1 for s in p.Y_in.stride0s)
    assert all(s == 1 for s in p.Y_in.stride1s)

    text = """
        __kernel void fn(
            __global int *items,
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
            const int bb = items[get_global_id(1)];
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
                    const int x_ji = X_js_data[ii];
                    const int a_ji = A_js_data[ii];
                    const int N_i = A_shape1s[a_ji];
                    const int x_offset = X_starts[x_ji];
                    const int a_offset = A_starts[a_ji];
                    const int AsM = A_stride0s[a_ji];
                    const int XsM = X_stride0s[x_ji];

                    for (int nn = 0; nn < N_i; ++nn)
                    {
                        y_sum += X_data[x_offset + nn * XsM]
                                 * A_data[a_offset + mm * AsM + nn];
                    }
                }
        % if float_alpha is not None:
                Y_data[y_offset + mm] += ${float_alpha} * y_sum;
        % elif cl_alpha is not None:
                Y_data[y_offset + mm] += alphas[bb] * y_sum;
        % endif
    % endif
            }

        }
    """

    text = Template(text, output_encoding='ascii').render(**p.__dict__)
    #print text

    gsize = (
        max(p.geometry[ii]['y_len'] for ii in items),
        len(items))
    lsize = None
    fn = cl.Program(p.queue.context, text).build().fn
    full_args = [cl_items]
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
        full_args += [p.clra_beta.cl_starts, p.clra_beta.cl_buf]

    if p.cl_gamma is not None:
        full_args += [p.cl_gamma]
    elif p.clra_gamma is not None:
        full_args += [p.clra_gamma.cl_starts, p.clra_gamma.cl_buf]

    full_args += [
        p.Y_in.cl_starts,
        p.Y_in.cl_buf,
        p.Y.cl_starts,
        p.Y.cl_shape0s,
        p.Y.cl_buf]

    #print [str(arr.dtype)[0] for arr in full_args]
    fn.set_args(*[arr.data for arr in full_args])
    rval = Plan(p.queue, fn, gsize, lsize, name="clra_gemv.ref_impl",
        tag=p.tag,
        bw_per_call=bw_from_geometry(p.geometry, items),
        flops_per_call=flops_from_geometry(p.geometry, items))
    rval.full_args = full_args  # prevent GC the args
    return rval


def reduce_impl(p, items,
                group_size=None,
                segment_size=None,
               ):

    #
    # Target use case: long inner products, small numbers of dots.
    #
    # Approach: each work-group computes a small number of gemv outputs
    #

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

    cl_gstructure, textconf = p.cl_geometry_and_textconf(items)
    max_n_dots = max([len(p.geometry[ii]['dots']) for ii in items])
    max_reduce_len = max(max([gg['a_shape1']
                              for gg in p.geometry[ii]['dots']])
                         for ii in items )
    max_y_len = max([p.geometry[ii]['y_len'] for ii in items])

    # segment means the piece of Y written by a work-group
    # group_size is the number of values that we're reducing over

    if len(items) < 4:
        if group_size is None:
            group_size = 32 # XXX
        if segment_size is None:
            segment_size = min(max_y_len, 2) # XXX
    else:
        if group_size is None:
            group_size = 32 # XXX
        if segment_size is None:
            segment_size = min(max_y_len, 4) # XXX
    g_segments = int(math.ceil(float(max_y_len) / segment_size))
    gsize = (group_size, g_segments * segment_size, len(items))
    lsize = (group_size, segment_size, 1)

    max_reduce_iters = int(math.ceil(float(max_reduce_len) / group_size))
    textconf.update({
        'n_items' : len(items),
        'gsize'   : gsize,
        'segment_size': segment_size,
        'max_y_len': max_y_len,
        'group_size' : group_size,
        'local_count': group_size * segment_size,
        'max_reduce_len': max_reduce_len,
        'N_cutoff': max_reduce_iters * group_size,
        'max_n_dots': max_n_dots,
    })
    if 0:
        for k, v in textconf.items():
            print k, v

    textconf.update(p.__dict__)

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
    % if segment_size > 1:
        // we'll cache X in shared memory so we load it only once
        // for the whole segment
        __local ${X.cl_buf.ocldtype} lX[${group_size}];
    % endif
        //Scratch space for the dot products
        __local ${Y.cl_buf.ocldtype}
            partialDotProduct[${segment_size}][${group_size}];
        __local ${Y.cl_buf.ocldtype}
            y_sum_pre[${segment_size}];
        const int local_idx = get_local_id(0)
            + get_local_id(1) * get_local_size(0);

        // load structure
    % if local_count < n_structure_vars:
        for (int ii = local_idx;
                 ii < ${n_structure_vars};
                 ii += ${local_count})
        {
            lstructure[ii] = gstructure[
                get_global_id(2) * ${structure_vars_stride} + ii];
        }
    % else :
        if (local_idx < ${n_structure_vars})
        {
            lstructure[local_idx] = gstructure[
                get_global_id(2) * ${structure_vars_stride} + local_idx];
        }
    % endif
        barrier(CLK_LOCAL_MEM_FENCE);

        if ((get_local_id(0) == 0) && (get_global_id(1) < ${y_len}))
        {
    % if float_beta is not None and float_beta != 0 :
            y_sum_pre[get_local_id(1)] = ${float_beta}
                * Y_in_data[${y_in_starts} + get_global_id(1)];
    % elif cl_beta is not None:
            y_sum_pre[get_local_id(1)] = betas[${bb}]
                * Y_in_data[${y_in_starts} + get_global_id(1)];
    % else :
            y_sum_pre[get_local_id(1)] = 0;
    % endif

    % if float_gamma is not None and float_gamma != 0:
            y_sum_pre[get_local_id(1)] += ${float_gamma};
    % endif
    // printf("betaY + gamma=%f\\n", y_sum_pre[get_local_id(1)]);
        }

        partialDotProduct[get_local_id(1)][get_local_id(0)] = 0;
    % if max_n_dots > 1:
        for (int ii = 0;
                 ii < ${n_dot_products};
                 ii += 1)
        {
    % else:
        const int ii = 0;
    % endif


        for (int nn = get_local_id(0);
                 nn < ${N_cutoff};
                 nn += get_local_size(0))
        {
    // segment_size = ${segment_size}
    % if (segment_size == 1):
            if ((nn < ${N_i}) && (get_global_id(1) < ${y_len}))
            {
            partialDotProduct[get_local_id(1)][get_local_id(0)] +=
                A_data[${a_starts} + get_global_id(1) * ${a_s0} + nn]
                * X_data[${x_starts} + nn];
            }
    % else:
            barrier(CLK_LOCAL_MEM_FENCE);
            if ((get_local_id(1) == 0) && (nn < ${N_i}))
            {
                lX[get_local_id(0)] = X_data[${x_starts} + nn];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if ((nn < ${N_i}) && (get_global_id(1) < ${y_len}))
            {
            partialDotProduct[get_local_id(1)][get_local_id(0)] +=
                A_data[${a_starts} + get_global_id(1) * ${a_s0} + nn]
                * lX[get_local_id(0)];
            }
    % endif
        }

    % if (max_n_dots > 1):
        }
    % endif

        // -- Parallel reduction long work-group dimension 0
        for (uint stride = 1;
                  stride < get_local_size(0);
                  stride *= 2)
        {
            barrier(CLK_LOCAL_MEM_FENCE);

            uint index = 2 * stride * get_local_id(0);
            if (index + stride < get_local_size(0))
            {
                partialDotProduct[get_local_id(1)][index] +=
                    partialDotProduct[get_local_id(1)][index + stride];
            }
        }
        // barrier(CLK_LOCAL_MEM_FENCE);
        if ((get_local_id(0) == 0) && (get_global_id(1) < ${y_len})) {
            Y_data[${y_offset} + get_global_id(1)] = y_sum_pre[get_local_id(1)]
                + ${float_alpha} * partialDotProduct[get_local_id(1)][0];
        }
    }
        """

    text = Template(text, output_encoding='ascii').render(**textconf)

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
    rval = Plan(p.queue, fn, gsize, lsize,
        name='clra_gemv.reduce_impl',
        tag=p.tag,
        bw_per_call=bw_from_geometry(p.geometry, items),
        flops_per_call=flops_from_geometry(p.geometry, items),
        )
    rval.full_args = full_args  # prevent GC the args
    return rval


def many_dots_impl(p, items):
    # target use case:
    # * several very shallow gemvs (short inner prods) into each target
    # * not all targets have the same size
    #p.print_geometry_summary(items, full=True)

    #
    # This algorithm is blocked out so that a work-group [i, j] computes
    # some segment of an output vector:
    # e.g. Y[i][ 32 * j : 32 * (j + 1)]
    #
    # This is done for two reasons:
    # - to increase occupancy when there are not so many vectors Y
    # - to handle long vectors Y

    #p.print_geometry_summary(items)

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

    if p.A_js is None:
        # -- easy probably, but not done
        raise NotImplementedError()
    A_js_shape0s = p.A_js.shape0s
    cl_gstructure, textconf = p.cl_geometry_and_textconf(items)

    #min_n_dots = min(A_js_shape0s)
    max_n_dots = max(A_js_shape0s)


    max_y_len = max(p.geometry[ii]['y_len'] for ii in items)
    MAX_SEGMENT_SIZE = 16 # tricky to tune?

    segment_size = min(
        max_y_len,
        MAX_SEGMENT_SIZE)
    dot_block_size = min(
        max_n_dots,
        int(p.queue.device.max_work_group_size / segment_size),
        )

    n_segments = int(math.ceil(float(max_y_len) / segment_size))
    gsize = (n_segments * segment_size, dot_block_size, len(items))
    lsize = (segment_size, dot_block_size, 1)

    textconf.update({
        'gsize': gsize,
        'lsize': lsize,
        'segment_size': segment_size,
        'dot_block_size': dot_block_size,
        'max_y_len': max_y_len,
        'n_locals': segment_size * dot_block_size,
        #'segment_idx': 'get_local_id(0)',
        #'dot_block_idx': 'get_local_id(1)',
        'segment_idx': 'segment_idx',
        'dot_block_idx': 'dot_block_idx',
        })
    if 0:
        for k, v in textconf.items():
            print k, v
    textconf.update(p.__dict__)
    #    print 'float_gamma', textconf['float_gamma']
    #    print 'cl_gamma', textconf['cl_gamma']
    #    print 'clra_gamma', textconf['clra_gamma']

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
        __local ${Y.cl_buf.ocldtype} y_sum_pre[${segment_size}];
        __local ${Y.cl_buf.ocldtype} \
            y_sum_post[${dot_block_size}][${segment_size}];
        const int local_idx = get_local_id(0) \
            + get_local_id(1) * get_local_size(0);

        int segment_idx = get_local_id(0);
        int dot_block_idx = get_local_id(1);

        for (int ii = local_idx; ii < ${n_structure_vars}; ii += ${n_locals})
        {
            lstructure[ii] = gstructure[
                get_global_id(2) * ${structure_vars_stride} + ii];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (get_global_id(0) < ${y_len})
        {

            if (dot_block_idx == 0)
            {
    % if float_beta is not None and float_beta != 0 :
                y_sum_pre[segment_idx]
                = ${float_beta} * Y_in_data[${y_in_starts} + get_global_id(0)];
    % elif cl_beta is not None:
                y_sum_pre[segment_idx]
                = betas[${bb}] * Y_in_data[${y_in_starts} + get_global_id(0)];
    % else :
                y_sum_pre[segment_idx] = 0;
    % endif

    % if float_gamma is not None:
        % if float_gamma != 0:
                y_sum_pre[segment_idx] += ${float_gamma};
        % endif
    % endif
            }
        //printf("betaY + gamma=%f\\n", y_sum_pre[segment_idx]);

            // XXX Move X into shared memory first
            y_sum_post[dot_block_idx][segment_idx] = 0;
            for (int ii = dot_block_idx;
                     ii < ${n_dot_products};
                     ii += ${dot_block_size})
            {
                for (int nn = 0; nn < ${N_i}; nn += 1)
                {
                    y_sum_post[dot_block_idx][segment_idx]
                    += A_data[${a_starts} + get_global_id(0) * ${a_s0} + nn]
                       * X_data[${x_starts} + nn];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        //printf("AX=%f\\n", y_sum_post[dot_block_idx][segment_idx]);
        if ((get_global_id(0) < ${y_len}) && (dot_block_idx == 0))
        {
            for (int ii = 1; ii < ${dot_block_size}; ++ii)
            {
                y_sum_post[0][segment_idx] += y_sum_post[ii][segment_idx];
            }
            Y_data[${y_offset} + get_global_id(0)]
                = y_sum_pre[segment_idx]
                  + ${float_alpha} * y_sum_post[0][segment_idx];
        //printf("Yout=%f\\n", Y_data[${y_offset} + get_global_id(0)]);
        }
    }
        """

    text = Template(text, output_encoding='ascii').render(**textconf)

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
    rval = Plan(p.queue, fn, gsize, lsize,
        name='clra_gemv.many_dots_impl',
        tag=p.tag,
        bw_per_call=bw_from_geometry(p.geometry, items),
        flops_per_call=flops_from_geometry(p.geometry, items),
        )
    rval.full_args = full_args  # prevent GC the args
    return rval


class plan_ref(gemv_prog):
    def choose_plans(self):
        return [ref_impl(self, range(len(self.Y)))]

class plan_many_dots(gemv_prog):
    def choose_plans(self):
        return [many_dots_impl(self, range(len(self.Y)))]

class plan_reduce(gemv_prog):
    def choose_plans(self):
        return [reduce_impl(self, range(len(self.Y)))]

class plan_ragged_gather_gemv(gemv_prog):

    def choose_plans(self):
        remaining_items = range(len(self.Y))
        plans = []

        long_dots = [ii
            for ii in remaining_items
            if len(self.geometry[ii]['dots']) <= 2
               and max([0] + [dct['a_shape1']
                              for dct in self.geometry[ii]['dots']]) > 16]
        if long_dots:
            try:
                long_plan = reduce_impl(self, long_dots)
            except NotImplementedError:
                long_plan = ref_impl(self, long_dots)
            long_plan.tag += '-long%i' % len(long_dots)
            plans.append(long_plan)
            remaining_items = [ii
                for ii in remaining_items
                if ii not in long_dots]

        #many_dots = [ii
            #for ii in remaining_items
            #if len(self.geometry[ii]['dots']) > 3]
        many_dots = remaining_items
        if many_dots:
            try:
                many_plan = many_dots_impl(self, many_dots)
                many_plan.tag += '-many%i' % len(many_dots)
                plans.append(many_plan)
                remaining_items = [ii
                    for ii in remaining_items
                    if ii not in many_dots]
            except NotImplementedError:
                pass

        if remaining_items:
            remaining_plan = ref_impl(self, remaining_items)
            remaining_plan.tag += '-remaining%i' % len(remaining_items)
            plans.append(remaining_plan)

        return plans

