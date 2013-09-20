import sys
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
        self.tag = str(tag)
        self.seq = seq

        self.geometry = self._geometry()
        plans = []
        remaining_items = range(len(Y))

        long_dots = [ii
            for ii in remaining_items
            if self.geometry[ii]['y_len'] == 1
                and len(self.geometry[ii]['dots'])
                and self.geometry[ii]['dots'][0]['a_shape1'] > 16]
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

        long_gemv2s = [ii
            for ii in remaining_items
            if self.geometry[ii]['y_len'] == 2
                and len(self.geometry[ii]['dots'])
                and self.geometry[ii]['dots'][0]['a_shape1'] > 16]
        if long_gemv2s:
            try:
                gemv2_plan = reduce_impl(self, long_gemv2s)
            except NotImplementedError:
                gemv2_plan = ref_impl(self, long_gemv2s)
            gemv2_plan.tag += '-gemv2-%i' % len(long_gemv2s)
            plans.append(gemv2_plan)
            remaining_items = [ii
                for ii in remaining_items
                if ii not in long_gemv2s]

        many_dots = [ii
            for ii in remaining_items
            if len(self.geometry[ii]['dots']) > 3]
        if many_dots:
            many_plan = many_dots_impl(self, many_dots)
            many_plan.tag += '-many%i' % len(many_dots)
            plans.append(many_plan)
            remaining_items = [ii
                for ii in remaining_items
                if ii not in many_dots]

        if remaining_items:
            remaining_plan = ref_impl(self, remaining_items)
            remaining_plan.tag += '-remaining%i' % len(remaining_items)
            plans.append(remaining_plan)
        Prog.__init__(self, plans)

    def print_geometry_summary(self, items=None, full=False):
        print 'geometry_summary: tag=%s' % self.tag
        if items is None:
            gg = self.geometry
        else:
            gg = map(self.geometry.__getitem__, items)
        print 'Y lens', dhist(ggi['y_len'] for ggi in gg)
        print 'ndots', dhist(len(ggi['dots']) for ggi in gg)
        Ks = []
        for ggi in gg:
            Ks.extend(ddb['a_shape1'] for ddb in ggi['dots'])
        print 'Ks', dhist(Ks)
        if full:
            for ggi in gg:
                tmp = dict(ggi)
                del tmp['dots']
                print tmp
                for dot in ggi['dots']:
                    print '  ', dot

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

    def cl_geometry_and_textconf(self, items):
        p = self
        max_n_dots = max(len(p.geometry[ii]['dots']) for ii in items)
        n_structure_vars = 4 * max_n_dots + 5
        gstructure = np.zeros((len(items), n_structure_vars), dtype='int32')
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
            tag=p.tag)
    rval.full_args = full_args  # prevent GC the args
    return rval


def reduce_impl(p, items):
    # TODO load X into shared if RPB > 1
    # TODO: tune group_size
    group_size = 32
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

    A_js_shape0s = p.A_js.shape0s
    max_n_dots = max(A_js_shape0s)
    min_n_dots = min(A_js_shape0s)
    n_structure_vars = 4 * max_n_dots + 5
    n_items = len(items)

    gstructure = np.zeros((n_items, n_structure_vars), dtype='int32')
    A_starts = p.A.starts
    X_starts = p.X.starts
    Y_starts = p.Y.starts
    Y_in_starts = p.Y_in.starts
    A_stride0s = p.A.stride0s
    A_shape1s = p.A.shape1s
    Y_shape0s = p.Y.shape0s

    max_reduce_len = 0
    max_y_len = 0
    min_y_len = sys.maxint
    for bbi, bb in enumerate(items):
        x_js_i = p.X_js[bb]
        A_js_i = p.A_js[bb]
        assert len(x_js_i) == len(A_js_i)
        for ii, (xi, ai) in enumerate(zip(x_js_i, A_js_i)):
            gstructure[bbi, 0 * max_n_dots + ii] = X_starts[xi]
            gstructure[bbi, 1 * max_n_dots + ii] = A_starts[ai]
            gstructure[bbi, 2 * max_n_dots + ii] = A_stride0s[ai]
            gstructure[bbi, 3 * max_n_dots + ii] = A_shape1s[ai]
            max_reduce_len = max(max_reduce_len, A_shape1s[ai])
        # -- offset of output and input buffers
        gstructure[bbi, 4 * max_n_dots + 0] = Y_in_starts[bb]
        gstructure[bbi, 4 * max_n_dots + 1] = Y_starts[bb]
        # -- number of dots for bb
        gstructure[bbi, 4 * max_n_dots + 2] = len(A_js_i)
        # -- length of Y[bb]
        gstructure[bbi, 4 * max_n_dots + 3] = Y_shape0s[bb]
        gstructure[bbi, 4 * max_n_dots + 4] = bb
        max_y_len = max(max_y_len, Y_shape0s[bb])
        min_y_len = min(min_y_len, Y_shape0s[bb])
    cl_gstructure = to_device(p.queue, gstructure)

    # Make the global size the closest multiple of the group size (ceiling)
    if max_y_len != min_y_len:
        raise NotImplementedError()
    reductions_per_block = max_y_len
    gsize = (group_size, reductions_per_block, n_items)
    lsize = (group_size, reductions_per_block, 1)

    textconf = {
        'n_items' : n_items,
        'RPB'     : reductions_per_block,
        'gsize'   : gsize,
        'group_size' : group_size,
        'local_count': group_size * reductions_per_block,
        'max_reduce_len': max_reduce_len,
        'n_structure_vars': n_structure_vars,
        'structure_stride': n_structure_vars,
        'max_n_dots': max_n_dots,
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
    textconf.update(p.__dict__)

    if n_structure_vars > group_size:
        raise NotImplementedError()
    if not (min_n_dots == max_n_dots == 1):
        raise NotImplementedError()

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
        //Scratch space for the dot products
        __local ${Y.cl_buf.ocldtype} partialDotProduct[${RPB}][${group_size}];
        __local ${Y.cl_buf.ocldtype} y_sum_pre[${RPB}];
        const int local_idx = get_local_id(0) + get_local_id(1) * get_local_size(0);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_idx < ${n_structure_vars})
        {
            lstructure[local_idx] = gstructure[
                get_global_id(2) * ${structure_stride} + local_idx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        const int ii = 0;
        const int mm = get_local_id(1);
        if (local_idx < ${RPB})
        {
    % if float_beta is not None and float_beta != 0 :
            y_sum_pre[local_idx] = ${float_beta} * Y_in_data[${y_in_starts} + local_idx];
    % elif cl_beta is not None:
            y_sum_pre[local_idx] = betas[${bb}] * Y_in_data[${y_in_starts} + local_idx];
    % else :
            y_sum_pre[local_idx] = 0;
    % endif

    % if float_gamma is not None and float_gamma != 0:
            y_sum_pre[local_idx] += ${float_gamma};
    % endif
        }

        partialDotProduct[mm][get_local_id(0)] = 0;
        for (int nn = get_local_id(0);
                 nn < ${N_i};
                 nn += get_local_size(0)) {
            partialDotProduct[mm][get_local_id(0)] +=
                A_data[${a_starts} + mm * ${a_s0} + nn] * X_data[${x_starts} + nn];
        }
        // -- Parallel reduction within local group sum registers
        for (uint stride = 1;
                  stride < get_local_size(0);
                  stride *= 2)
        {
            barrier(CLK_LOCAL_MEM_FENCE);

            uint index = 2 * stride * get_local_id(0);
            if (index + stride < get_local_size(0))
            {
                partialDotProduct[mm][index] +=
                    partialDotProduct[mm][index + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (get_local_id(0) == 0) {
            //printf("%i\\n", mm);
            Y_data[${y_offset} + mm] = y_sum_pre[mm]
                + ${float_alpha} * partialDotProduct[mm][0];
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
            tag=p.tag)
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

    n_segments = max_y_len // segment_size
    gsize = (n_segments * segment_size, dot_block_size, len(items))
    lsize = (segment_size, dot_block_size, 1)

    textconf.update({
        'segment_size': segment_size,
        'dot_block_size': dot_block_size,
        'max_y_len': max_y_len,
        'n_locals': segment_size * dot_block_size
        })
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
        ${Y.cl_buf.ocldtype} y_sum_pre;
        __local ${Y.cl_buf.ocldtype} \
            y_sum_post[${dot_block_size}][${segment_size}];
        const int local_idx = get_local_id(0) \
            + get_local_id(1) * get_local_size(0);

        for (int ii = local_idx; ii < ${n_structure_vars}; ii += ${n_locals})
        {
            lstructure[ii] = gstructure[
                get_global_id(2) * ${n_structure_vars} + ii];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        const int mm = get_global_id(0);
        if (mm < ${y_len})
        {

            if (get_local_id(1) == 0)
            {
    % if float_beta is not None and float_beta != 0 :
                y_sum_pre = ${float_beta} * Y_in_data[${y_in_starts} + mm];
    % elif cl_beta is not None:
                y_sum_pre = betas[${bb}] * Y_in_data[${y_in_starts} + mm];
    % else :
                y_sum_pre = 0;
    % endif

    % if float_gamma is not None:
        % if float_gamma != 0:
                y_sum_pre += ${float_gamma};
        % endif
    % endif
            }

            // XXX Move X into shared memory first
            y_sum_post[get_local_id(1)][get_local_id(0)] = 0;
            for (int ii = get_local_id(1);
                     ii < ${n_dot_products};
                     ii += ${dot_block_size})
            {
                for (int nn = 0; nn < ${N_i}; nn += 1)
                {
                    y_sum_post[get_local_id(1)][get_local_id(0)] +=
                        A_data[${a_starts} + mm * ${a_s0} + nn]
                        * X_data[${x_starts} + nn];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((mm < ${y_len}) && (get_local_id(1) == 0))
        {
            for (int ii = 1; ii < ${dot_block_size}; ++ii)
            {
                y_sum_post[0][get_local_id(0)] \
                    += y_sum_post[ii][get_local_id(0)];
            }
            Y_data[${y_offset} + mm] = y_sum_pre
                + ${float_alpha} * y_sum_post[0][get_local_id(0)];
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
            tag=p.tag)
    rval.full_args = full_args  # prevent GC the args
    return rval

