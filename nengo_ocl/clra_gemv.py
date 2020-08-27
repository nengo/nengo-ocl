from collections import defaultdict

import numpy as np
import pyopencl as cl
from mako.template import Template

from nengo_ocl.clraggedarray import CLRaggedArray, to_device
from nengo_ocl.plan import Plan
from nengo_ocl.utils import as_ascii


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
        for dotinfo in gi["dots"]:
            # -- for every value of A, we
            #    (1) mult with some x
            #    (2) add to a resulting inner-product
            flops += dotinfo["a_shape1"] * gi["y_len"] * 2
        # XXX Generously assuming alpha & beta in use
        flops += gi["y_len"] * 3
    return flops


def bw_from_geometry(geometry, items):
    n_bytes = 0
    elemsize = 4
    for ii in items:
        gi = geometry[ii]
        for dotinfo in gi["dots"]:
            # -- load A
            n_bytes += elemsize * dotinfo["a_shape1"] * gi["y_len"]
            # -- load X
            n_bytes += elemsize * dotinfo["a_shape1"]

        # -- load alpha scalar, beta scalar
        #    XXX: Account for a possible full vector read
        #    XXX: Account for a possible alpha vector read
        n_bytes += 2 * elemsize

        # -- load Y_in
        n_bytes += elemsize * gi["y_len"]

        # -- write Y_out
        n_bytes += elemsize * gi["y_len"]
    return n_bytes


class DotSignature(object):
    def __init__(self, dct):
        self.y_len = dct["y_len"]
        self.Ax_dims = tuple([(d["a_shape1"], d["a_stride0"]) for d in dct["dots"]])

    def __eq__(self, other):
        return (
            type(self) == type(other)
            and self.y_len == other.y_len
            and self.Ax_dims == other.Ax_dims
        )

    def __hash__(self):
        return hash((self.y_len, self.Ax_dims))

    def __str__(self):
        counts = defaultdict(lambda: 0)
        for dim_stride in self.Ax_dims:
            counts[dim_stride] += 1
        return "yd=%s <- %s" % (
            self.y_len,
            ", ".join(
                ("(%s x d=%s,s=%s)" % (counts[(d, s)], d, s)) for (d, s) in counts
            ),
        )


class gemv_prog(object):
    def __init__(
        self, queue, alpha, A, A_js, X, X_js, beta, Y, Y_in=None, gamma=0.0, tag=None
    ):

        self.float_alpha, self.cl_alpha, self.clra_alpha = float_cl_clra(
            queue, alpha, Y.dtype, len(Y)
        )
        self.float_beta, self.cl_beta, self.clra_beta = float_cl_clra(
            queue, beta, Y.dtype, len(Y)
        )
        self.float_gamma, self.cl_gamma, self.clra_gamma = float_cl_clra(
            queue, gamma, Y.dtype, len(Y)
        )

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

        self.geometry = self._geometry()
        self.plans = self.choose_plans()

    def geometry_summary(self, items=None):
        if items is None:
            gg = self.geometry
        else:
            gg = [self.geometry[i] for i in items]

        outputs = len(gg)
        dots = np.array([len(g["dots"]) for g in gg])
        shape0s = np.array([g["y_len"] for g in gg])
        shape1s = np.hstack([[d["a_shape1"] for d in g["dots"]] for g in gg])
        return (
            "outputs: %d; dots: %0.1f [%d, %d]; "
            "shape: %0.1f [%d, %d] x %0.1f [%d, %d]"
            % (
                outputs,
                dots.mean(),
                dots.min(),
                dots.max(),
                shape0s.mean(),
                shape0s.min(),
                shape0s.max(),
                shape1s.mean(),
                shape1s.min(),
                shape1s.max(),
            )
        )

    def print_geometry_summary(self, items=None, full=False):
        print("geometry_summary: tag=%s" % self.tag)
        if items is None:
            gg = self.geometry
        else:
            gg = map(self.geometry.__getitem__, items)

        ds = map(DotSignature, gg)
        counts = defaultdict(lambda: 0)
        for dsi in ds:
            counts[dsi] += 1
        for dsi in sorted(counts):
            print("  %6s\t%s" % (counts[dsi], dsi))

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
                "y_len": Y_shape0s[bb],
                "dots": [],
                "y_start": Y_starts[bb],
                "y_in_start": Y_in_starts[bb],
            }
            if self.X_js:
                x_js_i = self.X_js[bb]
                A_js_i = self.A_js[bb]
                assert len(x_js_i) == len(A_js_i)
                for jj, (xj, aj) in enumerate(zip(x_js_i, A_js_i)):
                    assert xj.size == 1 and aj.size == 1
                    xj, aj = xj[0], aj[0]  # to ignore numpy DeprecationWarning
                    dbb["dots"].append(
                        {
                            "j": jj,
                            "x_j": xj,
                            "a_j": aj,
                            "x_start": X_starts[xj],
                            "a_start": A_starts[aj],
                            "a_stride0": A_stride0s[aj],
                            "a_shape1": A_shape1s[aj],
                        }
                    )
            rval.append(dbb)

        return rval

    def cl_geometry_and_textconf(self, items, padding=4):
        p = self
        max_n_dots = max(len(p.geometry[ii]["dots"]) for ii in items)
        n_structure_vars = 4 * max_n_dots + 5
        structure_vars_stride = int(
            padding * np.ceil(float(n_structure_vars) / padding)
        )
        gstructure = np.zeros((len(items), structure_vars_stride), dtype="int32")
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
                assert xi.size == 1 and ai.size == 1
                xi, ai = xi[0], ai[0]  # to ignore numpy DeprecationWarning
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
            "n_structure_vars": n_structure_vars,
            "structure_vars_stride": structure_vars_stride,
            "x_starts": "lstructure[0 * %s + ii]" % max_n_dots,
            "a_starts": "lstructure[1 * %s + ii]" % max_n_dots,
            "a_s0": "lstructure[2 * %s + ii]" % max_n_dots,
            "N_i": "lstructure[3 * %s + ii]" % max_n_dots,
            "y_in_starts": "lstructure[4 * %s + 0]" % max_n_dots,
            "y_offset": "lstructure[4 * %s + 1]" % max_n_dots,
            "n_dot_products": "lstructure[4 * %s + 2]" % max_n_dots,
            "y_len": "lstructure[4 * %s + 3]" % max_n_dots,
            "bb": "lstructure[4 * %s + 4]" % max_n_dots,
        }
        return cl_gstructure, textconf


def ref_impl(p, items):  # noqa: C901
    """Return an OCL function to calculate ``items`` of gemv operation ``p``.

    In this reference implementation, we create a work item per output number,
    or more specifically, a work grid of shape ``(max_y_len, len(items))``.
    Each work item loops over the  dot products and the elements within
    each dot product to compute the output value
    ``Y[global_id(1)][global_id(0)]``.
    """

    if p.clra_alpha is not None:
        raise NotImplementedError()
    if p.clra_gamma is not None:
        raise NotImplementedError()
    cl_items = to_device(p.queue, np.asarray(items, dtype="int32"))
    if 0:
        if len(items) < 10:
            print("Falling back on reference implementation")
            p.print_geometry_summary(items, full=True)
        else:
            print("Falling back on reference implementation")
            p.print_geometry_summary(items)

    assert all(s == 1 for s in p.A.stride1s)
    assert all(s == 1 for s in p.X.stride1s)
    assert all(s == 1 for s in p.Y.stride0s)
    assert all(s == 1 for s in p.Y.stride1s)
    assert all(s == 1 for s in p.Y_in.stride0s)
    assert all(s == 1 for s in p.Y_in.stride1s)

    text = """
        __kernel void gemv_ref(
            __global int *items,
    % if cl_alpha is not None:
            __global ${cl_alpha.ctype} * alphas,
    % endif
    % if (A_js is not None):
            __global int *A_starts,
            __global int *A_shape1s,
            __global int *A_stride0s,
            __global ${A.cl_buf.ctype} *A_data,
            __global int *A_js_starts,
            __global int *A_js_shape0s,
            __global int *A_js_data,
            __global int *X_starts,
            __global int *X_stride0s,
            __global ${X.cl_buf.ctype} *X_data,
            __global int *X_js_starts,
            __global int *X_js_data,
    % endif
    % if cl_beta is not None:
            __global ${cl_beta.ctype} * betas,
    % endif
    % if clra_beta is not None:
            __global int *beta_starts,
            __global int *beta_data,
    % endif
    % if cl_gamma is not None:
            __global ${cl_gamma.ctype} * gammas,
    % endif
            __global int *Y_in_starts,
            __global ${Y_in.cl_buf.ctype} *Y_in_data,
            __global int *Y_starts,
            __global int *Y_shape0s,
            __global ${Y.cl_buf.ctype} *Y_data)
        {
            const int mm = get_global_id(0);
            const int bb = items[get_global_id(1)];
            const int M = Y_shape0s[bb];
            if (mm < M)
            {
                const int y_offset = Y_starts[bb];
                const int y_in_offset = Y_in_starts[bb];

    % if float_beta is not None:
                const ${Y.cl_buf.ctype} beta = ${float_beta};
    % elif cl_beta is not None:
                const ${cl_beta.ctype} beta = betas[bb];
    % elif clra_beta is not None:
                const int beta_offset = beta_starts[bb];
                const ${clra_beta.cl_buf.ctype} beta
                    = beta_data[beta_offset + mm];
    % endif

    % if float_gamma is not None:
                const ${Y.cl_buf.ctype} gamma = ${float_gamma};
    % elif cl_gamma is not None:
                const ${cl_gamma.ctype} gamma = gammas[bb];
    % endif

                Y_data[y_offset + mm] =
                    gamma + beta * Y_in_data[y_in_offset + mm];

    % if A_js is not None:
                const int n_dot_products = A_js_shape0s[bb];
                X_js_data += X_js_starts[bb];
                A_js_data += A_js_starts[bb];

                ${Y.cl_buf.ctype} y_sum = 0;
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

    text = as_ascii(Template(text, output_encoding="ascii").render(**p.__dict__))

    gsize = (max(p.geometry[ii]["y_len"] for ii in items), len(items))
    lsize = None
    fn = cl.Program(p.queue.context, text).build().gemv_ref
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
        p.Y.cl_buf,
    ]

    # print([str(arr.dtype)[0] for arr in full_args])
    fn.set_args(*[arr.data for arr in full_args])
    rval = Plan(
        p.queue,
        fn,
        gsize,
        lsize,
        name="clra_gemv.ref_impl",
        tag=p.tag,
        bw_per_call=bw_from_geometry(p.geometry, items),
        flops_per_call=flops_from_geometry(p.geometry, items),
    )
    rval.full_args = full_args  # prevent GC the args
    return rval


def reduce_impl(p, items, group_size=None, segment_size=None):  # noqa: C901

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
    max_n_dots = max([len(p.geometry[ii]["dots"]) for ii in items])
    max_reduce_len = max(
        max([gg["a_shape1"] for gg in p.geometry[ii]["dots"]]) for ii in items
    )
    max_y_len = max([p.geometry[ii]["y_len"] for ii in items])

    # segment means the piece of Y written by a work-group
    # group_size is the number of values that we're reducing over

    if len(items) < 4:
        if group_size is None:
            group_size = 32  # XXX
        if segment_size is None:
            segment_size = min(max_y_len, 2)  # XXX
    else:
        if group_size is None:
            group_size = 32  # XXX
        if segment_size is None:
            segment_size = min(max_y_len, 4)  # XXX
    g_segments = int(np.ceil(float(max_y_len) / segment_size))
    gsize = (group_size, g_segments * segment_size, len(items))
    lsize = (group_size, segment_size, 1)

    max_reduce_iters = int(np.ceil(float(max_reduce_len) / group_size))
    textconf.update(
        {
            "n_items": len(items),
            "gsize": gsize,
            "segment_size": segment_size,
            "max_y_len": max_y_len,
            "group_size": group_size,
            "local_count": group_size * segment_size,
            "max_reduce_len": max_reduce_len,
            "N_cutoff": max_reduce_iters * group_size,
            "max_n_dots": max_n_dots,
        }
    )
    if 0:
        for k, v in textconf.items():
            print(k, v)

    textconf.update(p.__dict__)

    text = """
        __kernel void gemv_reduce(
            const __global int *gstructure,
            const __global ${A.cl_buf.ctype} *A_data,
            const __global ${X.cl_buf.ctype} *X_data,
            % if cl_beta is not None:
            const __global ${cl_beta.ctype} * betas,
            % endif
            const __global ${Y_in.cl_buf.ctype} *Y_in_data,
            __global ${Y.cl_buf.ctype} *Y_data)
    {
        __local int lstructure[${n_structure_vars}];
    % if segment_size > 1:
        // we'll cache X in shared memory so we load it only once
        // for the whole segment
        __local ${X.cl_buf.ctype} lX[${group_size}];
    % endif
        //Scratch space for the dot products
        __local ${Y.cl_buf.ctype}
            partialDotProduct[${segment_size}][${group_size}];
        __local ${Y.cl_buf.ctype}
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

    text = as_ascii(Template(text, output_encoding="ascii").render(**textconf))

    fn = cl.Program(p.queue.context, text).build().gemv_reduce

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
    rval = Plan(
        p.queue,
        fn,
        gsize,
        lsize,
        name="clra_gemv.reduce_impl",
        tag=p.tag,
        bw_per_call=bw_from_geometry(p.geometry, items),
        flops_per_call=flops_from_geometry(p.geometry, items),
    )
    rval.full_args = full_args  # prevent GC the args
    rval.description = p.geometry_summary(items)
    return rval


def many_dots_impl(p, items):  # noqa: C901
    # target use case:
    # * several very shallow gemvs (short inner prods) into each target
    # * not all targets have the same size

    # p.print_geometry_summary(items, full=True)

    # This algorithm is blocked out so that a work-group [i, j] computes
    # some segment of an output vector:
    # e.g. Y[i][ 32 * j : 32 * (j + 1)]
    #
    # This is done for two reasons:
    # - to increase occupancy when there are not so many vectors Y
    # - to handle long vectors Y

    # p.print_geometry_summary(items)

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

    # min_n_dots = min(A_js_shape0s)
    max_n_dots = max(A_js_shape0s)

    max_y_len = max(p.geometry[ii]["y_len"] for ii in items)
    MAX_SEGMENT_SIZE = 16  # tricky to tune?

    segment_size = min(max_y_len, MAX_SEGMENT_SIZE)
    dot_block_size = min(
        max(max_n_dots, 1), int(p.queue.device.max_work_group_size / segment_size)
    )

    n_segments = int(np.ceil(float(max_y_len) / segment_size))
    gsize = (n_segments * segment_size, dot_block_size, len(items))
    lsize = (segment_size, dot_block_size, 1)

    textconf.update(
        {
            "gsize": gsize,
            "lsize": lsize,
            "segment_size": segment_size,
            "dot_block_size": dot_block_size,
            "max_y_len": max_y_len,
            "n_locals": segment_size * dot_block_size,
            # 'segment_idx': 'get_local_id(0)',
            # 'dot_block_idx': 'get_local_id(1)',
            "segment_idx": "segment_idx",
            "dot_block_idx": "dot_block_idx",
        }
    )
    if 0:
        for k, v in textconf.items():
            print(k, v)
    textconf.update(p.__dict__)
    #    print('float_gamma', textconf['float_gamma'])
    #    print('cl_gamma', textconf['cl_gamma'])
    #    print('clra_gamma', textconf['clra_gamma'])

    text = """
        __kernel void gemv_many_dots(
            const __global int *gstructure,
            const __global ${A.cl_buf.ctype} *A_data,
            const __global ${X.cl_buf.ctype} *X_data,
            % if cl_beta is not None:
            const __global ${cl_beta.ctype} * betas,
            % endif
            const __global ${Y_in.cl_buf.ctype} *Y_in_data,
            __global ${Y.cl_buf.ctype} *Y_data)
    {
        __local int lstructure[${n_structure_vars}];
        __local ${Y.cl_buf.ctype} y_sum_pre[${segment_size}];
        __local ${Y.cl_buf.ctype} \
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

    text = as_ascii(Template(text, output_encoding="ascii").render(**textconf))
    fn = cl.Program(p.queue.context, text).build().gemv_many_dots

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
    rval = Plan(
        p.queue,
        fn,
        gsize,
        lsize,
        name="clra_gemv.many_dots_impl",
        tag=p.tag,
        bw_per_call=bw_from_geometry(p.geometry, items),
        flops_per_call=flops_from_geometry(p.geometry, items),
    )
    rval.full_args = full_args  # prevent GC the args
    rval.description = p.geometry_summary(items)
    return rval


def block_impl(p, items):  # noqa: C901

    if p.clra_alpha is not None:
        raise NotImplementedError()
    if p.clra_gamma is not None:
        raise NotImplementedError()
    if p.clra_beta is not None:
        raise NotImplementedError()
    if p.cl_alpha is not None:
        raise NotImplementedError()
    if p.cl_beta is not None:
        raise NotImplementedError()
    if p.cl_gamma is not None:
        raise NotImplementedError()
    if not all(s == 1 for s in p.A.stride1s):
        raise NotImplementedError()

    if p.A_js is None:
        # -- easy probably, but not done
        raise NotImplementedError()

    # --- blocking
    # We want to group the dot products into blocks, so that each workgroup
    # is computing a (block_y, block_x) region of a dot product. To do this,
    # we create a temporary output buffer, compute each block to a separate
    # region of this buffer, then reduce across the buffer in a separate kernel

    # block_y = 8
    block_y = 32
    # block_x = 32
    block_x = 128

    shape0s = []
    shape1s = []
    Astride0s = []
    Astride1s = []
    Astarts = []
    Xstride0s = []
    Xstarts = []
    Ybufstarts = []
    Ybufstart = 0

    Yshape0s_reduce = []
    Yinstride0s_reduce = []
    Yinstarts_reduce = []
    Ystride0s_reduce = []
    Ystarts_reduce = []
    Ybufinds_reduce = []
    bw_reduce = 0

    for n in items:
        assert p.Y_in.shape0s[n] == p.Y.shape0s[n]
        shape0n = p.Y.shape0s[n]

        for i in range(0, shape0n, block_y):
            shape0i = min(shape0n - i, block_y)

            Ybufind_reduce = []

            # loop over dot products outputting to same Y
            assert len(p.A_js[n]) == len(p.X_js[n])
            for aj, xj in zip(p.A_js[n], p.X_js[n]):
                assert aj.size == 1 and xj.size == 1
                aj, xj = aj[0], xj[0]  # to ignore numpy DeprecationWarning

                assert p.A.shape0s[aj] == shape0n
                assert p.A.shape1s[aj] == p.X.shape0s[xj]
                assert p.X.shape1s[xj] == 1
                shape1n = p.A.shape1s[aj]

                for j in range(0, shape1n, block_x):
                    shape0s.append(shape0i)
                    shape1s.append(min(shape1n - j, block_x))
                    Astride0s.append(p.A.stride0s[aj])
                    Astride1s.append(p.A.stride1s[aj])
                    Astarts.append(
                        p.A.starts[aj] + i * p.A.stride0s[aj] + j * p.A.stride1s[aj]
                    )
                    Xstride0s.append(p.X.stride0s[xj])
                    Xstarts.append(p.X.starts[xj] + j * p.X.stride0s[xj])

                    Ybufstarts.append(Ybufstart)
                    Ybufind_reduce.append(Ybufstart)
                    # Ybufstart += shape0s[-1]
                    Ybufstart += block_y  # keep good offset

            # --- Y-blocking for reduce
            Yshape0s_reduce.append(shape0i)
            Yinstride0s_reduce.append(p.Y_in.stride0s[n])
            Yinstarts_reduce.append(p.Y_in.starts[n] + i * p.Y_in.stride0s[n])
            Ystride0s_reduce.append(p.Y.stride0s[n])
            Ystarts_reduce.append(p.Y.starts[n] + i * p.Y.stride0s[n])
            Ybufinds_reduce.append(Ybufind_reduce)
            bw_reduce += shape0i * (len(Ybufind_reduce) + 1) * p.Y.dtype.itemsize

    # --- create structure
    gstructure = np.column_stack(
        [
            shape0s,
            shape1s,
            Astride0s,
            Astride1s,
            Astarts,
            Xstride0s,
            Xstarts,
            Ybufstarts,
        ]
    )
    cl_gstructure = to_device(p.queue, gstructure.astype(np.int32))

    # --- create Y buffer
    clYbuf = to_device(p.queue, np.zeros(Ybufstart, dtype=p.Y.dtype))

    lsize0 = 4
    # lsize0 = 8
    lsize0_log2 = int(np.log2(lsize0))
    assert 2 ** lsize0_log2 == lsize0

    lsize = (lsize0, block_y, 1)
    gsize = (lsize[0], lsize[1], gstructure.shape[0])
    assert np.prod(lsize) >= block_x

    textconf = dict(
        A=p.A,
        X=p.X,
        Ybuf=clYbuf,
        n_structure_vars=gstructure.shape[1],
        shape0="lstructure[0]",
        shape1="lstructure[1]",
        Astride0="lstructure[2]",
        Astride1="lstructure[3]",
        Astart="lstructure[4]",
        Xstride0="lstructure[5]",
        Xstart="lstructure[6]",
        Ybufstart="lstructure[7]",
        block_y=block_y,
        block_x=block_x,
        lsize0=lsize0,
        lsize0_log2=lsize0_log2,
        float_alpha=p.float_alpha,
    )

    full_args = (
        cl_gstructure,
        p.A.cl_buf,
        p.X.cl_buf,
        clYbuf,
    )

    text = """
    __kernel void fn(
        __global const int *gstructure,
        __global const ${A.ctype} *Adata,
        __global const ${X.ctype} *Xdata,
        __global ${Ybuf.ctype} *Ybufdata
        )
    {
        const int j = get_global_id(0);
        const int i = get_global_id(1);
        const int n = get_global_id(2);

        // load structure
        __local int lstructure[${n_structure_vars}];
        const int local_idx =
            get_local_id(0) + get_local_id(1)*get_local_size(0);
        if (local_idx < ${n_structure_vars})
            lstructure[local_idx] = gstructure[
                n * ${n_structure_vars} + local_idx];
        barrier(CLK_LOCAL_MEM_FENCE);

        __global const ${X.ctype} *x = Xdata + ${Xstart};
        __global ${Ybuf.ctype} *ybuf = Ybufdata + ${Ybufstart};

        // load x into local memory
        __local ${X.ctype} xlocal[${block_x}];
        if (local_idx < ${shape1})
            xlocal[local_idx] = x[local_idx*${Xstride0}];
        barrier(CLK_LOCAL_MEM_FENCE);

        __local ${Ybuf.ctype} sums[${block_y}][${lsize0}];
        sums[i][j] = 0;

        if (i < ${shape0}) {
            __global const ${A.ctype} *Ai = Adata + ${Astart} + i*${Astride0};
            for(int jj = j; jj < ${shape1}; jj += get_global_size(0)) {
                sums[i][j] += Ai[jj*${Astride1}] * xlocal[jj];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

    % for k in range(lsize0_log2 - 1, 0, -1):
        if (j < ${2**k})
            sums[i][j] += sums[i][${2**k} + j];
        barrier(CLK_LOCAL_MEM_FENCE);
    % endfor

        if (i < ${shape0} && j == 0)
            ybuf[i] = ${float_alpha} * (sums[i][0] + sums[i][1]);
    }
    """

    text = as_ascii(Template(text, output_encoding="ascii").render(**textconf))
    kernel = cl.Program(p.queue.context, text).build().fn
    kernel.set_args(*[arr.data for arr in full_args])

    plan = Plan(
        p.queue,
        kernel,
        gsize,
        lsize,
        name="clra_gemv.block_impl",
        tag=p.tag,
        bw_per_call=bw_from_geometry(p.geometry, items),
        flops_per_call=flops_from_geometry(p.geometry, items),
    )
    plan.full_args = full_args  # prevent GC the args
    plan.description = p.geometry_summary(items)
    plan.Ybuf = clYbuf

    # --- Reduce kernel
    align = False

    Nreduce = len(Yshape0s_reduce)
    clYshape0s_reduce = to_device(p.queue, np.array(Yshape0s_reduce, dtype=np.int32))
    clYinstride0s_reduce = to_device(
        p.queue, np.array(Yinstride0s_reduce, dtype=np.int32)
    )
    clYinstarts_reduce = to_device(p.queue, np.array(Yinstarts_reduce, dtype=np.int32))
    clYstride0s_reduce = to_device(p.queue, np.array(Ystride0s_reduce, dtype=np.int32))
    clYstarts_reduce = to_device(p.queue, np.array(Ystarts_reduce, dtype=np.int32))
    clYbufinds_reduce = CLRaggedArray.from_arrays(
        p.queue, Ybufinds_reduce, dtype=np.int32, align=align
    )
    assert len(clYbufinds_reduce) == Nreduce
    assert (clYbufinds_reduce.shape1s == 1).all()

    textconf_reduce = dict(
        Ybuf=clYbuf,
        Yin=p.Y_in,
        Y=p.Y,
        float_beta=p.float_beta,
        float_gamma=p.float_gamma,
    )

    full_args_reduce = (
        clYshape0s_reduce,
        clYbufinds_reduce.cl_shape0s,
        clYbufinds_reduce.cl_starts,
        clYbufinds_reduce.cl_buf,
        clYbuf,
        clYinstride0s_reduce,
        clYinstarts_reduce,
        p.Y_in.cl_buf,
        clYstride0s_reduce,
        clYstarts_reduce,
        p.Y.cl_buf,
    )

    lsize_reduce = None
    gsize_reduce = (block_y, Nreduce)

    text_reduce = """
    __kernel void reduce(
        __global const int *shape0s,
        __global const int *Ishape0s,
        __global const int *Istarts,
        __global const int *Idata,
        __global ${Ybuf.ctype} *Ybufdata,
        __global const int *Yinstride0s,
        __global const int *Yinstarts,
        __global ${Yin.ctype} *Yindata,
        __global const int *Ystride0s,
        __global const int *Ystarts,
        __global ${Y.ctype} *Ydata
    )
    {
        const int i = get_global_id(0);
        const int n = get_global_id(1);
        if (i >= shape0s[n])
            return;

        const int Ishape0 = Ishape0s[n];

        __global const int *Ybufstart = Idata + Istarts[n];
        __global ${Yin.ctype} *yin = Yindata + Yinstarts[n];
        __global ${Y.ctype} *y = Ydata + Ystarts[n];

        ${Y.ctype} sum = ${float_beta} * yin[i*Yinstride0s[n]];
        for (int j = 0; j < Ishape0; j++) {
            sum += Ybufdata[Ybufstart[j] + i];
        }

        y[i*Ystride0s[n]] = sum + ${float_gamma};
    }
    """

    text_reduce = as_ascii(
        Template(text_reduce, output_encoding="ascii").render(**textconf_reduce)
    )
    kernel_reduce = cl.Program(p.queue.context, text_reduce).build().reduce
    kernel_reduce.set_args(*[arr.data for arr in full_args_reduce])

    plan_reduce = Plan(
        p.queue,
        kernel_reduce,
        gsize_reduce,
        lsize_reduce,
        name="clra_gemv.block_impl_reduce",
        tag=p.tag,
    )
    plan_reduce.full_args = full_args_reduce  # prevent GC of the args
    plan_reduce.bw_per_call = bw_reduce
    # plan_reduce.description = p.geometry_summary(items)

    return [plan, plan_reduce]


class plan_ref_gemv(gemv_prog):
    def choose_plans(self):
        return [ref_impl(self, range(len(self.Y)))]


class plan_many_dots_gemv(gemv_prog):
    def choose_plans(self):
        return [many_dots_impl(self, range(len(self.Y)))]


class plan_reduce_gemv(gemv_prog):
    def choose_plans(self):
        return [reduce_impl(self, range(len(self.Y)))]


class plan_block_gemv(gemv_prog):
    def choose_plans(self):
        return block_impl(self, list(range(len(self.Y))))


class plan_ragged_gather_gemv(gemv_prog):
    # EH: This heuristic was designed by James to get the best speeds, but for
    # large models (i.e. Spaun) just using block_impl seems to be faster.

    def choose_plans(self):
        remaining_items = range(len(self.Y))
        plans = []

        long_dots = [
            ii
            for ii in remaining_items
            if len(self.geometry[ii]["dots"]) <= 2
            and max([0] + [dct["a_shape1"] for dct in self.geometry[ii]["dots"]]) > 16
        ]
        if long_dots:
            try:
                long_plan = reduce_impl(self, long_dots)
            except NotImplementedError:
                long_plan = ref_impl(self, long_dots)
            long_plan.tag += "-long%i" % len(long_dots)
            plans.append(long_plan)
            remaining_items = [ii for ii in remaining_items if ii not in long_dots]

        # many_dots = [ii
        # for ii in remaining_items
        # if len(self.geometry[ii]['dots']) > 3]
        many_dots = remaining_items
        if many_dots:
            try:
                many_plan = many_dots_impl(self, many_dots)
                many_plan.tag += "-many%i" % len(many_dots)
                plans.append(many_plan)
                remaining_items = [ii for ii in remaining_items if ii not in many_dots]
            except NotImplementedError:
                pass

        if remaining_items:
            remaining_plan = ref_impl(self, remaining_items)
            remaining_plan.tag += "-remaining%i" % len(remaining_items)
            plans.append(remaining_plan)

        return plans


def plan_sparse_dot_inc(queue, A_indices, A_indptr, A_data, X, Y, inc=False, tag=None):
    """Implements a sparse matrix-vector multiply: Y += A * X or Y = A * X

    Parameters
    ----------
    A_indices, A_indptr : PyOpenCL array
        Column sparse row index specifications
    A_data : PyOpenCL array
        Matrix values at those indices
    X, Y : CLRaggedArrays of length 1
        Input/output data.
    inc : bool
        Whether to increment ``Y`` (True), or set it (False).

    Notes
    -----
    This function crashes when there are >10M nonzero weights. A potential solution
    would be some way to tell each work item to do multiple rows.
    """
    assert len(X) == len(Y) == 1

    for arr in [X, Y]:
        assert (arr.stride1s == 1).all()
        if not ((arr.shape1s == 1).all() and (arr.stride0s == 1).all()):
            raise NotImplementedError(
                "OCL SparseDot only supports matrix-vector currently, not matrix-matrix"
            )

    for arr in [A_indices, A_indptr, A_data]:
        assert len(arr.shape) == 1
        assert arr.strides[0] == arr.dtype.itemsize  # contiguous

    assert A_indices.size == A_data.size

    assert A_data.ctype == X.ctype == Y.ctype
    assert A_indices.ctype == A_indptr.ctype == "int"

    kern = """
    __kernel void sparsedot_inc(
        __global const int *A_indices,
        __global const int *A_indptr,
        __global const ${dtype} *A_data,
        __global const int *Xstarts,
        __global const ${dtype} *Xdata,
        __global const int *Ystarts,
        __global ${dtype} *Ydata
    )
    {
        // n can later be used to keep track of multiple arrays
        const int n = 0;
        const int irow = get_global_id(0);

        __global const ${dtype} *x = Xdata + Xstarts[n];
        __global ${dtype} *y = Ydata + Ystarts[n];

    %if not inc:
        y[irow] = 0;
    %endif
        const int end = A_indptr[irow + 1];
        for (int k = A_indptr[irow]; k < end; k++) {
            y[irow] += A_data[k] * x[A_indices[k]];
        }
    }
    """
    textconf = dict(dtype=A_data.ctype, IndType=A_indices.ctype, inc=inc)
    text = as_ascii(Template(kern, output_encoding="ascii").render(**textconf))
    full_args = (
        A_indices.base_data,
        A_indptr.base_data,
        A_data.base_data,
        X.cl_starts.data,
        X.cl_buf.data,
        Y.cl_starts.data,
        Y.cl_buf.data,
    )
    _fn = cl.Program(queue.context, text).build().sparsedot_inc
    _fn.set_args(*full_args)

    gsize = (Y.sizes[0], 1)  # this only works for a single operation
    lsize = None
    plan = Plan(queue, _fn, gsize, lsize=lsize, name="cl_sparsedot", tag=tag)
    plan.full_args = full_args  # prevent garbage-collection
    plan.flops_per_call = 2 * A_data.size
    plan.bw_per_call = A_data.nbytes * 3 + A_indices.nbytes + A_indptr.nbytes
    plan.description = "groups: %d; shape: (%d, %d); nonzeros: %d" % (
        1,
        Y.sizes[0],
        X.sizes[0],
        A_data.size,
    )
    return plan
