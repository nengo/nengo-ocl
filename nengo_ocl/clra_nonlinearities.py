import numpy as np
import pyopencl as cl
from plan import Plan
from mako.template import Template
from clarray import to_device
from .clraggedarray import CLRaggedArray


def all_equal(a, b):
    return (np.asarray(a) == np.asarray(b)).all()


def _indent(s, i):
    return '\n'.join([(' ' * i) + line for line in s.split('\n')])


def plan_elementwise_inc(queue, A, X, Y, tag=None):
    """Implements an element-wise increment Y += A * X"""
    N = len(X)
    assert len(Y) == N and len(A) == N

    for i in range(N):
        assert X.shape0s[i] in [1, Y.shape0s[i]]
        assert X.shape1s[i] in [1, Y.shape1s[i]]
        assert A.shape0s[i] in [1, Y.shape0s[i]]
        assert A.shape1s[i] in [1, Y.shape1s[i]]
        assert X.stride1s[i] == 1
        assert Y.stride1s[i] == 1
        assert A.stride1s[i] == 1

    assert X.cl_buf.ocldtype == Y.cl_buf.ocldtype
    assert A.cl_buf.ocldtype == Y.cl_buf.ocldtype

    text = """
        ${Ytype} get_element(
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
        __kernel void fn(
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
            for (int ij = get_global_id(0); ij < Ysize; ij += get_global_size(0))
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

    textconf = dict(Atype=A.cl_buf.ocldtype, Xtype=X.cl_buf.ocldtype,
                    Ytype=Y.cl_buf.ocldtype)
    text = Template(text, output_encoding='ascii').render(**textconf)

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
    _fn = cl.Program(queue.context, text).build().fn
    _fn.set_args(*[arr.data for arr in full_args])

    max_group = queue.device.max_work_group_size
    mn = min(max(max(Y.shape0s), max(Y.shape1s)), max_group)
    gsize = (mn, N)
    lsize = (mn, 1)
    rval = Plan(queue, _fn, gsize, lsize=lsize, name="cl_filter_synapses", tag=tag)
    rval.full_args = full_args     # prevent garbage-collection
    return rval


def plan_filter_synapse(queue, X, Y, A, B, tag=None):
    """
    Implements a filter of the form

        y[n+1] + a[0] y[n] + ... + a[i] y[n-i] = b[0] x[n] + ... + b[j] x[n-j]
    """
    N = len(X)
    assert len(Y) == N and len(A) == N and len(B) == N

    for i in range(N):
        assert X.shape0s[i] == Y.shape0s[i]
        assert X.shape1s[i] == 1
        assert Y.shape1s[i] == 1
        assert X.stride1s[i] == 1
        assert Y.stride1s[i] == 1
        assert A.shape1s[i] == 1
        assert B.shape1s[i] == 1

        # This currently assumes that each filter has one numerator coefficient
        # and one denominator coefficient. Generalized filters are on the TODO list.
        # Generalized filters will require buffering the data.
        assert A.shape0s[i] <= 1
        assert B.shape0s[i] == 1

    text = """
        ////////// MAIN FUNCTION //////////
        __kernel void fn(
            __global const int *shape0s,
            __global const int *Xstarts,
            __global const ${Xtype} *Xdata,
            __global const int *Ystarts,
            __global ${Ytype} *Ydata,
            __global const int *Astarts,
            __global const int *Ashape0s,
            __global const ${Atype} *Adata,
            __global const int *Bstarts,
            __global const int *Bshape0s,
            __global const ${Btype} *Bdata
        )
        {
            const int n = get_global_id(1);
            __global const ${Xtype} *x = Xdata + Xstarts[n];
            __global ${Ytype} *y = Ydata + Ystarts[n];
            __global const ${Atype} *a = Adata + Astarts[n];
            __global const ${Btype} *b = Bdata + Bstarts[n];

            const int na = Ashape0s[n];
            for (int i = get_global_id(0); i < shape0s[n]; i += get_global_size(0))
            {
                if (na == 0) {
                    y[i] = b[0] * x[i];
                } else {
                    // Filtering code assuming one A coeff and one B coeff
                    y[i] *= -a[0];
                    y[i] += b[0] * x[i];
                }
            }
        }
        """

    textconf = dict(
        Xtype=X.cl_buf.ocldtype, Ytype=Y.cl_buf.ocldtype,
        Atype=A.cl_buf.ocldtype, Btype=B.cl_buf.ocldtype
    )
    text = Template(text, output_encoding='ascii').render(**textconf)

    full_args = (
        X.cl_shape0s,
        X.cl_starts,
        X.cl_buf,
        Y.cl_starts,
        Y.cl_buf,
        A.cl_starts,
        A.cl_shape0s,
        A.cl_buf,
        B.cl_starts,
        B.cl_shape0s,
        B.cl_buf,
    )
    _fn = cl.Program(queue.context, text).build().fn
    _fn.set_args(*[arr.data for arr in full_args])

    max_len = min(queue.device.max_work_group_size, max(X.shape0s))
    gsize = (max_len, N)
    lsize = (max_len, 1)
    rval = Plan(queue, _fn, gsize, lsize=lsize, name="cl_filter_synapses", tag=tag)
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
    N = len(X)

    periods = np.asarray(periods, dtype='float32')
    cl_periods = to_device(queue, periods)
    cl_countdowns = to_device(queue, periods - 1)
    cl_bufpositions = to_device(queue, np.zeros(N, dtype='int32'))

    assert X.cl_buf.ocldtype == Y.cl_buf.ocldtype

    ### N.B.  X[i].shape = (M, N)
    ###       Y[i].shape = (buf_len, M * N)

    for i in xrange(N):
        assert X.shape0s[i] * X.shape1s[i] == Y.shape1s[i]
        assert X.stride0s[i] == X.shape1s[i]
        assert X.stride1s[i] == 1
        assert Y.stride0s[i] == Y.shape1s[i]
        assert Y.stride1s[i] == 1

    text = """
        ////////// MAIN FUNCTION //////////
        __kernel void fn(
            __global float *countdowns,
            __global int *bufpositions,
            __global const float *periods,
            __global const int *Xstarts,
            __global const int *Xshape0s,
            __global const int *Xshape1s,
            __global const ${Xtype} *Xdata,
            __global const int *Ystarts,
            __global ${Ytype} *Ydata
        )
        {
            const int n = get_global_id(1);
            const float countdown = countdowns[n];

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
            Xtype=X.cl_buf.ocldtype,
            Ytype=Y.cl_buf.ocldtype)
    text = Template(text, output_encoding='ascii').render(**textconf)

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
    _fn = cl.Program(queue.context, text).build().fn
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
    input_types = [x.cl_buf.ocldtype for x in inputs]
    output_type = output.cl_buf.ocldtype

    text = """
        ////////// MAIN FUNCTION //////////
        __kernel void fn(
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
    text = Template(text, output_encoding='ascii').render(**textconf)

    full_args = []
    for x in inputs:
        full_args.extend([x.cl_starts, x.cl_buf])
    full_args.extend([output.cl_starts, output.cl_buf])
    # full_args = (X.cl_starts, X.cl_buf, Y.cl_starts, Y.cl_buf)
    _fn = cl.Program(queue.context, text).build().fn
    _fn.set_args(*[arr.data for arr in full_args])

    gsize = (N,)
    rval = Plan(queue, _fn, gsize, lsize=None, name="cl_direct", tag=tag)
    rval.full_args = full_args     # prevent garbage-collection
    return rval


def plan_lif(queue, J, V, W, outV, outW, outS, ref, tau, dt,
             tag=None, n_elements=0, upsample=1):
    inputs = dict(j=J, v=V, w=W)
    outputs = dict(ov=outV, ow=outW, os=outS)
    parameters = dict(tau=tau, ref=ref)

    dt = float(dt)
    textconf = dict(upsample=upsample,
                    dtu=dt/upsample,
                    dtu_inv=upsample/dt,
                    dt_inv=1/dt,
                    V_threshold=1.)

    declares = """
            char spiked;
            %(Vtype)s dV, overshoot;
            """ % ({'Vtype': V.cl_buf.ocldtype})

    text = """
            spiked = 0;

% for ii in range(upsample):
            dV = (${dtu} / tau) * (j - v);
            v += dV;

            if (v < 0 || w > 2*${dtu})
                v = 0;
            else if (w > ${dtu})
                v *= 1.0 - (w - ${dtu}) * ${dtu_inv};

            if (v > ${V_threshold}) {
                overshoot = ${dtu} * (v - ${V_threshold}) / dV;
                w = ref - overshoot + ${dtu};
                v = 0.0;
                spiked = 1;
            } else {
                w -= ${dtu};
            }
% endfor
            ov = v;
            ow = w;
            os = (spiked) ? ${dt_inv} : 0.0f;
            """
    text = Template(text, output_encoding='ascii').render(**textconf)

    return _plan_template(
        queue, "cl_lif", text, declares=declares,
        tag=tag, n_elements=n_elements,
        inputs=inputs, outputs=outputs, parameters=parameters)


def plan_lif_rate(queue, J, R, ref, tau, dt, tag=None, n_elements=0):
    inputs = dict(j=J)
    outputs = dict(r=R)
    parameters = dict(tau=tau, ref=ref)
    text = """
            j = max(j - 1, 0.0f);
            r = 1.0 / (ref + tau * log1p(1.0/j));
            """

    return _plan_template(
        queue, "cl_lif_rate", text, tag=tag, n_elements=n_elements,
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

    base = inputs.values()[0]   # input to use as reference (for lengths)
    N = len(base)

    ### split parameters into static and updated params
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
    for vname, v in inputs.items() + outputs.items():
        assert vname not in avars, "Name clash"
        assert len(v) == N
        assert all_equal(v.shape0s, base.shape0s)

        ### N.B. - we should be able to ignore ldas as long as all vectors
        assert all_equal(v.shape1s, 1)

        dtype = v.cl_buf.ocldtype
        offset = '%(name)s_starts[n]' % {'name': vname}
        avars[vname] = (dtype, offset)

    for vname, v in params.items():
        assert vname not in avars, "Name clash"
        assert len(v) == N
        for i in xrange(N):
            assert v.shape0s[i] == base.shape0s[i] or v.shape0s[i] == 1, \
                "%s.shape0s[%d] must be 1 or %d (not %d)" % \
                (vname, i, base.shape0s[i], v.shape0s[i])
            assert v.shape1s[i] == 1

        dtype = v.cl_buf.ocldtype
        offset = '%(name)s_starts[n]' % {'name': vname}
        avars[vname] = (dtype, offset)

    ivars = dict((k, avars[k]) for k in inputs.keys())
    ovars = dict((k, avars[k]) for k in outputs.keys())
    pvars = dict((k, avars[k]) for k in params.keys())

    textconf = dict(N=N, n_elements=n_elements, tag=str(tag),
                    declares=declares, core_text=core_text,
                    ivars=ivars, ovars=ovars, pvars=pvars,
                    static_params=static_params)

    if n_elements > 0:
        ### Allocate the exact number of required kernels in a vector
        gsize = (int(np.ceil(np.sum(base.shape0s) / float(n_elements))),)
        text = """
        ////////// MAIN FUNCTION //////////
        __kernel void fn(
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
% for name, [type, offset] in ivars.items() + ovars.items() + pvars.items():
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

    % for name, [type, offset] in ivars.items() + ovars.items() + pvars.items():
                cur_${name} = in_${name} + ${offset};
    % endfor
    % for name, [type, offset] in pvars.items():
                ${name}_isvector = ${name}_shape0s[n] > 1;
                if (!${name}_isvector) ${name} = *cur_${name};
    % endfor
            } else {
    % for name, [type, offset] in ivars.items() + ovars.items():
                cur_${name}++;
    % endfor
    % for name, [type, offset] in pvars.items():
                if (${name}_isvector) cur_${name}++;
    % endfor
            }
  % endif
% endfor
        }
        """
    else:
        ### Allocate more than enough kernels in a matrix
        gsize = (int(np.max(base.shape0s)), int(N))
        text = """
        ////////// MAIN FUNCTION //////////
        __kernel void fn(
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

    text = Template(text, output_encoding='ascii').render(**textconf)
    if 0:
        for i, line in enumerate(text.split('\n')):
            print "%3d %s" % (i + 1, line)

    full_args = []
    for vname, v in inputs.items() + outputs.items():
        full_args.extend([v.cl_starts, v.cl_buf])
    for vname, v in params.items():
        full_args.extend([v.cl_starts, v.cl_shape0s, v.cl_buf])
    full_args.append(base.cl_shape0s)
    full_args = tuple(full_args)

    _fn = cl.Program(queue.context, text).build().fn
    _fn.set_args(*[arr.data for arr in full_args])

    rval = Plan(queue, _fn, gsize, lsize=None, name=name, tag=tag)
    rval.full_args = full_args     # prevent garbage-collection
    return rval
