
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

def plan_probes(queue, sim_step, P, X, Y, tag=None):
    """
    Parameters
    ----------
    P : raggedarray of ints
        The period (in time-steps) of each probe
    """

    assert len(X) == len(Y)
    assert len(X) == len(P)
    N = len(X)

    assert len(sim_step) == 1
    assert sim_step.shape0s[0] == 1 and sim_step.shape1s[0] == 1
    assert sim_step.ldas[0] == 1

    assert P.cl_buf.ocldtype == 'int'
    assert X.cl_buf.ocldtype == Y.cl_buf.ocldtype

    ### N.B.  X[i].shape = (ndims[i], )
    ###       Y[i].shape = (buf_ndims[i], buf_len)
    for i in xrange(N):
        assert X.shape0s[i] == Y.shape0s[i]
        assert X.shape1s[i] == 1

    text = """
        ////////// MAIN FUNCTION //////////
        __kernel void fn(
            __global const ${Stype} *step_data,
            __global const int *Pstarts,
            __global const int *Pdata,
            __global const int *Xstarts,
            __global const int *Xshape0s,
            __global const ${Xtype} *Xdata,
            __global const int *Ystarts,
            __global const int *Yshape1s,
            __global const int *Yldas,
            __global ${Ytype} *Ydata
        )
        {
            const int n = get_global_id(0);
            if (n >= ${N}) return;

            const int sim_step = (int)step_data[${step_start}];
            const int period = Pdata[Pstarts[n]];

            if ((sim_step % period) == 0) {
                const int n_dims = Xshape0s[n];
                __global const ${Xtype} *x = Xdata + Xstarts[n];

                const int probe_step = sim_step / period;
                const int buf_len = Yshape1s[n];
                __global ${Ytype} *y = Ydata + Ystarts[n]
                                     + Yldas[n] * (probe_step % buf_len);

                for (int i = 0; i < n_dims; i++)
                    y[i] = x[i];
            }
        }
        """

    textconf = dict(N=N, step_start=sim_step.starts[0],
                    Stype=sim_step.cl_buf.ocldtype,
                    Xtype=X.cl_buf.ocldtype, Ytype=Y.cl_buf.ocldtype)
    text = Template(text, output_encoding='ascii').render(**textconf)

    full_args = (
        sim_step.cl_buf,
        P.cl_starts, P.cl_buf,
        X.cl_starts, X.cl_shape0s, X.cl_buf,
        Y.cl_starts, Y.cl_shape1s, Y.cl_ldas, Y.cl_buf,
        )
    _fn = cl.Program(queue.context, text).build().fn
    _fn.set_args(*[arr.data for arr in full_args])

    gsize = (N,)
    rval = Plan(queue, _fn, gsize, lsize=None, name="probes", tag=tag)
    rval.full_args = full_args     # prevent garbage-collection
    return rval

def plan_direct(queue, code, init, Xname, X, Y, tag=None):
    from . import ast_conversion

    assert len(X) == len(Y)
    N = len(X)

    text = """
        ////////// MAIN FUNCTION //////////
        __kernel void fn(
            __global const int *${IN}starts,
            __global const ${INtype} *${IN}data,
            __global const int *${OUT}starts,
            __global ${OUTtype} *${OUT}data
        )
        {
            const int n = get_global_id(0);
            if (n >= ${N}) return;

            __global const ${INtype} *${arg} = ${IN}data + ${IN}starts[n];
            __global ${OUTtype} *${OUT} = ${OUT}data + ${OUT}starts[n];

            //vvvvv USER DECLARATIONS BELOW vvvvv
${init}
            //^^^^^ USER DECLARATIONS ABOVE ^^^^^

            /////vvvvv USER COMPUTATIONS BELOW vvvvv
${code}
            /////^^^^^ USER COMPUTATIONS ABOVE ^^^^^
        }
        """

    textconf = dict(init=_indent(init, 12),
                    code=_indent(code, 12), N=N, arg=Xname,
                    IN=ast_conversion.INPUT_NAME, INtype=X.cl_buf.ocldtype,
                    OUT=ast_conversion.OUTPUT_NAME, OUTtype=Y.cl_buf.ocldtype,
                    )
    text = Template(text, output_encoding='ascii').render(**textconf)

    full_args = (X.cl_starts, X.cl_buf, Y.cl_starts, Y.cl_buf)
    _fn = cl.Program(queue.context, text).build().fn
    _fn.set_args(*[arr.data for arr in full_args])

    gsize = (N,)
    rval = Plan(queue, _fn, gsize, lsize=None, name="direct_mode", tag=tag)
    rval.full_args = full_args     # prevent garbage-collection
    return rval

def plan_lif(queue, J, V, W, OV, OW, OS, ref, tau, dt,
             tag=None, n_elements=0, upsample=1):
    inputs = dict(j=J, v=V, w=W)
    outputs = dict(ov=OV, ow=OW, os=OS)
    parameters = dict(tau=tau, ref=ref)

    dt = float(dt)
    textconf = dict(upsample=upsample, dt=dt/upsample, dt_inv=upsample/dt,
                    V_threshold=1.)

    declares = """
            char spiked;
            %(Vtype)s dV, overshoot;
            """ % ({'Vtype': V.cl_buf.ocldtype})

    text = """
            spiked = 0;

% for ii in range(upsample):
            dV = (${dt} / tau) * (j - v);
            v += dV;

            if (v < 0 || w > 2*${dt})
                v = 0;
            else if (w > ${dt})
                v *= 1.0 - (w - ${dt}) * ${dt_inv};

            if (v > ${V_threshold}) {
                overshoot = ${dt} * (v - ${V_threshold}) / dV;
                w = ref - overshoot + ${dt};
                v = 0.0;
                spiked = 1;
            } else {
                w -= ${dt};
            }
% endfor
            ov = v;
            ow = w;
            os = (spiked) ? 1.0f : 0.0f;
            """
    text = Template(text, output_encoding='ascii').render(**textconf)

    return _plan_template(
        queue, "lif_step", text, declares=declares,
        tag=tag, n_elements=n_elements,
        inputs=inputs, outputs=outputs, parameters=parameters)

def plan_lif_rate(queue, J, R, ref, tau, tag=None, n_elements=0):
    inputs = dict(j=J)
    outputs = dict(r=R)
    parameters = dict(tau=tau, ref=ref)
    text = """
            j = max(j - 1, 0.0f);
            r = 1.0 / (ref + tau * log1p(1.0/j));
            """
    return _plan_template(
        queue, "lif_rate", text, tag=tag, n_elements=n_elements,
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

