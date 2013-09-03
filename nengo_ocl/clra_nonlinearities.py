
import numpy as np
import pyopencl as cl
from plan import Plan
from mako.template import Template
from clarray import to_device
from .clraggedarray import CLRaggedArray

def equal_arrays(a, b):
    return (np.asarray(a) == np.asarray(b)).all()

def plan_lif(queue, J, V, W, OV, OW, OS, ref, tau, dt, tag=None, upsample=1):
    """
    """
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
        queue, "lif_step", text, declares=declares, tag=tag, n_elements=0,
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
        Parameters to the function. RaggedArrays must be a list of vectors.
        Providing a float instead of a RaggedArray makes that parameter
        constant.

    """

    base = inputs.values()[0]   # input to use as reference (for lengths)
    N = len(base)

    ### split parameters into static and updated params
    sparams = {}  # static params (hard-coded)
    uparams = {}  # variable params (updated)
    for k, v in parameters.items():
        if isinstance(v, CLRaggedArray):
            uparams[k] = v
        else:
            try:
                sparams[k] = ('float', float(v))
            except TypeError:
                raise

    ivars = dict(inputs.items() + uparams.items())
    ovars = dict(outputs)
    avars = ivars.items() + ovars.items()

    variables = {}
    for vname, v in avars:
        assert vname not in variables, "Name clash"
        assert len(v) == N
        assert equal_arrays(v.shape0s, base.shape0s)

        ### N.B. - we should be able to ignore ldas as long as all vectors
        assert equal_arrays(v.shape1s, 1)

        dtype = v.cl_buf.ocldtype
        offset = '%(name)s_starts[n]' % {'name': vname}
        variables[vname] = (dtype, offset)

    ivariables = dict((k, variables[k]) for k in ivars.keys())
    ovariables = dict((k, variables[k]) for k in ovars.keys())

    textconf = dict(N=N, n_elements=n_elements, tag=str(tag),
                    declares=declares, core_text=core_text,
                    variables=variables, sparams=sparams,
                    ivariables=ivariables, ovariables=ovariables)

    if n_elements > 0:
        ### Allocate the exact number of required kernels in a vector
        gsize = (int(np.ceil(np.sum(base.shape0s) / float(n_elements))),)
        text = """
        ////////// MAIN FUNCTION //////////
        __kernel void fn(
% for name, [type, offset] in ivariables.items():
            __global const int *${name}_starts,
            __global const ${type} *in_${name},
% endfor
% for name, [type, offset] in ovariables.items():
            __global const int *${name}_starts,
            __global ${type} *in_${name},
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

% for name, [type, offset] in ivariables.items():
            __global const ${type} *cur_${name} = in_${name} + ${offset} + m;
% endfor
% for name, [type, offset] in ovariables.items():
            __global ${type} *cur_${name} = in_${name} + ${offset} + m;
% endfor
% for name, [type, offset] in variables.items():
            ${type} ${name};
% endfor
% for name, [type, value] in sparams.items():
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
  % for name, [type, offset] in ivariables.items():
            ${name} = *cur_${name};
  % endfor

            /////vvvvv USER COMPUTATIONS BELOW vvvvv
            ${core_text}
            /////^^^^^ USER COMPUTATIONS ABOVE ^^^^^

  % for name, [type, offset] in ovariables.items():
            *cur_${name} = ${name};
  % endfor

  % if ii + 1 < n_elements:
            m++;
            if (m >= lengths[n]) {
                n++;
                m = 0;
                if (n >= ${N}) return;

    % for name, [type, offset] in variables.items():
                cur_${name} = in_${name} + ${offset};
    % endfor
            } else {
    % for name, [type, offset] in variables.items():
                cur_${name}++;
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
% for name, [type, offset] in ivariables.items():
            __global const int *${name}_starts,
            __global const ${type} *in_${name},
% endfor
% for name, [type, offset] in ovariables.items():
            __global const int *${name}_starts,
            __global ${type} *in_${name},
% endfor
            __global const int *lengths
        )
        {
            const int m = get_global_id(0);
            const int n = get_global_id(1);
            const int M = lengths[n];
            if (m >= M) return;

% for name, [type, offset] in ivariables.items():
            ${type} ${name} = in_${name}[${offset} + m];
% endfor
% for name, [type, offset] in ovariables.items():
            ${type} ${name};
% endfor
% for name, [type, value] in sparams.items():
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

% for name, [type, offset] in ovariables.items():
            in_${name}[${offset} + m] = ${name};
% endfor
        }
        """

    text = Template(text, output_encoding='ascii').render(**textconf)
    if 0:
        for i, line in enumerate(text.split('\n')):
            print "%3d %s" % (i + 1, line)

    full_args = []
    for name, v in avars:
        full_args.extend([v.cl_starts, v.cl_buf])
    full_args.append(base.cl_shape0s)
    full_args = tuple(full_args)

    _fn = cl.Program(queue.context, text).build().fn
    _fn.set_args(*[arr.data for arr in full_args])

    rval = Plan(queue, _fn, gsize, lsize=None, name=name, tag=tag)
    rval.full_args = full_args     # prevent garbage-collection
    return rval

