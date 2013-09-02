
import numpy as np
import pyopencl as cl
from plan import Plan
from mako.template import Template
from clarray import to_device

def equal_arrays(a, b):
    return (np.asarray(a) == np.asarray(b)).all()

def plan_lif(queue, J, V, W, OV, OW, OS, ref, tau, dt, tag=None):
    """
    """

    ### EH: ideas for speed
    ### - make separate inline function for each population, switch statement
    ### - flatten grid, so we don't have so many returning kernel calls

    V_threshold = 1.

    N = len(J)

    textconf = {
        'ref': ref,
        'tau': tau,
        'dt': dt,
        'dt_inv': 1. / dt,
        'V_threshold': V_threshold,
        'tag': str(tag),
    }


    for vname in ['J', 'V', 'W', 'OV', 'OW', 'OS']:
        v = locals()[vname]
        textconf[vname + 'type'] = v.cl_buf.ocldtype
        textconf[vname + 'offset'] = '%(name)s_starts[n] + m' % {'name': vname}
        assert len(v) == N
        assert equal_arrays(v.shape0s, J.shape0s)

        ### N.B. - we should be able to ignore ldas as long as all vectors
        assert equal_arrays(v.shape1s, 1)

    text = """
        __kernel void lif_step(
            __global const int *lengths,
            __global const int *J_starts,
            __global const ${Jtype} *J,
            __global const int *V_starts,
            __global const ${Vtype} *V,
            __global const int *W_starts,
            __global const ${Wtype} *W,
            __global const int *OV_starts,
            __global ${Vtype} *OV,
            __global const int *OW_starts,
            __global ${Wtype} *OW,
            __global const int *OS_starts,
            __global ${OStype} *OS
                     )
        {
            const int m = get_global_id(0);
            const int n = get_global_id(1);
            const int M = lengths[n];
            if (m >= M) return;

            ${Jtype} j = J[${Joffset}];
            ${Vtype} v = V[${Voffset}];
            ${Wtype} w = W[${Woffset}];

            char spiked = 0;
            ${Vtype} dV, overshoot;
            ${Wtype} post_ref, spiketime;

            dV = (${dt} / ${tau}) * (j - v);
            v += dV;

            post_ref = 1.0 - (w - ${dt}) * ${dt_inv};
            v = v > 0 ?
                v * (post_ref < 0 ? 0.0 : post_ref < 1 ? post_ref : 1.0)
                : 0;
            spiked = v > ${V_threshold};
            if (v > ${V_threshold}) {
                overshoot = (v - ${V_threshold}) / dV;
                spiketime = ${dt} * (1.0f - overshoot);
                w = ${ref} + spiketime;
                v = 0.0;
            } else {
                w -= ${dt};
            }

            OV[${OVoffset}] = v;
            OW[${OWoffset}] = w;
            OS[${OSoffset}] = spiked ? 1.0f : 0.0f;
        }
        """

    text = Template(text, output_encoding='ascii').render(**textconf)
    # for i, line in enumerate(text.split('\n')):
    #     print "%2d %s" % (i, line)

    gsize = (int(max(J.shape0s)), int(N))
    lsize = None
    _fn = cl.Program(queue.context, text).build().lif_step
    full_args = (
        J.cl_shape0s,
        J.cl_starts,
        J.cl_buf,
        V.cl_starts,
        V.cl_buf,
        W.cl_starts,
        W.cl_buf,
        OV.cl_starts,
        OV.cl_buf,
        OW.cl_starts,
        OW.cl_buf,
        OS.cl_starts,
        OS.cl_buf
        )

    #print [str(arr.dtype)[0] for arr in full_args]
    _fn.set_args(*[arr.data for arr in full_args])
    rval = Plan(queue, _fn, gsize, lsize,
                name='lif_step',
                tag=tag,
               )
    # prevent garbage-collection
    rval.full_args = full_args
    return rval



# from mako.template import Template
# import pyopencl as cl
# from plan import Plan

# def plan_lif(queue, V, RT, J, OV, ORT, OS,
#              dt, tau_rc, tau_ref, V_threshold, upsample):
#     L, = V.shape

#     config = {
#         'tau_ref': tau_ref,
#         'tau_rc_inv': 1.0 / tau_rc,
#         'V_threshold': V_threshold,
#         'upsample': upsample,
#         'upsample_dt': dt / upsample,
#         'upsample_dt_inv': upsample / dt,
#     }
#     for vname in 'V', 'RT', 'J', 'OV', 'ORT', 'OS':
#         v = locals()[vname]
#         config[vname + 'type'] = v.ocldtype
#         config[vname + 'offset'] = 'gid+%s' % int(v.offset // v.dtype.itemsize)
#         assert v.shape == (L,)
#         assert v.strides == (v.dtype.itemsize,)


#     text = """
#         __kernel void foo(
#             __global const ${Jtype} *J,
#             __global const ${Vtype} *voltage,
#             __global const ${RTtype} *refractory_time,
#             __global ${Vtype} *out_voltage,
#             __global ${RTtype} *out_refractory_time,
#             __global ${OStype} *out_spiked
#                      )
#         {
#             const int gid = get_global_id(0);
#             ${Vtype} v = voltage[${Voffset}];
#             ${RTtype} rt = refractory_time[${RToffset}];
#             ${Jtype} j = J[${Joffset}];
#             char spiked = 0;
#             ${Vtype} dV, overshoot;
#             ${RTtype} post_ref, spiketime;

#           % for ii in range(upsample):
#             dV = ${upsample_dt} * ${tau_rc_inv} * (j - v);
#             post_ref = 1.0 - (rt - ${upsample_dt}) * ${upsample_dt_inv};
#             v += dV;
#             v = v > 0 ?
#                 v * (post_ref < 0 ? 0.0 : post_ref < 1 ? post_ref : 1.0)
#                 : 0;
#             spiked |= v > ${V_threshold};
#             overshoot = (v - ${V_threshold}) / dV;
#             spiketime = ${upsample_dt} * (1.0f - overshoot);
#             rt = (v > ${V_threshold}) ?
#                 spiketime + ${tau_ref}
#                 : rt - ${upsample_dt};
#             v = (v > ${V_threshold}) ? 0.0f: v;
#           % endfor

#             out_voltage[${OVoffset}] = v;
#             out_refractory_time[${ORToffset}] = rt;
#             out_spiked[${OSoffset}] = spiked ? 1.0f : 0.0f;
#         }
#         """

#     text = Template(text, output_encoding='ascii').render(**config)
#     build_options = [
#             '-cl-fast-relaxed-math',
#             '-cl-mad-enable',
#             #'-cl-strict-aliasing',
#             ]
#     fn = cl.Program(queue.context, text).build(build_options).foo

#     fn.set_args(J.data, V.data, RT.data, OV.data, ORT.data,
#                 OS.data)

#     return Plan(queue, fn, (L,), None, name='lif')

