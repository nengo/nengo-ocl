from mako.template import Template
import pyopencl as cl
from plan import Plan

def plan_lif(queue, V, RT, J, OV, ORT, OS,
             dt, tau_rc, tau_ref, V_threshold, upsample):
    L, = V.shape

    config = {
        'tau_ref': tau_ref,
        'tau_rc_inv': 1.0 / tau_rc,
        'V_threshold': V_threshold,
        'upsample': upsample,
        'upsample_dt': dt / upsample,
        'upsample_dt_inv': upsample / dt,
    }
    for vname in 'V', 'RT', 'J', 'OV', 'ORT', 'OS':
        v = locals()[vname]
        config[vname + 'type'] = v.ocldtype
        config[vname + 'offset'] = 'gid+%s' % int(v.offset // v.dtype.itemsize)
        assert v.shape == (L,)
        assert v.strides == (v.dtype.itemsize,)


    text = """
        __kernel void foo(
            __global const ${Jtype} *J,
            __global const ${Vtype} *voltage,
            __global const ${RTtype} *refractory_time,
            __global ${Vtype} *out_voltage,
            __global ${RTtype} *out_refractory_time,
            __global ${OStype} *out_spiked
                     )
        {
            const int gid = get_global_id(0);
            ${Vtype} v = voltage[${Voffset}];
            ${RTtype} rt = refractory_time[${RToffset}];
            ${Jtype} j = J[${Joffset}];
            char spiked = 0;
            ${Vtype} dV, overshoot;
            ${RTtype} post_ref, spiketime;

          % for ii in range(upsample):
            dV = ${upsample_dt} * ${tau_rc_inv} * (j - v);
            post_ref = 1.0 - (rt - ${upsample_dt}) * ${upsample_dt_inv};
            v += dV;
            v = v > 0 ?
                v * (post_ref < 0 ? 0.0 : post_ref < 1 ? post_ref : 1.0)
                : 0;
            spiked |= v > ${V_threshold};
            overshoot = (v - ${V_threshold}) / dV;
            spiketime = ${upsample_dt} * (1.0f - overshoot);
            rt = (v > ${V_threshold}) ?
                spiketime + ${tau_ref}
                : rt - ${upsample_dt};
            v = (v > ${V_threshold}) ? 0.0f: v;
          % endfor

            out_voltage[${OVoffset}] = v;
            out_refractory_time[${ORToffset}] = rt;
            out_spiked[${OSoffset}] = spiked ? 1.0f : 0.0f;
        }
        """

    text = Template(text, output_encoding='ascii').render(**config)
    build_options = [
            '-cl-fast-relaxed-math',
            '-cl-mad-enable',
            #'-cl-strict-aliasing',
            ]
    fn = cl.Program(queue.context, text).build(build_options).foo

    fn.set_args(J.data, V.data, RT.data, OV.data, ORT.data,
                OS.data)

    return Plan(queue, fn, (L,), None, name='lif')

