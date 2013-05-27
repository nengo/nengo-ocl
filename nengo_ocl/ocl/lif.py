import math
from mako.template import Template
import pyopencl as cl
from plan import Plan

def plan_lif(queue, V, RT, J, OV, ORT, OS,
             dt, tau_rc, tau_ref, V_threshold, upsample):
    tau_rc_inv = 1.0 / tau_rc

    upsample_dt = dt / upsample
    upsample_dt_inv = 1.0 / upsample_dt

    Jtype = J.ocldtype
    Vtype = V.ocldtype
    RTtype = RT.ocldtype
    OStype = OS.ocldtype

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
            ${Vtype} v = voltage[gid];
            ${RTtype} rt = refractory_time[gid];
            ${Jtype} j = J[gid];
            ${OStype} spiked = 0;
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
            spiketime = ${upsample_dt} * (1.0 - overshoot);
            rt = (v > ${V_threshold}) ?
                spiketime + ${tau_ref}
                : rt - ${upsample_dt};
            v = (v > ${V_threshold}) ? 0.0: v;
          % endfor

            out_voltage[gid] = v;
            out_refractory_time[gid] = rt;
            out_spiked[gid] = spiked;
        }
        """

    text = Template(text, output_encoding='ascii').render(**locals())
    build_options = [
            '-cl-fast-relaxed-math',
            '-cl-mad-enable',
            '-cl-strict-aliasing',
            ]
    fn = cl.Program(queue.context, text).build(build_options).foo

    fn.set_args(J.data, V.data, RT.data, OV.data, ORT.data,
                OS.data)

    # XXX ASSERT ALL CONTIGUOUS WITH IDENTICAL LAYOUT
    # TODO: Solve by compiling kernel using elemwise.py
    L, = V.shape
    return Plan(queue, fn, (L,), None, name='lif')

