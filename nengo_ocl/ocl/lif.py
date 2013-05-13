import pyopencl as cl
from plan import Plan

def plan_lif(queue, V, RT, J, OV, ORT, OS, dt,
        tau_rc, tau_ref, V_threshold, upsample):
    V_threshold = 1.0
    tau_rc_inv = 1.0 / tau_rc

    upsample_dt = dt / upsample
    upsample_dt_inv = 1.0 / upsample_dt

    Jtype = J.ocldtype
    Vtype = V.ocldtype
    RTtype = RT.ocldtype
    OStype = OS.ocldtype

    _fn = cl.Program(queue.context, """
        __kernel void foo(
            __global const %(Jtype)s *J,
            __global const %(Vtype)s *voltage,
            __global const %(RTtype)s *refractory_time,
            __global %(Vtype)s *out_voltage,
            __global %(RTtype)s *out_refractory_time,
            __global %(OStype)s *out_spiked
                     )
        {
            const %(RTtype)s dt = %(upsample_dt)s;
            const %(RTtype)s dt_inv = %(upsample_dt_inv)s;
            const %(RTtype)s tau_ref = %(tau_ref)s;
            const %(Vtype)s tau_rc_inv = %(tau_rc_inv)s;
            const %(Vtype)s V_threshold = %(V_threshold)s;

            const int gid = get_global_id(0);
            %(Vtype)s v = voltage[gid];
            %(RTtype)s rt = refractory_time[gid];
            %(Jtype)s input_current = J[gid];
            int spiked = 0;

            for (int ii = 0; ii < %(upsample)s; ++ii)
            {
              %(Vtype)s dV = dt * tau_rc_inv * (input_current - v);
              %(RTtype)s post_ref = 1.0 - (rt - dt) * dt_inv;
              v += dV;
              v = v > 0 ?
                  v * (post_ref < 0 ? 0.0 : post_ref < 1 ? post_ref : 1.0)
                  : 0;
              const int spiked_ii = v > V_threshold;
              %(Vtype)s overshoot = (v - V_threshold) / dV;
              %(RTtype)s spiketime = dt * (1.0 - overshoot);

              if (spiked_ii)
              {
                v = 0.0;
                rt = spiketime + tau_ref;
                spiked = 1;
              }
              else
              {
                rt -= dt;
              }
            }

            out_voltage[gid] = v;
            out_refractory_time[gid] = rt;
            out_spiked[gid] = spiked ? (%(OStype)s) 1 : (%(OStype)s) 0;
        }
        """ % locals()).build().foo

    # XXX ASSERT ALL CONTIGUOUS WITH IDENTICAL LAYOUT

    _fn_args = (queue, (V.size,), None,
                J.data, V.data, RT.data,
                OV.data, ORT.data, OS.data)
    return Plan(locals())

