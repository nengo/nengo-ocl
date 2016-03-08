import inspect
import logging
import os
import warnings
from collections import OrderedDict

import numpy as np
import pyopencl as cl

from nengo.synapses import LinearFilter
from nengo.utils.progress import ProgressTracker
from nengo.utils.stdlib import groupby

from nengo_ocl import sim_npy
from nengo_ocl.clraggedarray import CLRaggedArray, to_device
from nengo_ocl.clra_gemv import plan_ragged_gather_gemv
from nengo_ocl.clra_nonlinearities import (
    plan_timeupdate, plan_reset, plan_copy, plan_slicedcopy,
    plan_direct, plan_lif, plan_lif_rate,
    plan_probes, plan_linear_synapse, plan_elementwise_inc,
    init_rng, get_dist_enums_params, plan_whitenoise, plan_presentinput,
    plan_conv2d, plan_pool2d)
from nengo_ocl.plan import BasePlan, PythonPlan, Plans
from nengo_ocl.ast_conversion import OCL_Function
from nengo_ocl.utils import indent, split

logger = logging.getLogger(__name__)
PROFILING_ENABLE = cl.command_queue_properties.PROFILING_ENABLE


def get_closures(f):
    return OrderedDict(zip(
        f.__code__.co_freevars, (c.cell_contents for c in f.__closure__)))


class Simulator(sim_npy.Simulator):

    def Array(self, val, dtype=np.float32):
        return to_device(self.queue, np.asarray(val, dtype=dtype))

    def RaggedArray(self, listofarrays, **kwargs):
        return CLRaggedArray.from_arrays(self.queue, listofarrays, **kwargs)

    def __init__(self, network, dt=0.001, seed=None, model=None, context=None,
                 n_prealloc_probes=32, profiling=None, ocl_only=False):
        if context is None:
            print('No context argument was provided to sim_ocl.Simulator')
            print("Calling pyopencl.create_some_context() for you now:")
            context = cl.create_some_context()
        if profiling is None:
            profiling = int(os.getenv("NENGO_OCL_PROFILING", 0))
        self.context = context
        self.profiling = profiling
        if self.profiling:
            self.queue = cl.CommandQueue(context, properties=PROFILING_ENABLE)
        else:
            self.queue = cl.CommandQueue(context)

        self.n_prealloc_probes = n_prealloc_probes
        self.ocl_only = ocl_only
        self._cl_rng_state = None

        # -- allocate data
        sim_npy.Simulator.__init__(
            self, network=network, dt=dt, seed=seed, model=model)

        # -- create object to execute list of plans
        self._plans = Plans(self._plan, self.profiling)

    def _init_cl_rng(self):
        if self._cl_rng_state is None:
            self._cl_rng_state = init_rng(self.queue, self.seed)

    def _prep_all_data(self):
        # -- replace the numpy-allocated RaggedArray with OpenCL one
        self.all_data = CLRaggedArray(self.queue, self.all_data)

    def plan_ragged_gather_gemv(self, *args, **kwargs):
        return plan_ragged_gather_gemv(self.queue, *args, **kwargs)

    def plan_TimeUpdate(self, ops):
        op, = ops
        step = self.all_data[[self.sidx[op.step]]]
        time = self.all_data[[self.sidx[op.time]]]
        return [plan_timeupdate(self.queue, step, time, self.model.dt)]

    def plan_Reset(self, ops):
        targets = self.all_data[[self.sidx[op.dst] for op in ops]]
        values = self.Array([op.value for op in ops])
        return [plan_reset(self.queue, targets, values)]

    def plan_SlicedCopy(self, ops):
        copies, ops = split(
            ops, lambda op: op.a_slice is Ellipsis and op.b_slice is Ellipsis)

        plans = []
        if copies:
            A = self.all_data[[self.sidx[op.a] for op in copies]]
            B = self.all_data[[self.sidx[op.b] for op in copies]]
            incs = np.array([op.inc for op in copies], dtype=np.int32)
            plans.append(plan_copy(self.queue, A, B, incs))

        if ops:
            A = self.all_data[[self.sidx[op.a] for op in ops]]
            B = self.all_data[[self.sidx[op.b] for op in ops]]
            inds = lambda ary, i: np.arange(ary.size, dtype=np.int32)[i]
            Ainds = self.RaggedArray([inds(op.a, op.a_slice) for op in ops])
            Binds = self.RaggedArray([inds(op.b, op.b_slice) for op in ops])
            incs = np.array([op.inc for op in ops], dtype=np.int32)
            plans.append(plan_slicedcopy(self.queue, A, B, Ainds, Binds, incs))

        return plans

    def plan_ElementwiseInc(self, ops):
        A = self.all_data[[self.sidx[op.A] for op in ops]]
        X = self.all_data[[self.sidx[op.X] for op in ops]]
        Y = self.all_data[[self.sidx[op.Y] for op in ops]]
        return [plan_elementwise_inc(self.queue, A, X, Y)]

    def plan_SimPyFunc(self, ops):
        # TODO: test with a hybrid program (Python and OCL)

        # group nonlinearities
        unique_ops = OrderedDict()
        for op in ops:
            # assert op.n_args in (1, 2), op.n_args
            op_key = (op.fn, op.t_in, op.x is not None)
            if op_key not in unique_ops:
                unique_ops[op_key] = {'in': [], 'out': []}
            unique_ops[op_key]['in'].append(op.x)
            unique_ops[op_key]['out'].append(op.output)

        # make plans
        plans = []
        for (fn, t_in, x_in), signals in unique_ops.items():
            fn_name = (fn.__name__ if inspect.isfunction(fn) else
                       fn.__class__.__name__)
            if fn_name == "<lambda>":
                fn_name += "%d" % len(plans)

            # check signal input and output shape (implicitly checks
            # for indexing errors)
            vector_dims = lambda shape, dim: len(
                shape) == 1 and shape[0] == dim
            unit_stride = lambda es: len(es) == 1 and es[0] == 1

            if x_in:
                in_dim = signals['in'][0].size
                for sig_in in signals['in']:
                    assert sig_in.size == in_dim
                    assert vector_dims(sig_in.shape, in_dim)
                    assert unit_stride(sig_in.elemstrides)
            else:
                in_dim = None

            # if any functions have no output, must do them in Python
            if any(s is None for s in signals['out']):
                assert all(s is None for s in signals['out'])
                warnings.warn(
                    "Function '%s' could not be converted to OCL since it has "
                    "no outputs." % (fn_name), RuntimeWarning)
                plans.append(self._plan_pythonfn(
                    fn, t_in, signals, fn_name=fn_name))
                continue

            out_dim = signals['out'][0].size
            for sig_out in signals['out']:
                assert sig_out.size == out_dim
                assert vector_dims(sig_out.shape, out_dim)
                assert unit_stride(sig_out.elemstrides)

            # try to get OCL code
            try:
                in_dims = [1] if t_in else []
                in_dims += [in_dim] if x_in else []
                ocl_fn = OCL_Function(fn, in_dims=in_dims, out_dim=out_dim)
                input_names = ocl_fn.translator.arg_names
                inputs = []
                if t_in:  # append time
                    inputs.append(self.all_data[
                        [self.sidx[self._time] for i in signals['out']]])
                if x_in:  # append x
                    inputs.append(self.all_data[
                        [self.sidx[i] for i in signals['in']]])
                output = self.all_data[[self.sidx[i] for i in signals['out']]]
                plan = plan_direct(self.queue, ocl_fn.code, ocl_fn.init,
                                   input_names, inputs, output, tag=fn_name)
                plans.append(plan)
            except Exception as e:
                if self.ocl_only:
                    raise

                warnings.warn(
                    "Function '%s' could not be converted to OCL due to %s%s"
                    % (fn_name, e.__class__.__name__, e.args), RuntimeWarning)

                # not successfully translated to OCL, so do it in Python
                plans.append(self._plan_pythonfn(
                    fn, t_in, signals, fn_name=fn_name))

        return plans

    def _plan_pythonfn(self, fn, t_in, signals, fn_name=""):
        t_idx = self.sidx[self._time]
        in_idx = [self.sidx[s] if s else None for s in signals['in']]
        out_idx = [self.sidx[s] if s else None for s in signals['out']]
        assert len(in_idx) == len(out_idx)
        ix_iy = list(zip(in_idx, out_idx))

        def step():
            t = float(self.all_data[t_idx][0, 0] if t_in else 0)
            for ix, iy in ix_iy:
                if ix is not None:
                    x = self.all_data[ix]
                    if x.ndim == 2 and x.shape[1] == 1:
                        x = x[:, 0]
                    y = fn(t, x) if t_in else fn(x)
                else:
                    y = fn(t) if t_in else fn()
                if iy is not None:
                    y = np.asarray(y)
                    if y.ndim == 1:
                        y = y[:, None]
                    self.all_data[iy] = y

        return PythonPlan(step, name='py_function', tag=fn_name)

    def plan_SimNeurons(self, all_ops):
        groups = groupby(all_ops, lambda op: op.neurons.__class__)
        plans = []
        for neuron_class, ops in groups:
            attr_name = '_plan_%s' % neuron_class.__name__
            if hasattr(self, attr_name):
                plans.extend(getattr(self, attr_name)(ops))
            else:
                raise ValueError("Unsupported neuron type '%s'"
                                 % neuron_class.__name__)
        return plans

    def _plan_LIF(self, ops):
        if not all(op.neurons.min_voltage == 0 for op in ops):
            raise NotImplementedError("LIF min voltage")
        dt = self.model.dt
        J = self.all_data[[self.sidx[op.J] for op in ops]]
        V = self.all_data[[self.sidx[op.states[0]] for op in ops]]
        W = self.all_data[[self.sidx[op.states[1]] for op in ops]]
        S = self.all_data[[self.sidx[op.output] for op in ops]]
        ref = self.RaggedArray([op.neurons.tau_ref * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        tau = self.RaggedArray([op.neurons.tau_rc * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        return [plan_lif(self.queue, dt, J, V, W, S, ref, tau)]

    def _plan_LIFRate(self, ops):
        dt = self.model.dt
        J = self.all_data[[self.sidx[op.J] for op in ops]]
        R = self.all_data[[self.sidx[op.output] for op in ops]]
        ref = self.RaggedArray([op.neurons.tau_ref * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        tau = self.RaggedArray([op.neurons.tau_rc * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        return [plan_lif_rate(self.queue, dt, J, R, ref, tau)]

    def _plan_AdaptiveLIF(self, ops):
        dt = self.model.dt
        J = self.all_data[[self.sidx[op.J] for op in ops]]
        V = self.all_data[[self.sidx[op.states[0]] for op in ops]]
        W = self.all_data[[self.sidx[op.states[1]] for op in ops]]
        N = self.all_data[[self.sidx[op.states[2]] for op in ops]]
        S = self.all_data[[self.sidx[op.output] for op in ops]]
        ref = self.RaggedArray([op.neurons.tau_ref * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        tau = self.RaggedArray([op.neurons.tau_rc * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        tau_n = self.RaggedArray([op.neurons.tau_n * np.ones(op.J.size)
                                  for op in ops], dtype=J.dtype)
        inc_n = self.RaggedArray([op.neurons.inc_n * np.ones(op.J.size)
                                  for op in ops], dtype=J.dtype)
        return [plan_lif(self.queue, dt, J, V, W, S, ref, tau,
                         N=N, tau_n=tau_n, inc_n=inc_n)]

    def _plan_AdaptiveLIFRate(self, ops):
        dt = self.model.dt
        J = self.all_data[[self.sidx[op.J] for op in ops]]
        R = self.all_data[[self.sidx[op.output] for op in ops]]
        N = self.all_data[[self.sidx[op.states[0]] for op in ops]]
        ref = self.RaggedArray([op.neurons.tau_ref * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        tau = self.RaggedArray([op.neurons.tau_rc * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        tau_n = self.RaggedArray([op.neurons.tau_n * np.ones(op.J.size)
                                  for op in ops], dtype=J.dtype)
        inc_n = self.RaggedArray([op.neurons.inc_n * np.ones(op.J.size)
                                  for op in ops], dtype=J.dtype)
        return [plan_lif_rate(self.queue, dt, J, R, ref, tau,
                              N=N, tau_n=tau_n, inc_n=inc_n)]

    def plan_SimSynapse(self, ops):
        for op in ops:
            if not isinstance(op.synapse, LinearFilter):
                raise NotImplementedError(
                    "%r synapses" % type(op.synapse).__name__)
            if op.input.ndim != 1:
                raise NotImplementedError("Can only filter vectors")
        steps = [op.synapse.make_step(self.model.dt, []) for op in ops]
        A = self.RaggedArray([f.den for f in steps], dtype=np.float32)
        B = self.RaggedArray([f.num for f in steps], dtype=np.float32)
        X = self.all_data[[self.sidx[op.input] for op in ops]]
        Y = self.all_data[[self.sidx[op.output] for op in ops]]
        Xbuf = self.RaggedArray([np.zeros((b.size, x.size))
                                 for b, x in zip(B, X)], dtype=np.float32)
        Ybuf = self.RaggedArray([np.zeros((a.size, y.size))
                                 for a, y in zip(A, Y)], dtype=np.float32)
        return [plan_linear_synapse(self.queue, X, Y, A, B, Xbuf, Ybuf)]

    def plan_SimProcess(self, all_ops):
        groups = groupby(all_ops, lambda op: op.process.__class__)
        plans = []
        for process_class, ops in groups:
            attrname = '_plan_' + process_class.__name__
            if hasattr(self, attrname):
                plans.extend(getattr(self, attrname)(ops))
            else:
                raise NotImplementedError("Unsupported process type '%s'"
                                          % process_class.__name__)

        return plans

    def _plan_WhiteNoise(self, ops):
        assert all(op.input is None for op in ops)
        if any(op.process.seed is not None for op in ops):
            raise NotImplementedError("Seeds not supported for WhiteNoise")

        self._init_cl_rng()
        Y = self.all_data[[self.sidx[op.output] for op in ops]]
        scale = self.RaggedArray([op.process.scale for op in ops],
                                 dtype=np.int32)
        enums, params = get_dist_enums_params([op.process.dist for op in ops])
        enums = CLRaggedArray(self.queue, enums)
        params = CLRaggedArray(self.queue, params)
        dt = self.model.dt
        return [plan_whitenoise(self.queue, Y, enums, params, scale, dt,
                                self._cl_rng_state)]

    def _plan_FilteredNoise(self, ops):
        raise NotImplementedError()

    def _plan_WhiteSignal(self, ops):
        Y = self.all_data[[self.sidx[op.output] for op in ops]]
        t = self.all_data[[self.sidx[self._step] for _ in ops]]

        dt = self.model.dt
        signals = []
        for op in ops:
            assert op.input is None and op.output is not None
            rng = op.process.get_rng(self.rng)
            f = op.process.make_step(0, op.output.size, dt, rng)
            signals.append(get_closures(f)['signal'])

        signals = self.RaggedArray(signals, dtype=np.float32)
        return [plan_presentinput(self.queue, Y, t, signals, dt)]

    def _plan_PresentInput(self, ops):
        ps = [op.process for op in ops]
        Y = self.all_data[[self.sidx[op.output] for op in ops]]
        t = self.all_data[[self.sidx[self._step] for _ in ops]]
        inputs = self.RaggedArray([p.inputs.reshape(p.inputs.shape[0], -1)
                                   for p in ps], dtype=np.float32)
        pres_t = self.Array([p.presentation_time for p in ps])
        dt = self.model.dt
        return [plan_presentinput(self.queue, Y, t, inputs, dt, pres_t=pres_t)]

    def _plan_Conv2d(self, ops):
        plans = []
        for op in ops:
            p, f, b = op.process, op.process.filters, op.process.biases
            assert f.ndim in [4, 6]
            conv = (f.ndim == 4)
            X = self.all_data.getitem_device(self.sidx[op.input])
            Y = self.all_data.getitem_device(self.sidx[op.output])
            f = np.array(np.transpose(
                f, (1, 2, 3, 0) if conv else (3, 4, 5, 0, 1, 2)), order='C')
            F = self.Array(f.ravel())
            B = self.Array((np.zeros(p.shape_out) + b).ravel())
            plans.append(plan_conv2d(
                self.queue, X, Y, F, B, p.shape_in, p.shape_out,
                p.filters.shape[-2:], conv, p.padding, p.stride,
                tag="shape_in=%s, shape_out=%s, kernel=%s, conv=%s" % (
                    p.shape_in, p.shape_out, f.shape[-2:], conv)))

        return plans

    def _plan_Pool2d(self, ops):
        plans = []
        for op in ops:
            assert op.process.kind == 'avg'
            p = op.process
            X = self.all_data.getitem_device(self.sidx[op.input])
            Y = self.all_data.getitem_device(self.sidx[op.output])
            shape = p.shape_out + p.shape_in[1:]
            plans.append(plan_pool2d(
                self.queue, X, Y, shape, p.size, p.stride))

        return plans

    def plan_SimBCM(self, ops):
        raise NotImplementedError("BCM learning rule")

    def plan_SimOja(self, ops):
        raise NotImplementedError("Oja's learning rule")

    def plan_probes(self):
        if len(self.model.probes) == 0:
            self._max_steps_between_probes = self.n_prealloc_probes
            self._cl_probe_plan = None
            return []
        else:
            n_prealloc = self.n_prealloc_probes

            probes = self.model.probes
            periods = [1 if p.sample_every is None else
                       p.sample_every / self.dt
                       for p in probes]

            X = self.all_data[
                [self.sidx[self.model.sig[p]['in']] for p in probes]]
            Y = self.RaggedArray(
                [np.zeros((n_prealloc, self.model.sig[p]['in'].size))
                 for p in probes], dtype=np.float32)

            cl_plan = plan_probes(self.queue, periods, X, Y)
            self._max_steps_between_probes = n_prealloc * int(min(periods))
            self._cl_probe_plan = cl_plan
            return [cl_plan]

    def drain_probe_buffers(self):
        self.queue.finish()
        plan = self._cl_probe_plan
        bufpositions = plan.cl_bufpositions.get()
        for i, probe in enumerate(self.model.probes):
            shape = self.model.sig[probe]['in'].shape
            n_buffered = bufpositions[i]
            if n_buffered:
                # XXX: this syntax retrieves *ALL* of Y from the device
                #      because the :n_buffered only works on the ndarray
                #      *after* it has been transferred.
                raw = plan.Y[i][:n_buffered]
                shaped = raw.reshape((n_buffered,) + shape)
                self._probe_outputs[probe].extend(shaped)
        plan.cl_bufpositions.fill(0)
        self.queue.finish()

    def step(self):
        return self.run_steps(1, progress_bar=False)

    def run_steps(self, N, progress_bar=True):
        if self.closed:
            raise ValueError("Simulator cannot run because it is closed.")

        has_probes = self._cl_probe_plan is not None

        if has_probes:
            # -- precondition: the probe buffers have been drained
            bufpositions = self._cl_probe_plan.cl_bufpositions.get()
            assert np.all(bufpositions == 0)

        with ProgressTracker(N, progress_bar) as progress:
            # -- we will go through N steps of the simulator
            #    in groups of up to B at a time, draining
            #    the probe buffers after each group of B
            while N:
                B = min(N, self._max_steps_between_probes)
                self._plans.call_n_times(B)
                if has_probes:
                    self.drain_probe_buffers()
                N -= B
                self.n_steps += B
                progress.step(n=B)

        if self.profiling > 1:
            self.print_profiling()

    def close(self):
        super(Simulator, self).close()
        self.context = None
        self.queue = None
        self.all_data = None
        self._plan = []
        self._plans = None
        self._cl_rng_state = None
        self._cl_probe_plan = None

    def print_plans(self):
        print(" Plans ".center(80, '-'))
        for plan in self._plans.plans:
            print("%s" % plan)
            if hasattr(plan, 'description'):
                print(indent(plan.description, 4))

    def print_profiling(self, sort=None):
        """
        Parameters
        ----------
        sort : column to sort by (negative number sorts ascending)
            (0 = n_calls, 1 = runtime, 2 = q-time, 3 = subtime)
        """
        if not self.profiling:
            print("Profiling not enabled!")
            return

        # make and sort table
        table = []
        unknowns = []
        for p in self._plans.plans:
            if isinstance(p, BasePlan):
                t = sum(p.ctimes)
                calls_per_sec = p.n_calls / t if t > 0 else np.nan
                gfps = np.nan  # gigaflops / sec
                gbps = np.nan  # gigabytes / sec
                if p.flops_per_call is not None:
                    gfps = 1e-9 * p.flops_per_call * calls_per_sec
                if p.bw_per_call is not None:
                    gbps = 1e-9 * p.bw_per_call * calls_per_sec
                table.append((p.n_calls, t, gfps, gbps, str(p)))
            else:
                unknowns.append((str(p), getattr(p, 'cumtime', '<unknown>')))

        if sort is not None:
            reverse = sort >= 0
            table.sort(key=lambda x: x[abs(sort)], reverse=reverse)

        # print table
        print(" Profiling ".center(80, '-'))
        print('%s\t%s\t%s\t%s' % ('n_calls', 'runtime', 'GF/s', 'GB/s'))

        for r in table:
            print('%i\t%2.3f\t%2.3f\t%2.3f\t%s' % r)

        # totals totals
        print('-' * 80)
        col = lambda c: np.asarray(map(lambda x: x[c], table))
        times = col(1)

        def wmean(x):
            m = ~np.isnan(x)
            tm = times[m]
            return (x[m] * tm).sum() / tm.sum() if tm.size > 0 else np.nan

        print('totals:\t%2.3f\t%2.3f\t%2.3f' % (
            times.sum(), wmean(col(2)), wmean(col(3))))

        # print unknowns
        if len(unknowns) > 0:
            print('\n')
            for r in unknowns:
                print("%s %s" % r)
