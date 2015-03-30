import collections
import logging
import os

import numpy as np
import pyopencl as cl

from nengo.dists import Uniform, Gaussian
from nengo.neurons import LIF, LIFRate, Direct
from nengo.processes import StochasticProcess
from nengo.synapses import LinearFilter, Alpha, Lowpass
from nengo.utils.compat import OrderedDict
from nengo.utils.filter_design import cont2discrete
from nengo.utils.stdlib import groupby

from nengo_ocl import sim_npy
from nengo_ocl.raggedarray import RaggedArray
from nengo_ocl.clraggedarray import CLRaggedArray
from nengo_ocl.clra_gemv import plan_ragged_gather_gemv
from nengo_ocl.clra_nonlinearities import (
    plan_lif, plan_lif_rate, plan_direct, plan_probes,
    plan_filter_synapse, plan_elementwise_inc)
from nengo_ocl.plan import BasePlan, PythonPlan, DAG, Marker
from nengo_ocl.ast_conversion import OCL_Function

logger = logging.getLogger(__name__)
PROFILING_ENABLE = cl.command_queue_properties.PROFILING_ENABLE


class Simulator(sim_npy.Simulator):

    def RaggedArray(self, *args, **kwargs):
        val = RaggedArray(*args, **kwargs)
        return CLRaggedArray(self.queue, val)

    def __init__(self, network, dt=0.001, seed=None, model=None, context=None,
                 n_prealloc_probes=1000, profiling=None, ocl_only=False):
        if context is None:
            print 'No context argument was provided to sim_ocl.Simulator'
            print "Calling pyopencl.create_some_context() for you now:"
            context = cl.create_some_context()
        if profiling is None:
            profiling = int(os.getenv("NENGO_OCL_PROFILING", 0))
        self.context = context
        self.profiling = profiling
        if self.profiling:
            self.queue = cl.CommandQueue(context,
                                         properties=PROFILING_ENABLE)
        else:
            self.queue = cl.CommandQueue(context)

        self.n_prealloc_probes = n_prealloc_probes
        self.ocl_only = ocl_only

        # -- allocate data
        sim_npy.Simulator.__init__(
            self, network=network, dt=dt, seed=seed, model=model)

        # -- set up the DAG for executing OCL kernels
        self._plandict = OrderedDict()
        self.step_marker = Marker(self.queue)
        # -- marker is used to do the op_groups in order
        deps = []
        for op_type, op_list in self.op_groups:
            deps = self.plandict_op_group(op_type, op_list, deps)
        probe_plans = self.plan_probes()
        for p in probe_plans:
            self._plandict[p] = deps
        self._dag = DAG(context, self.step_marker,
                           self._plandict,
                           self.profiling)

    def plan_op_group(self, *args):
        # -- HACK: SLOWLY removing sim_npy from the project...
        return []

    def plandict_op_group(self, op_type, op_list, deps):
        plans = getattr(self, 'plan_' + op_type.__name__)(op_list)
        for p in plans:
            self._plandict[p] = deps
        return plans

    def _prep_all_data(self):
        # -- replace the numpy-allocated RaggedArray with OpenCL one
        self.all_data = CLRaggedArray(self.queue, self.all_data)

    def plan_ragged_gather_gemv(self, *args, **kwargs):
        return plan_ragged_gather_gemv(self.queue, *args, **kwargs)

    def plan_ElementwiseInc(self, ops):
        A = self.all_data[[self.sidx[op.A] for op in ops]]
        X = self.all_data[[self.sidx[op.X] for op in ops]]
        Y = self.all_data[[self.sidx[op.Y] for op in ops]]
        return [plan_elementwise_inc(self.queue, A, X, Y)]

    def plan_SimPyFunc(self, ops):
        ### TODO: test with a hybrid program (Python and OCL)

        ### group nonlinearities
        unique_ops = collections.OrderedDict()
        for op in ops:
            # assert op.n_args in (1, 2), op.n_args
            op_key = (op.fn, op.t_in, op.x is not None)
            if op_key not in unique_ops:
                unique_ops[op_key] = {'in': [], 'out': []}
            unique_ops[op_key]['in'].append(op.x)
            unique_ops[op_key]['out'].append(op.output)

        ### make plans
        plans = []
        for (fn, t_in, x_in), signals in unique_ops.items():
            fn_name = fn.__name__
            if fn_name == "<lambda>":
                fn_name += "%d" % len(plans)

            # check signal input and output shape (implicitly checks
            # for indexing errors)
            vector_dims = lambda shape, dim: len(shape) == 1 and shape[0] == dim
            unit_stride = lambda es: len(es) == 1 and es[0] == 1

            if x_in:
                in_dim = signals['in'][0].size
                for sig_in in signals['in']:
                    assert sig_in.size == in_dim
                    assert vector_dims(sig_in.shape, in_dim)
                    assert unit_stride(sig_in.elemstrides)
            else:
                in_dim = None

            out_dim = signals['out'][0].size
            for sig_out in signals['out']:
                assert sig_out.size == out_dim
                assert vector_dims(sig_out.shape, out_dim)
                assert unit_stride(sig_out.elemstrides)

            ### try to get OCL code
            code = None
            try:
                # in_dims = (1, in_dim) if n_args == 2 else (1, )
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
                logger.warning(
                    "Function '%s' could not be converted to OCL due to %s%s"
                    % (fn_name, e.__class__.__name__, e.args))

                if self.ocl_only:
                    raise

                # not successfully translated to OCL, so do it in Python
                dt = self.model.dt

                # Need make_step function so that variables get copied
                def make_step(t_in=t_in, x_in=x_in):
                    f = fn
                    t_idx = self.sidx[self._time]
                    out_idx = [self.sidx[s] for s in signals['out']]

                    if not x_in:
                        def step():
                            t = self.all_data[t_idx][0, 0] - dt
                            for sout in out_idx:
                                y = np.asarray(f(t) if t_in else f())
                                if y.ndim == 1:
                                    y = y[:, None]
                                self.all_data[sout] = y
                    else:
                        in_idx = [self.sidx[s] for s in signals['in']]

                        def step():
                            t = self.all_data[t_idx][0, 0]
                            for sin, sout in zip(in_idx, out_idx):
                                x = self.all_data[sin]
                                if x.ndim == 2 and x.shape[1] == 1:
                                    x = x[:, 0]
                                y = np.asarray(f(t, x) if t_in else f(x))
                                if y.ndim == 1:
                                    y = y[:, None]
                                self.all_data[sout] = y

                    return step

                plans.append(PythonPlan(make_step(), name=fn_name, tag=fn_name))

        return plans

    def plan_SimNeurons(self, all_ops):
        groups = groupby(all_ops, lambda op: op.neurons.__class__)
        plans = []
        for neuron_class, ops in groups:
            if neuron_class is LIF:
                plans.extend(self.plan_SimLIF(ops))
            elif neuron_class is LIFRate:
                plans.extend(self.plan_SimLIFRate(ops))
            else:
                raise ValueError("Unsupported neuron type '%s'"
                                 % neuron_class.__name__)

        return plans

    def plan_SimLIF(self, ops):
        J = self.all_data[[self.sidx[op.J] for op in ops]]
        V = self.all_data[[self.sidx[op.states[0]] for op in ops]]
        W = self.all_data[[self.sidx[op.states[1]] for op in ops]]
        S = self.all_data[[self.sidx[op.output] for op in ops]]
        ref = self.RaggedArray([op.neurons.tau_ref for op in ops])
        tau = self.RaggedArray([op.neurons.tau_rc for op in ops])
        dt = self.model.dt
        return [plan_lif(self.queue, J, V, W, V, W, S, ref, tau, dt,
                         tag="lif", n_elements=10)]

    def plan_SimLIFRate(self, ops):
        J = self.all_data[[self.sidx[op.J] for op in ops]]
        R = self.all_data[[self.sidx[op.output] for op in ops]]
        ref = self.RaggedArray([op.neurons.tau_ref for op in ops])
        tau = self.RaggedArray([op.neurons.tau_rc for op in ops])
        dt = self.model.dt
        return [plan_lif_rate(self.queue, J, R, ref, tau, dt,
                              tag="lif_rate", n_elements=10)]

    def plan_SimSynapse(self, ops):
        for op in ops:
            assert isinstance(op.synapse, LinearFilter)
        X = self.all_data[[self.sidx[op.input] for op in ops]]
        Y = self.all_data[[self.sidx[op.output] for op in ops]]

        dt = self.model.dt
        nums = []
        dens = []
        for s in (op.synapse for op in ops):
            if isinstance(s, (Alpha, Lowpass)) and s.tau <= 0.03 * dt:
                num, den = [1], [1, 0]  # simply copies the first value
            else:
                num, den, _ = cont2discrete((s.num, s.den), dt, method='zoh')
                num = num.flatten()
            nums.append(num[1:] if num[0] == 0 else num)
            dens.append(den[1:])  # drop first element (equal to 1)

        A = self.RaggedArray(dens)
        B = self.RaggedArray(nums)
        return [plan_filter_synapse(self.queue, X, Y, A, B)]

    def plan_SimNoise(self, ops):
        def split(iterable, condition):
            a = []
            b = []
            for i in iterable:
                if condition(i):
                    a.append(i)
                else:
                    b.append(i)
            return a, b

        ops, badops = split(ops, lambda x: x.__class__ is StochasticProcess)
        if len(badops) > 0:
            raise NotImplementedError("Can only simulate StochasticProcess base class")

        ops, badops = split(ops, lambda x: x.synapse is None)
        if len(badops) > 0:
            raise NotImplementedError("Can only simulate noise with no synapse")

        uniform, ops = split(ops, lambda x: isinstance(x.dist, Uniform))
        gaussian, ops = split(ops, lambda x: isinstance(x.dist, Gaussian))
        if len(ops) > 0:
            raise NotImplementedError("Can only simulate Uniform or Gaussian noise")

        raise NotImplementedError("TODO")

    def plan_probes(self):
        if len(self.model.probes) == 0:
            return []

        n_prealloc = self.n_prealloc_probes

        probes = self.model.probes
        periods = [1 if p.sample_every is None else
                   int(np.round(float(p.sample_every) / self.model.dt))
                   for p in probes]

        X = self.all_data[
            [self.sidx[self.model.sig[p]['in']] for p in probes]]
        Y = self.RaggedArray(
            [np.zeros((n_prealloc, self.model.sig[p]['in'].size))
             for p in probes])

        cl_plan = plan_probes(self.queue, periods, X, Y, tag="probes")
        self._max_steps_between_probes = n_prealloc * min(periods)

        cl_plan.Y = Y
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
        return self.run_steps(1)

    def run_steps(self, N, verbose=False):
        has_probes = hasattr(self, '_cl_probe_plan')

        if has_probes:
            # -- precondition: the probe buffers have been drained
            bufpositions = self._cl_probe_plan.cl_bufpositions.get()
            assert np.all(bufpositions == 0)
        # -- we will go through N steps of the simulator
        #    in groups of up to B at a time, draining
        #    the probe buffers after each group of B
        while N:
            B = min(N, self._max_steps_between_probes) if has_probes else N
            self._dag.call_n_times(B)
            if has_probes:
                self.drain_probe_buffers()
            N -= B
            self.n_steps += B
        if self.profiling > 1:
            self.print_profiling()

    def print_profiling(self, sort=None):
        """
        Parameters
        ----------
        sort : indicates the column to sort by (negative number sorts ascending)
            (0 = n_calls, 1 = runtime, 2 = q-time, 3 = subtime)
        """
        ### make and sort table
        table = []
        unknowns = []
        for p in self._dag.order:
            gflops_per_sec = 0
            gbytes_per_sec = 0
            if isinstance(p, BasePlan):
                if p.flops_per_call is not None:
                    gflops_per_sec = (p.n_calls * p.flops_per_call
                                      / (sum(p.ctimes) * 1.0e9))
                if p.bw_per_call is not None:
                    gbytes_per_sec = (p.n_calls * p.bw_per_call
                                      / (sum(p.ctimes) * 1.0e9))
                table.append((
                    p.n_calls,
                    sum(p.ctimes),
                    gflops_per_sec,
                    gbytes_per_sec,
                    p.name,
                    p.tag))
            else:
                unknowns.append((str(p), getattr(p, 'cumtime', '<unknown>')))

        if sort is not None:
            reverse = sort >= 0
            table.sort(key=lambda x: x[abs(sort)], reverse=reverse)

        ### printing
        print '-' * 80
        print '%s\t%s\t%s\t%s' % ('n_calls', 'runtime', 'GF/s', 'GB/s')

        for r in table:
            print '%i\t%2.3f\t%2.3f\t%2.3f\t<%s, tag=%s>' % r

        print '-' * 80
        col_sum = lambda c: sum(map(lambda x: x[c], table))
        print 'totals:\t%2.3f\t%2.3f\t%2.3f' % (
            col_sum(1), col_sum(2), col_sum(3))

        if len(unknowns) > 0:
            print
            for r in unknowns:
                print "%s %s" % r
