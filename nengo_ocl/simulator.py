from __future__ import print_function

import inspect
import logging
import os
import warnings
from collections import defaultdict, OrderedDict

import numpy as np
import pyopencl as cl

import nengo
import nengo.version
import nengo.utils.numpy as npext
from nengo.cache import get_default_decoder_cache
from nengo.exceptions import SimulatorClosed
from nengo.simulator import ProbeDict
from nengo.builder.builder import Model
from nengo.builder.operator import Operator, Copy, DotInc, Reset
from nengo.builder.signal import Signal, SignalDict
from nengo.utils.compat import iteritems, StringIO
from nengo.utils.progress import ProgressTracker
from nengo.utils.stdlib import groupby

from nengo_ocl.raggedarray import RaggedArray
from nengo_ocl.clraggedarray import CLRaggedArray, to_device
from nengo_ocl.clra_gemv import plan_block_gemv
from nengo_ocl.clra_nonlinearities import (
    plan_timeupdate, plan_reset, plan_copy, plan_slicedcopy,
    plan_direct, plan_lif, plan_lif_rate,
    plan_probes, plan_linearfilter, plan_elementwise_inc,
    create_rngs, init_rngs, get_dist_enums_params, plan_whitenoise,
    plan_presentinput, plan_conv2d, plan_pool2d, plan_bcm, plan_oja, plan_voja)
from nengo_ocl.plan import BasePlan, PythonPlan, Plans
from nengo_ocl.planners import greedy_planner
from nengo_ocl.ast_conversion import OCL_Function
from nengo_ocl.utils import get_closures, indent, split, stable_unique, Timer
from nengo_ocl.version import latest_nengo_version, latest_nengo_version_info

logger = logging.getLogger(__name__)
PROFILING_ENABLE = cl.command_queue_properties.PROFILING_ENABLE


class MultiDotInc(Operator):
    """ y <- gamma + beta * y_in + \sum_i dot(A_i, x_i) """

    def __init__(self, Y, Y_in, beta, gamma, tag=None):
        assert Y.ndim == 1
        self.Y = Y
        self.Y_in = Y_in
        if Y.shape != Y_in.shape:
            raise TypeError()
        try:
            if hasattr(beta, 'value'):
                self._float_beta = float(beta.value)
            else:
                self._float_beta = float(beta)
            self._signal_beta = None
        except:
            assert isinstance(beta, Signal)
            self._float_beta = None
            self._signal_beta = beta
            if beta.shape != Y.shape:
                raise NotImplementedError('', (beta.shape, Y.shape))
        self.gamma = float(gamma)
        self.tag = tag
        self.As = []
        self.Xs = []
        self._incs_Y = (self._signal_beta is None and self._float_beta == 1 and
                        self.Y_in is self.Y)

    @classmethod
    def convert_to(cls, op):
        if isinstance(op, Copy):
            rval = cls(op.dst, op.src, beta=1, gamma=0, tag=op.tag)
        elif isinstance(op, DotInc):
            rval = cls(op.Y, op.Y, beta=1, gamma=0, tag=op.tag)
            rval.add_AX(op.A, op.X)
        else:
            return op

        assert set(op.reads).issuperset(rval.reads), (rval.reads, op.reads)
        assert rval.incs == op.incs
        assert rval.sets == op.sets
        assert all(s.size for s in rval.all_signals), op
        assert set(rval.updates) == set(op.updates), (rval.updates, op.updates)
        return rval

    def add_AX(self, A, X):
        assert X.ndim == 1
        self.As.append(A)
        self.Xs.append(X)

    @property
    def reads(self):
        rval = self.As + self.Xs
        if not self._incs_Y:
            if self._signal_beta is None:
                if self._float_beta != 0 and self.Y_in is not self.Y:
                    rval += [self.Y_in]
            else:
                rval += [self._signal_beta]
                if self.Y_in is not self.Y:
                    rval += [self.Y_in]
        return rval

    @property
    def incs(self):
        return [self.Y] if self._incs_Y else []

    @property
    def sets(self):
        return [] if self._incs_Y else [self.Y]

    @property
    def updates(self):
        return []

    def __str__(self):
        beta = (self._float_beta if self._signal_beta is None else
                self._signal_beta)
        dots = ['dot(%s, %s)' % (A, X) for A, X in zip(self.As, self.Xs)]
        return ('<MultiDotInc(tag=%s, Y=%s, Y_in=%s, beta=%s, gamma=%s, '
                'dots=[%s]) at 0x%x>' % (
                    self.tag, self.Y, self.Y_in, beta, self.gamma,
                    ', '.join(dots), id(self)))

    def __repr__(self):
        return self.__str__()

    @classmethod
    def compress(cls, operators):
        sets = OrderedDict()
        incs = OrderedDict()
        rval = []
        for op in operators:
            if isinstance(op, cls):
                if op.sets:
                    assert len(op.sets) == 1 and len(op.incs) == 0
                    sets.setdefault(op.sets[0], []).append(op)
                else:
                    assert len(op.incs) == 1 and len(op.sets) == 0
                    incs.setdefault(op.incs[0], []).append(op)
            else:
                rval.append(op)

        # combine incs into sets if on same view
        for view, set_ops in iteritems(sets):
            set_op, = set_ops
            inc_ops = incs.get(view, [])
            for inc_op in inc_ops[:]:
                set_op.As.extend(inc_op.As)
                set_op.Xs.extend(inc_op.Xs)
                inc_ops.remove(inc_op)
            rval.append(set_op)

        # combine remaining incs if on same view
        for view, inc_ops in iteritems(incs):
            if len(inc_ops) > 0:
                inc_op0 = inc_ops[0]
                for inc_op in inc_ops[1:]:
                    inc_op0.As.extend(inc_op.As)
                    inc_op0.Xs.extend(inc_op.Xs)
                rval.append(inc_op0)

        return rval

    @staticmethod
    def _as2d(view):
        if view.ndim == 0:
            return view.reshape(1, 1)
        elif view.ndim == 1:
            return view.reshape(view.shape[0], 1)
        elif view.ndim == 2:
            return view
        else:
            raise ValueError(
                "No support for tensors with %d dimensions" % view.ndim)

    def get_views(self):
        Y_view = self._as2d(self.Y)
        Y_in_view = self._as2d(self.Y_in)
        beta_view = (self._as2d(self._signal_beta)
                     if self._signal_beta is not None else None)

        A_views = []
        X_views = []
        for A, X in zip(self.As, self.Xs):
            X_view = self._as2d(X)
            if A.ndim == 1 and X.ndim == 1:
                A_view = A.reshape((1, A.shape[0]))  # vector dot
            else:
                A_view = self._as2d(A)

            if A_view.shape == (1, 1):
                # -- scalar AX_views can be done as reverse multiplication
                A_view, X_view = X_view, A_view
            elif not (X_view.shape[0] == A_view.shape[1] and
                      X_view.shape[1] == Y_view.shape[1] and
                      A_view.shape[0] == Y_view.shape[0]):
                raise ValueError('shape mismach (A: %s, X: %s, Y: %s)' %
                                 (A.shape, X.shape, self.Y.shape))

            A_views.append(A_view)
            X_views.append(X_view)

        return A_views, X_views, Y_view, Y_in_view, beta_view


class ViewBuilder(object):

    def __init__(self, bases, rarray):
        self.sidx = {bb: ii for ii, bb in enumerate(bases)}
        assert len(bases) == len(self.sidx)
        self.rarray = rarray

        self.starts = []
        self.shape0s = []
        self.shape1s = []
        self.stride0s = []
        self.stride1s = []
        self.names = []

        self._A_views = {}
        self._X_views = {}
        self._YYB_views = {}

    def append_view(self, obj):
        if obj in self.sidx:
            return  # we already have this view

        if not obj.is_view:
            # -- it is not a view, and not OK
            raise ValueError('can only append views of known signals', obj)

        assert obj.size and obj.ndim <= 2
        idx = self.sidx[obj.base]
        self.starts.append(self.rarray.starts[idx] + obj.elemoffset)
        self.shape0s.append(obj.shape[0] if obj.ndim > 0 else 1)
        self.shape1s.append(obj.shape[1] if obj.ndim > 1 else 1)
        self.stride0s.append(obj.elemstrides[0] if obj.ndim > 0 else 1)
        self.stride1s.append(obj.elemstrides[1] if obj.ndim > 1 else 1)
        self.names.append(getattr(obj, 'name', ''))
        self.sidx[obj] = len(self.sidx)

    def add_views_to(self, rarray):
        rarray.add_views(self.starts, self.shape0s, self.shape1s,
                         self.stride0s, self.stride1s, names=self.names)

    def setup_views(self, ops):
        all_views = [sig for op in ops for sig in op.all_signals]
        for op in (op for op in ops if isinstance(op, MultiDotInc)):
            A_views, X_views, Y_view, Y_in_view, beta_view = op.get_views()
            all_views.extend(A_views + X_views + [Y_view, Y_in_view] +
                             ([beta_view] if beta_view else []))
            self._A_views[op] = A_views
            self._X_views[op] = X_views
            self._YYB_views[op] = [Y_view, Y_in_view, beta_view]

        for view in all_views:
            self.append_view(view)


class Simulator(nengo.Simulator):

    unsupported = []

    def Array(self, val, dtype=np.float32):
        return to_device(self.queue, np.asarray(val, dtype=dtype))

    def RaggedArray(self, listofarrays, **kwargs):
        return CLRaggedArray.from_arrays(self.queue, listofarrays, **kwargs)

    def __init__(self, network, dt=0.001, seed=None, model=None, context=None,
                 n_prealloc_probes=32, profiling=None, ocl_only=False,
                 planner=greedy_planner):
        # --- create these first since they are used in __del__
        self.closed = False
        self.model = None

        # --- check version
        if nengo.version.version_info[:2] != latest_nengo_version_info[:2]:
            raise ValueError(
                "This simulator only supports Nengo %s.x (got %s)" %
                ('.'.join(str(i) for i in latest_nengo_version_info[:2]),
                 nengo.__version__))
        elif nengo.version.version_info > latest_nengo_version_info:
            warnings.warn("This version of `nengo_ocl` has not been tested "
                          "with your `nengo` version (%s). The latest fully "
                          "supported version is %s" % (
                              nengo.__version__, latest_nengo_version))

        # --- arguments/attributes
        if context is None:
            print('No context argument was provided to nengo_ocl.Simulator')
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

        # --- Nengo build
        with Timer() as nengo_timer:
            if model is None:
                self.model = Model(dt=float(dt),
                                   label="%s, dt=%f" % (network, dt),
                                   decoder_cache=get_default_decoder_cache())
            else:
                self.model = model

            if network is not None:
                # Build the network into the model
                self.model.build(network)

        logger.info("Nengo build in %0.3f s" % nengo_timer.duration)

        # --- operators
        with Timer() as planner_timer:
            operators = list(self.model.operators)

            # convert DotInc and Copy to MultiDotInc
            operators = list(map(MultiDotInc.convert_to, operators))
            operators = MultiDotInc.compress(operators)

            # plan the order of operations, combining where appropriate
            op_groups = planner(operators)
            assert len([typ for typ, _ in op_groups if typ is Reset]) < 2, (
                "All resets not planned together")

            self.operators = operators
            self.op_groups = op_groups

        logger.info("Planning in %0.3f s" % planner_timer.duration)

        with Timer() as signals_timer:
            # Initialize signals
            all_signals = stable_unique(
                sig for op in operators for sig in op.all_signals)
            all_bases = stable_unique(sig.base for sig in all_signals)

            sigdict = SignalDict()  # map from Signal.base -> ndarray
            for op in operators:
                op.init_signals(sigdict)

            # Add built states to the probe dictionary
            self._probe_outputs = dict(self.model.params)

            # Provide a nicer interface to probe outputs
            self.data = ProbeDict(self._probe_outputs)

            # Create data on host and add views
            self.all_data = RaggedArray(
                [sigdict[sb] for sb in all_bases],
                names=[getattr(sb, 'name', '') for sb in all_bases],
                dtype=np.float32)

            view_builder = ViewBuilder(all_bases, self.all_data)
            view_builder.setup_views(operators)
            for probe in self.model.probes:
                view_builder.append_view(self.model.sig[probe]['in'])
            view_builder.add_views_to(self.all_data)

            self.all_bases = all_bases
            self.sidx = {
                k: np.int32(v) for k, v in iteritems(view_builder.sidx)}
            self._A_views = view_builder._A_views
            self._X_views = view_builder._X_views
            self._YYB_views = view_builder._YYB_views
            del view_builder

            # Copy data to device
            self.all_data = CLRaggedArray(self.queue, self.all_data)

        logger.info("Signals in %0.3f s" % signals_timer.duration)

        # --- set seed
        self.seed = np.random.randint(npext.maxint) if seed is None else seed
        self._reset_rng()

        # --- create list of plans
        self._raggedarrays_to_reset = {}
        self._cl_rngs = {}

        with Timer() as plans_timer:
            self._plan = []
            for op_type, op_list in op_groups:
                self._plan.extend(self.plan_op_group(op_type, op_list))
            self._plan.extend(self.plan_probes())

        logger.info("Plans in %0.3f s" % plans_timer.duration)

        # -- create object to execute list of plans
        self._plans = Plans(self._plan, self.profiling)

        self._reset_cl_rngs()
        self._probe_step_time()

    def _reset_rng(self):
        self.rng = np.random.RandomState(self.seed)

    def _reset_cl_rngs(self):
        for rngs, seeds in iteritems(self._cl_rngs):
            seeds = [self.rng.randint(npext.maxint) if s is None else s
                     for s in seeds]
            init_rngs(self.queue, rngs, seeds)

    def plan_op_group(self, op_type, ops):
        return getattr(self, 'plan_' + op_type.__name__)(ops)

    def plan_PreserveValue(self, ops):
        return []  # do nothing

    def plan_MultiDotInc(self, ops):
        constant_bs = [op for op in ops if op._float_beta is not None]
        vector_bs = [op for op in ops
                     if op._signal_beta is not None
                     and op._signal_beta.ndim == 1]
        if len(constant_bs) + len(vector_bs) != len(ops):
            raise NotImplementedError()

        args = (
            lambda op: self._A_views[op],
            lambda op: self._X_views[op],
            lambda op: self._YYB_views[op][0],
            lambda op: self._YYB_views[op][1],
            )
        constant_b_gemvs = self._sig_gemv(
            constant_bs, *args,
            beta=[op._float_beta for op in constant_bs],
            gamma=[op.gamma for op in constant_bs],
            tag='c-beta-%d' % len(constant_bs))
        vector_b_gemvs = self._sig_gemv(
            vector_bs, *args,
            beta=lambda op: self._YYB_views[op][2],
            gamma=[op.gamma for op in vector_bs],
            tag='v-beta-%d' % len(vector_bs))
        return constant_b_gemvs + vector_b_gemvs

    def _sig_gemv(self, ops, A_js_fn, X_js_fn, Y_fn, Y_in_fn=None,
                  alpha=1.0, beta=1.0, gamma=0.0, tag=None):
        if len(ops) == 0:
            return []

        all_data, sidx = self.all_data, self.sidx
        A_js = RaggedArray([[sidx[ss] for ss in A_js_fn(op)] for op in ops])
        X_js = RaggedArray([[sidx[ss] for ss in X_js_fn(op)] for op in ops])
        Y_sigs = [Y_fn(item) for item in ops]
        Y_in_sigs = [Y_in_fn(item) for item in ops] if Y_in_fn else Y_sigs
        Y = all_data[[sidx[sig] for sig in Y_sigs]]
        Y_in = all_data[[sidx[sig] for sig in Y_in_sigs]]
        if callable(beta):
            beta = RaggedArray([sidx[beta(o)] for o in ops], dtype=np.float32)

        rval = plan_block_gemv(
            self.queue, alpha, all_data, A_js, all_data, X_js, beta, Y,
            Y_in=Y_in, gamma=gamma, tag=tag)
        return rval.plans

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
        copies, ops = split(ops, lambda op: (op.src_slice is Ellipsis and
                                             op.dst_slice is Ellipsis))

        plans = []
        if copies:
            A = self.all_data[[self.sidx[op.src] for op in copies]]
            B = self.all_data[[self.sidx[op.dst] for op in copies]]
            incs = np.array([op.inc for op in copies], dtype=np.int32)
            plans.append(plan_copy(self.queue, A, B, incs))

        if ops:
            A = self.all_data[[self.sidx[op.src] for op in ops]]
            B = self.all_data[[self.sidx[op.dst] for op in ops]]
            inds = lambda ary, i: np.arange(ary.size, dtype=np.int32)[i]
            Ainds = self.RaggedArray(
                [inds(op.src, op.src_slice) for op in ops])
            Binds = self.RaggedArray(
                [inds(op.dst, op.dst_slice) for op in ops])
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
            op_key = (op.fn, op.t is not None, op.x is not None)
            if op_key not in unique_ops:
                unique_ops[op_key] = {'in': [], 'out': []}
            unique_ops[op_key]['in'].append(op.x)
            unique_ops[op_key]['out'].append(op.output)

        # make plans
        plans = []
        for (fn, t_in, x_in), signals in iteritems(unique_ops):
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
                        [self.sidx[self.model.time] for i in signals['out']]])
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
        t_idx = self.sidx[self.model.time]
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

    def plan_SimProcess(self, all_ops):
        class_groups = groupby(all_ops, lambda op: op.process.__class__)
        plan_groups = defaultdict(list)

        for process_class, ops in class_groups:
            for cls in process_class.__mro__:
                attrname = '_plan_' + cls.__name__
                if hasattr(self, attrname):
                    plan_groups[attrname].extend(ops)
                    break
            else:
                raise NotImplementedError("Unsupported process type '%s'"
                                          % process_class.__name__)

        return [p for attr, ops in iteritems(plan_groups)
                for p in getattr(self, attr)(ops)]

    def _plan_LinearFilter(self, ops):
        for op in ops:
            if op.input.ndim != 1:
                raise NotImplementedError("Can only filter vectors")
        steps = [op.process.make_step(op.input.shape, op.output.shape,
                                      self.model.dt, rng=None) for op in ops]
        A = self.RaggedArray([f.den for f in steps], dtype=np.float32)
        B = self.RaggedArray([f.num for f in steps], dtype=np.float32)
        X = self.all_data[[self.sidx[op.input] for op in ops]]
        Y = self.all_data[[self.sidx[op.output] for op in ops]]
        Xbuf0 = RaggedArray(
            [np.zeros(shape) for shape in zip(B.sizes, X.sizes)],
            dtype=np.float32)
        Ybuf0 = RaggedArray(
            [np.zeros(shape) for shape in zip(A.sizes, Y.sizes)],
            dtype=np.float32)
        Xbuf = CLRaggedArray(self.queue, Xbuf0)
        Ybuf = CLRaggedArray(self.queue, Ybuf0)
        self._raggedarrays_to_reset[Xbuf] = Xbuf0
        self._raggedarrays_to_reset[Ybuf] = Ybuf0
        return [plan_linearfilter(self.queue, X, Y, A, B, Xbuf, Ybuf)]

    def _plan_WhiteNoise(self, ops):
        assert all(op.input is None for op in ops)

        rngs = create_rngs(self.queue, len(ops))
        self._cl_rngs[rngs] = [op.process.seed for op in ops]

        Y = self.all_data[[self.sidx[op.output] for op in ops]]
        scale = self.Array([op.process.scale for op in ops], dtype=np.int32)
        inc = self.Array([op.mode == 'inc' for op in ops], dtype=np.int32)
        enums, params = get_dist_enums_params([op.process.dist for op in ops])
        enums = CLRaggedArray(self.queue, enums)
        params = CLRaggedArray(self.queue, params)
        dt = self.model.dt
        return [plan_whitenoise(
            self.queue, Y, enums, params, scale, inc, dt, rngs)]

    def _plan_FilteredNoise(self, ops):
        raise NotImplementedError()

    def _plan_WhiteSignal(self, ops):
        Y = self.all_data[[self.sidx[op.output] for op in ops]]
        t = self.all_data[[self.sidx[self.model.step] for _ in ops]]

        dt = self.model.dt
        signals = []
        for op in ops:
            assert op.input is None and op.output is not None
            rng = op.process.get_rng(self.rng)
            f = op.process.make_step((0,), op.output.shape, dt, rng)
            signals.append(get_closures(f)['signal'])

        signals = self.RaggedArray(signals, dtype=np.float32)
        return [plan_presentinput(self.queue, Y, t, signals, dt)]

    def _plan_PresentInput(self, ops):
        ps = [op.process for op in ops]
        Y = self.all_data[[self.sidx[op.output] for op in ops]]
        t = self.all_data[[self.sidx[self.model.step] for _ in ops]]
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
            kernel_shape = f.shape[-2:]
            X = self.all_data.getitem_device(self.sidx[op.input])
            Y = self.all_data.getitem_device(self.sidx[op.output])
            ftrans = np.asarray(np.transpose(
                f, (0, 1, 2, 3) if conv else (0, 3, 4, 5, 1, 2)), order='C')
            F = self.Array(ftrans.ravel())
            B = self.Array((np.zeros(p.shape_out) + b).ravel())
            plans.append(plan_conv2d(
                self.queue, X, Y, F, B, p.shape_in, p.shape_out,
                kernel_shape, conv, p.padding, p.strides))

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
                self.queue, X, Y, shape, p.pool_size, p.strides))

        return plans

    def plan_SimBCM(self, ops):
        pre = self.all_data[[self.sidx[op.pre_filtered] for op in ops]]
        post = self.all_data[[self.sidx[op.post_filtered] for op in ops]]
        theta = self.all_data[[self.sidx[op.theta] for op in ops]]
        delta = self.all_data[[self.sidx[op.delta] for op in ops]]
        alpha = self.Array([op.learning_rate * self.model.dt for op in ops])
        return [plan_bcm(self.queue, pre, post, theta, delta, alpha)]

    def plan_SimOja(self, ops):
        pre = self.all_data[[self.sidx[op.pre_filtered] for op in ops]]
        post = self.all_data[[self.sidx[op.post_filtered] for op in ops]]
        weights = self.all_data[[self.sidx[op.weights] for op in ops]]
        delta = self.all_data[[self.sidx[op.delta] for op in ops]]
        alpha = self.Array([op.learning_rate * self.model.dt for op in ops])
        beta = self.Array([op.beta for op in ops])
        return [plan_oja(self.queue, pre, post, weights, delta, alpha, beta)]

    def plan_SimVoja(self, ops):
        pre = self.all_data[[self.sidx[op.pre_decoded] for op in ops]]
        post = self.all_data[[self.sidx[op.post_filtered] for op in ops]]
        encoders = self.all_data[[self.sidx[op.scaled_encoders] for op in ops]]
        delta = self.all_data[[self.sidx[op.delta] for op in ops]]
        learning_signal = self.all_data[
            [self.sidx[op.learning_signal] for op in ops]]
        scale = self.RaggedArray([op.scale for op in ops], dtype=np.float32)
        alpha = self.Array([op.learning_rate * self.model.dt for op in ops])
        return [plan_voja(self.queue, pre, post, encoders, delta,
                          learning_signal, scale, alpha)]

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

    def _probe(self):
        """Copy all probed signals to buffers"""
        self._probe_step_time()

        plan = self._cl_probe_plan
        if plan is None:
            return  # nothing to probe

        self.queue.finish()
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
            raise SimulatorClosed("Simulator cannot run because it is closed.")

        if self.n_steps + N >= 2**24:
            # since n_steps is float32, point at which `n_steps == n_steps + 1`
            raise ValueError("Cannot handle more than 2**24 steps")

        if self._cl_probe_plan is not None:
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
                self._probe()
                N -= B
                progress.step(n=B)

        if self.profiling > 1:
            self.print_profiling()

    def reset(self, seed=None):
        if self.closed:
            raise SimulatorClosed("Cannot reset closed Simulator.")

        if seed is not None:
            raise NotImplementedError("Seed changing not implemented")

        # reset signals
        for base in self.all_bases:
            # TODO: copy all data on at once
            if not base.readonly:
                self.all_data[self.sidx[base]] = base.initial_value

        for clra, ra in iteritems(self._raggedarrays_to_reset):
            # TODO: copy all data on at once
            for i in range(len(clra)):
                clra[i] = ra[i]

        # clear probe data
        if self._cl_probe_plan is not None:
            self._cl_probe_plan.cl_bufpositions.fill(0)

            for probe in self.model.probes:
                del self._probe_outputs[probe][:]

        self._reset_rng()
        self._reset_cl_rngs()
        self._probe_step_time()

    def close(self):
        self.closed = True
        self.context = None
        self.queue = None
        self.all_data = None
        self._plan = []
        self._plans = None
        self._raggedarrays_to_reset = None
        self._cl_rngs = None
        self._cl_probe_plan = None

    def __getitem__(self, item):
        """
        Return internally shaped signals, which are always 2d
        """
        return self.all_data[self.sidx[item]]

    @property
    def signals(self):
        """Get/set [properly-shaped] signal value (either 0d, 1d, or 2d)
        """
        class Accessor(object):

            def __iter__(_):
                return iter(self.all_bases)

            def __getitem__(_, item):
                raw = self.all_data[self.sidx[item]]
                if item.ndim == 0:
                    return raw[0, 0]
                elif item.ndim == 1:
                    return raw.ravel()
                elif item.ndim == 2:
                    return raw
                else:
                    raise NotImplementedError()

            def __setitem__(_, item, val):
                incoming = np.asarray(val)
                if item.ndim == 0:
                    assert incoming.size == 1
                    self.all_data[self.sidx[item]] = incoming
                elif item.ndim == 1:
                    assert (item.size,) == incoming.shape
                    self.all_data[self.sidx[item]] = incoming[:, None]
                elif item.ndim == 2:
                    assert item.shape == incoming.shape
                    self.all_data[self.sidx[item]] = incoming
                else:
                    raise NotImplementedError()

            def __str__(self_):
                sio = StringIO()
                for k in self_:
                    print(k, self_[k], file=sio)
                return sio.getvalue()

        return Accessor()

    def print_plans(self):
        print(" Plans ".center(80, '-'))
        for plan in self._plans.plans:
            print("%r" % plan)
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
        print('%8s|%10s|%10s|%10s|' % ('n_calls', 'runtime', 'GF/s', 'GB/s'))

        for r in table:
            print('%8d|%10.3f|%10.3f|%10.3f| %s' % r)

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


Simulator.unsupported.extend([
    # learning rules
    ('nengo/tests/test_learning_rules.py:test_dt_dependence*',
     "Filtering matrices (i.e. learned transform) not implemented"),
    ('nengo/tests/test_learning_rules.py:test_reset*',
     "Filtering matrices not implemented"),

    # neuron types
    ('nengo/tests/test_neurons.py:test_izhikevich',
     "Izhikevich neurons not implemented"),
    ('nengo/tests/test_neurons.py:test_lif_min_voltage*',
     "Min voltage not implemented"),

    # nodes
    ('nengo/tests/test_node.py:test_none',
     "No error if nodes output None"),

    # processes
    ('nengo/tests/test_processes.py:test_brownnoise',
     "Filtered noise processes not yet implemented"),
    ('nengo/tests/test_ensemble.py:test_noise_copies_ok*',
     "Filtered noise processes not yet implemented"),
    ('nengo/tests/test_simulator.py:test_noise_copies_ok',
     "Filtered noise processes not yet implemented"),

    # synapses
    ('nengo/tests/test_synapses.py:test_triangle',
     "Only linear filters implemented"),
])
