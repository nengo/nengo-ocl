"""
numpy Simulator in the style of the OpenCL one, to get design right.
"""
import logging
import time
from collections import defaultdict

import networkx as nx
import numpy as np

from nengo.cache import get_default_decoder_cache
from nengo.simulator import ProbeDict, Simulator
from nengo.builder.builder import Model
from nengo.builder.operator import Operator, Copy, DotInc, PreserveValue, Reset
from nengo.builder.signal import Signal, SignalView, SignalDict
from nengo.utils.compat import OrderedDict

from nengo_ocl.plan import PythonPlan
from nengo_ocl.ra_gemv import ragged_gather_gemv
from nengo_ocl.raggedarray import RaggedArray as _RaggedArray

logger = logging.getLogger(__name__)


class StepUpdate(Operator):

    """Does Y += 1 as an update"""

    def __init__(self, Y, one):
        self.Y = Y
        self.one = one
        self.reads = [self.one]
        self.updates = [self.Y]
        self.incs = []
        self.sets = []

    def __str__(self):
        return 'StepUpdate(%s)' % (self.Y)

    def make_step(self, signals, dt):
        Y = signals[self.Y]

        def step():
            Y[...] += 1
        return step


class MultiProdUpdate(Operator):

    """
    y <- gamma + beta * y_in + \sum_i dot(A_i, x_i)
    """

    def __init__(self, Y, Y_in, beta, gamma, tag, as_update):
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
            assert isinstance(beta, SignalView)
            self._float_beta = None
            self._signal_beta = beta
            if beta.shape != Y.shape:
                raise NotImplementedError('', (beta.shape, Y.shape))
        self.gamma = float(gamma)
        self.tag = tag
        self.as_update = as_update
        self.As = []
        self.Xs = []
        self._incs_Y = (
            self._signal_beta is None
            and self._float_beta == 1
            and self.Y_in is self.Y
            and not as_update)

    @classmethod
    def convert_to(cls, op):
        def assert_ok():
            assert set(op.reads).issuperset(rval.reads), (rval.reads, op.reads)
            assert rval.incs == op.incs
            assert rval.sets == op.sets
            assert all(s.size for s in rval.all_signals), op
            assert set(rval.updates) == set(
                op.updates), (rval.updates, op.updates)
        if isinstance(op, Reset):
            rval = cls(Y=op.dst, Y_in=op.dst, beta=0, gamma=op.value,
                       as_update=False, tag=getattr(op, 'tag', ''))
            assert_ok()
        elif isinstance(op, Copy):
            rval = cls(op.dst, op.src, beta=1, gamma=0,
                       as_update=op.as_update, tag=op.tag)
            assert_ok()
        elif isinstance(op, DotInc):
            rval = cls(op.Y, op.Y, beta=1, gamma=0,
                       as_update=op.as_update, tag=op.tag)
            rval.add_AX(op.A, op.X)
            assert_ok()
        elif isinstance(op, PreserveValue):
            rval = cls(op.dst, op.dst, beta=1, gamma=0, as_update=False,
                       tag=getattr(op, 'tag', ''))
            rval._incs_Y = False
            assert_ok()
        elif isinstance(op, StepUpdate):
            # TODO: get rid of `op.one` and `add_AX` here; use `gamma`
            rval = cls(op.Y, op.Y, beta=1, gamma=0, as_update=True, tag="")
            rval.add_AX(op.one, op.one)
            assert_ok()
        else:
            return op
        return rval

    def add_AX(self, A, X):
        self.As.append(A)
        self.Xs.append(X)

    @property
    def reads(self):
        rval = self.As + self.Xs
        if self._incs_Y:
            pass
        else:
            if self._signal_beta is None:
                if self._float_beta != 0:
                    if self.Y_in is self.Y:
                        pass
                    else:
                        rval += [self.Y_in]
            else:
                if self.Y_in is self.Y:
                    rval += [self._signal_beta]
                else:
                    rval += [self._signal_beta, self.Y_in]
        return rval

    @property
    def incs(self):
        return [self.Y] if self._incs_Y else []

    @property
    def sets(self):
        if self._incs_Y:
            return []
        return [] if self.as_update else [self.Y]

    @property
    def updates(self):
        if self._incs_Y:
            return []
        return [self.Y] if self.as_update else []

    def __str__(self):
        if self._signal_beta is None:
            beta = self._float_beta
        else:
            beta = self._signal_beta

        dots = []
        for A, X in zip(self.As, self.Xs):
            dots.append('dot(%s, %s)' % (A, X))
        return ('<MultiProdUpdate(tag=%s, as_update=%s, Y=%s,'
                ' Y_in=%s, beta=%s,'
                ' gamma=%s, dots=[%s]'
                ') at 0x%x>' % (
                    self.tag, self.as_update, self.Y,
                    self.Y_in, beta,
                    self.gamma,
                    ', '.join(dots),
                    id(self)))

    def __repr__(self):
        return self.__str__()

    @classmethod
    def compress(cls, operators):
        sets = OrderedDict()
        incs = OrderedDict()
        rval = []
        for op in operators:
            if isinstance(op, cls):
                if op.as_update:
                    rval.append(op)
                else:
                    assert op.sets or op.incs
                    if op.sets:
                        sets.setdefault(op.sets[0], []).append(op)
                    if op.incs:
                        incs.setdefault(op.incs[0], []).append(op)
            else:
                rval.append(op)
        done = set()
        for view, set_ops in sets.items():
            set_op, = set_ops
            done.add(set_op)
            for inc_op in incs.get(view, []):
                set_op.As.extend(inc_op.As)
                set_op.Xs.extend(inc_op.Xs)
                done.add(inc_op)
            rval.append(set_op)
        for view, inc_ops in incs.items():
            for inc_op in inc_ops:
                if inc_op not in done:
                    rval.append(inc_op)
        return rval


def is_op(op):
    return isinstance(op, Operator)


def exact_dependency_graph(operators):
    from nengo.utils.simulator import operator_depencency_graph

    edges = operator_depencency_graph(operators)
    dg = nx.DiGraph()

    for source, dests in edges.items():
        dg.add_edges_from((source, dest) for dest in dests)

    return dg


def greedy_planner(operators):
    """
    I feel like there might e a dynamic programming solution here, but I can't
    work it out, and I'm not sure. Even if a DP solution existed, we would
    need a function to estimate the goodness (e.g. neg wall time) of kernel
    calls, and  that function would need to be pretty fast.
    """
    dg = exact_dependency_graph(operators)

    # -- TODO: linear time algo
    ancestors_of = {}
    for op in operators:
        ancestors_of[op] = set(filter(is_op, nx.ancestors(dg, op)))

    cliques = {}
    scheduled = set()
    rval = []
    while len(scheduled) < len(operators):
        candidates = [op
                      for op, pre_ops in ancestors_of.items()
                      if not pre_ops and op not in scheduled]
        if len(candidates) == 0:
            raise ValueError("Cycles in the op graph")

        type_counts = defaultdict(int)
        for op in candidates:
            type_counts[type(op)] += 1
        chosen_type = sorted(type_counts.items(), key=lambda x: x[1])[-1][0]
        candidates = [op for op in candidates if isinstance(op, chosen_type)]

        cliques_by_base = defaultdict(dict)
        by_base = defaultdict(list)
        for op in candidates:
            for sig in op.incs + op.sets + op.updates:
                for cliq in cliques.get(sig.base, []):
                    if sig.structure in cliq:
                        cliques_by_base[sig.base].setdefault(id(cliq), [])
                        cliques_by_base[sig.base][id(cliq)].append(op)
                by_base[sig.base].append(op)

        chosen = []
        for base, ops_writing_to_base in by_base.items():
            if base in cliques_by_base:
                most_ops = sorted(
                    (len(base_ops), base_ops)
                    for cliq_id, base_ops in cliques_by_base[base].items())
                chosen.extend(most_ops[-1][1])
            else:
                chosen.append(ops_writing_to_base[0])

        # -- ops that produced multiple outputs show up multiple times
        chosen = stable_unique(chosen)
        assert chosen

        assert not any(cc in scheduled for cc in chosen)
        scheduled.update(chosen)
        rval.append((chosen_type, chosen))

        # -- prepare for next iteration
        for op in ancestors_of:
            ancestors_of[op].difference_update(chosen)

    assert len(operators) == sum(len(p[1]) for p in rval)
    # print 'greedy_planner: Program len:', len(rval)
    return rval


# def sequential_planner(operators):
#     dg = exact_dependency_graph(operators, share_memory)

#     # list of pairs: (type, [ops_of_type], set_of_ancestors,
#     # set_of_descendants)
#     topo_order = [op
#                   for op in nx.topological_sort(dg)
#                   if is_op(op)]

#     return [(type(op), [op]) for op in topo_order]


def isview(obj):
    return obj.base is not None and obj.base is not obj


def shape0(obj):
    try:
        return obj.shape[0]
    except IndexError:
        return 1


def shape1(obj):
    try:
        return obj.shape[1]
    except IndexError:
        return 1


def idxs(seq, offset):
    rval = dict((s, i + offset) for (i, s) in enumerate(seq))
    return rval, offset + len(rval)


def stable_unique(seq):
    seen = set()
    rval = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            rval.append(item)
    return rval


class ViewBuilder(object):

    def __init__(self, bases, rarray):
        self.bases = bases
        self.sidx = dict((bb, ii) for ii, bb in enumerate(bases))
        assert len(self.bases) == len(self.sidx)
        self.rarray = rarray

        self.starts = []
        self.shape0s = []
        self.shape1s = []
        self.stride0s = []
        self.stride1s = []
        self.names = []
        # self.orig_len = len(self.all_signals)
        # self.base_starts = self.all_data_A.starts

    def append_view(self, obj):
        assert obj.size
        shape0(obj)
        shape1(obj)

        if obj in self.sidx:
            return
            # raise KeyError('sidx already contains object', obj)

        if obj in self.bases:
            # -- it is not a view, but OK
            return

        if not isview(obj):
            # -- it is not a view, and not OK
            raise ValueError('can only append views of known signals', obj)

        idx = self.sidx[obj.base]
        self.starts.append(self.rarray.starts[idx] + obj.offset)
        self.shape0s.append(shape0(obj))
        self.shape1s.append(shape1(obj))
        if obj.ndim == 0:
            self.stride0s.append(1)
            self.stride1s.append(1)
        elif obj.ndim == 1:
            self.stride0s.append(obj.elemstrides[0])
            self.stride1s.append(1)
        elif obj.ndim == 2:
            self.stride0s.append(obj.elemstrides[0])
            self.stride1s.append(obj.elemstrides[1])
        else:
            raise NotImplementedError()
        self.names.append(getattr(obj, 'name', ''))
        self.sidx[obj] = len(self.sidx)

    def add_views_to(self, rarray):
        rarray.add_views(
            starts=self.starts,
            shape0s=self.shape0s,
            shape1s=self.shape1s,
            stride0s=self.stride0s,
            stride1s=self.stride1s,
            names=self.names)


def signals_from_operators(operators):
    def all_with_dups():
        for op in operators:
            for sig in op.all_signals:
                yield sig
    return stable_unique(all_with_dups())


class Simulator(Simulator):

    profiling = False

    def __init__(self, network, dt=0.001, seed=None, model=None,
                 planner=greedy_planner):
        assert seed is None, "Seed not used"
        dt = float(dt)

        if model is None:
            self.model = Model(dt=dt,
                               label="%s, dt=%f" % (network, dt),
                               decoder_cache=get_default_decoder_cache())
        else:
            self.model = model

        if network is not None:
            # Build the network into the model
            self.model.build(network)

        # -- map from Signal.base -> ndarray
        sigdict = SignalDict()
        self._step = Signal(np.array(0.0, dtype=np.float64), name='step')
        self._time = Signal(np.array(0.0, dtype=np.float64), name='time')
        self._one = Signal(np.array(1.0, dtype=np.float64), name='ONE')
        self._zero = Signal(np.array(0.0, dtype=np.float64), name='ZERO')
        self._dt = Signal(np.array(dt, dtype=np.float64), name='DT')

        operators = list(self.model.operators)

        # -- add some time-keeping to the copied model
        #    this will be used by e.g. plan_SimPyFunc
        operators.append(StepUpdate(self._step, self._one))
        operators.append(Reset(self._time))
        operators.append(DotInc(self._dt, self._step, self._time))
        operators.append(DotInc(self._dt, self._one, self._time))

        # -- convert DotInc, Reset, Copy, and ProdUpdate to MultiProdUpdate
        operators = map(MultiProdUpdate.convert_to, operators)
        operators = MultiProdUpdate.compress(operators)
        self.operators = operators
        all_signals = signals_from_operators(operators)
        all_bases = stable_unique([sig.base for sig in all_signals])

        op_groups = planner(operators)
        self.op_groups = op_groups  # debug
        # print '-' * 80
        # self.print_op_groups()

        for op in operators:
            op.init_signals(sigdict)

        self.n_steps = 0

        # Add built states to the probe dictionary
        self._probe_outputs = self.model.params

        # Provide a nicer interface to probe outputs
        self.data = ProbeDict(self._probe_outputs)

        self.all_data = _RaggedArray(
            [sigdict[sb] for sb in all_bases],
            [getattr(sb, 'name', '') for ss in all_bases]
        )

        builder = ViewBuilder(all_bases, self.all_data)
        # self._DotInc_views = {}
        self._AX_views = {}
        for op_type, op_list in op_groups:
            self.setup_views(builder, op_type, op_list)
        for probe in self.model.probes:
            builder.append_view(self.model.sig[probe]['in'])
        builder.add_views_to(self.all_data)
        self.sidx = builder.sidx

        self._prep_all_data()

        self._plan = []
        for op_type, op_list in op_groups:
            self._plan.extend(self.plan_op_group(op_type, op_list))
        self._plan.extend(self.plan_probes())
        self.all_bases = all_bases

    def _prep_all_data(self):
        pass

    def print_op_groups(self):
        for op_type, op_list in self.op_groups:
            print 'op_type', op_type.__name__
            for op in op_list:
                print '  ', op

    def plan_op_group(self, op_type, ops):
        return getattr(self, 'plan_' + op_type.__name__)(ops)

    def setup_views(self, view_builder, op_type, ops):
        if hasattr(self, 'setup_views_' + op_type.__name__):
            getattr(self, 'setup_views_' + op_type.__name__)(
                view_builder, ops)
        else:
            for op in ops:
                map(view_builder.append_view, op.all_signals)

    def setup_views_MultiProdUpdate(self, view_builder, ops):
        def as2d(view):
            if view.ndim == 0:
                return view.reshape(1, 1)
            elif view.ndim == 1:
                return view.reshape(view.shape[0], 1)
            elif view.ndim == 2:
                return view
            else:
                raise ValueError(
                    "No support for tensors with %d dimensions" % view.ndim)

        if not hasattr(self, '_YYB_views'):
            self._YYB_views = {}

        for op in ops:
            Y_view = as2d(op.Y)
            Y_in_view = as2d(op.Y_in)
            if op._signal_beta is not None:
                YYB_views = [Y_view, Y_in_view, as2d(op._signal_beta)]
            else:
                YYB_views = [Y_view, Y_in_view]
            AX_views = []
            for A, X in zip(op.As, op.Xs):
                X_view = as2d(X)
                if A.ndim == 1 and X.ndim == 1:
                    A_view = A.reshape((1, A.shape[0]))  # vector dot
                else:
                    A_view = as2d(A)

                if A_view.shape == (1, 1):
                    # -- scalar AX_views can be done as reverse multiplication
                    A_view, X_view = X_view, A_view
                elif not (X_view.shape[0] == A_view.shape[1] and
                          X_view.shape[1] == Y_view.shape[1] and
                          A_view.shape[0] == Y_view.shape[0]):
                    raise ValueError('shape mismach (A: %s, X: %s, Y: %s)' %
                                     (A.shape, X.shape, op.Y.shape))

                AX_views.extend([A_view, X_view])

            map(view_builder.append_view,
                op.all_signals + AX_views + YYB_views)
            self._AX_views[op] = AX_views
            self._YYB_views[op] = YYB_views

    def plan_MultiProdUpdate(self, ops):
        constant_bs = [op for op in ops
                       if op._float_beta is not None]
        vector_bs = [op for op in ops
                     if op._signal_beta is not None
                     and op._signal_beta.ndim == 1]

        if len(constant_bs) + len(vector_bs) != len(ops):
            raise NotImplementedError()

        if len(ops) == 1:
            tag = ops[0].tag
        else:
            tag = 'ProdUpdate-scalar-constant-beta-%i' % len(ops)

        constant_b_gemvs = self.sig_gemv(
            constant_bs,
            1.0,
            A_js_fn=lambda op: self._AX_views[op][0::2],
            X_js_fn=lambda op: self._AX_views[op][1::2],
            beta=[op._float_beta for op in constant_bs],
            Y_sig_fn=lambda op: self._YYB_views[op][0],
            Y_in_sig_fn=lambda op: self._YYB_views[op][1],
            gamma=[op.gamma for op in constant_bs],
            verbose=0,
            tag=tag
        )

        vector_b_gemvs = self.sig_gemv(
            vector_bs,
            1.0,
            A_js_fn=lambda op: self._AX_views[op][0::2],
            X_js_fn=lambda op: self._AX_views[op][1::2],
            beta=lambda op: self._YYB_views[op][2],
            Y_sig_fn=lambda op: self._YYB_views[op][0],
            Y_in_sig_fn=lambda op: self._YYB_views[op][1],
            gamma=[op.gamma for op in vector_bs],
            verbose=0,
            tag='ProdUpdate-vector-beta'
        )
        return constant_b_gemvs + vector_b_gemvs

    def plan_ElementwiseInc(self, ops):
        sidx = self.sidx

        def elementwise_inc(profiling=False):
            for op in ops:
                A = self.all_data[sidx[op.A]]
                X = self.all_data[sidx[op.X]]
                Y = self.all_data[sidx[op.Y]]
                Y[...] += A * X
        return [elementwise_inc]

    def plan_SimDirect(self, ops):
        sidx = self.sidx

        def direct(profiling=False):
            for op in ops:
                J = self.all_data[sidx[op.J]]
                output = op.fn(J)
                self.all_data[sidx[op.output]] = output
        return [direct]

    def plan_SimPyFunc(self, ops):
        sidx = self.sidx
        t = self.all_data[sidx[self._time]]

        def pyfunc(profiling=False):
            for op in ops:
                output = self.all_data[sidx[op.output]]
                args = [t[0, 0]] if op.t_in else []
                args += [self.all_data[sidx[op.x]]] if op.x is not None else []
                out = np.asarray(op.fn(*args))
                if out.ndim == 1:
                    output[...] = out[:, None]
                else:
                    output[...] = out
        return [pyfunc]

    def plan_SimNeurons(self, ops):
        dt = self.model.dt
        sidx = self.sidx

        def neurons(profiling=False):
            for op in ops:
                J = self.all_data[sidx[op.J]]
                output = self.all_data[sidx[op.output]]
                states = [self.all_data[sidx[s]] for s in op.states]
                op.neurons.step_math(dt, J, output, *states)
        return [neurons]

    def plan_SimFilterSynapse(self, ops):
        assert all(len(op.num) == 1 and len(op.den) == 1 for op in ops)

        def synapse(profiling=False):
            for op in ops:
                x = self.all_data[self.sidx[op.input]]
                y = self.all_data[self.sidx[op.output]]
                y *= -op.den[0]
                y += op.num[0] * x
        return [synapse]

    def RaggedArray(self, *args, **kwargs):
        return _RaggedArray(*args, **kwargs)

    def sig_gemv(self, seq, alpha, A_js_fn, X_js_fn, beta, Y_sig_fn,
                 Y_in_sig_fn=None,
                 gamma=None,
                 verbose=0,
                 tag=None
                 ):
        if len(seq) == 0:
            return []
        sidx = self.sidx

        if callable(beta):
            beta_sigs = map(beta, seq)
            beta = self.RaggedArray(
                map(sidx.__getitem__, beta_sigs))

        Y_sigs = [Y_sig_fn(item) for item in seq]
        if Y_in_sig_fn is None:
            Y_in_sigs = Y_sigs
        else:
            Y_in_sigs = [Y_in_sig_fn(item) for item in seq]
        Y_idxs = [sidx[sig] for sig in Y_sigs]
        Y_in_idxs = [sidx[sig] for sig in Y_in_sigs]

        # -- The following lines illustrate what we'd *like* to see...
        #
        # A_js = self.RaggedArray(
        #   [[sidx[ss] for ss in A_js_fn(item)] for item in seq])
        # X_js = self.RaggedArray(
        #   [[sidx[ss] for ss in X_js_fn(item)] for item in seq])
        #
        # -- ... but the simulator supports broadcasting. So in fact whenever
        #    a signal in X_js has shape (N, 1), the corresponding A_js signal
        #    can have shape (M, N) or (1, 1).
        #    Fortunately, scalar multiplication of X by A can be seen as
        #    Matrix multiplication of A by X, so we can still use gemv,
        #    we just need to reorder and transpose.
        A_js = []
        X_js = []
        for ii, item in enumerate(seq):
            A_js_i = []
            X_js_i = []
            A_sigs_i = A_js_fn(item)
            X_sigs_i = X_js_fn(item)
            assert len(A_sigs_i) == len(X_sigs_i)
            for asig, xsig in zip(A_sigs_i, X_sigs_i):
                A_js_i.append(sidx[asig])
                X_js_i.append(sidx[xsig])
            A_js.append(A_js_i)
            X_js.append(X_js_i)

        if verbose:
            print "in sig_vemv"
            print "print A", A_js
            print "print X", X_js

        A_js = self.RaggedArray(A_js)
        X_js = self.RaggedArray(X_js)
        Y = self.all_data[Y_idxs]
        Y_in = self.all_data[Y_in_idxs]

        # if tag == 'transforms':
        #     print '=' * 70
        #     print A_js
        #     print X_js

        rval = self.plan_ragged_gather_gemv(
            alpha=alpha,
            A=self.all_data, A_js=A_js,
            X=self.all_data, X_js=X_js,
            beta=beta,
            Y=Y,
            Y_in=Y_in,
            tag=tag,
            seq=seq,
            gamma=gamma,
        )

        try:
            return rval.plans
        except AttributeError:
            return [rval]

    def plan_ragged_gather_gemv(self, alpha, A, A_js, X, X_js,
                                beta, Y, Y_in=None, tag=None, seq=None,
                                gamma=None):
        fn = lambda: ragged_gather_gemv(alpha, A, A_js, X, X_js, beta, Y,
                                        Y_in=Y_in, gamma=gamma,
                                        tag=tag,
                                        use_raw_fn=False)
        return PythonPlan(fn, name="npy_ragged_gather_gemv", tag=tag)

    def __getitem__(self, item):
        """
        Return internally shaped signals, which are always 2d
        """
        try:
            return self.all_data[self.sidx[item]]
        except KeyError:
            return self.all_data[self.sidx[self.model.memo[id(item)]]]

    @property
    def signals(self):
        """Get/set [properly-shaped] signal value (either 0d, 1d, or 2d)
        """
        class Accessor(object):

            def __iter__(_):
                return iter(self.all_bases)

            def __getitem__(_, item):
                # -- handle a few special keys
                item = {
                    '__time__': self._time,
                    '__step__': self._step,
                }.get(item, item)

                try:
                    raw = self.all_data[self.sidx[item]]
                except KeyError:
                    raw = self.all_data[self.sidx[self.model.memo[id(item)]]]
                assert raw.ndim == 2
                if item.ndim == 0:
                    return raw[0, 0]
                elif item.ndim == 1:
                    return raw.ravel()
                elif item.ndim == 2:
                    return raw
                else:
                    raise NotImplementedError()

            def __setitem__(_, item, val):
                if item not in self.sidx:
                    item = self.model.memo[id(item)]
                raw = self.all_data[self.sidx[item]]
                assert raw.ndim == 2
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
                import StringIO
                sio = StringIO.StringIO()
                for k in self_:
                    print >> sio, k, self_[k]
                return sio.getvalue()

        return Accessor()

    def plan_probes(self):
        if self.model.probes:
            def probe_fn(profiling=False):
                t0 = time.time()
                probes = self.model.probes
                for probe in probes:
                    period = (1 if probe.sample_every is None
                              else int(probe.sample_every / self.dt))
                    if self.n_steps % period == 0:
                        self._probe_outputs[probe].append(
                            self.signals[self.model.sig[probe]['in']].copy())
                t1 = time.time()
                probe_fn.cumtime += t1 - t0
            probe_fn.cumtime = 0.0
            return [probe_fn]
        else:
            return []

    def step(self):
        profiling = self.profiling
        for fn in self._plan:
            fn(profiling)
        self.n_steps += 1

    def run(self, time_in_seconds):
        """Simulate for the given length of time."""
        steps = int(np.round(float(time_in_seconds) / self.model.dt))
        logger.debug("Running %s for %f seconds, or %d steps",
                     self.model.label, time_in_seconds, steps)
        self.run_steps(steps)

    def run_steps(self, N, verbose=False):
        for i in xrange(N):
            self.step()

    def reset(self):
        raise NotImplementedError("Resetting not implemented")
