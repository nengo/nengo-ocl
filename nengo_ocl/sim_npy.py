"""
numpy Simulator in the style of the OpenCL one, to get design right.
"""
from __future__ import print_function

import logging
from collections import defaultdict, OrderedDict

import numpy as np

import nengo
from nengo.simulator import ProbeDict
from nengo.cache import get_default_decoder_cache
from nengo.builder.builder import Model
from nengo.builder.operator import Operator, Copy, DotInc, Reset
from nengo.builder.signal import Signal, SignalDict
from nengo.utils.compat import iteritems
import nengo.utils.numpy as npext
from nengo.utils.stdlib import Timer

from nengo_ocl.raggedarray import RaggedArray

logger = logging.getLogger(__name__)


class MultiProdUpdate(Operator):
    """ y <- gamma + beta * y_in + \sum_i dot(A_i, x_i) """

    def __init__(self, Y, Y_in, beta, gamma, tag, as_update=False):
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
        for view, set_ops in iteritems(sets):
            set_op, = set_ops
            done.add(set_op)
            for inc_op in incs.get(view, []):
                set_op.As.extend(inc_op.As)
                set_op.Xs.extend(inc_op.Xs)
                done.add(inc_op)
            rval.append(set_op)
        for view, inc_ops in iteritems(incs):
            for inc_op in inc_ops:
                if inc_op not in done:
                    rval.append(inc_op)
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


def greedy_planner(operators):
    from nengo.utils.simulator import operator_depencency_graph
    edges = operator_depencency_graph(operators)

    is_op = lambda op: isinstance(op, Operator)
    for op, dests in iteritems(edges):
        assert is_op(op) and all(is_op(op2) for op2 in dests)

    # map unscheduled ops to their direct predecessors and successors
    predecessors_of = {}
    successors_of = {}
    for op in operators:
        predecessors_of[op] = set()
        successors_of[op] = set()
    for op, dests in iteritems(edges):
        for op2 in dests:
            predecessors_of[op2].add(op)
        successors_of[op].update(dests)

    # available ops are ready to be scheduled (all predecessors scheduled)
    available = defaultdict(set)
    for op in (op for op, dep in iteritems(predecessors_of) if not dep):
        available[type(op)].add(op)

    rval = []
    while len(predecessors_of) > 0:
        if len(available) == 0:
            raise ValueError("Cycles in the op graph")

        chosen_type = sorted(available.items(), key=lambda x: len(x[1]))[-1][0]
        candidates = available[chosen_type]

        # --- greedily pick non-overlapping ops
        chosen = []
        base_sets = defaultdict(set)
        base_incs = defaultdict(set)
        base_updates = defaultdict(set)

        def overlaps(op):
            for s in op.sets:
                if any(s.may_share_memory(s2) for s2 in base_sets[s.base]):
                    return True
            for s in op.incs:
                if any(s.may_share_memory(s2) for s2 in base_incs[s.base]):
                    return True
            for s in op.updates:
                if any(s.may_share_memory(s2) for s2 in base_updates[s.base]):
                    return True
            return False

        for op in candidates:
            if not overlaps(op):
                # add op
                chosen.append(op)
                for s in op.sets:
                    base_sets[s.base].add(s)
                for s in op.incs:
                    base_incs[s.base].add(s)
                for s in op.updates:
                    base_updates[s.base].add(s)

        # --- schedule ops
        assert chosen
        rval.append((chosen_type, chosen))

        # --- update predecessors and successors of unsheduled ops
        available[chosen_type].difference_update(chosen)
        if not available[chosen_type]:
            del available[chosen_type]

        for op in chosen:
            for op2 in successors_of[op]:
                preds = predecessors_of[op2]
                preds.remove(op)
                if len(preds) == 0:
                    available[type(op2)].add(op2)
            del predecessors_of[op]
            del successors_of[op]

    assert len(operators) == sum(len(p[1]) for p in rval)
    # print('greedy_planner: Program len:', len(rval))
    return rval


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
        self.sidx = dict((bb, ii) for ii, bb in enumerate(bases))
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
        for op in (op for op in ops if isinstance(op, MultiProdUpdate)):
            A_views, X_views, Y_view, Y_in_view, beta_view = op.get_views()
            all_views.extend(A_views + X_views + [Y_view, Y_in_view] +
                             ([beta_view] if beta_view else []))
            self._A_views[op] = A_views
            self._X_views[op] = X_views
            self._YYB_views[op] = [Y_view, Y_in_view, beta_view]

        for view in all_views:
            self.append_view(view)


class Simulator(nengo.Simulator):

    profiling = False

    def __init__(self, network, dt=0.001, seed=None, model=None,
                 planner=greedy_planner):
        self.closed = False

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

        # --- set seed
        seed = np.random.randint(npext.maxint) if seed is None else seed
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

        # --- operators
        with Timer() as planner_timer:
            operators = list(self.model.operators)

            # convert DotInc, Reset, Copy, and ProdUpdate to MultiProdUpdate
            operators = list(map(MultiProdUpdate.convert_to, operators))
            operators = MultiProdUpdate.compress(operators)

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
            self._probe_outputs = self.model.params

            # Provide a nicer interface to probe outputs
            self.data = ProbeDict(self._probe_outputs)

            self.all_data = RaggedArray(
                [sigdict[sb] for sb in all_bases],
                [getattr(sb, 'name', '') for sb in all_bases],
                dtype=np.float32)

            builder = ViewBuilder(all_bases, self.all_data)
            builder.setup_views(operators)
            for probe in self.model.probes:
                builder.append_view(self.model.sig[probe]['in'])
            builder.add_views_to(self.all_data)

            self.all_bases = all_bases
            self.sidx = dict(builder.sidx)
            self._A_views = builder._A_views
            self._X_views = builder._X_views
            self._YYB_views = builder._YYB_views
            del builder

            self._prep_all_data()

        logger.info("Signals in %0.3f s" % signals_timer.duration)

        # --- create list of plans
        with Timer() as plans_timer:
            self._plan = []
            for op_type, op_list in op_groups:
                self._plan.extend(self.plan_op_group(op_type, op_list))
            self._plan.extend(self.plan_probes())

        logger.info("Plans in %0.3f s" % plans_timer.duration)

    def _prep_all_data(self):
        pass

    def plan_op_group(self, op_type, ops):
        return getattr(self, 'plan_' + op_type.__name__)(ops)

    def plan_PreserveValue(self, ops):
        return []  # do nothing

    def plan_MultiProdUpdate(self, ops):
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
            tag='ProdUpdate-constant-beta-%d' % len(constant_bs))
        vector_b_gemvs = self._sig_gemv(
            vector_bs, *args,
            beta=lambda op: self._YYB_views[op][2],
            gamma=[op.gamma for op in vector_bs],
            tag='ProdUpdate-vector-beta-%d' % len(vector_bs))
        return constant_b_gemvs + vector_b_gemvs

    def _sig_gemv(self, ops, A_js_fn, X_js_fn, Y_fn, Y_in_fn=None,
                  alpha=1.0, beta=1.0, gamma=0.0, tag=None):
        if len(ops) == 0:
            return []

        all_data, sidx = self.all_data, self.sidx
        A_js = RaggedArray(
            [[sidx[ss] for ss in A_js_fn(op)] for op in ops], dtype=np.int32)
        X_js = RaggedArray(
            [[sidx[ss] for ss in X_js_fn(op)] for op in ops], dtype=np.int32)
        Y_sigs = [Y_fn(item) for item in ops]
        Y_in_sigs = [Y_in_fn(item) for item in ops] if Y_in_fn else Y_sigs
        Y = all_data[[sidx[sig] for sig in Y_sigs]]
        Y_in = all_data[[sidx[sig] for sig in Y_in_sigs]]
        if callable(beta):
            beta = RaggedArray([sidx[beta(o)] for o in ops], dtype=np.float32)

        rval = self.plan_ragged_gather_gemv(
            alpha=alpha, A=all_data, A_js=A_js, X=all_data, X_js=X_js,
            beta=beta, Y=Y, Y_in=Y_in, gamma=gamma, tag=tag)
        return rval.plans

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
                from nengo.utils.compat import StringIO
                sio = StringIO()
                for k in self_:
                    print(k, self_[k], file=sio)
                return sio.getvalue()

        return Accessor()

    def reset(self, seed=None):
        if self.closed:
            raise ValueError("Simulator closed.")

        raise NotImplementedError("Resetting not implemented")
