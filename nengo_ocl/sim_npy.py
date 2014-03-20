"""
numpy Simulator in the style of the OpenCL one, to get design right.
"""

import time
from collections import defaultdict
import itertools
import logging
import networkx as nx

from tricky_imports import OrderedDict

logger = logging.getLogger(__name__)
info = logger.info
warn = logger.warn
error = logger.error
critical = logger.critical

import numpy as np

import nengo
from nengo import builder as nb
from nengo.nonlinearities import LIF, LIFRate, Direct

from .ra_gemv import ragged_gather_gemv
from .raggedarray import RaggedArray as _RaggedArray

from .plan import PythonPlan


class MultiProdUpdate(nb.Operator):
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
            assert isinstance(beta, nb.SignalView)
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
            assert set(rval.updates) == set(op.updates), (rval.updates, op.updates)
        if isinstance(op, nb.Reset):
            rval = cls(Y=op.dst, Y_in=op.dst, beta=0, gamma=op.value, as_update=False,
                    tag=getattr(op, 'tag', ''))
            assert_ok()
        elif isinstance(op, nb.Copy):
            rval = cls(op.dst, op.src, beta=1, gamma=0,
                    as_update=len(op.updates) == 1, tag=op.tag)
            assert_ok()
        elif isinstance(op, nb.DotInc):
            rval = cls(op.Y, op.Y, beta=1, gamma=0, as_update=False,
                    tag=op.tag)
            rval.add_AX(op.A, op.X)
            assert_ok()
        elif isinstance(op, nb.ProdUpdate):
            rval = cls(op.Y, op.Y, beta=op.B, gamma=0, as_update=True,
                    tag=op.tag)
            rval.add_AX(op.A, op.X)
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
        sets = defaultdict(list)
        incs = defaultdict(list)
        rval = []
        for op in operators:
            if isinstance(op, cls):
                if op.as_update:
                    rval.append(op)
                else:
                    assert op.sets or op.incs
                    if op.sets:
                        sets[op.sets[0]].append(op)
                    if op.incs:
                        incs[op.incs[0]].append(op)
            else:
                rval.append(op)
        done = set()
        for view, set_ops in sets.items():
            set_op, = set_ops
            done.add(set_op)
            for inc_op in incs[view]:
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
    return isinstance(op, nb.Operator)


_TimerCumulative = defaultdict(float)
_TimerCalls = defaultdict(int)
class Timer(object):
    enabled = False
    def __init__(self, msg, print_freq=0, enabled=enabled):
        self.msg = msg
        self.print_freq = print_freq
        self.enabled = enabled
    def __enter__(self):
        self.t0 = time.time()
    def __exit__(self, *args):
        if self.enabled:
            self.t1 = time.time()
            _TimerCumulative[self.msg] += self.t1 - self.t0
            _TimerCalls[self.msg] += 1
            if (self.print_freq == 0
                    or _TimerCalls[self.msg] % self.print_freq == 0):
                print 'Timer: "%s" took %f (cumulative: %f, calls: %i)' % (
                        self.msg,
                        self.t1 - self.t0,
                        _TimerCumulative[self.msg],
                        _TimerCalls[self.msg])


def exact_dependency_graph(operators, share_memory):
    dg = nx.DiGraph()

    for op in operators:
        dg.add_edges_from(itertools.product(op.reads + op.updates, [op]))
        dg.add_edges_from(itertools.product([op], op.sets + op.incs))

    #print ' .. adding edges for %i nodes' % len(dg.nodes())

    # -- all views of a base object in a particular dictionary
    by_base_writes = defaultdict(list)
    by_base_reads = defaultdict(list)
    reads = defaultdict(list)
    sets = defaultdict(list)
    incs = defaultdict(list)
    ups = defaultdict(list)

    for op in operators:
        for node in op.sets + op.incs:
            by_base_writes[node.base].append(node)

        for node in op.reads:
            by_base_reads[node.base].append(node)

        for node in op.reads:
            reads[node].append(op)

        for node in op.sets:
            sets[node].append(op)

        for node in op.incs:
            incs[node].append(op)

        for node in op.updates:
            ups[node].append(op)

    # -- assert that only one op sets any particular view
    for node in sets:
        assert len(sets[node]) == 1, (node, sets[node])

    # -- assert that no two views are both set and aliased
    if len(sets) >= 2:
        for node, other in itertools.combinations(sets, 2):
            assert not share_memory(node, other)

    # -- incs depend on sets
    #    Create an edge between any two ops (a, b)
    #    if `a` sets a signal `u` that is [aliased to]
    #    a signal `v` incremented by `b`
    for inc_view, inc_ops in incs.items():
        for set_view, set_ops in sets.items():
            if share_memory(inc_view, set_view):
                dg.add_edges_from(itertools.product(set_ops, inc_ops))


    # -- reads depend on writes (sets and incs)
    for node, post_ops in reads.items():
        pre_ops = []
        for other in by_base_writes[node.base]:
            if share_memory(node, other):
                pre_ops += sets[other] + incs[other]
        dg.add_edges_from(itertools.product(set(pre_ops), post_ops))

    # -- assert that only one op updates any particular view
    for node in ups:
        assert len(ups[node]) == 1, (node, ups[node])

    # -- assert that no two views are both updated and aliased
    if len(ups) >= 2:
        for node, other in itertools.combinations(ups, 2):
            assert not share_memory(node, other), (
                    node, other)

    # -- updates depend on reads, sets, and incs.
    for node, post_ops in ups.items():
        pre_ops = []
        others = by_base_writes[node.base] + by_base_reads[node.base]
        for other in others:
            if share_memory(node, other):
                pre_ops += sets[other] + incs[other] + reads[other]
        dg.add_edges_from(itertools.product(set(pre_ops), post_ops))

    for op in operators:
        assert op in dg

    return dg


def greedy_planner(operators, share_memory, cliques):
    """
    I feel like there might e a dynamic programming solution here, but I can't
    work it out, and I'm not sure. Even if a DP solution existed, we would
    need a function to estimate the goodness (e.g. neg wall time) of kernel
    calls, and  that function would need to be pretty fast.
    """
    dg = exact_dependency_graph(operators, share_memory)
    #depth = {}
    #ops_by_depth = defaultdict(list)

    # -- TODO: linear time algo
    ancestors_of = {}
    for op in operators:
        ancestors_of[op] = set(filter(is_op, nx.ancestors(dg, op)))

    scheduled = set()
    rval = []
    #for k in cliques:
        #print k, cliques[k]
    while len(scheduled) < len(operators):
        candidates = [op
            for op, pre_ops in ancestors_of.items()
            if not pre_ops and op not in scheduled]

        type_counts = defaultdict(int)
        for op in candidates:
            type_counts[type(op)] += 1
        chosen_type = sorted(type_counts.items(), key=lambda x: x[1])[-1][0]
        #print chosen_type
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

        #print by_base.keys()
        chosen = []
        for base, ops_writing_to_base in by_base.items():
            if base in cliques_by_base:
                most_ops = sorted((len(base_ops), base_ops)
                    for cliq_id, base_ops in cliques_by_base[base].items())
                chosen.extend(most_ops[-1][1])
            else:
                chosen.append(ops_writing_to_base[0])
        # -- ops that produced multiple outputs show up multiple times
        chosen = stable_unique(chosen)

        assert chosen

        #print clique_counts.keys()
        #print by_base
        assert not any(cc in scheduled for cc in chosen)
        scheduled.update(chosen)
        rval.append((chosen_type, chosen))
        # -- prepare for next iteration
        for op in ancestors_of:
            ancestors_of[op].difference_update(chosen)

    #print sum(len(p[1]) for p in rval)
    assert len(operators) == sum(len(p[1]) for p in rval)
    #print 'greedy_planner: Program len:', len(rval)
    return rval


def sequential_planner(operators, share_memory):
    """
    I feel like there might e a dynamic programming solution here, but I can't
    work it out, and I'm not sure. Even if a DP solution existed, we would
    need a function to estimate the goodness (e.g. neg wall time) of kernel
    calls, and  that function would need to be pretty fast.
    """
    dg = exact_dependency_graph(operators, share_memory)

    # list of pairs: (type, [ops_of_type], set_of_ancestors, set_of_descendants)
    topo_order = [op
        for op in nx.topological_sort(dg)
        if is_op(op)]

    return [(type(op), [op]) for op in topo_order]


class SignalDict(dict):
    """
    Map from Signal -> ndarray

    SignalDict overrides __getitem__ for two reasons:
    1. so that scalars are returned as 0-d ndarrays
    2. so that a SignalView lookup returns a views of its base

    """
    def __getitem__(self, obj):
        if obj in self:
            return dict.__getitem__(self, obj)
        elif obj.base in self:
            # look up views as a fallback
            # --work around numpy's special case behaviour for scalars
            base_array = self[obj.base]
            try:
                # for some installations, this works
                itemsize = int(obj.dtype.itemsize)
            except TypeError:
                # other installations, this...
                itemsize = int(obj.dtype().itemsize)
            byteoffset = itemsize * obj.offset
            bytestrides = [itemsize * s for s in obj.elemstrides]
            view = np.ndarray(shape=obj.shape,
                              dtype=obj.dtype,
                              buffer=base_array.data,
                              offset=byteoffset,
                              strides=bytestrides,
                             )
            return view
        else:
            raise KeyError(obj)


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
        #self.orig_len = len(self.all_signals)
        #self.base_starts = self.all_data_A.starts

    def append_view(self, obj):
        assert obj.size
        shape0(obj)
        shape1(obj)

        if obj in self.sidx:
            return
            #raise KeyError('sidx already contains object', obj)

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


class Simulator(object):

    profiling = False

    def __init__(self, model, dt=0.001, seed=None, builder=None,
            planner=greedy_planner,
            ):

        if builder is None:
            builder = nb.Builder(copy=True)

        # -- map from Signal.base -> ndarray
        sigdict = SignalDict()
        self._step = nb.Signal(np.asarray(0.0, dtype=np.float64), name='step')
        self._time = nb.Signal(np.asarray(0.0, dtype=np.float64), name='time')
        self._one = nb.Signal(np.asarray(1.0, dtype=np.float64), name='ONE')
        self._zero = nb.Signal(np.asarray(0.0, dtype=np.float64), name='ZERO')
        self._dt = nb.Signal(np.asarray(dt, dtype=np.float64), name='DT')

        # -- possibly make a copy of the model
        self.model = builder(model, dt)

        # -- add some time-keeping to the copied model
        #    this will be used by e.g. plan_SimPyFunc
        self.model.operators.append(
            nb.ProdUpdate(self._one, self._one, self._one, self._step))
        self.model.operators.append(
            nb.Reset(self._time))
        self.model.operators.append(
            nb.DotInc(self._dt, self._step, self._time))
        self.model.operators.append(
            nb.DotInc(self._dt, self._one, self._time))

        # -- convert DotInc, Reset, Copy, and ProdUpdate to MultiProdUpdate
        operators = map(MultiProdUpdate.convert_to, self.model.operators)
        operators = MultiProdUpdate.compress(operators)
        self.operators = operators
        all_signals = signals_from_operators(operators)
        all_bases = stable_unique([sig.base for sig in all_signals])
        # -- print ORIGINAL operators
        #print '=' * 80
        #for op in self.model.operators:
            #print op

        _shares_memory_with = getattr(self.model, '_shares_memory_with', {})
        indep_cliques = getattr(self.model, '_independent_view_cliques', {})
        def share_memory(a, b):
            if a.base is not b.base:
                return False
            base = a.base
            astruct = a.structure
            bstruct = b.structure
            key0 = (base, astruct, bstruct)
            key1 = (base, bstruct, astruct)
            try:
                return _shares_memory_with[key0]
            except KeyError:
                pass
            try:
                return _shares_memory_with[key1]
            except KeyError:
                pass
            if any(astruct in cc and bstruct in cc
                    for cc in indep_cliques.get(base, [])):
                rval = True
            else:
                rval = a.shares_memory_with(b)
            _shares_memory_with[key0] = rval
            _shares_memory_with[key1] = rval
            return rval

        op_groups = planner(operators, share_memory, indep_cliques)
        self.op_groups = op_groups # debug
        #print '-' * 80
        #self.print_op_groups()

        for op in operators:
            op.init_sigdict(sigdict, self.model.dt)


        self.n_steps = 0
        self.probe_outputs = dict((probe, []) for probe in self.model.probes)

        self.all_data = _RaggedArray(
                [sigdict[sb] for sb in all_bases],
                [getattr(sb, 'name', '') for ss in all_bases]
                )
        #for k in all_bases:
            #print k, k.shape#, sigdict[k]

        builder = ViewBuilder(all_bases, self.all_data)
        #self._DotInc_views = {}
        self._AX_views = {}
        for op_type, op_list in op_groups:
            self.setup_views(builder, op_type, op_list)
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
                    A_view = A.reshape((1, A.shape[0])) # vector dot
                else:
                    A_view = as2d(A)

                if A_view.shape == (1, 1):
                    # -- scalar AX_views can be done as reverse multiplication
                    A_view, X_view = X_view, A_view

                if not (X_view.shape[0] == A_view.shape[1] and
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
        constant_bs = [op
            for op in ops
            if op._float_beta is not None]

        vector_bs = [op for op in ops
            if op._signal_beta is not None and op._signal_beta.ndim == 1]

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

    def plan_SimDirect(self, ops):
        sidx = self.sidx
        def direct(profiling=False):
            with Timer('simdirect', 1000):
                for op in ops:
                    J = self.all_data[sidx[op.J]]
                    output = op.fn(J)
                    self.all_data[sidx[op.output]] = output
                    #print 'direct',
                    #print op.J, self.all_data[sidx[op.J]],
                    #print op.output, self.all_data[sidx[op.output]]
        return [direct]

    def plan_SimPyFunc(self, ops):
        dt = self.model.dt
        sidx = self.sidx
        t = self.all_data[sidx[self._time]]
        def pyfunc(profiling=False):
            for op in ops:
                output = self.all_data[sidx[op.output]]
                # -- YEP, subtracting off DT is crazy
                #    but it makes nengo's tests pass.
                #    See nengo ticket #234 for potential resolution.
                args = [t[0, 0] - dt] if op.t_in else []
                args += [self.all_data[sidx[op.x]]] if op.x is not None else []
                out = np.asarray(op.fn(*args))
                if out.ndim == 1:
                    output[...] = out[:, None]
                else:
                    #print output.shape, out.shape, op.fn, op
                    #print self._time.shape, t.shape
                    output[...] = out
        return [pyfunc]

    def plan_SimLIF(self, ops):
        dt = self.model.dt
        sidx = self.sidx
        def lif(profiling=False):
            for op in ops:
                J = self.all_data[sidx[op.J]]
                voltage = self.all_data[sidx[op.voltage]]
                reftime = self.all_data[sidx[op.refractory_time]]
                output = self.all_data[sidx[op.output]]
                op.nl.step_math(dt, J, voltage, reftime, output)
        return [lif]

    def plan_SimLIFRate(self, ops):
        dt = self.model.dt
        def lif_rate(profiling=False):
            for op in ops:
                J = self.all_data[self.sidx[op.J]]
                output = self.all_data[self.sidx[op.output]]
                op.nl.step_math(dt, J, output)
        return [lif_rate]

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
            ysig = Y_sigs[ii]
            yidx = Y_idxs[ii]
            yM = self.all_data.shape0s[yidx]
            yN = self.all_data.shape1s[yidx]
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

    def get(self, obj):
        """Get the simulator's copy of a model object.

        Parameters
        ----------
        obj : Nengo object
            A model from the original model

        Returns
        -------
        sim_obj : Nengo object
            The simulator's copy of `obj`.

        Examples
        --------
        Get the simulator's version of an ensemble
        in order to plot tuning curves

        >>> model = nengo.Model()
        >>> model.make_ensemble("A", nengo.LIF(4), 1)
        >>> sim = model.simulator()
        >>> A = sim.get("A")
        >>> from nengo.helpers import tuning_curves
        >>> print tuning_curves(A)
        """
        toret = self.model.get(obj, "NotFound")
        if toret == "NotFound":
            toret = self.model.memo[id(obj)]
        return toret

    def plan_probes(self):
        if self.model.probes:
            def probe_fn(profiling=False):
                t0 = time.time()
                probes = self.model.probes
                for probe in probes:
                    period = int(probe.dt // self.model.dt)
                    if self.n_steps % period == 0:
                        self.probe_outputs[probe].append(
                            self.signals[probe.sig].copy())
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

    # XXX there is both .signals and .signal and they are pretty different
    def signal(self, sig):
        probes = [sp for sp in self.model.probes if sp.sig == sig]
        if len(probes) == 0:
            raise KeyError()
        elif len(probes) > 1:
            raise KeyError()
        else:
            return self.signal_probe_output(probes[0])

    def probe_data(self, probe):
        return np.asarray(self.probe_outputs[probe])

    def data(self, probe):
        """Get data from signals that have been probed.

        Parameters
        ----------
        probe : Probe
            TODO

        Returns
        -------
        data : ndarray
            TODO: what are the dimensions?
        """
        probe_cpy = self.model.memo[id(probe)]
        return self.probe_data(probe_cpy.probe)

    def trange(self, dt=None):
        last_t = (self.n_steps - 1) * self.model.dt
        dt = self.model.dt if dt is None else dt
        n_steps = self.n_steps if dt is None else int(
            self.n_steps / (dt / self.model.dt))
        return np.linspace(0, last_t, n_steps)

# -- for flake-8
