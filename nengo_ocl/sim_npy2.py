"""
numpy Simulator in the style of the OpenCL one, to get design right.
"""

from collections import defaultdict
import itertools
import logging
import networkx
from networkx import ancestors, topological_sort

from tricky_imports import OrderedDict

logger = logging.getLogger(__name__)
info = logger.info
warn = logger.warn
error = logger.error
critical = logger.critical

import numpy as np

from nengo.core import Signal
from nengo.core import SignalView
from nengo.core import LIF, LIFRate, Direct
from nengo import simulator as sim

from ra_gemv import ragged_gather_gemv
from raggedarray import RaggedArray


def exact_dependency_graph(operators):
    dg = networkx.DiGraph()

    for op in operators:
        dg.add_edges_from(itertools.product(op.reads + op.updates, [op]))
        dg.add_edges_from(itertools.product([op], op.sets + op.incs))

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
    for node, other in itertools.combinations(sets, 2):
        assert not node.shares_memory_with(other)

    # -- incs depend on sets
    for node, post_ops in incs.items():
        pre_ops = []
        for other in by_base_writes[node.base]:
            if node.shares_memory_with(other):
                pre_ops += sets[other]
        dg.add_edges_from(itertools.product(set(pre_ops), post_ops))

    # -- reads depend on writes (sets and incs)
    for node, post_ops in reads.items():
        pre_ops = []
        for other in by_base_writes[node.base]:
            if node.shares_memory_with(other):
                pre_ops += sets[other] + incs[other]
        dg.add_edges_from(itertools.product(set(pre_ops), post_ops))

    # -- assert that only one op updates any particular view
    for node in ups:
        assert len(ups[node]) == 1, (node, ups[node])

    # -- assert that no two views are both updated and aliased
    for node, other in itertools.combinations(ups, 2):
        assert not node.shares_memory_with(other), (
                node, other)

    # -- updates depend on reads, sets, and incs.
    for node, post_ops in ups.items():
        pre_ops = []
        others = by_base_writes[node.base] + by_base_reads[node.base]
        for other in others:
            if node.shares_memory_with(other):
                pre_ops += sets[other] + incs[other] + reads[other]
        dg.add_edges_from(itertools.product(set(pre_ops), post_ops))

    for op in operators:
        assert op in dg

    return dg


def concurrent_ok(atable, op1, op2):
    return op1 not in atable[op2]\
            and op2 not in atable[op1]

def greedy_planner(dg, operators):
    """
    I feel like there might e a dynamic programming solution here, but I can't
    work it out, and I'm not sure. Even if a DP solution existed, we would
    need a function to estimate the goodness (e.g. neg wall time) of kernel
    calls, and  that function would need to be pretty fast.
    """

    # XXX: the `atable` can be built in linear rather than quadratic time
    atable = {}
    for op in operators:
        atable[op] = ancestors(dg, op)

    ops_by_type = defaultdict(list)
    for op in operators:
        ops_by_type[type(op)].append(op)

    concurrent_groups = []

    for op_type, ops in ops_by_type:
        concurrent_group = [ops[0]]
        #for other in ops[:]

    raise NotImplementedError()


def sequential_planner(dg, operators):
    """
    I feel like there might e a dynamic programming solution here, but I can't
    work it out, and I'm not sure. Even if a DP solution existed, we would
    need a function to estimate the goodness (e.g. neg wall time) of kernel
    calls, and  that function would need to be pretty fast.
    """

    def is_op(op):
        return isinstance(op, sim.Operator)

    # list of pairs: (type, [ops_of_type], set_of_ancestors, set_of_descendants)
    op_groups = []
    topo_order = [op
        for op in topological_sort(dg)
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

from sim_npy import stable_unique
from sim_npy import isview
from sim_npy import ViewBuilder

def signals_from_operators(operators):
    def all_with_dups():
        for op in operators:
            for sig in op.all_signals:
                yield sig
    return stable_unique(all_with_dups())


class Simulator(object):

    def __init__(self, model):

        if not hasattr(model, 'dt'):
            raise ValueError("Model does not appear to be built. "
                             "See Model.prep_for_simulation.")

        self.model = model
        dt = model.dt
        operators = model._operators
        all_signals = signals_from_operators(operators)
        all_bases = [sig for sig in all_signals if not isview(sig)]

        # -- map from Signal.base -> ndarray
        sigdict = SignalDict()
        for op in model._operators:
            op.init_sigdict(sigdict, model.dt)

        self.sim_step = 0
        self.probe_outputs = dict((probe, []) for probe in model.probes)

        self.all_data = RaggedArray(
                [sigdict[sb] for sb in all_bases],
                [getattr(sb, 'name', '') for ss in all_bases]
                )

        builder = ViewBuilder(all_bases, self.all_data)
        map(builder.append_view, all_signals)
        builder.add_views_to(self.all_data)
        self.sidx = builder.sidx

        dg = exact_dependency_graph(self.model._operators)
        op_groups = sequential_planner(dg, self.model._operators)
        self._plan = []
        for op_type, op_list in op_groups:
            self._plan.extend(self.plan_op_group(op_type, op_list))
        self.all_bases = all_bases
        self.op_groups = op_groups # debug

    def print_op_groups(self):
        for op_type, op_list in self.op_groups:
            print 'op_type', op_type.__name__
            for op in op_list:
                print '  ', op

    def plan_op_group(self, op_type, ops):
        return getattr(self, 'plan_' + op_type.__name__)(ops)

    def plan_Reset(self, ops):
        if not all(op.value == 0 for op in ops):
            raise NotImplementedError()

        return self.sig_gemv(
            ops,
            0.0,
            lambda op: [],
            lambda op: [],
            0.0,
            lambda op: op.dst,
            verbose=0,
            tag='Reset'
            )

    def plan_DotInc(self, ops):
        return self.sig_gemv(
            ops,
            1.0,
            lambda op: [op.A],
            lambda op: [op.X],
            1.0,
            lambda op: op.Y,
            verbose=0,
            tag='DotInc'
            )

    def plan_Copy(self, ops):
        return self.sig_gemv(
            ops,
            1.0,
            lambda op: [op.src],
            lambda op: [self.model.one],
            0.0,
            lambda op: op.dst,
            verbose=0,
            tag='Copy'
            )

    def plan_SimDirect(self, ops):
        sidx = self.sidx
        def direct():
            for op in ops:
                J = self.all_data[sidx[op.J]]
                output = op.fn(J)
                self.all_data[sidx[op.output]] = output
                #print 'direct',
                #print op.J, self.all_data[sidx[op.J]],
                #print op.output, self.all_data[sidx[op.output]]
        return [direct]

    def RaggedArray(self, *args, **kwargs):
        return RaggedArray(*args, **kwargs)

    def sig_gemv(self, seq, alpha, A_js_fn, X_js_fn, beta, Y_sig_fn,
                 Y_in_sig_fn=None,
                 verbose=0,
                 tag=None
                ):
        if len(seq) == 0:
            return []

        sidx = self.sidx
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
                aidx = sidx[asig]
                xidx = sidx[xsig]
                aM = self.all_data.shape0s[aidx]
                aN = self.all_data.shape1s[aidx]
                xM = self.all_data.shape0s[xidx]
                xN = self.all_data.shape1s[xidx]
                if aN == aM == 1:
                    # -- X must be column vector for this trick
                    if xN != 1 or xM != yM or xN != yN:
                        raise ValueError('shape mismatch in sig_gemv',
                                         ((asig, aM, aN),
                                          (xsig, xM, xN),
                                          (ysig, yM, yN),
                                         ))
                    A_js_i.append(xidx)
                    X_js_i.append(aidx)
                else:
                    if aN != xM or aM != yM or xN != yN:
                        raise ValueError('shape mismatch in sig_gemv',
                                         ((asig, aM, aN),
                                          (xsig, xM, xN),
                                          (ysig, yM, yN),
                                         ))
                    A_js_i.append(aidx)
                    X_js_i.append(xidx)
            A_js.append(A_js_i)
            X_js.append(X_js_i)

        if verbose:
            print 'in sig_vemv'
            print 'print A', A_js
            print 'print X', X_js

        A_js = self.RaggedArray(A_js)
        X_js = self.RaggedArray(X_js)
        Y = self.all_data[Y_idxs]
        Y_in = self.all_data[Y_in_idxs]

        # if tag == 'transforms':
        #     print '=' * 70
        #     print A_js
        #     print X_js

        return [self.plan_ragged_gather_gemv(
            alpha=alpha,
            A=self.all_data, A_js=A_js,
            X=self.all_data, X_js=X_js,
            beta=beta,
            Y=Y,
            Y_in=Y_in,
            tag=tag,
            seq=seq,
            )]

    def plan_ragged_gather_gemv(self, *args, **kwargs):
        return (lambda: ragged_gather_gemv(*args, **kwargs))

    def __getitem__(self, item):
        """
        Return internally shaped signals, which are always 2d
        """
        try:
            return self.all_data[self.sidx[item]]
        except KeyError:
            return self.all_data[self.sidx[self.copied(item)]]

    @property
    def signals(self):
        """Get/set [properly-shaped] signal value (either 0d, 1d, or 2d)
        """
        class Accessor(object):
            def __iter__(_):
                return iter(self.all_bases)

            def __getitem__(_, item):
                try:
                    raw = self.all_data[self.sidx[item]]
                except KeyError:
                    raw = self.all_data[self.sidx[self.copied(item)]]
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
                import StringIO
                sio = StringIO.StringIO()
                for k in self_:
                    print >> sio, k, self_[k]
                return sio.getvalue()

        return Accessor()

    def copied(self, obj):
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
        Manually set a raw signal value to ``5`` in the simulator
        (advanced usage). [TODO: better example]

        >>> model = nengo.Model()
        >>> foo = m.add(Signal(n=1))
        >>> sim = model.simulator()
        >>> sim.signals[sim.copied(foo)] = np.asarray([5])
        """
        return self.model.memo[id(obj)]

    def plan_probes(self):
        def fn():
            probes = self.model.probes
            #sidx = self.sidx
            for probe in probes:
                period = int(probe.dt // self.model.dt)
                if self.sim_step % period == 0:
                    self.probe_output[probe].append(
                        self.signals[probe.sig].copy())
        return fn

    def step(self):
        for fn in self._plan:
            fn()
        # print self.signals
        self.sim_step += 1

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
        return np.asarray(self.probe_output[probe])
