"""
Memory management, group signals on device based on what operators use them.
"""
from nengo_ocl.operators import MultiDotInc
from nengo_ocl.utils import OrderedSet


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
        shape0 = obj.shape[0] if obj.ndim > 0 else 1
        shape1 = obj.shape[1] if obj.ndim > 1 else 1
        self.starts.append(self.rarray.starts[idx] + obj.elemoffset)
        self.shape0s.append(shape0)
        self.shape1s.append(shape1)
        self.stride0s.append(obj.elemstrides[0] if shape0 > 1 else 1)
        self.stride1s.append(obj.elemstrides[1] if shape1 > 1 else 1)
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


def basic_grouper(op_groups):
    # operators = stable_unique(op for _, ops in op_groups for op in ops)
    # all_signals = stable_unique(s for op in operators for s in op.all_signals)
    # all_bases = stable_unique(s.base for s in all_signals)

    # operators = set(op for _, ops in op_groups for op in ops)
    # all_signals = set(s for op in operators for s in op.all_signals)
    # all_bases = set(s.base for s in all_signals)

    # --- group bases into sets based on what operators they appear in
    base_sets = []

    def augment(base_sets, ops, attr):
        n = len(getattr(ops[0], attr))
        assert all(len(getattr(op, attr)) == n for op in ops)
        for i in range(n):
            base_sets.append(
                OrderedSet(getattr(op, attr)[i].base for op in ops))

    for op_type, ops in op_groups:
        augment(base_sets, ops, 'sets')
        augment(base_sets, ops, 'incs')
        augment(base_sets, ops, 'reads')
        augment(base_sets, ops, 'updates')

    # --- combine sets that share bases
    groups = []
    for base_set in base_sets:
        for group in groups:
            if not group.isdisjoint(base_set):
                group.update(base_set)
                break
        else:
            groups.append(OrderedSet(base_set))

    return [list(group) for group in groups]
