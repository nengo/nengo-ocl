from collections import defaultdict

from nengo.builder.operator import Operator
from nengo.utils.compat import iteritems
from nengo.utils.simulator import operator_depencency_graph


class DependencyTracker(object):

    def __init__(self, ops, edges):
        # map unscheduled ops to their direct predecessors and successors
        predecessors_of = {}
        successors_of = {}
        for op in ops:
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

        self.predecessors_of = predecessors_of
        self.successors_of = successors_of
        self.available = available

    def copy(self):
        copy = type(self).__new__(type(self))
        copy.predecessors_of = {op: set(s) for op, s in iteritems(self.predecessors_of)}
        copy.successors_of = {op: set(s) for op, s in iteritems(self.successors_of)}
        copy.available = defaultdict(set)
        for op_type, s in iteritems(self.available):
            copy.available[op_type] = set(s)
        return copy

    def greedy_choose_type(self):
        chosen_type = sorted(
            self.available.items(), key=lambda x: len(x[1]))[-1][0]
        candidates = self.available[chosen_type]
        return chosen_type, candidates

    def remove_op(self, op, update_available=True):
        available = self._remove_op_from_predsucc(op)

        if update_available:
            if type(op) in self.available:
                self.available[type(op)].difference_update([op])
                if not self.available[type(op)]:
                    del self.available[type(op)]
            for op in available:
                self.available[type(op)].add(op)

    def remove_ops(self, ops, update_available=True):
        available = []
        for op in ops:
            available.extend(self._remove_op_from_predsucc(op))

        if update_available:
            for op in ops:
                if type(op) in self.available:
                    self.available[type(op)].difference_update([op])
            for op in available:
                self.available[type(op)].add(op)
            for op_type, s in self.available:
                if not s:
                    del self.available[op_type]

    def remove_op_type(self, chosen_type, chosen_ops, update_available=True):
        available = []
        for op in chosen_ops:
            available.extend(self._remove_op_from_predsucc(op))

        if update_available:
            self.available[chosen_type].difference_update(chosen_ops)
            for op in available:
                self.available[type(op)].add(op)
            if not self.available[chosen_type]:
                del self.available[chosen_type]

    def _remove_op_from_predsucc(self, op):
        available = []
        for op2 in self.successors_of[op]:
            preds = self.predecessors_of[op2]
            preds.remove(op)
            if len(preds) == 0:
                available.append(op2)
        for op2 in self.predecessors_of[op]:
            self.successors_of[op2].remove(op)
        del self.predecessors_of[op]
        del self.successors_of[op]
        return available


def _greedy_nonoverlapping(ops):
    chosen = []
    base_sets = defaultdict(set)
    base_incs = defaultdict(set)
    base_updates = defaultdict(set)

    def add_op(op):
        chosen.append(op)
        for s in op.sets:
            base_sets[s.base].add(s)
        for s in op.incs:
            base_incs[s.base].add(s)
        for s in op.updates:
            base_updates[s.base].add(s)

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

    for op in ops:
        if not overlaps(op):
            add_op(op)

    return chosen


def greedy_planner(operators):
    """Plan order of operators greedily (choosing most available operators).

    Plan the order of operators by iteratively determining which operators
    have all their predecessors planned, choosing the type with the highest
    number of such operators, and planning all non-overlapping operators of
    that type.
    """

    edges = operator_depencency_graph(operators)

    is_op = lambda op: isinstance(op, Operator)
    for op, dests in iteritems(edges):
        assert is_op(op) and all(is_op(op2) for op2 in dests)

    deps = DependencyTracker(operators, edges)

    rval = []
    while len(deps.predecessors_of) > 0:
        if len(deps.available) == 0:
            raise ValueError("Cycles in the op graph")

        chosen_type, candidates = deps.greedy_choose_type()

        # --- greedily pick non-overlapping ops
        chosen_ops = _greedy_nonoverlapping(candidates)

        # --- schedule ops
        assert chosen_ops
        rval.append((chosen_type, chosen_ops))

        # --- update predecessors and successors of unsheduled ops
        deps.remove_op_type(chosen_type, chosen_ops)

    assert len(operators) == sum(len(p[1]) for p in rval)
    # print('greedy_planner: Program len:', len(rval))
    return rval


def parallel_planner(operators):
    """Plan order of operators by determining parallel sets and optimizing.

    TODO
    """
    from hunse_tools.timing import tic, toc

    edges = operator_depencency_graph(operators)

    is_op = lambda op: isinstance(op, Operator)
    for op, dests in iteritems(edges):
        assert is_op(op) and all(is_op(op2) for op2 in dests)

    deps = DependencyTracker(operators, edges)

    # --- determine all successors of each op
    temp_deps = deps.copy()
    all_successors = {}
    while temp_deps.successors_of:
        queued = [
            op for op, succs in iteritems(temp_deps.successors_of) if not succs]
        assert queued

        for op in queued:
            op_preds = set()
            for a in deps.successors_of[op]:
                op_preds.add(a)
                op_preds.update(all_successors[a])

            all_successors[op] = op_preds

        temp_deps.remove_ops(queued, update_available=False)

    # --- determine which operators of the same type are independent (parallel)
    operators_by_type = defaultdict(set)
    for op in operators:
        operators_by_type[type(op)].add(op)

    parallel_by_type = {}
    for op_type, ops in iteritems(operators_by_type):
        ops = set(ops)

        groups = []
        while ops:
            op = ops.pop()
            group = set([op])
            for op2 in ops:
                if (op2 not in all_successors[op] and
                        op not in all_successors[op2]):
                    group.add(op2)
            groups.append(group)

        parallel_by_type[op_type] = groups

    rval = []
    while len(deps.predecessors_of) > 0:
        if len(deps.available) == 0:
            raise ValueError("Cycles in the op graph")

        fracs = {}
        for op_type, ops in iteritems(deps.available):
            for parallel_ops in parallel_by_type[op_type]:
                if ops.issubset(parallel_ops):
                    break
            else:
                raise ValueError("Could not find superset group")

            if len(ops) == len(parallel_ops):
                fracs[op_type] = -1  # to choose this first
            else:
                fracs[op_type] = float(len(ops)) / len(parallel_ops)

        # chosen_type = sorted(iteritems(fracs), key=lambda x: x[1])[-1][0]
        chosen_type = sorted(iteritems(fracs), key=lambda x: x[1])[0][0]
        candidates = deps.available[chosen_type]

        # --- greedily pick non-overlapping ops
        chosen_ops = _greedy_nonoverlapping(candidates)

        # --- schedule ops
        assert chosen_ops
        rval.append((chosen_type, chosen_ops))

        # --- update predecessors and successors of unsheduled ops
        deps.remove_op_type(chosen_type, chosen_ops)

    assert len(operators) == sum(len(p[1]) for p in rval)
    return rval
