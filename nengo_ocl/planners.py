from collections import defaultdict

from nengo.builder.operator import Operator
from nengo.utils.simulator import operator_dependency_graph


def greedy_planner(operators):  # noqa: C901
    edges = operator_dependency_graph(operators)

    is_op = lambda op: isinstance(op, Operator)
    for op, dests in edges.items():
        assert is_op(op) and all(is_op(op2) for op2 in dests)

    # map unscheduled ops to their direct predecessors and successors
    predecessors_of = {}
    successors_of = {}
    for op in operators:
        predecessors_of[op] = set()
        successors_of[op] = set()
    for op, dests in edges.items():
        for op2 in dests:
            predecessors_of[op2].add(op)
        successors_of[op].update(dests)

    # available ops are ready to be scheduled (all predecessors scheduled)
    available = defaultdict(set)
    for op in (op for op, dep in predecessors_of.items() if not dep):
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
