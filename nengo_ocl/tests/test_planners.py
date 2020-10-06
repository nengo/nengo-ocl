# pylint: disable=missing-module-docstring,missing-function-docstring

import nengo

import nengo_ocl
from nengo_ocl.planners import greedy_planner


def count_op_group(sim, op_group):
    return len([grp for grp, _ in sim.op_groups if grp is op_group])


def check_op_groups(sim):
    # [print(op_group) for op_group, _ in sim.op_groups]
    # sim.print_plans()

    # all resets planned together
    assert count_op_group(sim, nengo.builder.operator.Reset) == 1


def feedforward_network(extra_node=False):
    n = 5

    with nengo.Network(seed=0) as model:
        # u = nengo.Node([1] * n)
        nodes = [nengo.Node(nengo.processes.WhiteNoise()) for _ in range(n)]
        ensembles = [nengo.Ensemble(1, 1) for _ in range(n)]
        probes = [nengo.Probe(e, synapse=0.01) for e in ensembles]

        for i in range(n):
            nengo.Connection(nodes[i], ensembles[i], synapse=None)

        if extra_node:
            v = nengo.Node(lambda t, x: x ** 2, size_in=1)
            nengo.Connection(nodes[0], v, synapse=None)
            nengo.Connection(v, ensembles[0], synapse=None)

    return model, probes


def test_greedy_planner_feedforward():
    model, _ = feedforward_network()

    with nengo_ocl.Simulator(model, planner=greedy_planner) as sim:
        check_op_groups(sim)
        assert count_op_group(sim, nengo.builder.neurons.SimNeurons) == 1
        assert len(sim.op_groups) == 10
