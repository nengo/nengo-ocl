import time
import pyopencl as cl
from collections import defaultdict
import networkx as nx
PROFILING_ENABLE = cl.command_queue_properties.PROFILING_ENABLE


class BasePlan(object):
    def __init__(self, name="", tag="",
                 flops_per_call=None,
                 bw_per_call=None
                ):
        self.name = name
        self.tag = tag
        self.atimes = []
        self.btimes = []
        self.ctimes = []
        self.n_calls = 0
        # -- floating-point ops per call
        self.flops_per_call = flops_per_call
        # -- bandwidth requirement per call
        self.bw_per_call = bw_per_call

    def __str__(self):
        return '%s{%s %s}' % (
            self.__class__.__name__,
            self.name,
            self.tag,
            )


class PythonPlan(BasePlan):
    def __init__(self, function, **kwargs):
        super(PythonPlan, self).__init__(**kwargs)
        self.function = function

    def __call__(self, profiling=False):
        if profiling:
            t0 = time.time()
        self.function()
        if profiling:
            t1 = time.time()
            self.atimes.append(0)
            self.btimes.append(0)
            self.ctimes.append(t1 - t0)
            self.n_calls += 1

    def update_from_event(self, ev):
        self.atimes.append(1e-9 * (ev.profile.submit - ev.profile.queued))
        self.btimes.append(1e-9 * (ev.profile.start - ev.profile.submit))
        self.ctimes.append(1e-9 * (ev.profile.end - ev.profile.start))
        self.n_calls += 1


class Plan(BasePlan):
    def __init__(self, queue, kern, gsize, lsize, **kwargs):
        super(Plan, self).__init__(**kwargs)
        assert 0 not in gsize, gsize
        if lsize is not None:
            assert len(lsize) > 0
            assert 0 not in lsize, lsize
        self.queue = queue
        self.kern = kern
        self.gsize = gsize
        self.lsize = lsize
        self._evs = []

    def __call__(self, profiling=False):
        ev = self.enqueue()
        ev.wait()

    def update_from_enqueued_events(self, profiling):
        if profiling:
            for ev in self._evs:
                self.atimes.append(
                    1e-9 * (ev.profile.submit - ev.profile.queued))
                self.btimes.append(
                    1e-9 * (ev.profile.start - ev.profile.submit))
                self.ctimes.append(
                    1e-9 * (ev.profile.end - ev.profile.start))
                self.n_calls += 1
        self._evs[:] = []

    def enqueue(self, wait_for=None):
        ev = cl.enqueue_nd_range_kernel(
            self.queue, self.kern, self.gsize, self.lsize,
            wait_for=wait_for)
        self._evs.append(ev)
        return ev

    def __str__(self):
        return '%s{%s, %s, %s, %s, tag=%s, name=%s}' % (
            self.__class__.__name__,
            self.queue,
            self.kern,
            self.gsize,
            self.lsize, 
            self.tag,
            self.name)


class Marker(Plan):
    def __init__(self, queue):
        dummy = cl.Program(queue.context, """
        __kernel void dummy() {}
        """).build().dummy
        Plan.__init__(self, queue, dummy, (1,), None)


class DAG(object):
    def __init__(self, context, marker, plandict, profiling, overlap=False):
        self.overlap = overlap
        self.context = context
        self.marker = marker
        self.plandict = plandict
        self.clients = defaultdict(list)
        self.profiling = profiling
        self.dg = nx.DiGraph()
        for plan, waits_on in plandict.items():
            if self.overlap:
                # XXX THIS IS PROBABLY A BUG WAITING TO HAPPEN!! #XXX
                plan.queue = cl.CommandQueue(context)
            self.dg.add_node(plan)
            for other in waits_on:
                self.clients[other].append(plan)
                self.dg.add_edge(other, plan)
        self.order = nx.topological_sort(self.dg)
        #for plan in self.order:
            #print 'order', plan

    def __call__(self):
        return self.call_n_times(1)

    def call_n_times(self, n):
        if all(hasattr(p, 'enqueue') for p in self.order):
            last_ev, all_evs = self.enqueue_n_times(n)
            last_ev.wait()
            if self.profiling:
                for p in self.order:
                    p.update_from_enqueued_events(self.profiling)
        else:
            for ii in range(n):
                for p in self.order:
                    p(self.profiling)

    def enqueue_n_times(self, n):
        if self.overlap:
            return self._enqueue_n_times_parallel(n)
        else:
            return self._enqueue_n_times_serial(n)

    def _enqueue_n_times_serial(self, n):
        all_evs = []
        for ii in range(n):
            evs_ii = []
            for plan in self.order:
                ev = plan.enqueue()
                evs_ii.append(ev)
            plan.queue.flush()
            all_evs.append(evs_ii)
        return ev, all_evs

    def _enqueue_n_times_parallel(self, n):
        boundary = self.marker.enqueue()
        all_evs = [boundary]
        stuff = []
        def remember(obj):
            stuff.append(obj)
        for ii in range(n):
            preconditions = defaultdict(list)
            evs_ii = []
            for plan in self.order:
                if self.overlap:
                    tmp = [boundary] + preconditions[plan]
                else:
                    tmp = None
                #print 'tmp', tmp
                ev = plan.enqueue(wait_for=tmp)
                plan.queue.flush()
                for client in self.clients[plan]:
                    preconditions[client].append(ev)
                evs_ii.append(ev)
                remember(tmp)
            if self.overlap:
                boundary = self.marker.enqueue(wait_for=evs_ii)
            all_evs.append(evs_ii)
        if not self.overlap:
            # -- the last `ev` we created
            boundary = ev
        return boundary, (all_evs, stuff)




