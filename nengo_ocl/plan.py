import pyopencl as cl

class Plan(object):

    def __init__(self, queue, kern, gsize, lsize, **kwargs):
        self.queue = queue
        self.kern = kern
        self.gsize = gsize
        self.lsize = lsize
        self.kwargs = kwargs
        self.atimes = []
        self.btimes = []
        self.ctimes = []
        self.n_calls = 0

    def __call__(self, profiling=False):
        ev = self.enqueue()
        self.queue.finish()
        if profiling:
            self.update_from_event(ev)

    def update_from_event(self, ev):
        self.atimes.append(1e-9 * (ev.profile.submit - ev.profile.queued))
        self.btimes.append(1e-9 * (ev.profile.start - ev.profile.submit))
        self.ctimes.append(1e-9 * (ev.profile.end - ev.profile.start))
        self.n_calls += 1

    def enqueue(self):
        return cl.enqueue_nd_range_kernel(
            self.queue, self.kern, self.gsize, self.lsize)

    def __str__(self):
        return '%s{%s, %s, %s, %s, %s}' % (
            self.__class__.__name__,
            self.queue,
            self.kern,
            self.gsize,
            self.lsize,
            self.kwargs)

class Prog(object):
    def __init__(self, plans):
        self.plans = plans
        self.queues = [p.queue for p in self.plans]
        self.kerns = [p.kern for p in self.plans]
        self.gsizes = [p.gsize for p in self.plans]
        self.lsizes = [p.lsize for p in self.plans]
        self.map_args = (cl.enqueue_nd_range_kernel,
                         self.queues, self.kerns, self.gsizes, self.lsizes)

    def __call__(self, profiling=False):
        if profiling:
            for p in self.plans:
                p(profiling=profiling)
        else:
            map(*self.map_args)
            self.queues[-1].flush()

    def enqueue(self):
        return map(*self.map_args)

    def call_n_times(self, n, profiling=False):
        all_evs = self.enqueue_n_times(n, profiling)
        self.queues[-1].finish()
        if profiling:
            for evs in all_evs:
                for ev, plan in zip(evs, self.plans):
                    plan.update_from_event(ev)

    def enqueue_n_times(self, n, profiling=False):
        map_args = self.map_args
        flush = self.queues[-1].flush
        all_evs = []
        for ii in range(n):
            evs = map(*map_args)
            flush()
            if profiling:
                all_evs.append(evs)
        return all_evs
