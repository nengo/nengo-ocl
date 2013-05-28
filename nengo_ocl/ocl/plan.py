import pyopencl as cl

class Plan(object):

    def __init__(self, queue, kern, gsize, lsize, **kwargs):
        self.queue = queue
        self.kern = kern
        self.gsize = gsize
        self.lsize = lsize
        self.kwargs = kwargs
        self.atime = 0.0
        self.btime = 0.0
        self.ctime = 0.0
        self.n_calls = 0

    def __call__(self, profiling=False):
        ev = self.enqueue()
        self.queue.finish()
        if profiling:
            self.atime += 1e-9 * (ev.profile.submit - ev.profile.queued)
            self.btime += 1e-9 * (ev.profile.start - ev.profile.submit)
            self.ctime += 1e-9 * (ev.profile.end - ev.profile.start)
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
        self.gsize = [p.gsize for p in self.plans]
        self.lsize = [p.lsize for p in self.plans]

    def __call__(self, profiling=False):
        for p in self.plans:
            p(profiling=profiling)

    def call_n_times(self, n):
        self.enqueue_n_times(n)
        self.queues[-1].finish()

    def enqueue_n_times(self, n):
        clrk = cl.enqueue_nd_range_kernel
        qs, ks, gs, ls = self.queues, self.kerns, self.gsize, self.lsize
        for ii in range(n):
            map(clrk, qs, ks, gs, ls)
