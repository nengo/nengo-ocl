import pyopencl as cl

class Plan(object):

    def __init__(self, queue, kern, gsize, lsize, *kwargs):
        self.queue = queue
        self.kern = kern
        self.gsize = gsize
        self.lsize = lsize
        self.kwargs = kwargs

    def __call__(self):
        cl.enqueue_nd_range_kernel(
            self.queue, self.kern, self.gsize, self.lsize)
        self.queue.finish()


class Prog(object):
    def __init__(self, plans):
        self.plans = plans
        self.queues = [p.queue for p in self.plans]
        self.kerns = [p.kern for p in self.plans]
        self.gsize = [p.gsize for p in self.plans]
        self.lsize = [p.lsize for p in self.plans]

    def __call__(self):
        map(cl.enqueue_nd_range_kernel,
            self.queues, self.kerns, self.gsize, self.lsize)
        self.queues[-1].finish()
