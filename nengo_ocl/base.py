import numpy as np


class DuplicateFilter(Exception):
    pass


class Signal(object):
    def __init__(self, n=1):
        self.n = n


class SignalProbe(object):
    def __init__(self, sig, dt):
        self.sig = sig
        self.dt = dt


class Constant(Signal):
    def __init__(self, value):
        Signal.__init__(self, len(value))
        self.value = value


class Population(object):
    def __init__(self, n, bias=None):
        self.n = n
        if bias is None:
            bias = np.zeros(n)
        else:
            bias = np.asarray(bias, dtype=np.float64)
            if bias.shape != (n,):
                raise ValueError('shape', (bias.shape, n))
        self.bias = bias

class Transform(object):
    def __init__(self, alpha, insig, outsig):
        self.alpha = alpha
        self.insig = insig
        self.outsig = outsig


class CustomTransform(object):
    def __init__(self, func, insig, outsig):
        self.func = func
        self.insig = insig
        self.outsig = outsig


class Filter(object):
    def __init__(self, alpha, oldsig, newsig):
        self.oldsig = oldsig
        self.newsig = newsig
        self.alpha = alpha


class Encoder(object):
    def __init__(self, sig, pop, weights=None):
        self.sig = sig
        self.pop = pop
        self.weights = weights
        assert isinstance(sig, Signal)
        assert isinstance(pop, Population)
        if weights is None:
            weights = np.random.randn(pop.n, sig.n)
        else:
            weights = np.asarray(weights)
            if weights.shape != (pop.n, sig.n):
                raise ValueError('weight shape', weights.shape)


class Decoder(object):
    def __init__(self, pop, sig, weights=None):
        self.pop = pop
        self.sig = sig
        self.weights = weights
        if weights is None:
            weights = np.random.randn(sig.n, pop.n)
        else:
            weights = np.asarray(weights)
            if weights.shape != (sig.n, pop.n):
                raise ValueError('weight shape', weights.shape)


class Model(object):
    def __init__(self, dt):
        self.dt = dt
        self.signals = []
        self.populations = []
        self.encoders = []
        self.decoders = []
        self.transforms = []
        self.filters = []
        self.custom_transforms = []
        self.signal_probes = []

    def signal(self, value=None):
        if value is None:
            rval = Signal()
        else:
            rval = Constant([value])
        self.signals.append(rval)
        return rval

    def signal_probe(self, sig, dt):
        rval = SignalProbe(sig, dt)
        self.signal_probes.append(rval)
        return rval

    def population(self, *args, **kwargs):
        rval = Population(*args, **kwargs)
        self.populations.append(rval)
        return rval

    def encoder(self, sig, pop, weights=None):
        rval = Encoder(sig, pop, weights=weights)
        self.encoders.append(rval)
        return rval

    def decoder(self, pop, sig, weights=None):
        rval = Decoder(pop, sig, weights=weights)
        self.decoders.append(rval)
        return rval

    def transform(self, alpha, insig, outsig):
        rval = Transform(alpha, insig, outsig)
        self.transforms.append(rval)
        return rval

    def filter(self, alpha, oldsig, newsig):
        rval = Filter(alpha, oldsig, newsig)
        self.filters.append(rval)
        return rval

    def custom_transform(self, func, insig, outsig):
        rval = CustomTransform(func, insig, outsig)
        self.custom_transforms.append(rval)
        return rval


# ----------------------------------------------------------------------
# nengo_theano interface utilities
# ----------------------------------------------------------------------


def net_get_decoders(net, obj, signalname):
    origin = net.get_object(obj).origin[signalname]
    rval = origin.decoders.get_value().astype('float32')
    r = origin.ensemble.radius
    rval = rval * r / net.dt
    rval.shape = rval.shape[:-1]
    return rval

def net_get_encoders(net, obj):
    ensemble = net.get_object(obj)
    encoders = ensemble.shared_encoders.get_value().astype('float32')
    # -- N.B. shared encoders already have "alpha" factored in
    return encoders

def net_get_bias(net, obj):
    ensemble = net.get_object(obj)
    return ensemble.bias.astype('float32')
