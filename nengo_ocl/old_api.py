import numpy as np
import base
import simulator


class Ensemble(object):
    def __init__(self):
        pass


class Probe(object):
    def __init__(self, probe, net):
        self.probe = probe
        self.net = net

    def get_data(self):
        sim = self.net.sim
        return sim.signal_probe_output(self.probe)


class Network(object):

    def __init__(self, name, seed=None, dt=0.001,
                 Simulator=simulator.Simulator):
        self.model = base.Model(dt)
        self.ensembles = {}
        self.inputs = {}

        self.steps = self.model.signal()
        self.simtime = self.model.signal()
        self.one = self.model.signal(value=1.0)

        # -- hold 1.0 in self.one
        self.model.filter(1.0, self.one, self.one)

        # -- steps counts by 1.0
        self.model.filter(1.0, self.steps, self.steps)
        self.model.filter(1.0, self.one, self.steps)

        # simtime <- dt * steps
        self.model.filter(dt, self.steps, self.simtime)

        self.Simulator = Simulator

    @property
    def dt(self):
        return self.model.dt

    def make_input(self, name, value):
        if callable(value):
            rval = self.model.signal()
            self.model.custom_transform(value, self.simtime, rval)
            self.inputs[name] = rval
        else:
            rval = self.model.signal(value=value)
            self.inputs[name] = rval
        return rval

    def make_array(self, name, num_neurons, num_dimensions,
                   radius=1.0,
                   neuron_type='lif',
                   encoders=None):
        raise NotImplementedError()

    def make(self, name, num_neurons, num_dimensions):
        rval = Ensemble()
        rval.pop = self.model.population(num_neurons)
        rval.sig = self.model.signal(n=num_dimensions)
        rval.encoder = self.model.encoder(rval.sig, rval.pop)
        rval.origin = {
            'X':self.model.signal(n=num_dimensions),
        }
        self.model.decoder(rval.pop, rval.origin['X'])
        self.ensembles[name] = rval
        return rval

    def connect(self, name1, name2,
                func=None,
                transform=1.0,
                index_post=None):
        n = 1 # XXX how to know this
        decoded = self.model.signal(n=n)
        if name1 in self.ensembles:
            src = self.ensembles[name1]
            dst = self.ensembles[name2]
            decoder = self.model.decoder(src.pop, decoded)
            decoder.desired_function = func
            self.model.transform(np.asarray(transform), decoded, dst.sig)
        elif name1 in self.inputs:
            src = self.inputs[name1]
            dst = self.ensembles[name2]
            if func is not None:
                raise NotImplementedError()
            self.model.transform(np.asarray(transform), decoded, dst.sig)
        else:
            raise NotImplementedError()
        
    def make_probe(self, name, dt_sample, pstc):
        src = self.ensembles[name].sig
        sig = self.model.signal(src.n)
        # XXX use pstc to calculate decay
        self.model.filter(.9, sig, sig)
        self.model.transform(.1, src, sig)
        return Probe(self.model.signal_probe(sig, dt_sample),
                     self)

    def run(self, simtime):
        try:
            self.sim
        except:
            sim = self.Simulator(self.model)
            self.sim = sim
        n_steps = int(simtime // self.dt)
        self.sim.run_steps(n_steps)



