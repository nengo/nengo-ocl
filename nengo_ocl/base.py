from ocl import array
from ocl.plan import Prog
from ocl.lif import plan_lif
from ocl.gemv_batched import plan_misc_gemv


class LIFMultiEnsemble(object):
    def __init__(self, n_populations, n_neurons_per, n_signals, signal_size,
            n_dec_per_signal=1,
            n_enc_per_population=1,
            lif_tau_rc=0.02,
            lif_tau_ref=0.002,
            lif_V_threshold=1.0,
            lif_output_pstc=0.01,
            lif_upsample=2,
            noise=None,
            queue=None):
        self.__dict__.update(locals())
        del self.self

        self.lif_v = array.zeros(queue,
                shape=(n_populations, n_neurons_per),
                dtype='float32')

        self.lif_rt = array.zeros(queue,
                shape=(n_populations, n_neurons_per),
                dtype='float32')

        self.lif_ic = array.zeros(queue,
                shape=(n_populations, n_neurons_per),
                dtype='float32')

        self.lif_bias = array.zeros(queue,
                shape=(n_populations, n_neurons_per),
                dtype='float32')

        self.lif_output = array.zeros(queue,
                shape=(n_populations, n_neurons_per),
                dtype='float32')

        self.lif_output_filter = array.zeros(queue,
                shape=(n_populations, n_neurons_per),
                dtype='float32')

        self.signals = array.zeros(queue,
                shape=(n_signals, signal_size),
                dtype='float32')

        self.encoders = array.zeros(queue,
                shape=(n_populations, n_neurons_per, n_enc_per_population,
                    signal_size),
                dtype='float32')

        self.encoders_signal_idx = array.zeros(queue,
                shape=(n_populations, n_enc_per_population),
                dtype='int32')

        self.decoders = array.zeros(queue,
                shape=(n_signals, signal_size, n_dec_per_signal,
                    n_neurons_per),
                dtype='float32')

        self.decoders_population_idx = array.zeros(queue,
                shape=(n_signals, n_dec_per_signal),
                dtype='int32')

    def _randomize_encoders(self, rng):
        encoders = rng.randn(*self.encoders.shape).astype('float32')
        self.encoders.set(encoders)

    def _randomize_decoders(self, rng):
        decoders = rng.randn(*self.decoders.shape).astype('float32')
        self.decoders.set(decoders)

    def neuron_plan(self, dt):
        # XXX add support for built-in filtering
        rval = plan_lif(self.queue,
                V=self.lif_v,
                RT=self.lif_rt,
                J=self.lif_ic,
                OV=self.lif_v,
                ORT=self.lif_rt,
                OS=self.lif_output,
                OSfilt=self.lif_output_filter,
                dt=dt,
                tau_rc=self.lif_tau_rc,
                tau_ref=self.lif_tau_ref,
                V_threshold=self.lif_V_threshold,
                upsample=self.lif_upsample,
                pstc=self.lif_output_pstc,
                )
        return rval

    def decoder_plan(self, signals_beta):
        if self.n_dec_per_signal != 1:
            raise NotImplementedError()
        rval = plan_misc_gemv(self.queue,
                alpha=1.0,
                A=self.decoders[:, :, 0],
                X=self.lif_output_filter,
                Xi=self.decoders_population_idx[:, 0],
                beta=signals_beta,
                Y=self.signals,)
        return rval

    def encoder_plan(self, lif_ic_beta):
        if self.n_enc_per_population != 1:
            raise NotImplementedError()
        rval = plan_misc_gemv(self.queue,
                alpha=1.0,
                A=self.encoders[:, :, 0],
                X=self.signals, # use filtered here
                Xi=self.encoders_signal_idx[:, 0],
                beta=lif_ic_beta,
                Y_in=self.lif_bias,
                Y=self.lif_ic)
        return rval

    def prog(self, dt):
        return Prog([
            self.encoder_plan(lif_ic_beta=1.0),
            self.neuron_plan(dt=dt),
            self.decoder_plan(signals_beta=0.0),
        ])


