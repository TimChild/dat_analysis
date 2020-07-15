from src.Scripts.StandardImports import *

from src.DatObject.Attributes.Transition import i_sense
from src.DatObject.Attributes import AWG
from src.DatObject.Attributes import Logs

class TransitionModel(object):
    def __init__(self):
        self.mid = 0
        self.theta = 0.5
        self.amp = 0.5
        self.lin = 0.01
        self.const = 8
        self.func = i_sense

    def eval(self, x):
        return self.func(x, self.mid, self.theta, self.amp, self.lin, self.const)




class SquareTransitionModel(TransitionModel):
    def __init__(self, square_wave):
        super().__init__()
        self.cross_cap = 0
        self.heat_factor = 0
        self.square_wave: SquareWave = square_wave

        self.start = square_wave.start
        self.fin = square_wave.fin
        self.numpts = square_wave.numpts
        self.x = square_wave.x

    def eval(self, x):
        x = np.asarray(x)
        if x.shape == self.x.shape:
            pass  # Just use full waves because it will be faster
        else:
            pass
        heating_v = self.square_wave.eval(x)

        func = self.func
        x = x-self.cross_cap*heating_v
        thetas = self.theta+self.heat_factor*(heating_v**2)  # assuming quadratic heating
        z = func(x, self.mid, thetas, self.amp, self.lin, self.const)
        return z


def i_sense_heated():
    pass  # TODO: Make function which includes entropy and everything to fit data to


class SquareWave(AWG.AWG):
    def __init__(self, measure_freq, start, fin, sweeprate):
        self.measure_freq = measure_freq
        self.start = start
        self.fin = fin
        self.sweeprate = sweeprate

        self.v0 = 0
        self.vp = 100  # voltage positive
        self.vm = -100  # voltage negative

        self.step_dur = 0.1  # Won't be exactly this (see props)

        self.num_steps = None
        self.num_steps = self.get_numsteps()
        self.x = np.linspace(self.start, self.fin, self.numpts)


    @property
    def step_dur(self):
        return self._step_dur

    @step_dur.setter
    def step_dur(self, value):
        self._step_dur = round(value*self.measure_freq)/self.measure_freq

    @property
    def AWs(self):
        step_samples = round(self.step_dur * self.measure_freq)
        assert np.isclose(round(step_samples), step_samples, atol=0.00001)  # Should be an int
        return [np.array([[self.v0, self.vp, self.v0, self.vm],
                          [int(step_samples)] * 4])]

    @property
    def info(self):
        wave_len = self.step_dur*self.measure_freq*4
        assert np.isclose(round(wave_len), wave_len, atol=0.00001)  # Should be an int
        return Logs.AWGtuple(outputs={0:[0]},  # wave 0 output 0
                                 wave_len=int(wave_len),
                                 num_adcs=1,
                                 samplingFreq=self.measure_freq,
                                 measureFreq=self.measure_freq,
                                 num_cycles=1,
                                 num_steps=self.num_steps)

    @property
    def numpts(self):
        info = self.info
        return info.wave_len * info.num_cycles * info.num_steps

    def eval(self, x):
        x_arr = np.linspace(self.start, self.fin, self.numpts)
        idx = np.array(CU.get_data_index(x_arr, x))
        wave = self.get_full_wave(0)
        return wave[idx]

    def get_numsteps(self):
        # Similar to process that happens in IGOR
        target_numpts = numpts_from_sweeprate(self.sweeprate, self.measure_freq, self.start, self.fin)
        return round(target_numpts / self.info.wave_len * self.info.num_cycles)


def numpts_from_sweeprate(sweeprate, measure_freq, start, fin):
    return round(abs(fin-start)*measure_freq/sweeprate)


if __name__ == '__main__':
    import src.Main_Config as cfg
    cfg.PF_num_points_per_row = 2000
    fig, ax = plt.subplots(1)
    measure_freq = 2016.1
    sweeprate = 0.5
    mid = -1007.91086
    start = -1034.0
    fin = -984.0

    sqw = SquareWave(measure_freq, start, fin, sweeprate)
    sqw.step_dur = 0.25
    sqw.num_steps = sqw.get_numsteps()
    t = SquareTransitionModel(sqw)

    dat = get_dat(500)
    z = dat.Data.i_sense[0]
    x = dat.Data.x_array
    nz, f = CU.decimate(z, dat.Logs.Fastdac.measure_freq, 20, return_freq=True)
    nx = np.linspace(x[0], x[-1], nz.shape[-1])


    ax.cla()
    PF.display_1d(nx, nz, ax, label='Data', marker='', linewidth=1)
    t.amp = 0.3
    t.lin = 0.0149
    t.theta = 0.5
    t.mid = mid
    t.const = 7.8538
    # PF.display_1d(t.x, t.eval(t.x), ax, 'Plunger /mV', 'Current /nA', auto_bin=False, label='base', marker='',
    #               linewidth=1, color='k')

    ax.lines[-1].remove()
    t.cross_cap = 0.025
    t.heat_factor = 0.0001
    PF.display_1d(t.x, t.eval(t.x), ax, label=f'{t.cross_cap:.2g}, {t.heat_factor:.2g}', marker='')
    ax.legend(title='cross_cap, heat_factor')
