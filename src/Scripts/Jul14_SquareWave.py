from src.Scripts.StandardImports import *

from src.DatObject.Attributes.Transition import i_sense
from src.DatObject.Attributes import AWG
from src.DatObject.Attributes import Logs
import src.Main_Config as cfg


class TransitionModel(object):
    def __init__(self, mid=0, amp=0.5, theta=0.5, lin=0.01, const=8):
        self.mid = mid
        self.amp = amp
        self.theta = theta
        self.lin = lin
        self.const = const

    def eval(self, x):
        return i_sense(x, self.mid, self.theta, self.amp, self.lin, self.const)


class SquareTransitionModel(TransitionModel):
    def __init__(self, square_wave, mid=0, amp=0.5, theta=0.5, lin=0.01, const=8, cross_cap=0, heat_factor=0.0001, dS=0):
        super().__init__(mid, amp, theta, lin, const)
        self.cross_cap = cross_cap
        self.heat_factor = heat_factor
        self.dS = dS
        self.square_wave: SquareWave = square_wave

        self.start = square_wave.start
        self.fin = square_wave.fin
        self.numpts = square_wave.numpts
        self.x = square_wave.x

    def eval(self, x, no_heat=False):
        x = np.asarray(x)
        if no_heat:
            return super().eval(x)
        else:
            if x.shape == self.x.shape:
                pass  # Just use full waves because it will be faster
            else:
                pass
            heating_v = self.square_wave.eval(x)
            z = i_sense_square_heated(x, self.mid, self.theta, self.amp, self.lin, self.const, heating_v,
                          self.cross_cap, self.heat_factor, self.dS)
            return z


def i_sense_square_heated(x, mid, theta, amp, lin, const, hv, cc, hf, dS):
    """ Full transition signal with square wave heating and entropy change

    Args:
        x (Union[float, np.ndarray]):
        mid (Union[float, np.ndarray]):
        theta (Union[float, np.ndarray]):
        amp (Union[float, np.ndarray]):
        lin (Union[float, np.ndarray]):
        const (Union[float, np.ndarray]):
        hv (Union[float, np.ndarray]): Heating Voltage
        cc (Union[float, np.ndarray]): Cross Capacitance of HV and Plunger gate (i.e. shift middle)
        hf (Union[float, np.ndarray]): Heat Factor (i.e. how much HV increases theta)
        dS (Union[float, np.ndarray]): Change in entropy between N -> N+1

    Returns:
        (Union[float, np.ndarray]): evaluated function at x value(s)
    """
    heat = hf*hv**2  # Heating proportional to hv^2
    T = theta + heat  # theta is base temp theta, so T is that plus any heating
    X = x - mid + dS*heat - hv*cc  # X is x position shifted by entropy when heated and cross capacitance of heating
    arg = X/(2*T)
    return -amp/2 * np.tanh(arg) + lin*(x-mid) + const  # Linear term is direct interaction with CS


class SquareWave(AWG.AWG):
    def __init__(self, measure_freq=1000, start=-10, fin=10, sweeprate=1, step_dur=0.1, vheat=100):
        self.measure_freq = measure_freq
        self.start = start
        self.fin = fin
        self.sweeprate = sweeprate

        self.v0 = 0
        self.vp = vheat  # voltage positive
        self.vm = -vheat  # voltage negative

        self.step_dur = step_dur  # Won't be exactly this (see props)

        self.num_steps = None  # Because self.info is called in get_numsteps and includes num_steps
        self.num_steps = self.get_numsteps()
        self.x = np.linspace(self.start, self.fin, self.numpts)

    @property
    def step_dur(self):
        return self._step_dur

    @step_dur.setter
    def step_dur(self, value):  # Always an integer number of samples
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
        if np.all(np.isclose(x, x_arr)):  # If full wave, don't bother searching for points
            idx = np.arange(self.numpts)
        else:
            idx = np.array(CU.get_data_index(x_arr, x))
        wave = self.get_full_wave(0)
        return wave[idx]

    def get_numsteps(self):
        # Similar to process that happens in IGOR
        target_numpts = numpts_from_sweeprate(self.sweeprate, self.measure_freq, self.start, self.fin)
        return round(target_numpts / self.info.wave_len * self.info.num_cycles)


def numpts_from_sweeprate(sweeprate, measure_freq, start, fin):
    return round(abs(fin-start)*measure_freq/sweeprate)


def line_exists_in_ax(ax: plt.Axes, label: str) -> bool:
    exists = False
    for line in ax.lines:
        l = line.get_label()
        if l.lower() == label.lower():
            exists = True
            break
    return exists


def remove_line(ax:plt.Axes, label:str) -> bool:
    if line_exists_in_ax(ax, label):
        i = None
        for i, line in enumerate(ax.lines):
            if line.get_label().lower() == label.lower():
                break
        ax.lines[i].remove()
    else:
        logger.info(f'"{label}" not found in ax.lines')


if __name__ == '__main__':
    cfg.PF_num_points_per_row = 2000  # Otherwise binning data smears out square steps too much
    fig, ax = plt.subplots(1)
    ax.cla()

    # Get real data (get up here so I can use value to make models)
    dat = get_dat(500)
    fit_values = dat.Transition.all_fits[0].best_values  # Shorten accessing these values

    # Params for sqw model
    measure_freq = dat.Logs.Fastdac.measure_freq  # NOT sample freq, actual measure freq
    sweeprate = CU.get_sweeprate(measure_freq, dat.Data.x_array)  # mV/s sweeping
    start = dat.Data.x_array[0]  # Start of plunger sweep
    fin = dat.Data.x_array[-1]  # End of plunger sweep
    step_dur = 0.25  # Duration of each step of the square wave
    vheat = 800  # Heating voltage applied (divide/10 to convert to nA). Voltage useful for cross capacitance

    # Initial params for transition model (gets info from sqw by default)
    mid = fit_values.mid  # Middle of transition
    amp = fit_values.amp
    theta = fit_values.theta
    lin = fit_values.lin
    const = fit_values.const
    cross_cap = 0
    heat_factor = 0.0001
    dS = np.log(2)

    # Make the square wave
    sqw = SquareWave(measure_freq, start, fin, sweeprate, step_dur, vheat)

    # Make Transition model
    t = SquareTransitionModel(sqw, mid, amp, theta, lin, const, cross_cap, heat_factor, dS)

    ax.cla()

    # Data from Dat
    z = dat.Data.i_sense[0]
    x = dat.Data.x_array
    nz, f = CU.decimate(z, dat.Logs.Fastdac.measure_freq, 20, return_freq=True)
    nx = np.linspace(x[0], x[-1], nz.shape[-1])

    # Plot Data
    remove_line(ax, 'Data')  # Remove if already exists
    PF.display_1d(nx, nz, ax, label='Data', marker='', linewidth=1)

    # Plot cold transition only (no heating)
    remove_line(ax, 'Base')
    PF.display_1d(t.x, t.eval(x, no_heat=True), ax, 'Plunger /mV', 'Current /nA', auto_bin=False, label='Base', marker='',
                  linewidth=1, color='k')

    # Plot with heating
    line: Union[None, plt.Line2D] = None

    t.cross_cap = 0.002
    t.heat_factor = 0.0000018
    t.theta = 0.5
    if line:
        line.remove()
    PF.display_1d(t.x, t.eval(t.x), ax, label=f'{t.cross_cap:.2g}, {t.heat_factor:.2g}', marker='')
    line = ax.lines[-1]
    ax.legend(title='cross_cap, heat_factor')
