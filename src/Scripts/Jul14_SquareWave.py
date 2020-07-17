from src.Scripts.StandardImports import *

from src.DatObject.Attributes import Transition as T, AWG, Logs, DatAttribute as DA, Entropy as E
import src.Main_Config as cfg


class TransitionModel(object):
    def __init__(self, mid=0, amp=0.5, theta=0.5, lin=0.01, const=8):
        self.mid = mid
        self.amp = amp
        self.theta = theta
        self.lin = lin
        self.const = const

    def eval(self, x):
        return T.i_sense(x, self.mid, self.theta, self.amp, self.lin, self.const)


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


def get_zs(data, awg):
    """
    Breaks up data into chunks which make more sense for square wave heating datasets.
    Args:
        data (np.ndarray): 1D or 2D data (full data)
        awg (AWG.AWG): AWG part of dat which has all square wave info in.

    Returns:
        List[np.ndarray]: z0_1, zp, z0_2, zm -- where each has dimensions ([ylen], num_steps, sp_len).

        Only has ylen if 2D data passed in.
        Has to be a list returned and not a ndarray because sp_len may vary per step!
    """
    wave_num = 0
    masks = awg.get_full_wave_masks(wave_num)
    AW = awg.AWs[wave_num]  # [[setpoints],[lengths]]
    num_steps = awg.info.num_steps
    zs = []
    for mask, sp_len in zip(masks, AW[1].astype(int)):
        # Data per row
        z = np.atleast_2d(data)

        zm = z * mask  # Mask data
        zm = zm[~np.isnan(zm)]  # remove blanks
        zm = zm.reshape(z.shape[0], num_steps, sp_len)
        if zm.shape[0] == 1:  # If was 1D array initially
            zm = zm.squeeze(axis=0)  # Remove first dimension
        # Convert to array with shape = ([ylen], num_steps, samples_per_step)
        # ylen only if it was 2D data to start with
        zs.append(zm)
    return zs


def bin_zs(zs, s=None, f=None):
    """ Averages last index of AWG data passed in from index s to f.

    Args:
        zs (list): List of datas chunked nicely for AWG data
        s (Union[int, None]): Start index to average in each setpoint chunk
        f (Union[int, None]): Final index to average to in each setpoint chunk (can be negative)

    Returns:
        np.ndarray: Array of zs with averaged last dimension. Can be an array here because will always have 1 value per
        averaged chunk of data (i.e. can't have different last dimension any more)
    """
    # TODO: Can be improved by allowing tuple of s, f to be passed in for different averaging for each setpoint

    zs = [np.atleast_3d(z) for z in zs]  # So will work for 1D or 2D zs data (has extra dimension because of chunks)
    nz = [np.mean(z[:, :, s:f], axis=2) for z in zs]  # Average the last dimension from s:f
    if nz[0].shape[0] == 1:
        nz = [np.squeeze(z, axis=0) for z in nz]
    return np.array(nz)


if __name__ == '__main__':
    run = 'fitting_data'
    if run == 'modelling':
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

    elif run == 'fitting_data':
        cfg.PF_num_points_per_row = 2000  # Otherwise binning data smears out square steps too much
        fig, axs = PF.make_axes(2)
        ax = axs[0]
        ax.cla()

        # Get data
        dats = get_dats(range(500, 515))  # Various square wave tests (varying freq, cycles, amplitude)
        
        dat = dats[0]
        zs = get_zs(dat.Data.i_sense, dat.AWG)

        # Bin data from s to f
        nx = dat.AWG.true_x_array
        nzs = bin_zs(zs, s=None, f=None)

        nnzs = []
        nxs = []
        for z in nzs:
            fits = T.transition_fits(nx, z, func=T.i_sense)
            fis = [DA.FitInfo.from_fit(f) for f in fits]

            nnz, nx = CU.center_data(nx, z, [fi.best_values.mid for fi in fis], return_x=True)
            nnz = np.mean(nnz, axis=0)
            nnzs.append(nnz)
            nxs.append(nx)

        ax.cla()
        for x, z, label in zip(nxs, nnzs, ['v0_1', 'vp', 'v0_2', 'vm']):
            ax.plot(nx, z, label=label, marker='+')

        PF.ax_setup(ax, 'Separately Centered then averaged data', 'Plunger /mV', 'Current /nA', True)


        # Harmonic
        ax = axs[1]
        ax.cla()
        x = nxs[0]
        from scipy.interpolate import interp1d
        data = []
        for nx, z in zip(nxs, nnzs):
            interper = interp1d(nx, z, bounds_error=False)
            data.append(interper(x))
        data = np.array(data)  # Data which shares the same x axis

        harm2 = -1*(np.mean(data[(1, 3), :], axis=0) - np.mean(data[(0, 2), :], axis=0))
        ent_fit = E.entropy_fits(x, harm2)[0]
        efi = DA.FitInfo.from_fit(ent_fit)
        # efi.edit_params('const', 0, True)
        # efi.recalculate_fit(x, harm2)

        ax.plot(x, harm2, label=f'data')
        ax.plot(x, efi.eval_init(x), label='init')
        ax.plot(x, efi.eval_fit(x), label='fit')
        PF.ax_setup(ax, f'Entropy: dS = {efi.best_values.dS:.3f}', 'Plunger /mV', 'Current /nA')
        ax.legend(title='Entropy /kB')




        fig.tight_layout()
        # # Plot data averaged over rows (not Centered here, but gives an idea...
        # ax.cla()
        # for z, label in zip(nzs, ['v0_1', 'vp', 'v0_2', 'vm']):
        #     avg = np.average(z, axis=0)
        #     ax.plot(nx, avg, label=label)
        # ax.legend()
        #
        #