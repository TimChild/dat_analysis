from src.PlottingFunctions import remove_line
from src.Scripts.StandardImports import *

from src.DatObject.Attributes import Transition as T, AWG, Logs, DatAttribute as DA, Entropy as E
import src.Main_Config as cfg
from scipy.interpolate import interp1d

import napari


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
    def __init__(self, square_wave, mid=0, amp=0.5, theta=0.5, lin=0.01, const=8, cross_cap=0, heat_factor=0.0001,
                 dS=0):
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
    heat = hf * hv ** 2  # Heating proportional to hv^2
    T = theta + heat  # theta is base temp theta, so T is that plus any heating
    X = x - mid + dS * heat - hv * cc  # X is x position shifted by entropy when heated and cross capacitance of heating
    arg = X / (2 * T)
    return -amp / 2 * np.tanh(arg) + lin * (x - mid) + const  # Linear term is direct interaction with CS


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
        self._step_dur = round(value * self.measure_freq) / self.measure_freq

    @property
    def AWs(self):
        step_samples = round(self.step_dur * self.measure_freq)
        assert np.isclose(round(step_samples), step_samples, atol=0.00001)  # Should be an int
        return [np.array([[self.v0, self.vp, self.v0, self.vm],
                          [int(step_samples)] * 4])]

    @property
    def info(self):
        wave_len = self.step_dur * self.measure_freq * 4
        assert np.isclose(round(wave_len), wave_len, atol=0.00001)  # Should be an int
        return Logs.AWGtuple(outputs={0: [0]},  # wave 0 output 0
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
    return round(abs(fin - start) * measure_freq / sweeprate)


def chunk_awg_data(data, awg):
    """
    Breaks up data into chunks which make more sense for square wave heating datasets.
    Args:
        data (np.ndarray): 1D or 2D data (full data to match original x_array).
            Note: will return with y dim regardless of 1D or 2D
        awg (AWG.AWG): AWG part of dat which has all square wave info in.

    Returns:
        List[np.ndarray]: Data broken up into chunks (setpoints, (ylen, num_steps, num_cycles, sp_len)).

            NOTE: Has to be a list returned and not a ndarray because sp_len may vary per steps

            NOTE: This is the only step where setpoints should come first, once sp_len binned it should be ylen first
    """
    wave_num = 0
    masks = awg.get_full_wave_masks(wave_num)
    AW = awg.AWs[wave_num]  # [[setpoints],[lengths]]
    num_steps = awg.info.num_steps
    num_cycles = awg.info.num_cycles
    zs = []
    for mask, sp_len in zip(masks, AW[1].astype(int)):
        z = np.atleast_2d(data)  # Always assume 2D data
        zm = z * mask  # Mask data
        zm = zm[~np.isnan(zm)]  # remove blanks
        zm = zm.reshape(z.shape[0], num_steps, num_cycles, sp_len)
        zs.append(zm)
    return zs


def bin_awg_data(chunked_data, start_index=None, fin_index=None):
    """ Averages last index of AWG data passed in from index s to f.

    Args:
        chunked_data (List[np.ndarray]): List of datas chunked nicely for AWG data.
            dimensions (num_setpoints_per_cycle, (len(y), num_steps, num_cycles, sp_len))
        start_index (Union[int, None]): Start index to average in each setpoint chunk
        fin_index (Union[int, None]): Final index to average to in each setpoint chunk (can be negative)

    Returns:
        np.ndarray: Array of zs with averaged last dimension. ([ylen], setpoints, num_steps, num_cycles)
        Can be an array here because will always have 1 value per
        averaged chunk of data (i.e. can't have different last dimension any more)
    """

    assert np.all([arr.ndim == 4 for arr in chunked_data])  # Assumes [setpoints, (ylen, num_steps, num_cycles, sp_len)]
    nz = []
    for z in chunked_data:
        z = np.moveaxis(z, -1, 0)  # move sp_len to first axis to make mean nicer
        nz.append(np.mean(z[start_index:fin_index], axis=0))

    # nz = [np.mean(z[:, :, :, start_index:fin_index], axis=3) for z in chunked_data]  # Average the last dimension
    nz = np.moveaxis(np.array(nz), 0, 1)  # So that ylen is first now
    # (ylen, setpoins, num_steps, num_cycles)

    if nz.shape[0] == 1:  # Remove ylen dimension if len == 1
        nz = np.squeeze(nz, axis=0)
    return np.array(nz)


def average_cycles_awg_data(binned_data, start_cycle=None, fin_cycle=None):
    """
    Average values from cycles from start_cycle to fin_cycle
    Args:
        binned_data (np.ndarray): Binned AWG data with shape ([ylen], setpoints, num_steps, num_cycles)
        start_cycle (Union[int, None]): Cycle to start averaging from
        fin_cycle (Union[int, None]): Cycle to finish averaging on (can be negative to count backwards)

    Returns:
        np.ndarray: Averaged data with shape ([ylen], setpoints, num_steps)

    """
    # [y], setpoints, numsteps, cycles
    data = np.array(binned_data, ndmin=4)  # [y], setpoints, numsteps, cycles
    averaged = np.mean(np.moveaxis(data, -1, 0)[start_cycle:fin_cycle], axis=0)
    if averaged.shape[0] == 1:  # Return 1D or 2D depending on y_len
        averaged = np.squeeze(averaged, axis=0)
    return averaged


def average_2D_awg_data(x, data):
    """
    Averages data in y direction after centering using fits to v0 parts of square wave. Returns 1D data unchanged
    Args:
        x (np.ndarray): Original x_array for data
        data (np.ndarray): Data after binning and cycle averaging. Shape ([ylen], setpoints, num_steps)

    Returns:
        Tuple[np.ndarray, np.ndarray]: New x_array, averaged_data (shape (setpoints, num_steps))
    """
    if data.ndim == 3:
        z0s = data[:, (0, 2)]
        z0_avg_per_row = np.mean(z0s, axis=1)
        fits = T.transition_fits(x, z0_avg_per_row)
        fit_infos = [DA.FitInfo.from_fit(fit) for fit in fits]  # Has my functions associated
        centers = [fi.best_values.mid for fi in fit_infos]
        nzs = []
        nxs = []
        for z in np.moveaxis(data, 1, 0):  # For each of v0_0, vP, v0_1, vM
            nz, nx = CU.center_data(x, z, centers, return_x=True)
            nzs.append(nz)
            nxs.append(nx)
        assert (nxs[0] == nxs).all()  # Should all have the same x_array
        ndata = np.array(nzs)
        ndata = np.mean(ndata, axis=1)  # Average centered data
        nx = nxs[0]
    else:
        nx = x
        ndata = data
        logger.info(f'Data passed in was likely 1D already, same values returned')
    return nx, ndata


def align_setpoint_data(xs, data, nx=None):
    """
    In case want to realign data where each setpoint of heating has a different x_array (i.e. taking into account some
    additional shifts)
    Args:
        xs (np.ndarray):  x_array for each heating setpoint in data
        data (np.ndarray):  data with shape (setpoints, num_steps)
        nx (np.ndarray): New x_array to put data on, or will use first of xs by default

    Returns:
        Tuple[np.ndarray, np.ndarray]: new x_array, interpolated data with same shape as original
    """
    assert xs.ndim == 2  # different x_array for each heating setpoint
    assert xs.shape[0] == data.shape[0]
    oxs = xs  # Old xs
    if nx is None:
        nx = xs[0]  # New x
    ndata = []  # New data
    for ox, z in zip(oxs, data):
        interper = interp1d(xs, z, bounds_error=False)
        ndata.append(interper(nx))
    data = np.array(ndata)  # Data which shares the same x axis
    return nx, data


def entropy_signal_awg_data(data):
    """
    Calculates equivalent of second harmonic from data with v0_0, vP, v0_1, vM as first dimension
    Note: Data should be aligned for same x_array before doing this
    Args:
        data (np.ndarray): Data with first dimension corresponding to v0_0, vP, v0_1, vM. Can be any dimensions for rest

    Returns:
        np.ndarray: Entropy signal array with same shape as data minus the first axis

    """
    assert data.shape[0] == 4
    entropy_signal = -1 * (np.mean(data[(1, 3),], axis=0) - np.mean(data[(0, 2),], axis=0))
    return entropy_signal


class SquarePlotInfo(object):

    def __init__(self):
        self.dat: Union[DatHDF, None] = None
        self.axs: Union[np.ndarray[plt.Axes], None] = None

        self.plots = {'info': True,
                      'raw': False,
                      'binned': False,
                      'cycle_averaged': False,
                      'averaged': True,
                      'entropy': True}

        # Attrs that will be set from dat if possible
        self.raw_data = None  # basic 1D or 2D data (Needs to match original x_array for awg)
        self.orig_x_array = None  # original x_array (needs to be original for awg)
        self.awg = None  # AWG class from dat for chunking data AND for plot_info
        self.datnum = None  # For plot_info plot title
        self.x_label = None  # For plots

        # Additional params that can be changed
        self.bin_start = None  # Index to start binning
        self.bin_fin = None  # Index to stop binning
        self.cycle_start = None  # Index to start averaging cycles
        self.cycle_fin = None  # Index to stop averaging cycles
        self.decimate_freq = 50  # Target decimation freq when plotting raw. Set to None for no decimation
        self.plot_row_num = 0  # Which row of data to plot where this is an option

        # Data that will be calculated
        self.x = None  # x_array with length of num_steps (for cycled, averaged, entropy)
        self.chunked = None  # Data broken in to chunks based on AWG (just plot raw_data on orig_x_array)
        self.binned = None  # binned data only
        self.binned_x = None  # x_array
        self.cycled = None  # binned and then cycles averaged data (same x as average_data)
        self.average_data = None  # Binned, cycle_avg, then averaged in y
        self.entropy_signal = None  # Entropy signal data (same x as averaged data)

        self.entropy_fit = None  # FitInfo of entropy fit to self.entropy_signal

    @classmethod
    def from_plot_fn(cls, dat, axs, info, raw, binned, cycle_averaged, averaged, entropy):
        inst = cls()
        inst.update(dat, axs, info, raw, binned, cycle_averaged, averaged, entropy)
        return inst

    def update(self, dat=None, axs=None, info=None, raw=None, binned=None, cycle_averaged=None, averaged=None,
               entropy=None):
        if dat is not None:
            self.orig_x_array = dat.Data.x_array
            self.raw_data = dat.Data.i_sense
            self.awg = dat.AWG
            self.datnum = dat.datnum
            self.x_label = dat.Logs.x_label
        else:
            logger.info(f'Still need to set [orig_x_array, raw_data, awg, datnum, x_label]')

        if type(axs) == str:
            self.axs = None
        elif isinstance(axs, (tuple, np.ndarray, list)):
            self.axs = axs
        elif axs is not None:
            raise ValueError(f'{axs} is not an expected entry for updating SPI')



        # Which plots to plot
        for item, key in zip((info, raw, binned, cycle_averaged, averaged, entropy), self.plots.keys()):
            if item is not None:
                assert type(item) == bool
                self.plots[key] = item


def plot_square_wave(SPI=None, dat=None, axs=None, info=None, raw=None, binned=None, cycle_averaged=None, averaged=None,
                     entropy=None, calculate=True, show_plots=True):
    """
    Plots square wave info. SPI can be used if re running (and will update any other variables set). None's default to
    defaults in SPI class init
    Args:
        SPI (SquarePlotInfo): To alter plots with previous SPI (and have control over finer behaviour)
        dat (DatHDF): Dat to use to get data, x_array, awg, x_label... Can set in SPI manually and not use dat
        axs (Union[List[plt.Axes], str]): list of axes to put plots on. Must match length of number of plots set to True
            or use any string to reset SPI.axs to None
        info (bool): Show info about dat and awg in first plot
        raw (bool): show data (will default to using decimation if decimation not set to None in SPI
        binned (bool): Just averaging data at each heating setpoint
        cycle_averaged (bool): Averaging cycles at each DAC setpoint
        averaged (bool): Averaging over y_array
        entropy (bool): The entropy signal of averaged data

    Returns:
        SquarePlotInfo: Object which holds the values of things calculated (not only plotted)
    """
    if SPI is None:
        SPI = SquarePlotInfo.from_plot_fn(dat, axs, info, raw, binned, cycle_averaged, averaged, entropy)
    else:
        SPI.update(dat, axs, info, raw, binned, cycle_averaged, averaged, entropy)

    if calculate is True:
        SPI.x = np.linspace(SPI.orig_x_array[0], SPI.orig_x_array[-1], SPI.awg.info.num_steps)

        # Get chunked data (setpoints, ylen, numsteps, numcycles, splen)
        SPI.chunked = chunk_awg_data(SPI.raw_data, SPI.awg)

        # Bin data ([ylen], setpoints, numsteps, numcycles)
        SPI.binned = bin_awg_data(SPI.chunked, start_index=SPI.bin_start, fin_index=SPI.bin_fin)
        SPI.binned_x = np.linspace(SPI.orig_x_array[0], SPI.orig_x_array[-1],
                                   SPI.awg.info.num_steps * SPI.awg.info.num_cycles)

        # Averaged cycles ([ylen], setpoints, numsteps)
        SPI.cycled = average_cycles_awg_data(SPI.binned, start_cycle=SPI.cycle_start, fin_cycle=SPI.cycle_fin)

        # Center and average 2D data or skip for 1D
        SPI.x, SPI.averaged_data = average_2D_awg_data(SPI.x, SPI.cycled)

        # region Use this if want to start shifting each heater setpoint of data left or right
        # Align data
        # SPI.x, SPI.averaged_data = align_setpoint_data(xs, SPI.averaged_data, nx=None)
        # endregion

        # Entropy signal
        SPI.entropy_signal = entropy_signal_awg_data(SPI.averaged_data)

        # Fit Entropy
        ent_fit = E.entropy_fits(SPI.x, SPI.entropy_signal)[0]
        SPI.entropy_fit = DA.FitInfo.from_fit(ent_fit)

    if show_plots is True:
        num_plots_for_dat = sum(SPI.plots.values())
        if SPI.axs is None:
            fig, SPI.axs = PF.make_axes(num_plots_for_dat)
        elif SPI.axs.size != num_plots_for_dat:
            raise ValueError(f'{len(axs)} axs passed to plot {num_plots_for_dat} figures')

        for ax in SPI.axs:
            ax.cla()

        """Plot things"""
        ax_index = 0
        axs = SPI.axs
        for ax in axs:
            ax.set_xlabel(SPI.x_label)
            ax.set_ylabel('Current /nA')

        if SPI.plots['info'] is True:
            ax = axs[ax_index]
            ax_index += 1
            awg = SPI.awg
            freq = awg.info.wave_len * awg.info.measureFreq
            sqw_info = f'Square Wave:\n' \
                       f'Output amplitude: {int(awg.AWs[0][0][1]):d}mV\n' \
                       f'Heating current: {awg.AWs[0][0][1] / 10:.1f}nA\n' \
                       f'Frequency: {freq:.1f}Hz\n' \
                       f'Measure Freq: {awg.info.measureFreq:.1f}Hz\n' \
                       f'Num Steps: {awg.info.num_steps:d}\n' \
                       f'Num Cycles: {awg.info.num_cycles:d}\n' \
                       f'SP samples: {int(awg.AWs[0][1][0]):d}\n'

            scan_info = f'Scan info:\n' \
                        f'Sweeprate: {CU.get_sweeprate(awg.info.measureFreq, SPI.orig_x_array):.2f}mV/s\n'

            PF.ax_text(ax, sqw_info, loc=(0.05, 0.05), fontsize=7)
            PF.ax_text(ax, scan_info, loc=(0.5, 0.05), fontsize=7)
            PF.ax_setup(ax, f'Dat{SPI.datnum}')
            ax.axis('off')

        if SPI.plots['raw'] is True:
            ax = axs[ax_index]
            ax_index += 1
            z = SPI.raw_data[SPI.plot_row_num]
            x = SPI.orig_x_array
            if SPI.decimate_freq is not None:
                z, freq = CU.decimate(z, SPI.awg.info.measureFreq, SPI.decimate_freq, return_freq=True)
                x = np.linspace(x[0], x[-1], z.shape[-1])
                title = f'Row {SPI.plot_row_num} of CS data:\nDecimated to ~{freq:.1f}/s'
            else:
                title = f'Row {SPI.plot_row_num} or CS data'
            PF.display_1d(x, z, ax, linewidth=1, marker='', auto_bin=False)
            PF.ax_setup(ax, title)

        if SPI.plots['binned'] is True:
            ax = axs[ax_index]
            ax_index += 1
            x = SPI.binned_x
            for z, label in zip(SPI.binned[SPI.plot_row_num], ['v0_1', 'vp', 'v0_2', 'vm']):
                ax.plot(x, z.flatten(), label=label)
            PF.ax_setup(ax, f'Row {SPI.plot_row_num} of binned only', legend=True)

        if SPI.plots['cycle_averaged'] is True:
            ax = axs[ax_index]
            ax_index += 1
            for z, label in zip(SPI.cycled[SPI.plot_row_num], ['v0_1', 'vp', 'v0_2', 'vm']):
                ax.plot(SPI.x, z, label=label)
            PF.ax_setup(ax, f'Row {SPI.plot_row_num} of\nBinned and cycles averaged', legend=True)

        if SPI.plots['averaged'] is True:
            # Plot averaged data
            ax = axs[ax_index]
            ax_index += 1
            for z, label in zip(SPI.averaged_data, ['v0_1', 'vp', 'v0_2', 'vm']):
                ax.plot(SPI.x, z, label=label, marker='+')
            PF.ax_setup(ax, f'Centered with v0\nthen averaged data', legend=True)

        if SPI.plots['entropy'] is True:
            # Plot harmonic
            ax = axs[ax_index]
            ax_index += 1
            ax.plot(SPI.x, SPI.entropy_signal, label=f'data')
            ax.plot(SPI.x, SPI.entropy_fit.eval_fit(SPI.x), label='fit')
            PF.ax_setup(ax, f'Entropy: dS = {SPI.entropy_fit.best_values.dS:.2f}'
                            f'{PM}{SPI.entropy_fit.params["dS"].stderr:.2f}', legend=True)
        axs[0].get_figure().tight_layout()
    return SPI


if __name__ == '__main__':
    run = 'napari'
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
        PF.display_1d(t.x, t.eval(x, no_heat=True), ax, 'Plunger /mV', 'Current /nA', auto_bin=False, label='Base',
                      marker='',
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
        fig, all_axs = plt.subplots(2)  # Not used but lets code run

        # Get data
        dats = get_dats(range(500, 515))  # Various square wave tests (varying freq, cycles, amplitude)

        for dat in dats:
            dat.Other.save_code(inspect.getsource(plot_square_wave), 'calc and plot square wave')

        # dats = dats[:5]
        # dats = dats[5:10]
        dats = dats[10:]

        plot_info = True
        plot_raw = True
        plot_binned = True
        plot_cycle_averaged = True
        plot_averaged = True
        plot_entropy = True

        plots_per_dat = sum([plot_info, plot_binned, plot_cycle_averaged, plot_raw, plot_averaged, plot_entropy])

        if len(fig.axes) != plots_per_dat * len(dats):
            plt.close(fig)
            fig, all_axs = plt.subplots(plots_per_dat, len(dats), figsize=(len(dats) * 3.5, plots_per_dat * 3))

        SPIs = [plot_square_wave(dat=dat, info=plot_info, raw=plot_raw, binned=plot_binned,
                                         cycle_averaged=plot_cycle_averaged, averaged=plot_averaged,
                                         entropy=plot_entropy, calculate=True, show_plots=False) for dat in dats]

        for SPI, axs in zip(SPIs, all_axs.T):
            plot_square_wave(SPI, axs=axs, calculate=False)

        for axs in all_axs:
            for ax in axs:
                if ax.get_legend() is not None:
                    ax.legend(fontsize=7)
        for ax in all_axs[:-1, :].flatten():
            ax.set_xlabel('')
        for ax in all_axs[:, 1:].flatten():
            ax.set_ylabel('')

        fig.tight_layout(pad=0.5, h_pad=0.05, w_pad=0.05)

    elif run == 'napari':
        from src.Scripts.Napari_test import View
        dats = get_dats(range(500, 505))
        SPIs = [plot_square_wave(dat=dat, calculate=True, show_plots=False) for dat in dats]


        data = []
        xs = []
        for i, spi in enumerate(SPIs):
            d = spi.raw_data
            x = spi.orig_x_array
            d, f = CU.decimate(d, spi.awg.measure_freq, desired_freq=20, return_freq=True)
            nx = np.linspace(x[0], x[-1], d.shape[-1])
            d, nx = CU.remove_nans(d, nx)
            if i == 0:
                final_x = nx
            else:
                interper = interp1d(nx, d)
                d = interper(final_x)
            data.append(d)
            xs.append(nx)

        data = np.array(data)
        x = final_x

        v = View()
        v.add_data(data)
        v.add_profile()

