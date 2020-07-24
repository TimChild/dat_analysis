from src.PlottingFunctions import remove_line
from src.Scripts.StandardImports import *

from src.DatObject.Attributes import DatAttribute as DA, Entropy as E
import src.Main_Config as cfg
from scipy.interpolate import interp1d
from src.DatObject.Attributes.SquareEntropy import SquareAWGModel as SquareWave, SquareTransitionModel, chunk_data, \
    average_setpoints, average_cycles, average_2D, entropy_signal
import copy

from dataclasses import dataclass

@dataclass
class IntegratedInfo:
    bias: int
    t_hot: float = None
    t_cold: float = None
    dt: float = None
    amp: float = None
    dx: float = None
    sf: float = None


class SquarePlotInfo(object):

    def __init__(self):
        self.axs: Union[np.ndarray[plt.Axes], None] = None

        self.plots = {'info': True,
                      'raw': False,
                      'binned': False,
                      'cycle_averaged': False,
                      'averaged': True,
                      'entropy': True,
                      'integrated': True}

        # Attrs that will be set from dat if possible
        self.raw_data = None  # basic 1D or 2D data (Needs to match original x_array for awg)
        self.orig_x_array = None  # original x_array (needs to be original for awg)
        self.awg = None  # AWG class from dat for chunking data AND for plot_info
        self.datnum = None  # For plot_info plot title
        self.x_label = None  # For plots
        self.bias = None  # Square wave bias applied (assumed symmetrical since only averaging for now anyway)
        self.transition_amplitude = None  # For calculating scaling factor for integrated entropy

        # Additional params that can be changed
        self.bin_start = None  # Index to start binning
        self.bin_fin = None  # Index to stop binning
        self.cycle_start = None  # Index to start averaging cycles
        self.cycle_fin = None  # Index to stop averaging cycles
        self.decimate_freq = 50  # Target decimation freq when plotting raw. Set to None for no decimation
        self.plot_row_num = 0  # Which row of data to plot where this is an option

        # Integrated Params
        self.bias_theta_lookup = {0: 0.4942, 30: 0.7608, 50: 1.0301, 80: 1.4497}  # Bias/nA: Theta/mV

        # Data that will be calculated
        self.x = None  # x_array with length of num_steps (for cycled, averaged, entropy)
        self.chunked = None  # Data broken in to chunks based on AWG (just plot raw_data on orig_x_array)
        self.binned = None  # binned data only
        self.binned_x = None  # x_array
        self.cycled = None  # binned and then cycles averaged data (same x as average_data)
        self.averaged_data = None  # Binned, cycle_avg, then averaged in y
        self.entropy_signal = None  # Entropy signal data (same x as averaged data)
        self.integrated_entropy = None  # Integrated entropy signal (same x as averaged data)

        self.entropy_fit = None  # FitInfo of entropy fit to self.entropy_signal
        self.integrated_info: IntegratedInfo = None  # Things like dt, sf, amp etc

    def update(self, dat=None, axs=None, info=None, raw=None, binned=None, cycle_averaged=None, averaged=None,
               entropy=None, integrated=None):
        if dat is not None:
            self.orig_x_array = dat.Data.x_array
            self.raw_data = dat.Data.i_sense
            self.awg = dat.AWG
            self.datnum = dat.datnum
            self.x_label = dat.Logs.x_label
            self.bias = dat.AWG.AWs[0][0][1]/10  # Wave 0, setpoints row, setpoint 1 (vP)/10 to go to nA
            self.transition_amplitude = dat.Transition.avg_fit.best_values.amp  # May want to update this before using since this is without filtering etc
        else:
            logger.info(f'Still need to set [orig_x_array, raw_data, awg, datnum, x_label, bias, transition_amplitude]')

        if type(axs) == str:
            self.axs = None
        elif isinstance(axs, (tuple, np.ndarray, list)):
            self.axs = axs
        elif axs is not None:
            raise ValueError(f'{axs} is not an expected entry for updating SPI')

        # Which plots to plot
        for item, key in zip((info, raw, binned, cycle_averaged, averaged, entropy, integrated), self.plots.keys()):
            if item is not None:
                assert type(item) == bool
                self.plots[key] = item

    @classmethod
    def from_plot_fn(cls, dat, axs, info, raw, binned, cycle_averaged, averaged, entropy, integrated):
        inst = cls()
        inst.update(dat, axs, info, raw, binned, cycle_averaged, averaged, entropy, integrated)
        return inst


def plot_square_wave(SPI=None, dat=None, axs=None, info=None, raw=None, binned=None, cycle_averaged=None, averaged=None,
                     entropy=None, integrated=None, calculate=True, show_plots=True):
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
        SPI = SquarePlotInfo.from_plot_fn(dat, axs, info, raw, binned, cycle_averaged, averaged, entropy, integrated)
    else:
        SPI.update(dat, axs, info, raw, binned, cycle_averaged, averaged, entropy, integrated)

    if calculate is True:
        SPI.x = np.linspace(SPI.orig_x_array[0], SPI.orig_x_array[-1], SPI.awg.info.num_steps)

        # Get chunked data (setpoints, ylen, numsteps, numcycles, splen)
        SPI.chunked = chunk_data(SPI.raw_data, SPI.awg)

        # Bin data ([ylen], setpoints, numsteps, numcycles)
        SPI.binned = average_setpoints(SPI.chunked, start_index=SPI.bin_start, fin_index=SPI.bin_fin)
        SPI.binned_x = np.linspace(SPI.orig_x_array[0], SPI.orig_x_array[-1],
                                   SPI.awg.info.num_steps * SPI.awg.info.num_cycles)

        # Averaged cycles ([ylen], setpoints, numsteps)
        SPI.cycled = average_cycles(SPI.binned, start_cycle=SPI.cycle_start, fin_cycle=SPI.cycle_fin)

        # Center and average 2D data or skip for 1D
        SPI.x, SPI.averaged_data = average_2D(SPI.x, SPI.cycled)

        # region Use this if want to start shifting each heater setpoint of data left or right
        # Align data
        # SPI.x, SPI.averaged_data = align_setpoint_data(xs, SPI.averaged_data, nx=None)
        # endregion

        # Entropy signal
        SPI.entropy_signal = entropy_signal(SPI.averaged_data)

        # Fit Entropy
        ent_fit = E.entropy_fits(SPI.x, SPI.entropy_signal)[0]
        SPI.entropy_fit = DA.FitInfo.from_fit(ent_fit)

        # Integrated Entropy
        def scaling(dt, amplitude, dx):
            return dx / amplitude / dt
        if SPI.bias is None or (heat_bias := int(round(SPI.bias))) not in SPI.bias_theta_lookup.keys():
            raise ValueError(f'SPI.bias = {SPI.bias}. Should be in nA, and will only work for [{SPI.bias_theta_lookup.keys()}]')
        SPI.integrated_info = IntegratedInfo(heat_bias)
        ii = SPI.integrated_info
        ii.dx = np.mean(np.diff(SPI.x))
        ii.t_hot = SPI.bias_theta_lookup[heat_bias]
        ii.t_cold = SPI.bias_theta_lookup[0]
        ii.dt = (ii.t_hot-ii.t_cold)
        ii.amp = SPI.transition_amplitude
        ii.sf = scaling(ii.dt, SPI.transition_amplitude, ii.dx)
        SPI.integrated_entropy = np.nancumsum(SPI.entropy_signal)*ii.sf

    if show_plots is True:
        num_plots_for_dat = sum(SPI.plots.values())
        if SPI.axs is None:
            fig, SPI.axs = PF.make_axes(num_plots_for_dat)
        elif SPI.axs.size < num_plots_for_dat:
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
            freq = awg.info.measureFreq / awg.info.wave_len
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
            temp_x = np.linspace(SPI.x[0], SPI.x[-1], 1000)
            ax.plot(temp_x, SPI.entropy_fit.eval_fit(temp_x), label='fit')
            PF.ax_setup(ax, f'Entropy: dS = {SPI.entropy_fit.best_values.dS:.2f}'
                            f'{PM}{SPI.entropy_fit.params["dS"].stderr:.2f}', legend=True)

        if SPI.plots['integrated'] is True:
            # Plot integrated
            ax = axs[ax_index]
            ax_index += 1
            ax.plot(SPI.x, SPI.integrated_entropy)
            ax.axhline(np.log(2), color='k', linestyle=':', label='Ln2')
            PF.ax_setup(ax, f'Integrated Entropy:\ndS = {SPI.integrated_entropy[-1]:.2f}, SF={SPI.integrated_info.sf:.3f}', y_label='Entropy /kB', legend=True)

        axs[0].get_figure().tight_layout()

    return SPI


def _plot_2d_i_sense(dats, axs=None):
    if axs is None:
        fig, axs = PF.make_axes(len(dats))
    else:
        fig = axs[0].figure
        assert len(axs) >= len(dats)

    fs = []
    for dat, ax in zip(dats, axs):
        ax.cla()
        data = dat.Data.i_sense
        data, f = CU.decimate(data, dat.Logs.Fastdac.measure_freq, 10 * dat.Logs.sweeprate, return_freq=True)
        fs.append(f)
        x_arr = dat.Data.x_array
        x = np.linspace(x_arr[0], x_arr[-1], data.shape[-1])
        PF.display_2d(x, dat.Data.y_array, data, ax, x_label=dat.Logs.x_label, y_label=dat.Logs.y_label)
        PF.ax_setup(ax, f'Dat{dat.datnum}: Bias={dat.Logs.fds["L2T(10M)"] / 10:.1f}nA')
    if np.all([np.isclose(fs[0], freq) for freq in fs]):
        freq = f'{fs[0]:.1f}/s'
    else:
        freq = f'various ({np.nanmin(fs):.1f} -> {np.nanmax(fs):.1f})/s'
    fig.suptitle(f'I_sense data decimated to {freq}')
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for dat in dats:
        dat: DatHDF
        dat.Other.save_code(inspect.getsource(_plot_2d_i_sense), 'isense_2d_plots')
    return axs


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
        # dats = dats[10:]

        plot_info = True
        plot_raw = False
        plot_binned = False
        plot_cycle_averaged = False
        plot_averaged = True
        plot_entropy = True
        plot_integrated = True

        plots_per_dat = sum([plot_info, plot_binned, plot_cycle_averaged, plot_raw, plot_averaged, plot_entropy, plot_integrated])

        SPIs = [plot_square_wave(dat=dat, info=plot_info, raw=plot_raw, binned=plot_binned,
                                 cycle_averaged=plot_cycle_averaged, averaged=plot_averaged,
                                 entropy=plot_entropy, integrated=plot_integrated, calculate=True, show_plots=False) for dat in dats]
        spis_to_plot = SPIs[0:5]
        if len(fig.axes) != plots_per_dat * len(spis_to_plot):
            plt.close(fig)
            fig, all_axs = plt.subplots(plots_per_dat, len(spis_to_plot), figsize=(len(spis_to_plot) * 3.5, plots_per_dat * 3))
        # for SPI, axs in zip(SPIs, all_axs.T):
        for SPI, axs in zip(spis_to_plot, all_axs.T):
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
        import napari
        from src.Scripts.Napari_test import View

        get_ipython().enable_gui('qt')

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

        with napari.gui_qt():
            v = View()
            v.add_data(data)
            # v.add_profile()

    elif run == 'dc_bias':
        bias_dats = get_dats(range(522, 528 + 1))  # bias scans
        dat = bias_dats[0]
        # axs = _plot_2d_i_sense(bias_dats, None)

        # Process data
        recalculate = False
        if recalculate is True:
            for dat in bias_dats:
                filtered, freq = CU.decimate(dat.Data.i_sense, dat.Logs.Fastdac.measure_freq, 10 * dat.Logs.sweeprate,
                                             return_freq=True)
                filt_x = np.linspace(dat.Data.x_array[0], dat.Data.x_array[-1], filtered.shape[-1])
                filtered, filt_x = CU.remove_nans(filtered, filt_x, verbose=False)

                dat.Other.set_data('filtered_i_sense', filtered)
                dat.Other.set_data('filt_x', filt_x)
                dat.Other.filtered_freq = freq

                # Refit and save
                # Refit row fits first
                row_fits = copy.copy(dat.Transition.all_fits)

                for fit, d in zip(row_fits, filtered):
                    fit.recalculate_fit(filt_x, d)
                frf = dat.Other.group.require_group('filtered_row_fits')
                DA.row_fits_to_group(frf, row_fits, dat.Data.y_array)

                # Use row fits to average data
                avg_data = np.mean(CU.center_data(filt_x, filtered, [fit.best_values.mid for fit in row_fits]), axis=0)

                # Refit avg_fit and save
                avg_fit = copy.copy(dat.Transition.avg_fit)
                dat.Other.set_data('filtered_i_sense_avg', avg_data)
                avg_fit.recalculate_fit(filt_x, avg_data)
                afg = dat.Other.group.require_group('filtered_avg_fit')
                avg_fit.save_to_hdf(afg)

                # Make sure datHDF is updated
                dat.hdf.flush()

        # Plot waterfall fits to data
        plot_waterfall = False
        if plot_waterfall:
            fig, axs = PF.make_axes(len(bias_dats), single_fig_size=(3, 3))
            every_nth = 5
            for dat, ax in zip(bias_dats, axs):
                ax.cla()
                data = dat.Other.Data['filtered_i_sense']
                x = dat.Other.Data['filt_x']
                fits = DA.rows_group_to_all_FitInfos(dat.Other.group['filtered_row_fits'])
                y = dat.Data.y_array
                y_add, x_add = PF.waterfall_plot(x, data, ax, 4, None, 0, None, every_nth, auto_bin=False)
                best_fits = np.array([fit.eval_fit(x) for fit in fits])
                PF.waterfall_plot(x, best_fits, ax, y_add=y_add, x_add=x_add, every_nth=every_nth, auto_bin=False,
                                  color='red', plot_args={'linewidth': 1})
                PF.ax_setup(ax, f'Dat{dat.datnum}: Bias={dat.Logs.fds["L2T(10M)"] / 10:.1f}nA',
                            x_label=dat.Logs.x_label, y_label='Current /nA')
            fig.suptitle(f'Fits to DCbias data: Every {every_nth:d}th row')
            fig.tight_layout(rect=[0, 0, 1, 0.95])

        # Plot fit parameters
        plot_fit_params = True
        if plot_fit_params:
            # fig, axs = plt.subplots(1, 2, figsize=(12, 3.5))
            fig, axs = PF.make_axes(2)
            for ax in axs:
                ax.cla()
            for dat in bias_dats:
                data = dat.Other.Data['filtered_i_sense']
                x = dat.Other.Data['filt_x']
                fits = DA.rows_group_to_all_FitInfos(dat.Other.group['filtered_row_fits'])
                avg_fit = DA.fit_group_to_FitInfo(dat.Other.group['filtered_avg_fit'])
                bias = dat.Logs.fds['L2T(10M)'] / 10

                for ax, key in zip(axs, ['theta', 'mid']):
                    ax: plt.Axes
                    err = np.nanstd([getattr(fit.best_values, key) for fit in fits])
                    # err = avg_fit.params[key].stderr
                    avg_value = getattr(avg_fit.best_values, key)
                    if avg_value:
                        ax.errorbar(bias, getattr(avg_fit.best_values, key), yerr=err, label=f'{dat.datnum}', marker='+')

            PF.ax_setup(axs[0], f'Theta vs DCbias', 'DCbias /nA', 'Theta /mV')
            PF.ax_setup(axs[1], 'Center vs DCbias', 'DCbias /nA', 'Center /mV', legend=True)

            for ax in axs:
                ax.legend(title='Datnum')

            fig.suptitle(f'Fixed Bias: Standard deviation of\nfits to each row as error bars')

        # Print theta values
        print_values = False
        if print_values:
            theta_v_bias = {}
            for dat in bias_dats:
                avg_fit = DA.fit_group_to_FitInfo(dat.Other.group['filtered_avg_fit'])
                theta_v_bias[int(round(dat.Logs.fds['L2T(10M)']/10))] = avg_fit.best_values.theta

            print('Bias/nA: Theta/mV')
            for k, v in theta_v_bias.items():
                print(f'{k}: {v:.4f}')

            print('\nAveraged\nBias/nA: Theta/mV')
            print(f'0: {theta_v_bias[0]:.4f}')
            for p in [30, 50, 80]:
               print(f'{abs(p):d}: {np.average([theta_v_bias[p], theta_v_bias[-p]]):.4f}')

    elif run == 'temp_working':
        dat = get_dat(500)
        cfg.PF_binning = False
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
        sqw = SquareWave(measure_freq=measure_freq, start=start, fin=fin, sweeprate=sweeprate, v0=0, vp=vheat,
                         vm=-vheat, step_duration=step_dur)

        # Make Transition model
        t = SquareTransitionModel(mid, amp, theta, lin, const, sqw, cross_cap, heat_factor, dS)

        # heat = hf * hv ** 2  # Heating proportional to hv^2
        # T = theta + heat  # theta is base temp theta, so T is that plus any heating
        t.cross_cap = 0.001  # From quadratic fit to centers of DC bias dats 522->528
        t.heat_factor = 0.000001493  # For 800mV heat_factor = 0.9555/(800**2) (full_dT/mV**2)
        t.theta = 0.4942

        fig, ax = plt.subplots(1)
        line: Union[None, plt.Line2D] = None

        # t.cross_cap = 0.001
        # t.heat_factor = 0.000004
        # t.theta = 0.5
        if line:
            line.remove()
        PF.display_1d(t.x, t.eval(t.x), ax, label=f'{t.cross_cap:.2g}, {t.heat_factor:.2g}', marker='')
        line = ax.lines[-1]
        ax.legend(title='cross_cap, heat_factor')

        model_spi = SquarePlotInfo()
        model_spi.update(None, None, True, False, False, False, True, True, True)
        model_spi.orig_x_array = t.x
        model_spi.raw_data = np.tile(t.eval(t.x), (2,1))
        model_spi.awg = t.square_wave
        model_spi.datnum = 0
        model_spi.x_label = 'Plunger /mV'
        model_spi.bias = t.square_wave.AWs[0][0][1]/10
        model_spi.transition_amplitude = t.amp



        plot_square_wave(model_spi, None, None, calculate=True, show_plots=False)

        fig, axs = PF.make_axes(7)
        for ax in axs:
            ax.cla()

        model_spi.decimate_freq = None
        plot_square_wave(model_spi, axs=axs, info=True, raw=True, binned=True, cycle_averaged=True, averaged=True, entropy=True, integrated=True,
                         calculate=False, show_plots=True)

        line = lm.models.LinearModel()
        idx = CU.get_data_index(model_spi.x, -10)
        lfit = line.fit(model_spi.integrated_entropy[:idx], x=model_spi.x[:idx])
        lfi = DA.FitInfo.from_fit(lfit)
        ax.plot(model_spi.x, model_spi.integrated_entropy - lfi.eval_fit(model_spi.x))
        slope_subtracted = model_spi.integrated_entropy - lfi.eval_fit(model_spi.x)
        print(f'dS={slope_subtracted[-1]:.3f}')

    elif run == 'temp2':
        fig, ax = plt.subplots(1)
        ax.cla()
        for z in np.arange(11):
            ax.axhline(z, c='k', linestyle=':', linewidth=1)
            ax.axvline(z, c='k', linestyle=':', linewidth=1)
        num = 5
        a = np.linspace(0, 10, num)
        dx = (a[-1]-a[0])/num/2
        # dx = 0
        b = np.linspace(0, 10, 100)
        interper = interp1d(np.linspace(a[0]+dx, a[-1]-dx, num), a, kind='nearest', bounds_error=False, fill_value='extrapolate')
        c = interper(b)
        ax.plot(b, c, label='c on b')
        ax.plot(b, b, label='b on b')

        truey = [0,0,2.5,2.5,5,5,7.5,7.5,10,10]
        truex = [0,2,2,4,4,6,6,8,8,10]
        ax.plot(truex, truey, label='true steps')
        ax.legend()


