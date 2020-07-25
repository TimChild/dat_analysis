from src.PlottingFunctions import remove_line
from src.Scripts.StandardImports import *
from src.DatObject.Attributes.SquareEntropy import *
import copy


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
    run = 'modelling_array'
    if run == 'modelling':
        cfg.PF_num_points_per_row = 2000  # Otherwise binning data smears out square steps too much
        ax = None
        ax = PF.require_axs(1, ax, clear=True)[0]

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
        heat_factor = 1.5e-6
        dS = np.log(2)

        # Make the square wave
        sqw = SquareAWGModel(measure_freq=measure_freq, start=start, fin=fin, sweeprate=sweeprate,
                             v0=0, vp=vheat, vm=-vheat, step_duration=step_dur)

        # Make Transition model
        t = SquareTransitionModel(mid, amp, theta, lin, const, sqw, cross_cap, heat_factor, dS)

        # Init SquareProcessed class but suppress processing for now
        model_sp = SquareProcessed.from_info(np.tile(t.eval(t.x), (2, 1)), t.x, t.square_wave,
                                             bias=t.square_wave.AWs[0][0][1] / 10, amplitude=t.amp,
                                             datnum=0, x_label='Plunger /mV', calculate=False)

        # Change any processing params here
        pass

        # Run processing
        model_sp.calculate()

        # Which plots to plot
        model_sp.plot_info.show = ShowPlots(info=True,
                                            raw=True,
                                            setpoint_averaged=True,
                                            cycle_averaged=True,
                                            averaged=True,
                                            entropy=True,
                                            integrated=True)
        model_sp.plot_info.decimate_freq = None

        model_sp.plot_info.axs = PF.require_axs(model_sp.plot_info.num_plots, model_sp.plot_info.axs, clear=True)

        plot_square_entropy(model_sp)

    elif run == 'fitting_data':
        fig, all_axs = plt.subplots(2)  # Not used but lets code run

        # Get data
        dats = get_dats(range(500, 515))  # Various square wave tests (varying freq, cycles, amplitude)

        # Make SquareProcess class but suppress processing here
        SPs = [SquareProcessed.from_dat(dat, calculate=False) for dat in dats]

        # Make any changes to process params etc here
        for sp in SPs:
            pass

        # Run processing
        for sp in SPs: sp.calculate()

        # Choose plots
        for sp in SPs:
            sp.plot_info.show = ShowPlots(info=True,
                                          raw=False,
                                          setpoint_averaged=False,
                                          cycle_averaged=False,
                                          averaged=True,
                                          entropy=True,
                                          integrated=True)

        # Which SPs to plot
        sps_to_plot = SPs[0:5]

        # Make figure, axs
        plots_per_dat = sps_to_plot[0].plot_info.num_plots
        if len(fig.axes) != plots_per_dat * len(sps_to_plot):
            plt.close(fig)
            fig, all_axs = plt.subplots(plots_per_dat, len(sps_to_plot),
                                        figsize=(len(sps_to_plot) * 3.5, plots_per_dat * 3))

        # Plot
        for sp, axs in zip(sps_to_plot, all_axs.T):
            sp.plot_info.axs = axs
            plot_square_entropy(sp)

        # Adjust plots afterwards
        for axs in all_axs:
            for ax in axs:
                if ax.get_legend() is not None:
                    ax.legend(fontsize=7)
        for ax in all_axs[:-1, :].flatten():
            ax.set_xlabel('')
        for ax in all_axs[:, 1:].flatten():
            ax.set_ylabel('')

        fig.tight_layout(pad=0.5, h_pad=0.05, w_pad=0.05)

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
                        ax.errorbar(bias, getattr(avg_fit.best_values, key), yerr=err, label=f'{dat.datnum}',
                                    marker='+')

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
                theta_v_bias[int(round(dat.Logs.fds['L2T(10M)'] / 10))] = avg_fit.best_values.theta

            print('Bias/nA: Theta/mV')
            for k, v in theta_v_bias.items():
                print(f'{k}: {v:.4f}')

            print('\nAveraged\nBias/nA: Theta/mV')
            print(f'0: {theta_v_bias[0]:.4f}')
            for p in [30, 50, 80]:
                print(f'{abs(p):d}: {np.average([theta_v_bias[p], theta_v_bias[-p]]):.4f}')

    elif run == 'modelling DAC steps test':
        fig, ax = plt.subplots(1)
        ax.cla()
        for z in np.arange(11):
            ax.axhline(z, c='k', linestyle=':', linewidth=1)
            ax.axvline(z, c='k', linestyle=':', linewidth=1)
        num = 5
        a = np.linspace(0, 10, num)
        dx = (a[-1] - a[0]) / num / 2
        # dx = 0
        b = np.linspace(0, 10, 100)
        interper = interp1d(np.linspace(a[0] + dx, a[-1] - dx, num), a, kind='nearest', bounds_error=False,
                            fill_value='extrapolate')
        c = interper(b)
        ax.plot(b, c, label='c on b')
        ax.plot(b, b, label='b on b')

        truey = [0, 0, 2.5, 2.5, 5, 5, 7.5, 7.5, 10, 10]
        truex = [0, 2, 2, 4, 4, 6, 6, 8, 8, 10]
        ax.plot(truex, truey, label='true steps')
        ax.legend()

    elif run == 'modelling_array':
        import sys
        import os
        napari_py = "../../../Napari_interface/"
        sys.path.append(os.path.abspath(napari_py))
        from Modelling_example import Window
        get_ipython().enable_gui('qt')


        # Params for both sqw and transition
        start = -20
        fin = 20
        measure_freq = 1000
        sweeprate = 0.5

        # Params for transition model
        mid = 0
        amp = 0.5
        theta = np.linspace(0.3, 1.0, 11)
        lin = np.linspace(0, 0.03,  11)
        const = 8
        cross_cap = np.linspace(0, 0.004, 11)
        heat_factor = np.linspace(0, 3e-6, 11)
        dS = np.log(2)

        # Params for square wave
        vheat = np.linspace(0, 1000, 11)
        step_dur = np.linspace(0.25, 1, 11)

        variables = ['step_dur', 'vheat', 'theta', 'lin', 'cross cap', 'heating']

        recalculate = False
        if recalculate or os.path.isfile(r'C:\Users\Child\Downloads\nD_data.npy') is False:
            def get_sqw(vheat, step_dur) -> SquareAWGModel:
                return SquareAWGModel(measure_freq=1000, start=start, fin=fin, sweeprate=sweeprate,
                                      v0=0, vp=vheat, vm=-vheat, step_duration=step_dur)

            vheat, step_dur = np.meshgrid(vheat, step_dur)
            sqws = np.ndarray(vheat.shape, dtype=object)
            for i, (vrow, srow) in enumerate(zip(vheat, step_dur)):
                for j, (v, s) in enumerate(zip(vrow, srow)):
                    sqws[i, j] = get_sqw(v, s)

            tmods = np.array([[SquareTransitionModel(mid=mid, amp=amp, theta=theta, lin=lin, const=const,
                                         square_wave=sqw, cross_cap=cross_cap, heat_factor=heat_factor,
                                         dS=dS) for sqw in row] for row in sqws])

            data = np.array([[t.eval(np.linspace(start, fin, 400)) for t in row] for row in tmods])
            np.save(r'C:\Users\Child\Downloads\nD_data.npy', data.astype(np.float32))
        else:
            data = np.load(r'C:\Users\Child\Downloads\nD_data.npy')

        w = Window()
        w.add_data(data, x=np.linspace(start, fin, 1000))
        w.add_profile()

        for i, label in enumerate(variables):
            w.viewer.dims.set_axis_label(i, label)
            w.viewer.dims.set_point(i, 2)

        # fig, ax = plt.subplots(1)
        # x = np.linspace(start, fin, 300)
        # z = data[2,2,2,2,2,2,0]
        # ax.plot(x, z)
        d = np.load(r'C:\Users\Child\Downloads\nD_data.npy')