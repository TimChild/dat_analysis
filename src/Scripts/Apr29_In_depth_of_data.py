from src.Scripts.StandardImports import *
import src.DatCode.Entropy as E
import src.DatCode.Transition as T
from scipy.signal import savgol_filter


def _recalculate_dats(datdf: DF.DatDF, datnums: list, datname='base', make_dat_function=make_dat_standard, dattypes: set = None, save=True):
    """
    Just a quick fn to recalculate and save all dats to given datDF
    """
    # if datdf.config_name != cfg.current_config.__name__.split('.')[-1]:
    #     print('WARNING[_recalculate_given_dats]: Need to change config while running this. No dats changed')
    #     return
    for datnum in datnums:
        dat = make_dat_function(datnum, datname=datname, dfname=datdf.name, dfoption='overwrite', dattypes=dattypes)
        datdf.update_dat(dat, yes_to_all=True)
    if save is True:
        datdf.save()


datdf = DF.DatDF(dfname='Apr20')
datnums = [1533, 1501]
_recalculate_dats(datdf, datnums, datname='digamma_quad', make_dat_function=make_dat_standard, dattypes={'suppress_auto_calculate'}, save=True)
dats = make_dats(datnums)


run = False
if run is True:
    dats = [make_dat_standard(num, datname='digamma', dfoption='load', dfname='Apr20') for num in [1533, 1501]]
    dc = make_dat_standard(1529, dfname='Apr20', dfoption='load')

    # region Setup data to look at
    dat = dats[0]
    every_nth = 5  # every nth row of data
    from_to = (None, None)
    thin_data_points = 10  # only look at every 10th datapoint (faster plotting)
    i_spacing_y = -2
    e_spacing_y = -3
    smoothing_num = 11
    view_width = 20
    fig_size = (4.5,7)

    # dat = dats[1]
    # every_nth = 1  # every nth row of data
    # from_to = (-6, -1)
    # thin_data_points = 10  # only look at every 10th datapoint (faster plotting)
    # i_spacing_y = -1
    # e_spacing_y = -3
    # smoothing_num = 11
    # view_width = 20
    # fig_size = (4.5,7)
    # endregion

    show_plots = {'i_sense':False, 'i_sense_avg':False, 'entr': False, 'avg_entr': True, 'int_ent':True, 'tables':False}


    # region Select data to look at
    x = dat.Data.x_array[::thin_data_points]
    y_isense = dat.Data.i_sense[from_to[0]:from_to[1]:every_nth, ::thin_data_points]
    y_entr = dat.Entropy._data[from_to[0]:from_to[1]:every_nth, ::thin_data_points]
    dx = (x[-1] - x[0]) / len(x)
    # endregion

    # region Match up with existing fit data
    fits_isense = dat.Transition._full_fits[from_to[0]:from_to[1]:every_nth]
    fits_entr = dat.Entropy._full_fits[from_to[0]:from_to[1]:every_nth]
    # endregion


    # region Average of data being looked at ONLY
    i_y_avg, _ = np.array(
        CU.average_data(y_isense, [CU.get_data_index(x, fit.best_values['mid']) for fit in fits_isense]))
    e_y_avg, _ = np.array(
        CU.average_data(y_entr, [CU.get_data_index(x, fit.best_values['mid']) for fit in fits_isense]))
    # endregion

    # region Fits to average of data being looked at ONLY
    i_fit_avg = T.transition_fits(x, i_y_avg, params=[dat.Transition.avg_params], func=T.i_sense_digamma)[0]
    x_i_fit_avg = i_fit_avg.userkws['x']
    e_fit_avg = E.entropy_fits(x, e_y_avg, params=[dat.Entropy.avg_params])[0]
    x_e_fit_avg = e_fit_avg.userkws['x']
    # endregion


    # region Integrated Entropy with standard best guess
    dt = dc.DCbias.get_dt_at_current(dat.Instruments.srs1.out / 50 * np.sqrt(2))
    sf = E.scaling(dt, i_fit_avg.best_values['amp'], dx)
    int_avg = E.integrate_entropy_1d(e_y_avg, sf)  # Not sure why const/2 has to be subtracted....
    int_of_fit = E.integrate_entropy_1d(e_fit_avg.best_fit, sf)
    # endregion

    # region Force dS to value that results in Ln2 entropy
    params = CU.edit_params(e_fit_avg.params, 'dS', np.log(2), vary=False)
    e_fit_ln2 = E.entropy_fits(x, e_y_avg, params=[params])[0]
    # endregion

    # region E fit with dT forced s.t. integrated(data) = Ln2
    # dt_from_fit = e_fit_avg.best_values['dT']
    new_dt = dt*int_of_fit[-1]/np.log(2)
    params = CU.edit_params(e_fit_avg.params, 'dT', dt, False)
    e_fit_dt_ln2 = E.entropy_fits(x, e_y_avg, params=[params])[0]
    # endregion

    # region Integrated E of fit with dT forced s.t. integrated(data) = Ln2
    sf_dt_forced = E.scaling(new_dt, i_fit_avg.best_values['amp'], dx)
    int_of_fit_dt_ln2 = E.integrate_entropy_1d(e_fit_dt_ln2.best_fit, sf_dt_forced)
    # endregion


    # region Force amplitude to value that results in Ln2 integrated entropy
    new_amp = i_fit_avg.best_values['amp'] * int_avg[-1] / np.log(2)
    params = CU.edit_params(i_fit_avg.params, 'amp', new_amp, vary=False)
    i_fit_ln2 = T.transition_fits(x, i_y_avg, params=[params])[0]
    # endregion

    #  PLOTTING BELOW HERE

    # region I_sense by row plots
    if show_plots['i_sense'] is True:
        fig, axs = plt.subplots(2,1,figsize=fig_size)
        axs = axs.flatten()
        ax = axs[0]
        y_add, x_add = PF.waterfall_plot(x, y_isense, ax=ax, y_spacing=i_spacing_y, x_add=0, every_nth=1, plot_args={'s': 1},
                       ptype='scatter', label=True)
        PF.ax_setup(ax, f'I_sense data for dat[{dat.datnum}]', dat.Logs.x_label, 'I_sense /nA', legend=True)

        ax = axs[1]
        ysmooth = savgol_filter(y_isense, smoothing_num, 1)
        xi = (CU.get_data_index(x, dat.Transition.mid-view_width), CU.get_data_index(x, dat.Transition.mid+view_width))
        PF.waterfall_plot(x[xi[0]:xi[1]], ysmooth[:,xi[0]:xi[1]], ax=ax, y_add=y_add, x_add=x_add, every_nth=1, plot_args={'s':1}, ptype='scatter', label=True)
        y_fits = np.array([fit.eval(x=x[xi[0]:xi[1]]) for fit in fits_isense])
        PF.waterfall_plot(x[xi[0]:xi[1]], y_fits, ax=ax, y_add=y_add, x_add=x_add, color='C3', ptype='plot')
        PF.ax_setup(ax, f'Smoothed I_sense data for dat[{dat.datnum}]\nwith fits', dat.Logs.x_label, 'I_sense /nA', legend=True)
        PF.add_standard_fig_info(fig)

        if show_plots['tables'] is True:
            df = CU.fit_info_to_df(fits_isense)
            PF.plot_df_table(df, title=f'I_sense_fit info for dat[{dat.datnum}]')
    # endregion


    # region Average I_sense plots
    if show_plots['i_sense_avg'] is True:
        fig, axs = plt.subplots(2, 1, figsize=fig_size)
        axs = axs.flatten()
        ax = axs[0]
        PF.display_1d(x, i_y_avg, ax, scatter=True, label='Averaged data')
        ax.plot(x_i_fit_avg, i_fit_avg.best_fit, c='C3', label='Best fit')
        PF.ax_setup(ax, f'Dat[{dat.datnum}]:Averaged data with fit', dat.Logs.x_label, 'I_sense /nA', legend=True)

        if show_plots['tables'] is True:
            df = CU.fit_info_to_df([i_fit_avg])
            df.pop('index')
            PF.plot_df_table(df, title=f'Avg I_sense fit info for dat[{dat.datnum}]')

        ax = axs[1]
        PF.display_1d(x, i_y_avg, ax, scatter=True, label='Averaged data')
        ax.plot(x_i_fit_avg, i_fit_ln2.best_fit, c='C3', label='Ln(2) amplitude fit')
        PF.ax_setup(ax, f'Dat[{dat.datnum}]:Averaged I_sense data with Ln(2) amp fit', dat.Logs.x_label, 'I_sense /nA', legend=True)
        PF.add_standard_fig_info(fig)

        if show_plots['tables'] is True:
            df = CU.fit_info_to_df([i_fit_ln2])
            df.pop('index')
            PF.plot_df_table(df, title=f'Avg I_sense Ln(2) amplitude fit info for dat[{dat.datnum}]')
    # endregion


    # region Entropy by row plots
    if show_plots['entr'] is True:
        fig, axs = plt.subplots(2, 1, figsize=fig_size)
        axs = axs.flatten()
        ax = axs[0]
        y_add, x_add = PF.waterfall_plot(x, y_entr, ax=ax, y_spacing=e_spacing_y, x_add=0, every_nth=1, plot_args={'s': 1},
                       ptype='scatter', label=True)
        PF.ax_setup(ax, f'Entropy_r data for dat[{dat.datnum}]', dat.Logs.x_label, 'Entr /nA', legend=True)

        ax = axs[1]
        ysmooth = savgol_filter(y_entr, smoothing_num, 1)  # same xi as for i_sense
        xi = (CU.get_data_index(x, dat.Transition.mid - view_width), CU.get_data_index(x, dat.Transition.mid + view_width))
        PF.waterfall_plot(x[xi[0]:xi[1]], ysmooth[:, xi[0]:xi[1]], ax=ax, y_add=y_add, x_add=x_add, every_nth=1,
                       plot_args={'s': 1}, ptype='scatter', label=True)
        y_fits = np.array([fit.eval(x=x[xi[0]:xi[1]]) for fit in fits_entr])
        PF.waterfall_plot(x[xi[0]:xi[1]], y_fits, ax=ax, y_add=y_add, x_add=x_add, color='C3', ptype='plot')
        PF.ax_setup(ax, f'Dat[{dat.datnum}]:Smoothed entropy_r data\nwith fits', dat.Logs.x_label,
                    'Entr /nA', legend=True)
        PF.add_standard_fig_info(fig)

        if show_plots['tables'] is True:
            df = CU.fit_info_to_df(fits_entr)
            PF.plot_df_table(df, title=f'Entropy_R_fit info for dat[{dat.datnum}]')
    # endregion


    # region Average Entropy Plots
    if show_plots['avg_entr'] is True:
        fig, axs = plt.subplots(2,2,figsize=(fig_size[0]*2, fig_size[1]))
        axs = axs.flatten()
        ax = axs[0]

        PF.display_1d(x, e_y_avg, ax, scatter=True, label='Averaged data')
        ax.plot(x_e_fit_avg, e_fit_avg.best_fit, c='C3', label='Best fit')
        PF.ax_setup(ax, f'Dat[{dat.datnum}]:Averaged Entropy R data with fit', dat.Logs.x_label, 'Entropy R /nA', legend=True)

        if show_plots['tables'] is True:
            df = CU.fit_info_to_df([e_fit_avg])
            df.pop('index')
            PF.plot_df_table(df, title=f'Avg Entropy R fit info for dat[{dat.datnum}]')

        ax = axs[1]
        PF.display_1d(x, e_y_avg, ax, scatter=True, label='Averaged data')
        ax.plot(x_e_fit_avg, e_fit_ln2.best_fit, c='C3', label='Ln(2) fit')
        PF.ax_setup(ax, f'Dat[{dat.datnum}]:Averaged Entropy R data with Ln(2) fit', dat.Logs.x_label, 'Entropy R /nA', legend=True)
        PF.add_standard_fig_info(fig)

        if show_plots['tables'] is True:
            df = CU.fit_info_to_df([e_fit_ln2])
            df.pop('index')
            PF.plot_df_table(df, title=f'Avg Entropy R Ln(2) fit info for dat[{dat.datnum}]')

        ax = axs[2]
        PF.display_1d(x, e_y_avg, ax, scatter=True, label='Averaged data')
        ax.plot(x_e_fit_avg, e_fit_dt_ln2.best_fit, c='C3', label='dT forced fit')
        PF.ax_setup(ax, f'Dat[{dat.datnum}]:Averaged Entropy R data\nwith dT forced fit', dat.Logs.x_label, 'Entropy R /nA',
                    legend=True)
        PF.add_standard_fig_info(fig)

        if show_plots['tables'] is True:
            df = CU.fit_info_to_df([e_fit_dt_ln2])
            df.pop('index')
            PF.plot_df_table(df, title=f'Avg Entropy R dT forced fit info for dat[{dat.datnum}]')
    # endregion


    # region Integrated Entropy Plots
    if show_plots['int_ent'] is True:
        fig, axs = plt.subplots(2, 1, figsize=fig_size)
        axs = axs.flatten()
        ax = axs[0]

        PF.display_1d(x, int_avg, ax, label='Averaged data')
        ax.plot(x_e_fit_avg, int_of_fit, c='C3', label='integrated best fit')
        PF.ax_setup(ax, f'Dat[{dat.datnum}]:Integrated Entropy - int of fit', dat.Logs.x_label, 'Entropy /kB', legend=True)
        PF.ax_text(ax, f'int_dS={int_avg[-1]:.3f}kB\n'
                       f'int_fit_dS={int_of_fit[-1]:.3f}kB')

        ax = axs[1]
        PF.display_1d(x, int_avg, ax, label='Averaged data')
        ax.plot(x_e_fit_avg, int_of_fit_dt_ln2, c='C3', label='integrated fit\nwith dT forced')
        PF.ax_setup(ax, f'Dat[{dat.datnum}]:Integrated Entropy - dT forced', dat.Logs.x_label, 'Entropy /kB', legend=True)
        PF.ax_text(ax, f'int_dS={int_avg[-1]:.3f}kB\n'
                       f'int_fit_dT_forced_dS={int_of_fit_dt_ln2[-1]:.3f}kB')

        PF.add_standard_fig_info(fig)
    # endregion