from src.Scripts.StandardImports import *
import src.DatCode.Entropy as E
import src.DatCode.Transition as T
from scipy.signal import savgol_filter


def _recalculate_dats(datdf: DF.DatDF, datnums: list, datname='base', dattypes: set = None, setupdf=None, config=None, transition_func = None, save=True):
    """
    Just a quick fn to recalculate and save all dats to given datDF
    """
    # if datdf.config_name != cfg.current_config.__name__.split('.')[-1]:
    #     print('WARNING[_recalculate_given_dats]: Need to change config while running this. No dats changed')
    #     return
    if transition_func is not None:
        dattypes = CU.ensure_set(dattypes)
        dattypes.add('suppress_auto_calculate')  # So don't do pointless calculation of transition when first initializing
    for datnum in datnums:
        dat = make_dat_standard(datnum, datname=datname, datdf=datdf, dfoption='overwrite', dattypes=dattypes, setupdf=setupdf, config=config)
        if transition_func is not None:
            dat._reset_transition(fit_function=transition_func)
            if 'entropy' in dat.dattype:
                dat._reset_entropy()
        datdf.update_dat(dat, yes_to_all=True)
    if save is True:
        datdf.save()


def _add_ln3_ln2(ax:plt.Axes):
    ax.axhline(np.log(2), linestyle=':', color='grey')
    ax.axhline(np.log(3), linestyle=':', color='grey')


def _add_peak_final_text(ax, data, fit):
    PF.ax_text(ax, f'Peak/Final /(Ln3/Ln2)\ndata, fit={(np.nanmax(data) / data[-1])/(np.log(3)/np.log(2)):.2f}, '
                   f'{(np.nanmax(fit) / fit[-1])/(np.log(3)/np.log(2)):.2f}',
               loc=(0.6, 0.8), fontsize=8)

# datdf = DF.DatDF(dfname='Apr20')
# assert datdf.config_name == 'Jan20Config'
# datnums = [1533, 1501]
# _recalculate_dats(datdf, datnums, datname='digamma_quad', dattypes={'entropy', 'transition', 'suppress_auto_calculate'}, config=cfg.current_config, transition_func=T.i_sense_digamma_quad, save=True)

run = True
if run is True:
    datdf = DF.DatDF(dfname='Apr20')
    assert datdf.config_name == 'Jan20Config'
    dats = [make_dat_standard(num, datname='digamma_quad', dfoption='load', datdf=datdf) for num in [1533, 1501]]
    dc = make_dat_standard(1529, datdf=datdf, dfoption='load')

    # region Setup data to look at
    # dat = dats[0]
    # every_nth = 5  # every nth row of data
    # from_to = (None, None)
    # thin_data_points = 10  # only look at every 10th datapoint (faster plotting)
    # i_spacing_y = -2
    # e_spacing_y = -3
    # smoothing_num = 11
    # view_width = 20
    # beta = 0.82423 / 0.1  # Theta in mV at min point of DCbias / 100mK in K.  T(K) = beta*theta(mV)
    # fig_size = (4.5,7)


    dat = dats[1]
    every_nth = 1  # every nth row of data
    from_to = (-6, -1)
    thin_data_points = 10  # only look at every 10th datapoint (faster plotting)
    i_spacing_y = -1
    e_spacing_y = -3
    smoothing_num = 11
    view_width = 20
    beta = 0.82423 / 0.1  # Theta in mV at min point of DCbias / 100mK in K.  T(K) = beta*theta(mV)
    fig_size = (4.5,7)
    # endregion

    show_plots = {'i_sense': True, 'i_sense_avg': True, 'entr': True, 'avg_entr': True, 'int_ent': True, 'tables': True}
    uncertainties = True  # Whether to show uncertainties in tables or not
    use_existing_fits = False
    transition_fit_func = T.i_sense_digamma_quad

    # region Select data to look at
    x = dat.Data.x_array[::thin_data_points]
    y_isense = dat.Data.i_sense[from_to[0]:from_to[1]:every_nth, ::thin_data_points]
    y_entr = dat.Entropy._data[from_to[0]:from_to[1]:every_nth, ::thin_data_points]
    dx = (x[-1] - x[0]) / len(x)
    # endregion

    # region Get or make fit data
    if use_existing_fits is True:  # Match up to rows of data chosen
        # region Get Existing fits
        fits_isense = dat.Transition._full_fits[from_to[0]:from_to[1]:every_nth]
        fits_entr = dat.Entropy._full_fits[from_to[0]:from_to[1]:every_nth]
        # endregion
    else:  # Make new fits
        # region Make Transition fits
        params = T.get_param_estimates(x, y_isense)
        for par in params:
            T._append_param_estimate_1d(par, ['g', 'quad'])

        # Edit fit pars here
        params = [CU.edit_params(par, param_name='g', value=0, vary=False, min_val=None, max_val=None) for par in params]

        fits_isense = T.transition_fits(x, y_isense, params=params, func=transition_fit_func)
        # endregion
        # region Make Entropy fits
        mids = [fit.best_values['mid'] for fit in fits_isense]
        thetas = [fit.best_values['theta'] for fit in fits_isense]
        params = E.get_param_estimates(x, y_entr, mids, thetas)

        # Edit fit pars here
        params = [CU.edit_params(par, param_name='const', value=0, vary=False, min_val=None, max_val=None) for par in params]

        fits_entr = E.entropy_fits(x, y_entr, params=params)
        # endregion
    # endregion


    # region Average of data being looked at ONLY
    i_y_avg, _ = np.array(
        CU.average_data(y_isense, [CU.get_data_index(x, fit.best_values['mid']) for fit in fits_isense]))
    e_y_avg, _ = np.array(
        CU.average_data(y_entr, [CU.get_data_index(x, fit.best_values['mid']) for fit in fits_isense]))
    # endregion

    # region Fits to average of data being looked at ONLY
    i_fit_avg = T.transition_fits(x, i_y_avg, params=[fits_isense[0].params], func=transition_fit_func)[0]
    x_i_fit_avg = i_fit_avg.userkws['x']
    e_fit_avg = E.entropy_fits(x, e_y_avg, params=[fits_entr[0].params])[0]
    x_e_fit_avg = e_fit_avg.userkws['x']
    # endregion


    # region Integrated Entropy with dT from DCbias amp from i_sense (standard)
    dt = dc.DCbias.get_dt_at_current(dat.Instruments.srs1.out / 50 * np.sqrt(2))
    sf = E.scaling(dt, i_fit_avg.best_values['amp'], dx)
    int_avg = E.integrate_entropy_1d(e_y_avg, sf)
    int_of_fit = E.integrate_entropy_1d(e_fit_avg.best_fit, sf)
    # endregion

    # region E fit with dS forced = Ln2
    params = CU.edit_params(e_fit_avg.params, 'dS', np.log(2), vary=False)
    e_fit_ln2 = E.entropy_fits(x, e_y_avg, params=[params])[0]
    # endregion

    # region E fit with dT forced s.t. int_data dS = Ln2
    dt_ln2 = dt * int_avg[-1] / np.log(2)  # scaling prop to 1/dT
    params = CU.edit_params(e_fit_avg.params, 'dT', dt/beta, False)
    e_fit_dt_ln2 = E.entropy_fits(x, e_y_avg, params=[params])[0]
    # endregion

    # region Integrated E of fit with dT forced s.t. int_data dS = Ln2
    sf_dt_forced = E.scaling(dt_ln2, i_fit_avg.best_values['amp'], dx)
    int_of_fit_dt_ln2 = E.integrate_entropy_1d(e_fit_dt_ln2.best_fit, sf_dt_forced)
    int_avg_dt_ln2 = E.integrate_entropy_1d(e_y_avg, sf_dt_forced)
    # endregion

    # region Integrated E of data and best fit with dT from E_avg fit
    dt_from_fit = e_fit_avg.best_values['dT']*beta
    sf_dt_from_fit = E.scaling(dt_from_fit, i_fit_avg.best_values['amp'], dx)
    int_avg_dt_from_fit = E.integrate_entropy_1d(e_y_avg, sf_dt_from_fit)
    int_of_fit_dt_from_fit = E.integrate_entropy_1d(e_fit_avg.best_fit, sf_dt_from_fit)
    # endregion

    # region I_sense with amp forced s.t. int_data dS = Ln2 with dT from DCbias
    amp_forced_ln2 = i_fit_avg.best_values['amp'] * int_avg[-1] / np.log(2)
    params = CU.edit_params(i_fit_avg.params, 'amp', amp_forced_ln2, vary=False)
    i_fit_ln2 = T.transition_fits(x, i_y_avg, params=[params], func=transition_fit_func)[0]
    # endregion

    # region I_sense with amp forced s.t. int_avg_fit dS = E_avg_fit dS with dT from E_avg fit.
    amp_forced_fit_ds = i_fit_avg.best_values['amp'] * int_of_fit_dt_from_fit[-1] / e_fit_avg.best_values['dS']  # sf prop to 1/amp
    params = CU.edit_params(i_fit_avg.params, 'amp', amp_forced_fit_ds, vary=False)
    i_fit_ds = T.transition_fits(x, i_y_avg, params=[params], func=transition_fit_func)[0]
    # endregion

    # region Integrated E of data and best fit with dT from E_avg fit and amp s.t. int_avg_fit dS = E_avg_fit dS
    # dt_from_fit = e_fit_avg.best_values['dT'] * beta  # Calculated above
    sf_from_fit = E.scaling(dt_from_fit, amp_forced_fit_ds, dx)
    int_avg_sf_from_fit = E.integrate_entropy_1d(e_y_avg, sf_from_fit)
    int_of_fit_sf_from_fit = E.integrate_entropy_1d(e_fit_avg.best_fit, sf_from_fit)
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
            df = CU.fit_info_to_df(fits_isense, uncertainties=uncertainties, sf=3)
            PF.plot_df_table(df, title=f'I_sense_fit info for dat[{dat.datnum}]')
    # endregion

    # region Average I_sense plots
    if show_plots['i_sense_avg'] is True:
        fig, axs = plt.subplots(2, 2, figsize=(fig_size[0]*2, fig_size[1]))
        axs = axs.flatten()

        # region No params forced
        ax = axs[0]
        # PF.display_1d(x, i_y_avg, ax, scatter=True, label='Averaged data')
        PF.display_1d(x, i_y_avg, ax, scatter=True, label='i_y_avg')
        # ax.plot(x_i_fit_avg, i_fit_avg.best_fit, c='C3', label='Best fit')
        ax.plot(x_i_fit_avg, i_fit_avg.best_fit, c='C3', label='i_fit_avg.best_fit')
        PF.ax_setup(ax, f'Dat[{dat.datnum}]:Averaged data with fit', dat.Logs.x_label, 'I_sense /nA', legend=True)

        if show_plots['tables'] is True:
            df = CU.fit_info_to_df([i_fit_avg], uncertainties=uncertainties, sf=3)
            df.pop('index')
            PF.plot_df_table(df, title=f'Dat[{dat.datnum}]:I_sense fit values no additional forcing')
        # endregion

        # region Amplitude forced s.t. integrated = Ln2 with dT from DCbias
        ax = axs[1]
        # PF.display_1d(x, i_y_avg, ax, scatter=True, label='Averaged data')
        PF.display_1d(x, i_y_avg, ax, scatter=True, label='i_y_avg')
        # ax.plot(x_i_fit_avg, i_fit_ln2.best_fit, c='C3', label='Ln(2) amplitude fit')
        ax.plot(x_i_fit_avg, i_fit_ln2.best_fit, c='C3', label='i_fit_ln2.best_fit')
        PF.ax_setup(ax, f'Dat[{dat.datnum}]:Averaged I_sense data with\nwith amp forced s.t. int_dS = Ln(2)', dat.Logs.x_label, 'I_sense /nA', legend=True)
        PF.add_standard_fig_info(fig)

        if show_plots['tables'] is True:
            df = CU.fit_info_to_df([i_fit_ln2], uncertainties=uncertainties, sf=3)
            df.pop('index')
            PF.plot_df_table(df, title=f'Dat[{dat.datnum}]:I_sense fit values with amp forced s.t. int_dS = Ln(2)')
        # endregion

        # region Amplitude forced s.t. integrated fit dS = fit dS (with dT from fit)
        ax = axs[2]
        # PF.display_1d(x, i_y_avg, ax, scatter=True, label='Averaged data')
        PF.display_1d(x, i_y_avg, ax, scatter=True, label='i_y_avg')
        # ax.plot(x_i_fit_avg, i_fit_ds.best_fit, c='C3', label='amp s.t.\nint_fit dS = fit_dS')
        ax.plot(x_i_fit_avg, i_fit_ds.best_fit, c='C3', label='i_fit_ds.best_fit')
        PF.ax_setup(ax, f'Dat[{dat.datnum}]:Averaged I_sense data\nwith amp forced s.t. int_fit dS=fit dS\n(with dT from fit)', dat.Logs.x_label,
                    'I_sense /nA', legend=True)
        PF.add_standard_fig_info(fig)

        if show_plots['tables'] is True:
            df = CU.fit_info_to_df([i_fit_ds], uncertainties=uncertainties, sf=3)
            df.pop('index')
            PF.plot_df_table(df, title=f'Dat[{dat.datnum}]:I_sense fit values with amp forced s.t. int_fit dS=fit dS (with dT from fit)')
        # endregion
        PF.add_standard_fig_info(fig)
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
            df = CU.fit_info_to_df(fits_entr, uncertainties=uncertainties, sf=3)
            PF.plot_df_table(df, title=f'Entropy_R_fit info for dat[{dat.datnum}]')
        PF.add_standard_fig_info(fig)
    # endregion

    # region Average Entropy Plots
    if show_plots['avg_entr'] is True:
        fig, axs = plt.subplots(2,2,figsize=(fig_size[0]*2, fig_size[1]))
        axs = axs.flatten()

        # region No params forced
        ax = axs[0]
        # PF.display_1d(x, e_y_avg, ax, scatter=True, label='Averaged data')
        PF.display_1d(x, e_y_avg, ax, scatter=True, label='e_y_avg')
        # ax.plot(x_e_fit_avg, e_fit_avg.best_fit, c='C3', label='Best fit')
        ax.plot(x_e_fit_avg, e_fit_avg.best_fit, c='C3', label='e_fit_avg.best_fit')
        PF.ax_text(ax, f'dT={dt / beta * 1000:.3f}mK', loc=(0.02, 0.6))
        PF.ax_setup(ax, f'Dat[{dat.datnum}]:Averaged Entropy R data with fit', dat.Logs.x_label, 'Entropy R /nA', legend=True)

        if show_plots['tables'] is True:
            df = CU.fit_info_to_df([e_fit_avg], uncertainties=uncertainties, sf=3)
            df.pop('index')
            PF.plot_df_table(df, title=f'Dat[{dat.datnum}]:Entropy R fit values with no additional forcing')
        # endregion

        # region Forced to dS = Ln2
        ax = axs[1]
        # PF.display_1d(x, e_y_avg, ax, scatter=True, label='Averaged data')
        PF.display_1d(x, e_y_avg, ax, scatter=True, label='e_y_avg')
        # ax.plot(x_e_fit_avg, e_fit_ln2.best_fit, c='C3', label='Ln(2) fit')
        ax.plot(x_e_fit_avg, e_fit_ln2.best_fit, c='C3', label='e_fit_ln2.best_fit')
        PF.ax_text(ax, f'dT={dt / beta * 1000:.3f}mK', loc=(0.02, 0.6))
        PF.ax_setup(ax, f'Dat[{dat.datnum}]:Averaged Entropy R data with Ln(2) fit', dat.Logs.x_label, 'Entropy R /nA', legend=True)
        PF.add_standard_fig_info(fig)

        if show_plots['tables'] is True:
            df = CU.fit_info_to_df([e_fit_ln2], uncertainties=uncertainties, sf=3)
            df.pop('index')
            PF.plot_df_table(df, title=f'Dat[{dat.datnum}]:Entropy R fit values with dS forced to Ln2')
        # endregion

        # region Forced dT s.t. int_data dS = ln2
        ax = axs[2]
        # PF.display_1d(x, e_y_avg, ax, scatter=True, label='Averaged data')
        PF.display_1d(x, e_y_avg, ax, scatter=True, label='e_y_avg')
        # ax.plot(x_e_fit_avg, e_fit_dt_ln2.best_fit, c='C3', label='dT forced fit')
        ax.plot(x_e_fit_avg, e_fit_dt_ln2.best_fit, c='C3', label='e_fit_dt_ln2.best_fit')
        PF.ax_text(ax, f'dT={dt_ln2 / beta * 1000:.3f}mK', loc=(0.02, 0.6))
        PF.ax_setup(ax, f'Dat[{dat.datnum}]:Averaged Entropy R data\nwith dT forced s.t. int_data dS = Ln2', dat.Logs.x_label, 'Entropy R /nA',
                    legend=True)
        PF.add_standard_fig_info(fig)

        if show_plots['tables'] is True:
            df = CU.fit_info_to_df([e_fit_dt_ln2], uncertainties=uncertainties, sf=3)
            df.pop('index')
            PF.plot_df_table(df, title=f'Dat[{dat.datnum}]:Entropy R fit values\nwith dT forced s.t. int_data dS = Ln2')
        # endregion
        PF.add_standard_fig_info(fig)
    # endregion


    # region Integrated Entropy Plots
    if show_plots['int_ent'] is True:
        fig, axs = plt.subplots(2, 2, figsize=(fig_size[0]*2, fig_size[1]))
        axs = axs.flatten()

        # region dT from DCbias, amp from I_sense, also int of e_fit_avg
        ax = axs[0]
        # PF.display_1d(x, int_avg, ax, label='Averaged data')
        PF.display_1d(x, int_avg, ax, label='int_avg')
        # ax.plot(x_e_fit_avg, int_of_fit, c='C3', label='integrated best fit')
        ax.plot(x_e_fit_avg, int_of_fit, c='C3', label='int_of_fit')
        PF.ax_setup(ax, f'Dat[{dat.datnum}]:Integrated Entropy\ndT from DCbias for data and fit', dat.Logs.x_label, 'Entropy /kB')
        _add_ln3_ln2(ax)
        _add_peak_final_text(ax, int_avg, int_of_fit)
        ax.legend(loc='lower right')
        PF.ax_text(ax, f'dT = {dt/beta*1000:.3f}mK\n'
                       f'amp = {i_fit_avg.best_values["amp"]:.3f}nA\n'
                       f'int_avg dS={int_avg[-1]/np.log(2):.3f}kBLn2\n'
                       f'int_of_fit dS={int_of_fit[-1]/np.log(2):.3f}kBLn2',
                   loc=(0.02, 0.7), fontsize=8)

        # endregion

        # region dT adjusted s.t. integrated_data has dS = ln2, fit with that dt forced then integrated
        ax = axs[1]
        # PF.display_1d(x, int_avg_dt_ln2, ax, label='Averaged data')
        PF.display_1d(x, int_avg_dt_ln2, ax, label='int_avg_dt_ln2')
        # ax.plot(x_e_fit_avg, int_of_fit_dt_ln2, c='C3', label='integrated fit\nwith dT forced')
        ax.plot(x_e_fit_avg, int_of_fit_dt_ln2, c='C3', label='int_of_fit_dt_ln2')
        PF.ax_setup(ax, f'Dat[{dat.datnum}]:Integrated Entropy\ndT forced s.t. int_ds=Ln2', dat.Logs.x_label, 'Entropy /kB')
        _add_ln3_ln2(ax)
        _add_peak_final_text(ax, int_avg_dt_ln2, int_of_fit_dt_ln2)
        ax.legend(loc='lower right')
        PF.ax_text(ax, f'dT of forced fit={dt_ln2 / beta * 1000:.3f}mK\n'
                       f'amp = {i_fit_avg.best_values["amp"]:.3f}nA\n'
                       f'int_avg_dt_ln2 dS={int_avg_dt_ln2[-1]/np.log(2):.3f}kBLn2\n'
                       f'int_fit_dt_ln2 dS={int_of_fit_dt_ln2[-1]/np.log(2):.3f}kBLn2',
                   loc=(0.02, 0.7), fontsize=8)
        # endregion

        # region dT from Entropy fit, also integration of best fit
        ax = axs[2]
        # PF.display_1d(x, int_avg_dt_from_fit, ax, label='Averaged data')
        PF.display_1d(x, int_avg_dt_from_fit, ax, label='int_avg_dt_from_fit')
        # ax.plot(x_e_fit_avg, int_of_fit_dt_from_fit, c='C3', label='integrated fit\nwith dT from fit')
        ax.plot(x_e_fit_avg, int_of_fit_dt_from_fit, c='C3', label='int_of_fit_dt_from_fit')
        PF.ax_setup(ax, f'Dat[{dat.datnum}]:Integrated Entropy\ndT from entropy fit', dat.Logs.x_label, 'Entropy /kB')
        _add_ln3_ln2(ax)
        _add_peak_final_text(ax, int_avg_dt_from_fit, int_of_fit_dt_from_fit)
        ax.legend(loc='lower right')
        PF.ax_text(ax, f'dT = {dt_from_fit/beta*1000:.3f}mK\n'
                       f'amp = {i_fit_avg.best_values["amp"]:.3f}nA\n'
                       f'int_avg_dt_from_fit dS=\n{int_avg_dt_from_fit[-1]/np.log(2):.3f}kBLn2\n'
                       f'int_of_fit_dt_fit dS=\n{int_of_fit_dt_from_fit[-1]/np.log(2):.3f}kBLn2',
                   loc=(0.02, 0.6), fontsize=8)
        # endregion

        # region dT from fit, amp s.t. int_fit dS = fit dS (scaling from fit)
        ax = axs[3]
        # PF.display_1d(x, int_avg_sf_from_fit, ax, label='Averaged data')
        PF.display_1d(x, int_avg_sf_from_fit, ax, label='int_avg_sf_from_fit')
        # ax.plot(x_e_fit_avg, int_of_fit_sf_from_fit, c='C3', label='integrated fit\nscaling from fit')
        ax.plot(x_e_fit_avg, int_of_fit_sf_from_fit, c='C3', label='int_of_fit_sf_from_fit')
        PF.ax_setup(ax, f'Dat[{dat.datnum}]:Integrated Entropy\nscaling from fit (dT from fit\namp s.t. int_fit dS = fit dS)', dat.Logs.x_label, 'Entropy /kB')
        ax.legend(loc='lower right')
        _add_ln3_ln2(ax)
        _add_peak_final_text(ax, int_avg_sf_from_fit, int_of_fit_sf_from_fit)
        PF.ax_text(ax, f'dT = {dt_from_fit / beta * 1000:.3f}mK\n'
                       f'amp = {amp_forced_fit_ds:.3f}nA\n'
                       f'int_avg_sf_from_fit dS=\n{int_avg_sf_from_fit[-1]/np.log(2):.3f}kBLn2\n'
                       f'int_of_fit_sf_from_fit dS=\n{int_of_fit_sf_from_fit[-1]/np.log(2):.3f}kBLn2',
                   loc=(0.02, 0.6), fontsize=8)
        # endregion
        PF.add_standard_fig_info(fig)

    # endregion


