from src.DatBuilder.InDepthData import InDepthData, get_exp_df
from src.Scripts.StandardImports import *
import os
import scipy.io as sio
import src.DatBuilder.InDepthData as IDD


def _recalculate_dats(datdf: DF.DatDF, datnums: list, datname='base', dattypes: set = None, setupdf=None, config=None,
                      transition_func=None, save=True):
    """
    Just a quick fn to recalculate and save all dats to given datDF
    """
    # if datdf.config_name != cfg.current_config.__name__.split('.')[-1]:
    #     print('WARNING[_recalculate_given_dats]: Need to change config while running this. No dats changed')
    #     return
    if transition_func is not None:
        dattypes = CU.ensure_set(dattypes)
        dattypes.add(
            'suppress_auto_calculate')  # So don't do pointless calculation of transition when first initializing
    for datnum in datnums:
        dat = make_dat_standard(datnum, datname=datname, datdf=datdf, dfoption='overwrite', dattypes=dattypes,
                                setupdf=setupdf, config=config)
        if transition_func is not None:
            dat._reset_transition(fit_function=transition_func)
            if 'entropy' in dat.dattype:
                dat._reset_entropy()
        datdf.update_dat(dat, yes_to_all=True)
    if save is True:
        datdf.save()


def _plot_rows_of_data(idd, row_range=(0, 5)):
    """For looking at bump in charge sensor data, looking at average of different chunks
    of rows to see if the bump is always there"""
    ax = plt.gca()
    ax.cla()
    cut = row_range
    ys = idd.y_isense[cut[0]:cut[1]]
    x = idd.x - np.average(idd.x)
    y = CU.average_data(ys, [fit.best_values['mid'] for fit in idd.i_fits[cut[0]:cut[1]]])[0].astype(np.float32)
    y = y - idd.i_avg_fit.quad * x ** 2 - idd.i_avg_fit.lin * x - idd.i_avg_fit.const
    pars = idd.i_avg_fit.params
    pars = CU.edit_params(pars, ['const', 'lin', 'quad', 'mid'], [0, 0, 0, 0], [True, True, True, True])
    fit = idd.i_avg_fit.fit.model.fit(y, params=pars, x=x, nan_policy='omit')
    ax.scatter(x, y, s=1)
    ax.plot(fit.userkws['x'], fit.best_fit, color='C3', label='Fit')
    PF.add_scatter_label(f'{cut[0]} to {cut[1]}')
    ax.legend()
    ax.legend().set_title('Data rows')
    PF.ax_setup(ax, f'Dat{idd.datnum}: I_sense poly subtracted\nLooking for bump on right side', 'Gate /mV',
                'Current (offset) /nA')
    plt.tight_layout()


def _plot_entropy_vs_gamma(IDDs, fig_title='Jan20 Entropy vs Gamma', gate_fn=(lambda x: getattr(x, 'fdacs')[4])):
    """
    4 axes: a few i_avg_fit.data/fit, a few e_avg_fit.integrated, entropy vs gamma, entropy vs gate where gate obtained
    applying gate_fn to dat.Logs

    @param IDDs:
    @type IDDs: list[src.DatAttributes.InDepthData.InDepthData]
    @param fig_title:
    @type fig_title: str
    @param gate_fn: function to apply to dat.Logs to get the value of gate responsible for coupling
    (lambda x: getattr(x, 'dacs')[13]) for sep19
    @type gate_fn: func
    @return: None
    @rtype: None
    """

    fig, axs = PF.make_axes(4)
    fig.suptitle(fig_title)
    ax = axs[0]
    for idd in IDDs:
        idd.Plot.plot_avg_i(idd, ax, True, True, True, True)
    PF.ax_setup(ax, 'Avg i_sense', 'Plunger /mV', 'Current /nA', legend=True)
    ax.legend().set_title('Dat')
    ax = axs[1]
    for idd in IDDs:
        idd.Plot.plot_int_e(idd, ax)
    PF.ax_setup(ax, 'Integrated_entropy', 'Plunger /mV', 'Entropy /kB', legend=True)
    ax.legend().set_title('Dat')
    axs[0].legend().set_title('Dat')
    ax = axs[2]
    xs = [idd.i_avg_fit.g for idd in IDDs]
    ys = [idd.e_avg_fit.integrated[-1] for idd in IDDs]
    ax.scatter(xs, ys, s=3)
    for x, y, idd in zip(xs, ys, IDDs):
        ax.text(x, y, f'{idd.datnum}', fontsize=6)

    PF.ax_setup(ax, 'Integrated entropy vs Gamma', 'Gamma /mV', 'Entropy /kB')
    ax = axs[3]
    xs = [gate_fn(idd.setup_meta.dat.Logs) for idd in IDDs]
    ys = [idd.e_avg_fit.integrated[-1] for idd in IDDs]
    ax.scatter(xs, ys, s=3)
    for x, y, idd in zip(xs, ys, IDDs):
        ax.text(x, y, f'{idd.datnum}', fontsize=6)

    PF.ax_setup(ax, 'Integrated entropy vs Coupling gate', 'Coupling Gate /mV', 'Entropy /kB')
    PF.add_standard_fig_info(fig)


def _i_sense_data_to_yigal(IDDs, show=True, save_to_file=False):
    if show is True:
        fig, axs = PF.make_axes(num=len(IDDs), single_fig_size=IDDs[0].fig_size,
                                plt_kwargs={'sharex': False, 'sharey': True})
    else:
        axs = np.zeros(len(IDDs))

    for idd, ax in zip(IDDs, axs):
        x = idd.x - idd.i_avg_fit.mid
        y = idd.i_avg
        x, y = map(np.array, zip(*([[x1, y1] for x1, y1 in zip(x, y) if not np.isnan(y1)])))
        subber = lambda x, y: y - idd.i_avg_fit.quad * x ** 2 - idd.i_avg_fit.lin * x - idd.i_avg_fit.const
        y = subber(x, y)
        x_fit = idd.i_avg_fit.x - idd.i_avg_fit.mid
        y_fit = subber(x_fit, idd.i_avg_fit.best_fit)

        if show is True:
            PF.display_1d(x, y, ax=ax, scatter=True)
            ax.plot(x_fit, y_fit, label='fit', color='C3')
            PF.ax_setup(ax, f'Dat[{idd.datnum}]: I_avg minus polynomial terms', 'Gate /mV', 'Current (offset to 0) /nA',
                        legend=True)
            PF.ax_text(ax, f'amp={idd.i_avg_fit.amp:.3f}nA\n'
                           f'theta={idd.i_avg_fit.theta:.3f}mV\n'
                           f'gamma={idd.i_avg_fit.g:.3f}mV', loc=(0.02, 0.05))
            # plt.tight_layout(rect=[0, 0.05, 1, 1])

        if save_to_file is True:
            datapath = os.path.normpath(
                r'D:\OneDrive\UBC LAB\My work\My_Papers\2Resources\Equations_and_Graphs\Yigal\i_sense_data_to_send')
            data = np.array([x, y])
            filepath = os.path.join(datapath, f'dat[{idd.datnum}]')
            sio.savemat(filepath + '.mat', {'x': x, 'i_sense': y})
            np.savetxt(filepath + '.csv', data, delimiter=',')


def _get_fit_params(IDDs):
    idd  # type: IDD.InDepthData
    fit_dfs = IDD.compare_IDDs(IDDs)
    i_df = fit_dfs.i_df_text
    e_df = fit_dfs.e_df_text
    # i_df = CU.fit_info_to_df([idd.i_avg_fit.fit for idd in IDDs], uncertainties=True, index=[f'{idd.setup_meta.datdf.config_name[0:5]}[{idd.datnum}]' for idd in IDDs])
    # e_df = CU.fit_info_to_df([idd.e_avg_fit.fit for idd in IDDs], uncertainties=True, index=[f'{idd.setup_meta.datdf.config_name[0:5]}[{idd.datnum}]' for idd in IDDs])
    PF.plot_df_table(i_df, 'I_sense Fit Info for Sep19 and Jan20')
    PF.plot_df_table(e_df, 'Entropy Fit Info for Sep19 and Jan20')
    print(i_df)
    print(e_df)
    return fit_dfs


sep_datdf = get_exp_df('sep19')

mar_datdf = get_exp_df('mar19')

jan_datdf = get_exp_df('jan20')

if __name__ == '__main__':
    plots = {
        'i_sense': True,  # waterfall w/fit
        'i_sense_raw': False,  # waterfall raw
        'i_sense_avg': True,  # averaged_i_sense
        'i_sense_avg_others': False,  # forced fits
        'entr': False,  # waterfall w/fit
        'entr_raw': False,  # waterfall raw
        'avg_entr': False,  # averaged_entr
        'avg_entr_others': False,  # forced fits
        'int_ent': False,  # integrated_entr
        'int_ent_others': False,  # forced fits
        'tables': True  # Fit info tables
    }
    run_mar = False
    run_sep = True
    run_jan1 = False
    run_jan2 = False

    if run_mar is True:
        m_datnums = InDepthData.get_datnums('mar19_gamma_entropy')
        m_IDDs = [
            InDepthData(num, plots_to_show=plots, set_name='mar19_gamma_entropy', run_fits=False, show_plots=False)
            for
            num in m_datnums]
        m_idd_dict = dict(zip(m_datnums, m_IDDs))
        e_params = m_IDDs[0].setup_meta.dat.Entropy.avg_params
        e_params = CU.edit_params(e_params, 'const', 0, False)
        for idd in m_IDDs:
            idd.run_all_fits(e_params=e_params)

    if run_sep is True:
        # f = InDepthData(1563, plots_to_show=plots, set_name='Jan20_gamma_2', run_fits=True, config=None)

        # dc = C.DatHandler.get_dat(1947, 'base', sep_datdf, config=Sep19Config)
        # dat = C.DatHandler.get_dat(2713, 'base', sep_datdf, config=Sep19Config)
        # dc.DCbias.plot_self(dc, dat)
        s_datnums = InDepthData.get_datnums('sep19_gamma')
        s_IDDs = [InDepthData(num, plots_to_show=plots, set_name='Sep19_gamma', run_fits=False, show_plots=False) for
                  num in
                  s_datnums[0:8]]
        s_idd_dict = dict(zip(s_datnums[0:8], s_IDDs))
        e_params = s_IDDs[0].setup_meta.dat.Entropy.avg_params
        cols = ['datnum', 'offset']
        data = [[]]
        for f in s_IDDs:
            # f.plot_integrated(avg=True, others=False)
            # i_params = f.i_fits[0].params
            # i_params = CU.edit_params(i_params, 'g', 0, False)
            f.fit_isenses()
            i_params = [fit.params for fit in f.i_fits]
            i_params = [CU.edit_params(param, 'theta', 27.3, False) for param in i_params]
            f.fit_isenses(params=i_params)
            f.make_averages()
            # offset = np.nanmean([f.e_avg[CU.get_data_index(f.x, -2000):CU.get_data_index(f.x, -1500)], f.e_avg[CU.get_data_index(f.x, 1500):CU.get_data_index(f.x, 2100)]] )
            # f.y_entr = f.y_entr - offset
            e_params = CU.edit_params(e_params, 'const', 0, False)
            # data.append([f.datnum, offset])
            #
            # f.run_all_fits(i_params=i_params, e_params=e_params)
            f.run_all_fits(i_params=None, e_params=e_params)
            # f.plot_integrated(avg=True, others=False)
            # f.plot_all_plots()

        # df = pd.DataFrame(data, columns=cols)
        # print(df)
        # PF.plot_df_table(df, sig_fig=4)

    if run_jan1 is True:
        j1_datnums = InDepthData.get_datnums('jan20_gamma')
        j1_IDDs = [InDepthData(num, plots, set_name='jan20_gamma', run_fits=False, show_plots=False) for num in
                   j1_datnums]
        j1_idd_dict = dict(zip(j1_datnums, j1_IDDs))
        e_params = j1_IDDs[0].setup_meta.dat.Entropy.avg_params
        for idd in j1_IDDs:
            idd.fit_isenses()
            i_params = idd.i_fits[0].params
            if idd.datnum not in [1492, 1495]:
                i_params = CU.edit_params(i_params, 'theta', 0.9765, False)
            else:
                i_params = CU.edit_params(i_params, 'g', 0, False)
            e_params = CU.edit_params(e_params, 'const', 0, False)
            e_params = CU.edit_params(e_params, 'mid', idd.i_fits[0].best_values['mid'])
            idd.run_all_fits(i_params=i_params, e_params=e_params)
        # Per IDD fixes.

    if run_jan2 is True:
        j2_datnums = InDepthData.get_datnums('jan20_gamma_2')
        j2_IDDs = [InDepthData(num, plots, set_name='jan20_gamma_2', run_fits=False, show_plots=False) for num in
                   j2_datnums]
        j2_idd_dict = dict(zip(j2_datnums, j2_IDDs))
        e_params = j2_IDDs[0].setup_meta.dat.Entropy.avg_params
        for idd in j2_IDDs:
            idd.fit_isenses()
            i_params = idd.i_fits[0].params
            if idd.datnum not in [1533, 1536]:
                i_params = CU.edit_params(i_params, 'theta', 0.976, False)
            else:
                i_params = CU.edit_params(i_params, 'g', 0, False)
            e_params = CU.edit_params(e_params, 'const', 0, False)
            e_params = CU.edit_params(e_params, 'mid', idd.i_fits[0].best_values['mid'])
            idd.run_all_fits(i_params=i_params, e_params=e_params)
