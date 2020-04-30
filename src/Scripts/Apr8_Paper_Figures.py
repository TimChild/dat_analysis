from src.CoreUtil import fit_info_to_df, switch_config_decorator_maker, wrapped_call
from src.Scripts.StandardImports import *

import src.DatCode.Transition as T
import src.DatCode.Entropy as E


def _recalculate_given_dats(datdf: DF.DatDF, make_dat_function, datnums: list, dattypes: set = None, save=True):
    """
    Just a quick fn to recalculate and save all dats to given datDF
    """
    # if datdf.config_name != cfg.current_config.__name__.split('.')[-1]:
    #     print('WARNING[_recalculate_given_dats]: Need to change config while running this. No dats changed')
    #     return
    for datnum in datnums:
        dat = make_dat_function(datnum, dfname=datdf.name, dfoption='overwrite', dattypes=dattypes)
        datdf.update_dat(dat, yes_to_all=True)
    if save is True:
        datdf.save()


def _recalculate_isens_digamma_constsubtracted(ent_datnums, dc_datnums, datdf):
    """Needs to be wrapped in config changer for make_dats.
    Only uses dc_datnums[0] for calculating int_entropy"""
    from src.DatCode.Transition import i_sense_digamma
    _recalculate_given_dats(datdf, ent_datnums, dattypes={'transition', 'entropy'}, save=False)
    _recalculate_given_dats(datdf, dc_datnums, dattypes={'transition', 'dcbias'})
    dats = make_dats(ent_datnums, dfname=datdf.name, dfoption='load')
    for dat in dats:
        dat.datname = 'digamma'
        dat.Transition.recalculate_fits(func=i_sense_digamma)
        DF.update_save(dat, update=True, save=False, datdf=datdf)
    datdf.save()

    dc = make_dat_standard(dc_datnums[0], dfoption='load', dfname=datdf.name)
    for dat in dats:
        E.recalculate_int_entropy_with_offset_subtracted(dat, dc, make_new=True, update=True, save=False,
                                                         dfname=datdf.name)
    datdf.save()


def _calculate_mar_int_entropy(mar_dat, mdcs=None, dfoption='load'):
    """Calculates integrated entropy for march dats where dcbias was done with repeat measurements for a few DC biases
    If no mdcs are passed in, they will be loaded from mar19_dcdats and optionally recalculated ('overwrite')"""
    import lmfit
    global m_result, mar19_datdf, m_xs, m_ys, m_i_heat, m_amplitude
    if mdcs is None:
        mdcs = [mar19_make_dat(num, dfname='Apr20', dattypes={'transition'}, dfoption=dfoption) for num in mar19_dcdats]
        if dfoption == 'overwrite':
            for dat in mdcs:
                mar19_datdf.update_dat(dat, yes_to_all=True)
            print(f'mar19_datdf updated but not saved!')
    m_xs = [dat.Logs.dacs[0] / 10 for dat in mdcs]
    m_ys = [dat.Transition.theta for dat in mdcs]
    quad = lmfit.models.QuadraticModel()
    m_result = quad.fit(m_ys, x=m_xs)
    m_i_heat = mar_dat.Instruments.srs1.out / 50 * np.sqrt(2)
    mx = np.average([m_result.eval(x=m_i_heat), m_result.eval(x=-m_i_heat)])
    mn = np.nanmin(m_result.eval(x=np.linspace(-5, 5, 1000)))
    dt = (mx - mn) / 2
    m_amplitude = mar_dat.Transition.amp
    mar_dat.Entropy.init_integrated_entropy_average(dT_mV=dt, amplitude=m_amplitude)
    return m_i_heat


def _recalculate_mar_sep(mar_recalculate=True, sep_recalculate=True, sep_dats=1):
    """
    Recalculates and saves ['base'] dats for all Mar19 and Sep19 entropy/dc dats to 'Apr20' df.
    Then calculates and saves ['digamma'] dats.
    Also calculates integrated_entropy for ['digamma'] dats.

    @param mar_recalculate: Whether to recalculate and save mar19 dats
    @type mar_recalculate: bool
    @param sep_recalculate: Whether to recalculate and save sep19 dats
    @type sep_recalculate: bool
    """
    from src.DatCode.Transition import i_sense_digamma

    if mar_recalculate is True:
        _recalculate_given_dats(mar19_datdf, mar19_make_dat, mar19_dcdats, dattypes={'transition'}, save=False)
        _recalculate_given_dats(mar19_datdf, mar19_make_dat, mar19_dcdats2, dattypes={'transition'}, save=False)
        _recalculate_given_dats(mar19_datdf, mar19_make_dat, mar19_ent_datnums, dattypes={'transition', 'entropy'},
                                save=False)

        mdats = [mar19_make_dat(num, dfname='Apr20', dfoption='load') for num in mar19_ent_datnums]
        mdcs = [mar19_make_dat(num, dfname='Apr20', dattypes={'transition'}, dfoption='load') for num in mar19_dcdats]

        for dat in mdats:
            dat.datname = 'digamma'
            dat.Transition.recalculate_fits(func=i_sense_digamma)
            _calculate_mar_int_entropy(dat, mdcs=mdcs)
            DF.update_save(dat, update=True, save=False, datdf=mar19_datdf)
        mar19_datdf.save()

    if sep_recalculate is True:
        if sep_dats == 1:
            sdat_nums, sdc_nums = sep19_ent_datnums, sep19_dcdat
        elif sep_dats == 2:
            sdat_nums, sdc_nums = sep19_ent_datnums2, sep19_dcdat2
        else:
            raise ValueError('Choose 1 or 2 for sep_dats according to which set of dats to recalculate')
        _recalculate_given_dats(sep19_datdf, sep19_make_dat, sdat_nums, dattypes={'entropy', 'transition'},
                                save=False)
        _recalculate_given_dats(sep19_datdf, sep19_make_dat, sdc_nums, dattypes={'transition', 'dcbias'}, save=False)

        sdats = [sep19_make_dat(num, dfname='Apr20', dfoption='load') for num in sdat_nums]
        sdc = sep19_make_dat(sdc_nums[0], dfname='Apr20', dfoption='load')
        for dat in sdats:
            dat.datname = 'digamma'
            dat.Transition.recalculate_fits(func=i_sense_digamma)
            dt = sdc.DCbias.get_dt_at_current(dat.Instruments.srs1.out / 50 * np.sqrt(2))
            dat.Entropy.init_integrated_entropy_average(dT_mV=dt, amplitude=dat.Transition.amp, dcdat=sdc)
            DF.update_save(dat, update=True, save=False, datdf=sep19_datdf)
        sep19_datdf.save()


def _plot_all_raw_entropy_things_22nd_apr():
    """two sets of plots, one showing raw entropy x and y signal and the cumulative sums for Mar19, Sep19, Jan20
    other showing scaled integrated entropy, raw and scaled entropy signal, scaled charge sensor signal"""
    fig, axs = PF.make_axes(9)

    jan_dfoption = 'load'
    mar_dfoption = 'load'
    sep_dfoption = 'load'

    datdf = DF.DatDF(dfname='Apr20')
    dat = make_dat_standard(jan20_ent_datnums[0], datname='digamma', dfname='Apr20', dfoption=jan_dfoption)
    dc = make_dat_standard(jan20_dcdat[0], dfname='Apr20', dfoption=jan_dfoption)
    i_heat = dat.Instruments.srs1.out / 50 * np.sqrt(2)
    dt = dc.DCbias.get_dt_at_current(i_heat)
    amp = dat.Transition.amp
    dat.Entropy.init_integrated_entropy_average(dT_mV=dt, amplitude=amp, dcdat=dc)
    if jan_dfoption == 'overwrite':
        datdf.update_dat(dat, yes_to_all=True)
        datdf.update_dat(dc, yes_to_all=True)
        datdf.save()
    dt = dat.Entropy._int_dt

    axs[6].plot(dat.Entropy.x_array, dat.Entropy.integrated_entropy)
    PF.ax_text(axs[6], f'dat[{dat.datnum}]\n'
                       f'dt = {dt:.3f}mV\n'
                       f'amp = {amp:.3f}nA\n'
                       f'i_heat = {i_heat:.2f}nA',
               loc=(0.35, 0.05))
    axs[6].set_title(f'Entropy for Jan20_dat[{dat.datnum}]')
    axs[6].legend()

    axs[7].plot(dat.Entropy.x_array, dat.Entropy._data_average, label=f'dat[{dat.datnum}] avg', color='C3')
    axs[7].legend(loc='upper left')
    twax1 = axs[7].twinx()
    twax1.plot(dat.Data.x_array, dat.Data.ADC2_2d[0], label=f'dat[{dat.datnum}] raw')
    twax1.legend(loc='lower right')

    axs[8].plot(dat.Transition._x_array, dat.Transition._avg_data, label=f'dat[{dat.datnum}] i_sense_avg')
    PF.ax_text(axs[8], f'Amplitude = {dat.Transition.amp:.3f}nA')

    mdat = mar19_make_dat(mar19_ent_datnums[0], dfname='Apr20', dattypes=['entropy', 'transition'],
                          dfoption=mar_dfoption)
    mdc = mar19_make_dat(mar19_dcdats[0], dfname='Apr20', dattypes=['transition'], dfoption=mar_dfoption)

    mi_heat = _calculate_mar_int_entropy(mdat, dfoption=mar_dfoption)
    mdt = mdat.Entropy._int_dt
    mamp = mdat.Transition.amp
    axs[0].plot(mdat.Entropy.x_array, mdat.Entropy.integrated_entropy)
    PF.ax_text(axs[0], f'mdat[{mdat.datnum}]\n'
                       f'dt = {mdt:.3f}mV\n'
                       f'amp = {mamp:.3f}nA\n'
                       f'i_heat = {mi_heat:.2f}nA',
               loc=(0.35, 0.05))
    axs[0].set_title(f'Entropy for March_dat[{mdat.datnum}]')
    axs[0].legend()

    axs[1].plot(mdat.Entropy.x_array, mdat.Entropy._data_average, label=f'mdat[{mdat.datnum}] avg', color='C3')
    axs[1].legend(loc='upper left')
    twax2 = axs[1].twinx()
    twax2.plot(mdat.Data.x_array, mdat.Data.g2x2d[0], label=f'mdat[{mdat.datnum}] raw')
    twax2.legend(loc='lower right')

    axs[2].plot(mdat.Transition._x_array, mdat.Transition._avg_data, label=f'mdat[{mdat.datnum}] i_sense_avg')
    PF.ax_text(axs[2], f'Amplitude = {mdat.Transition.amp:.3f}nA')

    if mar_dfoption == 'overwrite':
        mar19_datdf.update_dat(mdat, yes_to_all=True)
        mar19_datdf.update_dat(mdc, yes_to_all=True)
        mar19_datdf.save()

    sdat = sep19_make_dat(sep19_ent_datnums[0], dfname='Apr20', dattypes=['entropy', 'transition'],
                          dfoption=sep_dfoption)
    sdc = sep19_make_dat(sep19_dcdat[0], dfname='Apr20', dattypes=['transition', 'dcbias'], dfoption=sep_dfoption)
    si_heat = sdat.Instruments.srs1.out / 50 * np.sqrt(2)
    sdt = sdc.DCbias.get_dt_at_current(si_heat)
    samp = sdat.Transition.amp
    sdat.Entropy.init_integrated_entropy_average(dT_mV=sdt, amplitude=samp)
    sdt = sdat.Entropy._int_dt
    if sep_dfoption == 'overwrite':
        sep19_datdf.update_dat(sdat, yes_to_all=True)
        sep19_datdf.update_dat(sdc, yes_to_all=True)
        sep19_datdf.save()
    axs[3].plot(sdat.Entropy.x_array, sdat.Entropy.integrated_entropy)
    PF.ax_text(axs[3], f'sdat[{sdat.datnum}]\n'
                       f'dt = {sdt:.3f}mV\n'
                       f'amp = {samp:.3f}nA\n'
                       f'i_heat = {si_heat:.2f}nA',
               loc=(0.35, 0.05))
    axs[3].set_title(f'Entropy for Sep_dat[{sdat.datnum}]')
    axs[3].legend()
    axs[4].plot(sdat.Entropy.x_array, sdat.Entropy._data_average, label=f'sdat[{sdat.datnum}] avg', color='C3')
    axs[4].legend(loc='upper left')
    twax3 = axs[4].twinx()
    twax3.plot(sdat.Data.x_array, sdat.Data.FastScanCh1_2D[0], label=f'sdat[{sdat.datnum}] raw')
    twax3.legend(loc='lower right')

    axs[5].plot(sdat.Transition._x_array, sdat.Transition._avg_data, label=f'sdat[{sdat.datnum}] i_sense_avg')
    PF.ax_text(axs[5], f'Amplitude = {samp:.3f}nA')

    for a in [twax1, twax2, twax3]:
        a.set_ylabel('raw_signal')
    for a in [axs[1], axs[4], axs[7]]:
        a.set_ylabel('scaled_ent_signal')
    for a in [axs[2], axs[5], axs[8]]:
        a.set_ylabel('I_sens /nA')
        a.set_title('Charge sensor')
        a.set_xlabel('Plunger gate /mV')

    PF.add_standard_fig_info(fig)

    fig, axs = PF.make_axes(3)
    axs[0].set_title(f'Mar19_dat{mdat.datnum}')
    axs[0].plot(mdat.Data.x_array, mdat.Data.g2x2d[0], label='g2x2d[0]')
    axs[0].plot(mdat.Data.x_array, mdat.Data.g2y2d[0], label='g2y2d[0]')
    axs[0].legend(loc='upper left')
    axs0 = axs[0].twinx()
    axs0.plot(mdat.Data.x_array, np.nancumsum(mdat.Data.g2x2d[0]), label='sum_g2x2d[0]', color='C3')
    axs0.legend(loc='lower right')

    axs[1].set_title(f'Sep19_dat{sdat.datnum}')
    axs[1].plot(sdat.Data.x_array, sdat.Data.FastScanCh1_2D[0], label='FastScanCh1_2D[0]')
    axs[1].plot(sdat.Data.x_array, sdat.Data.FastScanCh2_2D[0], label='FastScanCh2_2D[0]')
    axs[1].legend(loc='upper left')
    axs1 = axs[1].twinx()
    axs1.plot(sdat.Data.x_array, np.nancumsum(sdat.Data.FastScanCh1_2D[0]), label='sum_FastScanCh1_2D[0]', color='C3')
    axs1.legend(loc='lower right')

    axs[2].set_title(f'Jan20_dat{dat.datnum}')
    axs[2].plot(dat.Data.x_array, dat.Data.entropy_x_2d[0], label='entropy_x_2d[0]')
    axs[2].plot(dat.Data.x_array, dat.Data.entropy_y_2d[0], label='entropy_y_2d[0]')
    axs[2].legend(loc='upper left')
    axs2 = axs[2].twinx()
    axs2.plot(dat.Data.x_array, np.nancumsum(dat.Data.entropy_y_2d[0]), label='sum_entropy_y_2d[0]', color='C3')
    axs2.legend(loc='lower right')

    for a in axs:
        a.set_xlabel('Plunger gate /mV')
        a.set_ylabel('Raw data (no units)')
    for a in [axs0, axs1, axs2]:
        a.set_ylabel('Cumulative sum of larger component (no units)')
    PF.add_standard_fig_info(fig)


def _plot_int_entropy_for_all(x_axis='gate'):
    """
    Plots scaled integrated_entropy for all experiments either against coupling gate or gamma

    @param x_axis: 'gate' or 'gamma' for whether to plot against coupling gate or against gamma broadening
    @type x_axis: str
    """
    if x_axis not in ['gate', 'gamma']:
        print(f'WARNING[_plot_int_entropy_for_all]: x_axis must be either "gate" or "gamma"')
        return None
    mdats = [mar19_make_dat(num, datname='digamma', dfname='Apr20', dfoption='load') for num in mar19_ent_datnums]
    sdats = [sep19_make_dat(num, datname='digamma', dfname='Apr20', dfoption='load') for num in sep19_ent_datnums]
    sdats2 = [sep19_make_dat(num, datname='digamma', dfname='Apr20', dfoption='load') for num in sep19_ent_datnums2]
    dats = make_dats(jan20_ent_datnums, datname='digamma', dfoption='load', dfname='Apr20')

    fig, axs = PF.make_axes(4)

    # Mar19 part
    ax = axs[0]
    if x_axis == 'gate':
        xs = [dat.Logs.dacs[13] for dat in mdats]
        PF.ax_setup(ax, 'Int_entropy_vs_Gate for Mar19', 'Coupling Gate /mV', 'Entropy /kB')
    elif x_axis == 'gamma':
        xs = [dat.Transition.g for dat in mdats]
        PF.ax_setup(ax, 'Int_entropy_vs_Gamma for Mar19', 'Gamma /mV', 'Entropy /kB')
    ys = [dat.Entropy.int_ds for dat in mdats]
    ax.scatter(xs, ys)

    # Sep19 part
    # Sep_1s
    ax = axs[1]
    for dat in sdats:
        if dat.datnum != 2229:
            if x_axis == 'gate':
                xs = dat.Logs.y_array
            elif x_axis == 'gamma':
                xs = dat.Transition.fit_values.gs
            ys = [line[-1] for line in dat.Entropy.int_entropy_per_line]
            ax.scatter(xs, ys, label=f'dat[{dat.datnum}]')
    if x_axis == 'gate':
        PF.ax_setup(ax, 'Int_entropy_vs_Gate for Sep19', 'Coupling Gate /mV', 'Entropy /kB', legend=True)
    elif x_axis == 'gamma':
        PF.ax_setup(ax, 'Int_entropy_vs_Gamma for Sep19', 'Gamma /mV', 'Entropy /kB', legend=True)

    # Sep_2s
    ax = axs[2]
    if x_axis == 'gate':
        xs = [dat.Logs.dacs[13] for dat in sdats2]
        PF.ax_setup(ax, 'Int_entropy_vs_Gate for Sep19', 'Coupling Gate /mV', 'Entropy /kB')
    elif x_axis == 'gamma':
        xs = [dat.Transition.g for dat in sdats2]
        PF.ax_setup(ax, 'Int_entropy_vs_Gamma for Sep19', 'Gamma /mV', 'Entropy /kB')
    ys = [dat.Entropy.int_ds for dat in sdats2]
    ax.scatter(xs, ys)

    # Jan20 part
    ax = axs[3]
    if x_axis == 'gate':
        xs = [dat.Logs.fdacs[4] for dat in dats]
        PF.ax_setup(ax, 'Int_entropy_vs_Gate for Jan20', 'Coupling Gate /mV', 'Entropy /kB')
    elif x_axis == 'gamma':
        xs = [dat.Transition.g for dat in dats]
        PF.ax_setup(ax, 'Int_entropy_vs_Gamma for Jan20', 'Gamma /mV', 'Entropy /kB')
    ys = [dat.Entropy.int_ds for dat in dats]
    ax.scatter(xs, ys)

    PF.add_standard_fig_info(fig)


def _plot_SRS_DCbias_Isens_for_all():
    """
    For looking at how parts of the integrated entropy are calculated. Shows SRS info, plots first row of i_sense for
    entropy data with fits_isense, plots DCbias data with markings where dT is calculated from. Does not show int_entropy
    """
    mdcs = [mar19_make_dat(num, datname='base', dfname='Apr20', dfoption='load') for num in mar19_dcdats]
    sdc = sep19_make_dat(sep19_dcdat[0], datname='base', dfname='Apr20', dfoption='load')
    sdc2 = sep19_make_dat(sep19_dcdat2[0], datname='base', dfname='Apr20', dfoption='load')
    jdc = make_dat_standard(jan20_dcdat[0], datname='base', dfname='Apr20', dfoption='load')

    mdat = mar19_make_dat(mar19_ent_datnums[0], datname='digamma', dfname='Apr20', dfoption='load')
    sdat = sep19_make_dat(sep19_ent_datnums[0], datname='digamma', dfname='Apr20', dfoption='load')
    sdat2 = sep19_make_dat(sep19_ent_datnums2[0], datname='digamma', dfname='Apr20', dfoption='load')
    jdat = make_dat_standard(jan20_ent_datnums[0], datname='digamma', dfname='Apr20', dfoption='load')
    all_ent_dat = [jdat, mdat, sdat, sdat2]

    show_metadata = True
    show_DCbias = True
    show_i_sense = True

    # Metadata
    if show_metadata is True:
        for dat in all_ent_dat:
            print_info.srss(dat, graphic=True)

    # DC bias
    if show_DCbias is True:
        for dc, dat in zip([jdc, sdc, sdc2], [jdat, sdat, sdat2]):
            fig, axs = dc.DCbias.plot_self(dc)
            i_heat_ac = dat.Instruments.srs1.out / 50 * np.sqrt(2)
            fit = dc.DCbias.full_fit
            y_min, _ = axs[1].get_ylim()
            x_min, x_max = axs[1].get_xlim()
            fit_min = np.nanmin(fit.eval(x=np.linspace(-10, 10, 10000)))
            dt = dat.Entropy._int_dt
            axs[1].margins(x=0, y=0)
            for i_heat in [i_heat_ac, -i_heat_ac]:
                y_val = fit.eval(x=i_heat)
                axs[1].plot([i_heat, i_heat], [y_min, y_val], color='k', linestyle=':')  # Vertical lines to fit
                axs[1].plot([x_min, i_heat], [y_val, y_val], color='k', linestyle='--')  # Horizontal lines to fit
                axs[1].plot([x_min, x_min + (x_max - x_min) / 10], [fit_min + dt * 2, fit_min + dt * 2],
                            color='C3')  # Lines near y_axis showing dT
                axs[1].plot([x_min, x_min + (x_max - x_min) / 10], [fit_min, fit_min],
                            color='C3')  # Lines near y_axis showing dT
            axs[1].plot([], [], color='C3', label=f'2*dT = {dt*2:.2f}')
            axs[1].legend()

            # Mar DC only
            # All dcdats for Mar
        fig, axs = PF.make_axes(len(mar19_dcdats), plt_kwargs={'sharex': True, 'sharey': True})
        for a, dat in zip(axs, mdcs):
            PF.display_2d(dat.Data.x_array, dat.Data.y_array, dat.Data.i_sense, ax=a, colorscale=True,
                          x_label=dat.Logs.x_label, y_label=dat.Logs.y_label, dat=dat)

            # Combined Mar dats
        fig, axs = PF.make_axes(2)
        _calculate_mar_int_entropy(mdat, mdcs=mdcs)  # Stores all calculations in m_VAL (e.g. m_xs, m_ys etc)
        m_ys_errs = [np.std(dat.Transition.fit_values.thetas) for dat in mdcs]
        PF.display_1d(m_xs, m_ys, axs[0], x_label=dat.Logs.y_label, y_label='Theta /mV', label='data', linewidth=1,
                      errors=m_ys_errs)
        _xs = np.linspace(np.nanmin(m_xs), np.nanmax(m_xs), 1000)
        axs[0].plot(_xs, m_result.eval(x=_xs), color='C3', label='best fit')
        i_heat_ac = m_i_heat
        y_min, _ = axs[0].get_ylim()
        x_min, x_max = axs[0].get_xlim()
        fit_min = np.nanmin(m_result.eval(x=np.linspace(-10, 10, 10000)))
        dt = mdat.Entropy._int_dt
        axs[0].margins(x=0, y=0)
        for i_heat in [i_heat_ac, -i_heat_ac]:
            y_val = m_result.eval(x=i_heat)
            axs[0].plot([i_heat, i_heat], [y_min, y_val], color='k', linestyle=':')  # Vertical lines to fit
            axs[0].plot([x_min, i_heat], [y_val, y_val], color='k', linestyle='--')  # Horizontal lines to fit
            axs[0].plot([x_min, x_min + (x_max - x_min) / 10], [fit_min + dt * 2, fit_min + dt * 2],
                        color='C3')  # Lines near y_axis showing dT
            axs[0].plot([x_min, x_min + (x_max - x_min) / 10], [fit_min, fit_min],
                        color='C3')  # Lines near y_axis showing dT
        axs[0].plot([], [], color='C3', label='2dT')
        PF.ax_setup(axs[0], f'Mar19 Theta vs Bias/nA', 'Current /nA', 'Theta /mV', legend=True)
        PF.plot_dac_table(axs[1], mdcs[0])
        fig.tight_layout()

    # DC charge sensor
    if show_i_sense is True:
        fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(10, 5))
        axs = axs.flatten()
        fig.suptitle(f'First row of I_sense for Entropy data')
        raw_i_senses = [jdat.Data.ADC0_2d[0], mdat.Data.v5dc2d[0], sdat.Data.FastScanCh0_2D[0],
                        sdat2.Data.FastScanCh0_2D[0]]
        pnum = len(raw_i_senses)
        for i, dat, raw_i_sense in zip(range(len(all_ent_dat)), all_ent_dat, raw_i_senses):
            PF.display_1d(dat.Data.x_array, raw_i_sense, ax=axs[i], x_label=dat.Logs.x_label,
                          y_label='I_sense raw data', scatter=True)
            PF.display_1d(dat.Data.x_array, dat.Data.i_sense[0], ax=axs[i + pnum], x_label=dat.Logs.x_label,
                          y_label='I_sense /nA', scatter=True)
            axs[i + pnum].plot(dat.Data.x_array, dat.Transition._full_fits[0].best_fit, color='C3', label='fit')
            axs[i + pnum].legend(title=f'Fit Func:\n[{dat.Transition.fit_func.__name__}]')
            PF.ax_text(axs[i + pnum], f'Amp: {dat.Transition.amp:.3f}nA')
            PF.ax_setup(axs[i], f'[{dat.config_name}][{dat.datnum}]')
        fig.tight_layout(rect=(0, 0, 1, 0.95))


# def _plot_entropy_along_transition():
#     """Not finished, was only comparing first two dats still"""
#     mdats = [mar19_make_dat(num, datname='digamma', dfname='Apr20', dfoption='load') for num in mar19_ent_datnums]
#     sdats = [sep19_make_dat(num, datname='digamma', dfname='Apr20', dfoption='load') for num in sep19_ent_datnums]
#     sdats2 = [sep19_make_dat(num, datname='digamma', dfname='Apr20', dfoption='load') for num in sep19_ent_datnums2]
#     dats = make_dats(jan20_ent_datnums, datname='digamma', dfoption='load', dfname='Apr20')
#
#     sdc2 = sep19_make_dat(2711, dfname='Apr20', dfoption='load')
#     sdc3 = sep19_make_dat(1945, dfname='Apr20', dfoption='load')
#
#     sdat = sdats2[0]
#
#     for dat in sdats2:
#         E.recalculate_int_entropy_with_offset_subtracted(dat, dc=sdc3, dT_mV=None, make_new=True, update=True,
#                                                          save=False, datdf=sep19_datdf)
#
#     sep19_datdf.save()
#
#     sdat = sdats2[0]
#     fig, axs = PF.make_axes(4)
#     plot_standard_entropy(sdat, axs, plots=[2, 3, 9, 10])
#     fig.suptitle(f'Sep19_Dat[{sdat.datnum}]: Const subtracted Entropy')
#
#     dats = make_dats(jan20_ent_datnums, datname='const_subtracted_entropy', dfoption='load', dfname='Apr20')
#     fig, axs = PF.make_axes(4)
#     plot_standard_entropy(dats[0], axs, plots=[2, 3, 9, 10])
#     fig.suptitle(f'Jan20_Dat[{dats[0].datnum}]: Const subtracted Entropy')


class print_info(object):
    """container for printing functions"""

    @staticmethod
    def srss(dat, graphic=False):
        from tabulate import tabulate
        columns = [key for key in dat.Instruments.srs1._fields]
        index = [f'SRS{i}' for i in range(1,4+1)]
        srss = [getattr(dat.Instruments, f'srs{i}', None) for i in range(1,4+1)]
        data = np.zeros((len(index), len(columns)))
        for i, srs in enumerate(srss):
            for j, key in enumerate(columns):
                if srs is not None:
                    data[i][j] = getattr(srs, key, None)
                else:
                    data[i][j] = None
        df = pd.DataFrame(data, index=index, columns=columns)
        if graphic is False:
            print(f'SRS info -- [{dat.config_name}] - Dat[{dat.datnum}]:\n')
            print(tabulate(df, headers=df.columns, tablefmt='simple'), end='\n\n')
        else:
            fig, ax = PF.plot_df_table(df, f'SRS info -- [{dat.config_name}] - Dat[{dat.datnum}]')


# region Experiment Inits [config switchers, datnums, dfs, etc]
jan20_ent_datnums = [1492, 1495, 1498, 1501, 1504, 1507, 1510, 1513, 1516, 1519, 1522, 1525, 1528, 1533, 1536, 1539,
                     1542, 1545, 1548, 1551, 1554, 1557, 1560, 1563, 1566]
jan20_dcdat = [1529]  # Theta vs DCbias
datdf = DF.DatDF(dfname='Apr20')

from src.Configs import Mar19Config
mar19_config_switcher = switch_config_decorator_maker(Mar19Config)
mar19_ent_datnums = list(range(2689, 2711 + 1, 2))  # Not in order
mar19_trans_datnums = list(range(2688, 2710 + 1, 2))  # Not in order
mar19_dcdats = list(range(3385, 3410 + 1, 2))  # 0.8mV steps. Repeat scan of theta at DC bias steps
mar19_dcdats2 = list(range(3386, 3410 + 1, 2))  # 3.1mV steps. Repeat scan of theta at DC bias steps
mar19_datdf = wrapped_call(mar19_config_switcher, (lambda: DF.DatDF(dfname='Apr20')))  # type: DF.DatDF
msf = wrapped_call(mar19_config_switcher, SF.SetupDF)  # type: SF.SetupDF
mar19_make_dat = mar19_config_switcher(make_dat_standard)  # type: C.make_dat_standard

from src.Configs import Sep19Config
sep19_config_switcher = switch_config_decorator_maker(Sep19Config)
sep19_ent_datnums = [2227, 2228] #, 2229]  # For more see OneNote (Integrated Entropy (Nik v2) - Jan 2020/General Notes/Datasets for Integrated Entropy Paper)
sep19_ent_datnums2 = list(range(2713, 2727+1))  # mix of 100/50 line repeats at 200/100mV/s along 0->1 transition?
sep19_dcdat = [1947]  # For more see OneNote "" 1947 is -980mV HPQC,
sep19_dcdat2 = [1945]  # 2711 is -960mV HQPC
sep19_datdf = wrapped_call(sep19_config_switcher, (lambda: DF.DatDF(dfname='Apr20')))  # type: DF.DatDF
ssf = wrapped_call(sep19_config_switcher, SF.SetupDF)  # type: SF.SetupDF
sep19_make_dat = sep19_config_switcher(make_dat_standard)  # type: C.make_dat_standard
#endregion


def _plot_gamma_of_peak(dats, ax, f_temp=100, mv_base=0.8):
    ax.cla()
    cols = ['datnum', 'G/mV', 'G/kbT', 'G/T']
    data = [[]]
    for dat in dats:
        if dat.Transition.g > mv_base and dat.Entropy.int_ds > 0:
            data.append([dat.datnum, dat.Transition.g, dat.Transition.g*f_temp/mv_base, dat.Transition.g/mv_base])
            ax.scatter(dat.Entropy.integrated_entropy_x_array - dat.Transition.mid, dat.Entropy.integrated_entropy, s=1)
            c = ax.collections[-1].get_fc()
            ax.scatter([],[], label=f'Dat[{dat.datnum}]', s=10, c=c)
            ax.set_xlabel(dat.Logs.x_label)
            ax.set_ylabel('Entropy /kB')
            print(
                f'Dat[{dat.datnum}]\n\tG={dat.Transition.g:.3f}mV\n\tT={dat.Transition.theta:.3f}mV\n\tG/T={dat.Transition.g / dat.Transition.theta:.3f}')
    df = pd.DataFrame(data[1:], columns=cols)
    df.set_index('datnum')
    ax.legend()
    return df

def _manipulate_avg_fit_of_dats(dats):
    """Used for emails to Yigal looking at G/T etc
    Only recalculates _avg_full_fit to save time, does not save results in df or anything
    returns df of fit results with datnum as index"""
    for dat in dats:
        param = CU.edit_params(dat.Transition.avg_params, 'theta', 0.92, False)
        dat.Transition._avg_full_fit = \
        T.transition_fits(dat.Transition._x_array, dat.Transition._avg_data, [param], T.i_sense_digamma)[0]

    df = fit_info_to_df([dat.Transition._avg_full_fit for dat in dats])
    df['index'] = [dat.datnum for dat in dats]
    return df



if __name__ == '__main__':
    # _recalculate_isens_digamma_constsubtracted(jan20_ent_datnums, jan20_dcdat, DF.DatDF(dfname='Apr20'))
    # _plot_all_raw_entropy_things_22nd_apr()
    # _recalculate_mar_sep(mar_recalculate=False, sep_recalculate=False)
    # _plot_int_entropy_for_all(x_axis='gate')
    # _plot_SRS_DCbias_Isens_for_all()
    # DONE
    pass













