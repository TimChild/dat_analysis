"""Scripts for scanning along transition either where each dat is a repeat along a transition, or where a single dat is a scan along transition"""
from src.Scripts.StandardImports import *
from src.DatCode.Entropy import plot_standard_entropy
from src.DatCode.Transition import plot_standard_transition
import src.Scripts.Mar3_Entropy_Vs_field as Mar3
import src.DatCode.Entropy as E


def plot_entropy_along_transition(dats, fig=None, axs=None, x_axis='RCSS', exclude=None):
    """
    For plotting dats along a transition. I.e. each dat is a repeat measurement somewhere along transition

    @param exclude: datnums to exclude from plot
    @type exclude: List[int]
    @param dats: list of dat objects
    @type dats: src.DatCode.Dat.Dat
    @param fig:
    @type fig: plt.Figure
    @param axs:
    @type axs: List[plt.Axes]
    @return:
    @rtype: plt.Figure, List[plt.Axes]
    """

    if exclude is not None:
        dats = [dat for dat in dats if dat.datnum not in exclude]  # remove excluded dats from plotting

    if axs is None:
        fig, axs = PF.make_axes(3)

    PF.add_standard_fig_info(fig)

    if x_axis.lower() == 'rct':
        xs = [dat.Logs.fdacs[4] for dat in dats]
    elif x_axis.lower() == 'rcss':
        xs = [dat.Logs.fdacs[6] for dat in dats]
    elif x_axis.lower() == 'gamma':
        xs = [dat.Transition.avg_fit_values.gs[0] for dat in dats]
    else:
        print('x_axis has to be one of [rct, gamma]')

    ax = axs[0]
    ax_setup(ax, title=f'Nik Entropy vs {x_axis}', x_label=f'{x_axis} /mV', y_label='Entropy /kB', legend=False, fs=10)
    for dat, x in zip(dats, xs):
        y = dat.Entropy.avg_fit_values.dSs[0]
        yerr = np.std(dat.Entropy.fit_values.dSs)
        ax.errorbar(x, y, yerr=yerr, linestyle=None, marker='x')

    ax = axs[1]
    ax_setup(ax, title=f'Integrated Entropy vs {x_axis}', x_label=f'{x_axis} /mV', y_label='Entropy /kB', legend=False,
             fs=10)
    for dat, x in zip(dats, xs):
        y = dat.Entropy.int_ds
        yerr = np.std(dat.Entropy.int_entropy_per_line[-1])
        ax.errorbar(x, y, yerr=yerr, linestyle=None, marker='x')

    ax = axs[2]
    for dat in dats:
        x = dat.Entropy.x_array - dat.Transition.mid
        ax.plot(x, dat.Entropy.integrated_entropy, linewidth=1)
    ax_setup(ax, title=f'Integrated Entropy vs {x_axis}', x_label=dats[0].Logs.x_label, y_label='Entropy /kB', fs=10)

    plt.tight_layout(rect=(0, 0.1, 1, 1))
    return fig, axs


def plot_entropy_with_mid(dat, dc=make_dat_standard(1689, dfoption='load')):
    """Where a single dat is along a transition"""
    Mar3.init_int_entropy(dat, recalc=False, dcdat=dc, update=True, savedf=True)
    Mar3.plot_full_entropy_info(dat)
    fig = plt.gcf()
    axs = fig.axes
    plot_standard_entropy(dat, [axs[2]], plots=[10])
    plot_standard_transition(dat, [axs[0]], plots=[1])
    ax = axs[4]
    ax.cla()
    x = dat.Transition.fit_values.mids
    xerr = [par['mid'].stderr for par in dat.Transition.params]
    y = dat.Data.y_array
    ax.errorbar(x, y, xerr=xerr)

    ax.set_title('Midpoint deviation\nfrom straight line')
    ax.set_xlabel(dat.Logs.x_label)
    ax.set_ylabel(dat.Logs.y_label)
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])


def plot_entropy_offset_comparison(dat):
    """Takes dat of either offset or not, assumes there will be a [base] and [const_subtracted_entropy] in df already"""
    if dat.datname == 'const_subtracted_entropy':
        cdat = dat
        dat = make_dat_standard(cdat.datnum, dfoption='load')
    elif dat.datname == 'base':
        if DF.dat_exists_in_df(dat.datnum, 'const_subtracted_entropy', datdf):
            cdat = make_dat_standard(dat.datnum, datname='const_subtracted_entropy', dfoption='load')
        else:
            ans = CU.option_input('Looks like [const_subtracted_entropy] doesnt exist, do you want to create it?', {'yes':True, 'no': False})
            if ans is False:
                return None
            elif ans is True:
                if dat.Entropy._dc_datnum is not None:
                    num = dat.Entropy._dc_datnum
                    print(f'using dat{dat.Entropy._dc_datnum} for dcbias (initializing_int_entropy on dat{dat.datnum})')
                else:
                    num = int(input(f'Enter a dcbias datnum to use for calculating integrated entropy for dat{dat.datnum}'))
                E.recalculate_int_entropy_with_offset_subtracted(dat, make_dat_standard(num, dfoption='load'), make_new=True, update=True, save=True)
                cdat = dat
                dat = make_dat_standard(dat.datnum, dfoption='load')  # original dat gets changed so need to reload
            else:
                raise NotImplementedError
    else:
        print(f'Dat{dat.datnum} is not the [base] or [const_subtracted_entropy] dat')
        raise ValueError

    fig, axs = plt.subplots(1, 2, figsize=(6,4))
    axs = axs.flatten()
    for d, ax in zip([dat, cdat], axs):
        plot_standard_entropy(d, [ax], plots=[10])
        ax_setup(ax, title=f'Dat{d.datnum}[{d.datname}]')
    PF.add_standard_fig_info(fig)



def _plot_6th_march():
    datnums8 = list(range(1490, 1566+1))  # All dats in scan along transition 6th March  1529 is the only DCbias dat here
    d8_entropy_dats = [1492, 1495, 1498, 1501, 1504, 1507, 1510, 1513, 1516, 1519, 1522, 1525, 1528, 1533, 1536, 1539, 1542, 1545,
           1548, 1551, 1554, 1557, 1560, 1563, 1566]  # Just the entropy scans of datnums8.


    dats = [make_dat_standard(num, datname='digamma', dfoption='load') for num in d8_entropy_dats]
    dc = make_dat_standard(1529, dfoption='load')


    for dat in dats:
        if dat.Entropy.int_entropy_initialized is False:
            Mar3.init_int_entropy(dat, recalc=True, dcdat=dc, update=True)
        datdf.save()

    fig, axs = plot_entropy_along_transition(dats, exclude=[1528, 1519, 1536])


def _plot_11_mar_RCT():
    dat = make_dat_standard(1695)
    dats = make_dats([1695, 1696, 1697, 1698, 1699], dfoption='sync')
    dc = make_dat_standard(1689, dfoption='load')
    plot_with_mid__offset__standard(dats, dc)


def plot_with_mid__offset__standard(dats, dc):
    for dat in dats:
        Mar3.init_int_entropy(dat, recalc=False, dcdat=dc, update=True, savedf=False)
        # plot_entropy_with_mid(dat, dc)
        # fig = plt.gcf()
        # axs = fig.axes
        # for a in [axs[3], axs[4]]:
        #     a.lines[0].set_marker('x')
        #     a.lines[0].set_markeredgecolor('C3')
    datdf.save()
    for dat in dats:
        plot_entropy_offset_comparison(dat)

    fig, axs = PF.make_axes(len(dats))
    for dat, ax in zip(dats, axs):
        if not DF.dat_exists_in_df(dat.datnum, 'const_subtracted_entropy', datdf):
            E.recalculate_int_entropy_with_offset_subtracted(dat, dc, make_new=True, update=True, save=False)
        plot_standard_entropy(dat, [ax], plots=[10], kwargs_list=[{'no_datnum': False}])
    datdf.save()


def _plot_11_mar_RCSS():
    dats = make_dats(list(range(1783, 1796+1)), dfoption='load')
    dc = make_dat_standard(1730, dfoption='load')
    plot_with_mid__offset__standard(dats, dc)


def _plot_dats1723to1760_vs_dat1684():
    dats = make_dats(list(range(1723, 1760 + 1)))
    dats = [dat for dat in dats if 'entropy' in dat.dattype]
    fig, ax = plt.subplots(1)
    x = [dat.Logs.fdacs[4] * 0.097 for dat in dats]
    y = [dat.Entropy.avg_fit_values.dSs[0] for dat in dats]
    yerr = [np.std(dat.Entropy.fit_values.dSs) for dat in dats]
    dat = make_dat_standard(1684, dfoption='load')
    ax.errorbar(x, y, yerr, color='C0', label=f'Dats{dats[0].datnum} to {dats[-1].datnum}\nin real mV', marker='x',
                linestyle='None')
    ax.errorbar(dat.Data.y_array, dat.Entropy.fit_values.dSs, [f.params['dS'].stderr for f in dat.Entropy._full_fits],
                color='C3', label=f'Dat{dat.datnum}', marker='+', linestyle='None')
    ax_setup(ax, title='Comparing dats1723-1760 to dat1684\nHad to convert RCT to real mV', x_label='RCT /mV', y_label='Entropy /kB', legend=True)
    PF.add_standard_fig_info(fig)


def _plot_dats1726to1756_at_RCT4450():
    fig, ax = plt.subplots(1)
    dats = make_dats(list(range(1723, 1760 + 1)))
    dats = [dat for dat in dats if np.isclose(dat.Logs.fdacs[4], -4450, atol=20)]
    x = [dat.datnum for dat in dats]
    y = [dat.Entropy.avg_fit_values.dSs[0] for dat in dats]
    yerr = [np.std(dat.Entropy.fit_values.dSs) for dat in dats]
    ax_setup(ax, f'Dats{dats[0].datnum} to {dats[-1].datnum}: Entropy vs datnum (time)\nRCT~-4450mV', 'datnum',
             'Entropy /kB')
    PF.add_standard_fig_info(fig)

    x2 = [dat.Logs.fdacs[4] for dat in dats]
    fig, ax = plt.subplots(1)
    ax.errorbar(x2, y, yerr, marker='x')
    ax_setup(ax, f'Dats{dats[0].datnum} to {dats[-1].datnum}: Entropy vs RCT\nRCT~-4450mV', 'RCT*0.097 /mV',
             'Entropy /kB')
    PF.add_standard_fig_info(fig)







def _plot_entropy_vs_RCSS_dats1783on():
    dats = make_dats(list(range(1783, 1815 + 1)))
    dc = make_dat_standard(1730)
    plot_entropy_vs_axis(dats, dc, 'RCSS')


def _plot_entropy_vs_RCT_dats1811on():
    dats = make_dats(list(range(1817, 1847+1)), dfoption='load')
    dats = [dat for dat in dats if 'entropy' in dat.dattype]
    dc = make_dat_standard(1730, dfoption='load')
    plot_entropy_vs_axis(dats, dc, x_axis='RCT')


def plot_entropy_vs_axis(dats, dc, x_axis='RCT'):
    # for dat in dats:
    #     if dat.Entropy.int_entropy_initialized is False:
    #         dt = dc.DCbias.get_dt_at_current(dat.Instruments.srs1.out / 50 * np.sqrt(2))
    #         dat.Entropy.init_integrated_entropy_average(dT_mV=dt / 2, amplitude=dat.Transition.avg_fit_values.amps[0])
    #         update_save(dat, update=True, save=False)
    # datdf.save()
    # plot_entropy_along_transition(dats, x_axis=x_axis, exclude=None)

    for dat in dats:
        if not DF.dat_exists_in_df(dat.datnum, 'const_subtracted_entropy', datdf):
            E.recalculate_int_entropy_with_offset_subtracted(dat, dc, make_new=True,
                                                             update=True, save=False)

    datdf.save()
    cdats = [make_dat_standard(dat.datnum, datname='const_subtracted_entropy', dfoption='load') for dat in dats]
    plot_entropy_along_transition(cdats, x_axis=x_axis, exclude=None)
    fig = plt.gcf()
    PF.add_standard_fig_info(fig)
    PF.add_to_fig_text(fig, f'dats{cdats[0].datnum} to {cdats[-1].datnum}')
    axs = fig.axes
    for a in axs[:3]:
        a.grid()



_plot_entropy_vs_RCT_dats1811on()

