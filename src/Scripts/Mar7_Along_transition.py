from src.Scripts.StandardImports import *
import src.Scripts.Mar3_Entropy_Vs_field as Mar3


def plot_entropy_along_transition(dats, fig=None, axs=None, x_axis = 'G', exclude = None):
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
    elif x_axis.lower() == 'g':
        xs = [dat.Transition.avg_fit_values.gs[0] for dat in dats]
    else:
        print('x_axis has to be one of [rct, g]')

    ax = axs[0]
    ax_setup(ax, title='Nik Entropy vs Gamma', x_label=f'{x_axis} /mV', y_label='Entropy /kB', legend=False, fs=10)
    for dat, x in zip(dats, xs):
        y = dat.Entropy.avg_fit_values.dSs[0]
        yerr = np.std(dat.Entropy.fit_values.dSs)
        ax.errorbar(x, y, yerr=yerr, linestyle=None, marker='x')

    ax = axs[1]
    ax_setup(ax, title='Integrated Entropy vs Gamma', x_label=f'{x_axis} /mV', y_label='Entropy /kB', legend=False, fs=10)
    for dat, x in zip(dats, xs):
        y = dat.Entropy.int_ds
        yerr = np.std(dat.Entropy.int_entropy_per_line[-1])
        ax.errorbar(x, y, yerr=yerr, linestyle=None, marker='x')

    ax = axs[2]
    for dat in dats:
        x = dat.Entropy.x_array-dat.Transition.mid
        ax.plot(x, dat.Entropy.integrated_entropy, linewidth=1)
    ax_setup(ax, title='Integrated Entropy vs Gamma', x_label=dats[0].Logs.x_label, y_label='Entropy /kB', fs=10)


    plt.tight_layout(rect=(0, 0.1, 1, 1))
    return fig, axs




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
