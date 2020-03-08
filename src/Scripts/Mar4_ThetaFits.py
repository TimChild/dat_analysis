from src.Scripts.StandardImports import *
from src.DatCode.DCbias import plot_standard_dcbias
from src.DatCode.Transition import plot_standard_transition


def plot_dc_bias(dat):
    fig, axs = PF.make_axes(2)
    plot_standard_dcbias(dat,axs, plots=[1,2])
    fig.suptitle(f'Dat{dat.datnum} - Field={dat.Instruments.magy.field:.1f}mT')

def plot_transition_params(dat):
    fig, axs = PF.make_axes(6)
    dat.display(dat.Data.i_sense, axs[0])
    fit_attrs = ['thetas', 'amps', 'mids', 'lins', 'consts']
    fit_attr = dat.Transition
    fit_values = fit_attr.fit_values
    for ax, fit_values_name in zip(axs[1:], fit_attrs):
        values = fit_values._asdict().get(fit_values_name, None)
        PF.display_1d(dat.Data.y_array, values, ax, dat.Logs.y_label, fit_values_name, swap_ax=True,
                       swap_ax_labels=True, no_datnum=True, scatter=True)

    PF.add_standard_fig_info(fig)
    PF.add_to_fig_text(fig, f'Field={dat.Instruments.magy.field:.0f}mT, sweeprate={dat.Logs.sweeprate:.1f}mV/s, ACheat={dat.Instruments.srs1.out/50:.1f}nA, HQPC={dat.Logs.fdacs[0]}mV')
    fig.suptitle(f'Dat{dat.datnum}')
    plt.tight_layout(rect=[0, 0.1, 1, 0.9])


def plot_single_transition_fit(dat, yval, y_is_index=False, ax:plt.Axes=None):
    if ax is None:
        fig, ax = plt.subplots(1)
    ax, yid = dat.display1D_slice(dat.Data.i_sense, yval, yisindex=y_is_index, ax=ax, scatter=True)
    ax.plot(dat.Transition._x_array, dat.Transition._full_fits[yid].best_fit, color='C3')
    ax.texts[-1].remove()
    ax.collections[0].set_sizes([1]*len(dat.Data.x_array))
    PF.ax_text(ax, f'yid = {yid}'
                   f'\nyval = {dat.Data.y_array[yid]:.1f}'
                   f'\ntheta = {dat.Transition.fit_values.thetas[yid]:.2f}mV'
                   f'\namp = {dat.Transition.fit_values.amps[yid]:.4f}nA'
                   f'\nmid = {dat.Transition.fit_values.mids[yid]:.1f}mV', loc=(0.7, 0.05))
    fig = plt.gcf()
    ax.set_title(f'Dat{dat.datnum}')
    PF.add_standard_fig_info(fig)
    PF.add_to_fig_text(fig,
                       f'Field={dat.Instruments.magy.field:.0f}mT, '
                       f'sweeprate={dat.Logs.sweeprate:.1f}mV/s, '
                       f'ACheat={dat.Instruments.srs1.out / 50:.1f}nA, '
                       f'HQPC={dat.Logs.fdacs[0]}mV')


def plot_multiple_fits(dat, y_indexes:list, ax=None, align=False, spacing_x=0, spacing_y=0):
    """Plots up to 10 1D datasets and fits on single axes with legend"""
    if ax is None:
        fig, ax = PF.make_axes(1)
        ax = ax[0]
    ax.set_title(f'Dat{dat.datnum}: Data and Fits')

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    i = 0
    for yid, c in zip(y_indexes, colors):
        x = dat.Transition._x_array
        data = dat.Transition._data[yid]
        fit = dat.Transition._full_fits[yid]
        best_fit = fit.best_fit
        if align is True:
            x = x-fit.best_values['mid']+spacing_x*i
            data = data-fit.best_values['const']+spacing_y*i
            best_fit = best_fit-fit.best_values['const']+spacing_y*i
            i+=1
        ax.scatter(x, data, s=1, color=c)
        ax.plot(x, best_fit, linewidth = 1, color=c, label=f'{yid}: {dat.Transition.fit_values.thetas[yid]:.2f}mV')
    ax.legend(title='y_index: theta')




#  1350 - 1358 are 20 repeats on transition with widths 70, 40, 10: speeds 30, 10, 1 mV/s

if __name__ == '__main__':
    # dc = make_dat_standard(1305, dfoption='load')
    # dat = make_dat_standard(1306, dfoption='load')
    pass
