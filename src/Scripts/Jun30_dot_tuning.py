import src.Plotting.Mpl.PlotUtil
import src.Plotting.Mpl.Plots
from src.Scripts.StandardImports import *

from scipy.signal import savgol_filter

# old_dat = make_dat(2713, 'base', overwrite=True, ESI_class=Jan20.JanESI, run_fits=False)


def _plot_dat(dat, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
    x = dat.Data.x_array
    y = dat.Data.y_array
    z = dat.Data.Exp_cscurrent_2d
    z_smooth = savgol_filter(z, 31, 2)
    z_diff = np.gradient(z_smooth, axis=1)
    src.Plotting.Mpl.Plots.display_2d(x, y, z_diff, ax)
    src.Plotting.Mpl.PlotUtil.ax_setup(ax, f'Dat{dat.datnum}: RP/0.16={dat.Logs.Fastdac.dacs[7]:.0f}, RCSS={dat.Logs.Fastdac.dacs[6]:.0f}',
                                       dat.Logs.x_label, dat.Logs.y_label)


def _plot_row(dat: DatHDF, yval, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
    ax: plt.Axes

    yid = CU.get_data_index(dat.Data.y_array, yval)

    dacs = dat.Logs.Fastdac.dacs
    ax.plot(dat.Data.x_array, dat.Data.Exp_cscurrent_2d[yid], label=f'{dat.datnum}: {dacs[7]}, {dacs[6]}')
    ax.legend(title='Datnum: RP/0.16, RCSS')


def _plot_dc2d(dat: DatHDF):
    """Mostly here just to use in wait_for fn"""
    fig, ax = plt.subplots(1)
    src.Plotting.Mpl.Plots.display_2d(dat.Data.x_array, dat.Data.y_array, dat.Data.i_sense, ax, x_label=dat.Logs.x_label, y_label=dat.Logs.y_label)
    return


def _plot_dat_array(dats: List[DatHDF], rows=4, cols=6, axs=None, fixed_scale=False, norm=None, left_side=False):
    if axs is None:
        fig, axs = plt.subplots(nrows=rows, ncols=cols)
        all_axs = axs.flatten()
    else:
        assert(len(axs) >= len(dats))
        all_axs = axs
        fig = axs[0].figure

    for ax in all_axs:
        ax.cla()

    if fixed_scale or norm:
        if not norm:
            norm = mpl.colors.Normalize(vmin=-0.02, vmax=0.01)
    else:
        norm = None

    if left_side is False:
        side = 'R'
    else:
        side = 'L'

    for i in range(rows):
        dat_chunk = dats[i*cols:(i+1)*cols]
        axs = all_axs[i*cols:(i+1)*cols]

        for dat, ax in zip(dat_chunk, axs):
            dat: DatHDF
            x = dat.Data.x_array
            y = dat.Data.y_array
            z = dat.Data.Exp_cscurrent_2d

            # z_smooth = savgol_filter(z, 31, 2)
            z_smooth = CU.decimate(z, dat.Logs.Fastdac.measure_freq, 30)
            x = np.linspace(x[0], x[-1], z_smooth.shape[-1])
            z_diff = np.gradient(z_smooth, axis=1)
            src.Plotting.Mpl.Plots.display_2d(x, y, z_diff, ax, norm=norm, colorscale=False)
            src.Plotting.Mpl.PlotUtil.ax_setup(ax, f'dat{dat.datnum}')
            if i == rows-1:
                if f'{side}P/0.16' in dat.Logs.fds:
                    ax.set_xlabel(f'{side}P/0.16 = {dat.Logs.fds[f"{side}P/0.16"]:.0f}mV')
                elif f'{side}P*2' in dat.Logs.fds:
                    ax.set_xlabel(f'{side}P*2 = {dat.Logs.fds[f"{side}P*2"]:.0f}mV')
                else:
                    raise KeyError(f"No {side}P key found, come add one here!!")

        axs[0].set_ylabel(f'{side}CSS = {dat_chunk[0].Logs.Fastdac.dacs[6]:.0f}mV')
    fig.suptitle(f'X axis = {side}CT /mV, Y axis = {side}CB /mV')

    # Save code to dats.Other.code
    for dat in dats:
        code = inspect.getsource(_plot_dat_array)
        dat.Other.code = code
        dat.Other.update_HDF()

    return axs



if __name__ == '__main__':

    # dats = [get_dat(num) for num in range(254, 277+1)]
    # _plot_dat_array(dats, rows=4, cols=6, axs = None)


    ########################################
    # dats = [get_dat(num) for num in [278, 279, 280, 281, 282, 283, 284, 285]]
    #
    # fig, axs = PF.make_axes(len(dats), single_fig_size=(4,4))
    # for dat, ax in zip(dats, axs):
    #     _plot_dat(dat, ax)
    #

    ###########################################
    fig, axs = plt.subplots(4, 6)
    axs = axs.flatten()

    dats = get_dats(range(328, 351+1))
    # norm = mpl.colors.Normalize(vmin=-0.02, vmax=0.01)
    norm = mpl.colors.Normalize(vmin=-0.008, vmax=0.008)
    _plot_dat_array(dats, rows=4, cols=6, axs=axs, fixed_scale=True, norm=norm)


    plt.tight_layout()
