from src.Scripts.StandardImports import *


from scipy.signal import savgol_filter

# old_dat = make_dat(2713, 'base', overwrite=True, ESI_class=Jan20.JanESI, run_fits=False)


import threading
import time
from src.DatObject.Make_Dat import default_ESI


def wait_for(datnum, ESI_class=default_ESI):
    def _wait_fn(num):
        esi = ESI_class(num)
        while True:
            found = esi.check_data_exists(supress_output=True)
            if found:
                print(f'Dat{num} is ready!!')
                break
            else:
                time.sleep(10)
    x = threading.Thread(target=_wait_fn, args=(datnum,))
    x.start()
    print(f'A thread is waiting on dat{datnum} to appear')
    return x


def _plot_dat(dat, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
    x = dat.Data.x_array
    y = dat.Data.y_array
    z = dat.Data.Exp_cscurrent_2d
    z_smooth = savgol_filter(z, 31, 2)
    z_diff = np.gradient(z_smooth, axis=1)
    PF.display_2d(x, y, z_diff, ax)
    PF.ax_setup(ax, f'Dat{dat.datnum}: RP/0.16={dat.Logs.Fastdac.dacs[7]:.0f}, RCSS={dat.Logs.Fastdac.dacs[6]:.0f}',
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
    PF.display_2d(dat.Data.x_array, dat.Data.y_array, dat.Data.i_sense, ax, x_label=dat.Logs.x_label, y_label=dat.Logs.y_label)
    return


if __name__ == '__main__':

    # dats = [get_dat(num) for num in range(254, 277+1)]
    #
    # fig, axs = plt.subplots(nrows=4, ncols=6)
    # all_axs = axs.flatten()
    #
    #
    # for ax in all_axs:
    #     ax.cla()
    # for i in range(4):
    #     dat_chunk = dats[i*6:(i+1)*6]
    #     axs = all_axs[i*6:(i+1)*6]
    #
    #     for dat, ax in zip(dat_chunk, axs):
    #         dat: DatHDF
    #         x = dat.Data.x_array
    #         y = dat.Data.y_array
    #         z = dat.Data.Exp_cscurrent_2d
    #         z_smooth = savgol_filter(z, 31, 2)
    #         z_diff = np.gradient(z_smooth, axis=1)
    #         PF.display_2d(x, y, z_diff, ax)
    #         PF.ax_setup(ax, f'dat{dat.datnum}')
    #         if i == 3:
    #             ax.set_xlabel(f'RP/0.16 = {dat.Logs.Fastdac.dacs[7]:.0f}mV')
    #
    #     axs[0].set_ylabel(f'RCSS = {dat_chunk[0].Logs.Fastdac.dacs[6]:.0f}mV')
    # # plt.tight_layout()

    ########################################
    # dats = [get_dat(num) for num in [278, 279, 280, 281, 282, 283, 284, 285]]
    #
    # fig, axs = PF.make_axes(len(dats), single_fig_size=(4,4))
    # for dat, ax in zip(dats, axs):
    #     _plot_dat(dat, ax)
    #

    ###########################################

    dcdats = [get_dat(num) for num in range(288, 293)]
    fig, axs = PF.make_axes(len(dcdats))

    for ax in axs:
        ax.cla()

    which = 'theta'
    for dat, ax in zip(dcdats, axs):
        x = dat.Data.y_array

        if which == 'theta':
            theta = [fit.best_values.theta for fit in dat.Transition.all_fits]
            theta = [v if v is not None and v < 5 else None for v in theta]
            ax.scatter(x, theta, s=2)
            PF.ax_setup(ax, f'Dat{dat.datnum}: LCSS/LCSQ={dat.Logs.Babydac.dacs[1]:.0f}mV', dat.Logs.y_label, 'Theta /mV')
        elif which == 'mid':
            mid = [fit.best_values.mid for fit in dat.Transition.all_fits]
            # mid = [v if v is not None and v < 5 else None for v in mid]
            ax.scatter(x, mid, s=2)
            PF.ax_setup(ax, f'Dat{dat.datnum}: LCSS/LCSQ={dat.Logs.Babydac.dacs[1]:.0f}mV', dat.Logs.y_label,
                        'Center /mV')



    plt.tight_layout()
