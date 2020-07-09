from src.Scripts.StandardImports import *

import inspect


def _plot_entropy(dats, axs=None):

    if axs is None:
        fig, axs = PF.make_axes(len(dats))
    else:
        assert len(axs) >= len(dats)

    for dat, ax in zip(dats, axs):
        ax.cla()
        x = dat.Data.x_array
        PF.display_1d(x, dat.Entropy.avg_data, ax, dat.Logs.x_label, 'Entropy /kB')

        fx = x[::10]
        fz = dat.Entropy.avg_fit.eval_fit(fx)
        ax.plot(fx, fz, label='Fit')
        PF.ax_setup(ax, f'dat{dat.datnum}: Average of {len(dat.Data.y_array)} rows')

        bv = dat.Entropy.avg_fit.best_values
        srs1 = dat.Logs.srs1
        PF.ax_text(ax, f'dS={bv.dS:.3f}\nSRS:\n  out={srs1.out:.0f}mV\n  tc={srs1.tc:.2f}s\n'
                       f'HQPC(LCB/LP):\n  {dat.Logs.bds["LCB"]:.0f}mV',
                   loc=(0.05, 0.6), fontsize=8)

    for dat in dats:
        dat.Other.save_code(inspect.getsource(_plot_entropy), 'avg_entropy')
    return axs


def _plot_dcbias(dats, which, axs = None):
    # gates = 'LCSS/LCSQ'
    # gate_dac = 1
    # dats = get_dats(range(352, 358+1))
    gates = 'LCB/LP'
    gate_dac = 0

    if axs is None:
        fig, axs = PF.make_axes(len(dats))
    else:
        assert (len(axs) >= len(dats))

    for ax in axs:
        ax.cla()

    for dat, ax in zip(dats, axs):
        x = dat.Data.y_array[:]

        if which == 'theta':
            theta = [fit.best_values.theta for fit in dat.Transition.all_fits]
            theta = [v if v is not None and v < 5 else None for v in theta]
            ax.scatter(x / 10, theta, s=2)
            PF.ax_setup(ax, f'Dat{dat.datnum}: {gates}={dat.Logs.Babydac.dacs[gate_dac]:.0f}mV', 'DCbias /nA',
                        'Theta /mV')
        elif which == 'mid':
            mid = [fit.best_values.mid for fit in dat.Transition.all_fits]
            # mid = [v if v is not None and v < 5 else None for v in mid]
            ax.scatter(x / 10, mid, s=2)
            PF.ax_setup(ax, f'Dat{dat.datnum}: {gates}={dat.Logs.Babydac.dacs[gate_dac]:.0f}mV', 'DCbias /nA',
                        'Center /mV')
        elif which == '2d':
            PF.display_2d(dat.Data.x_array, dat.Data.y_array, dat.Data.i_sense, ax, colorscale=True)
            PF.ax_setup(ax, f'Dat{dat.datnum}: {gates}={dat.Logs.Babydac.dacs[gate_dac]:.0f}mV', dat.Logs.x_label,
                        dat.Logs.y_label)

    for dat in dats:
        dat.Other.save_code(inspect.getsource(_plot_dcbias), 'DCbias')

    return axs


if __name__ == '__main__':









    # dats = get_dats(range(288, 292+1))
    # dats = [get_dat(368)]
    # dats = get_dats(range(361, 367+1))  # BAD because Babydac's possibly not working for these
    # dats = get_dats(range(369, 375+1))
    # _plot_dcbias(dats, 'theta')


    # nd = get_dat(295)
    # od = get_dat(2786, ESI_class=JanESI)
    #
    # # fig, axs = plt.subplots(2)
    # # for ax in axs:
    # #     ax.cla()
    # fig, ax = plt.subplots(1)
    # ax.cla()
    #
    # for dat, ax, name in zip([nd, od], [ax, ax], ['New', 'Old']):
    #     if name == 'New':
    #         srso = dat.Logs.srs1
    #         srsi = dat.Logs.srs1
    #     elif name == 'Old':
    #         srso = dat.Logs.srs1
    #         srsi = dat.Logs.srs3
    #     else:
    #         raise ValueError
    #
    #     text = f'{name}-Dat{dat.datnum}\n'\
    #            f'  Out:{srso.out}\n' \
    #            f'  Sens:{srsi.sens}\n' \
    #            f'  Freq:{srso.freq}\n' \
    #            f'  Tc:{srsi.tc}\n' \
    #            f'  Harm:{srsi.harm}'
    #
    #     x = dat.Data.x_array - dat.Transition.avg_fit.best_values.mid
    #     z = dat.Entropy.avg_data
    #     # z = dat.Data.Exp_entropy_y_2d_RAW
    #     PF.display_1d(x, z, ax, x_label=dat.Logs.x_label, y_label='Entropy signal /nA', label=text)
    #     # PF.ax_setup(ax, f'{name}-Dat{dat.datnum}')
    # ax.legend()
    # PF.ax_setup(ax, 'Comparing Entropy from Jan20 to Jun20')
    #
    #
    # # plt.tight_layout()
    #
    # nesi = JunESI(295)
    # oesi = JanESI(2786)
