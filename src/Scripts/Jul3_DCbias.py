from src.Scripts.StandardImports import *


import inspect


if __name__ == '__main__':
    dats = get_dats(range(288, 292+1))

    fig, axs = PF.make_axes(len(dats))

    for ax in axs:
        ax.cla()

    which = 'theta'
    for dat, ax in zip(dats, axs):
        x = dat.Data.y_array[:]

        if which == 'theta':
            theta = [fit.best_values.theta for fit in dat.Transition.all_fits]
            theta = [v if v is not None and v < 5 else None for v in theta]
            ax.scatter(x/10, theta, s=2)
            PF.ax_setup(ax, f'Dat{dat.datnum}: LCSS/LCSQ={dat.Logs.Babydac.dacs[1]:.0f}mV', 'DCbias /nA', 'Theta /mV')
        elif which == 'mid':
            mid = [fit.best_values.mid for fit in dat.Transition.all_fits]
            # mid = [v if v is not None and v < 5 else None for v in mid]
            ax.scatter(x/10, mid, s=2)
            PF.ax_setup(ax, f'Dat{dat.datnum}: LCSS/LCSQ={dat.Logs.Babydac.dacs[1]:.0f}mV', 'DCbias /nA', 'Center /mV')
