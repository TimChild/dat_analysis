import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.colors as colors
import numpy as np
from typing import Union

import src.Plotting.Mpl.PlotUtil as PU
import src.Plotting.Mpl.AddCopyFig

mpl.use('tkagg')
# mpl.interactive(True)

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (5, 4),
          'axes.labelsize': 16,
          'axes.titlesize': 20,
          'xtick.labelsize': 16,
          'xtick.direction': 'in',
          'xtick.minor.visible': True,
          'ytick.labelsize': 16,
          'ytick.direction': 'in',
          'ytick.minor.visible': True,

          }
pylab.rcParams.update(params)


# Just here as a guide for 2D plots
# fig, ax = plt.subplots(1, 1)
#
# # 2D - Reminder of setting edgecolors for export to pdf
# im = ax.pcolor(edgecolors='face')
#
# # Use these 4 numbers to control [x-pos, y-pos, length, width] of the colorbar
# cax = fig.add_axes([.6, 0.90, 0.3, 0.015])
# cb = fig.colorbar(im, cax=cax, orientation="horizontal")  # , ticks=[0, 0.5, 1, 1.5])
# cb.ax.xaxis.set_ticks_position('top')
# # Add mini ticks
# cb.minorticks_on()
#
# cb.ax.set_ylabel(r'$dI_{sense}/dV_{IP}$ [nA/mV]', rotation=0, labelpad=105)


def gamma_vs_coupling(ax: plt.Axes, coupling_gates: Union[list, np.ndarray],
                      gammas: Union[list, np.ndarray]) -> plt.Axes:
    """Adds Gamma vs Coupling plot to axes"""
    # ax.set_title('Gamma/T vs Coupling Gate')
    ax.set_xlabel('Coupling Gate /mV')
    ax.set_ylabel('Gamma/KbT')

    ax.plot(coupling_gates, gammas, marker='.', color='black')
    return ax


def entropy_vs_coupling(ax: plt.Axes, int_coupling: Union[list, np.ndarray], int_entropy: Union[list, np.ndarray],
                        fit_coupling: Union[list, np.ndarray], fit_entropy: Union[list, np.ndarray]) -> plt.Axes:
    # ax.set_title('Entropy vs Coupling Gate')
    ax.set_xlabel('Coupling Gate /mV')
    ax.set_ylabel('Entropy /kB')

    ax.set_ylim(0, None)

    ax.plot(int_coupling, int_entropy, marker='.', label='From integration')
    ax.plot(fit_coupling, fit_entropy, marker='+', label='From dN/dT fit')
    ax.axhline(y=np.log(2), color='black', linestyle=':')

    ax.legend()
    return ax


if __name__ == '__main__':
    from src.DatObject.Make_Dat import get_dats, get_dat, DatHDF

    fit_name = 'forced_theta_linear'
    dats = get_dats(range(2095, 2125 + 1, 2))  # Goes up to 2141 but the last few aren't great
    tonly_dats = get_dats(range(2096, 2126 + 1, 2))
    # Loading fitting done in Analysis.Feb2021.entropy_gamma_final

    gamma_cg_vals = [dat.Logs.fds['ESC'] for dat in tonly_dats]
    gammas = [dat.Transition.get_fit(name=fit_name).best_values.g for dat in tonly_dats]

    int_cg_vals = [dat.Logs.fds['ESC'] for dat in dats]
    # TODO: Need to make sure all these integrated entropies are being calculated at good poitns (i.e. not including slopes)
    integrated_entropies = [np.nanmean(
        dat.Entropy.get_integrated_entropy(name=fit_name,
                                           data=dat.SquareEntropy.get_Outputs(
                                               name=fit_name, check_exists=True).average_entropy_signal
                                           )[-10:]) for dat in dats]

    fit_cg_vals = [dat.Logs.fds['ESC'] for dat in dats if dat.Logs.fds['ESC'] < -260]
    fit_entropies = [dat.Entropy.get_fit(name=fit_name).best_values.dS for dat in dats if dat.Logs.fds['ESC'] < -260]


    fig, ax = plt.subplots(1, 1)
    ax = gamma_vs_coupling(ax, coupling_gates=gamma_cg_vals, gammas=gammas)
    plt.tight_layout()
    fig.show()

    fig, ax = plt.subplots(1, 1)
    # ax = ax.inset_axes((0.2, 0.25, 0.5, 0.7))
    ax = entropy_vs_coupling(ax, int_coupling=int_cg_vals, int_entropy=integrated_entropies,
                             fit_coupling=fit_cg_vals, fit_entropy=fit_entropies)
    plt.tight_layout()
    fig.show()
