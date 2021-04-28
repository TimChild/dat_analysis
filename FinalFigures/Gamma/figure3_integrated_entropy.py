import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.colors as colors
import src.Plotting.Mpl.AddCopyFig
import numpy as np
from typing import List, Union
from itertools import chain

from src.UsefulFunctions import save_to_igor_itx

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
          # 'text.usetex': False,
          # 'font.family': 'sans-serif',
          # 'font.sans-serif': ['Helvetica'],
          }
pylab.rcParams.update(params)


def integrated_entropy(ax: plt.Axes, xs: List[np.ndarray], datas: List[np.ndarray], gamma_over_ts: list) -> plt.Axes:
    """Plots integrated entropy vs gate voltage in real mV"""

    ax.set_xlabel('Sweep gate /mV')
    ax.set_ylabel('Entropy /kB')

    for x, data, gt in zip(xs, datas, gamma_over_ts):
        ax.plot(x, data, label=f'{gt:.1f}')

    leg = ax.legend()
    ax.get_legend().set_title('Gamma/T')
    return ax


def entropy_vs_coupling(ax: plt.Axes, int_coupling: Union[list, np.ndarray], int_entropy: Union[list, np.ndarray],
                        fit_coupling: Union[list, np.ndarray], fit_entropy: Union[list, np.ndarray]) -> plt.Axes:
    """Plots fit and integrated entropy vs coupling gate"""
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

    ##########################################################################
    # Data for integrated_entropy
    fit_name = 'forced_theta_linear'
    dats = get_dats(range(2164, 2170 + 1, 3)) + [get_dat(2216)]
    tonly_dats = get_dats([dat.datnum + 1 for dat in dats])

    outs = [dat.SquareEntropy.get_Outputs(name=fit_name) for dat in dats]
    int_infos = [dat.Entropy.get_integration_info(name=fit_name) for dat in dats]

    xs = [out.x for out in outs]
    int_entropies = [int_info.integrate(out.average_entropy_signal) for int_info, out in zip(int_infos, outs)]
    gts = [dat.Transition.get_fit(name=fit_name).best_values.g / dat.Transition.get_fit(name=fit_name).best_values.theta
           for dat in tonly_dats]

    save_to_igor_itx(file_path=f'fig3_integrated_entropy.itx',
                     xs=xs + [np.arange(len(gts))],
                     datas=int_entropies + [np.array(gts)],
                     names=[f'int_entropy_{i}' for i in range(len(int_entropies))] + ['gts_for_int_entropies'],
                     x_labels=['Sweep Gate /mV'] * len(int_entropies) + ['index'],
                     y_labels=['Entropy /kB']*len(int_entropies) + ['G/T'])

    # Plot Integrated Entropy
    fig, ax = plt.subplots(1, 1)
    integrated_entropy(ax, xs=xs, datas=int_entropies, gamma_over_ts=gts)
    plt.tight_layout()
    fig.show()

    ############################################################################
    # Data for entropy_vs_coupling
    fit_name = 'forced_theta_linear'
    dats = get_dats(range(2095, 2125 + 1, 2))  # Goes up to 2141 but the last few aren't great
    tonly_dats = get_dats(range(2096, 2126 + 1, 2))

    gamma_cg_vals = np.array([dat.Logs.fds['ESC'] for dat in tonly_dats])
    gammas = np.array([dat.Transition.get_fit(name=fit_name).best_values.g for dat in tonly_dats])

    int_cg_vals = np.array([dat.Logs.fds['ESC'] for dat in dats])
    # TODO: Need to make sure all these integrated entropies are being calculated at good poitns (i.e. not including slopes)
    integrated_entropies = np.array([np.nanmean(
        dat.Entropy.get_integrated_entropy(name=fit_name,
                                           data=dat.SquareEntropy.get_Outputs(
                                               name=fit_name, check_exists=True).average_entropy_signal
                                           )[-10:]) for dat in dats])

    fit_cg_vals = np.array([dat.Logs.fds['ESC'] for dat in dats if dat.Logs.fds['ESC'] < -260])
    fit_entropies = np.array(
        [dat.Entropy.get_fit(name=fit_name).best_values.dS for dat in dats if dat.Logs.fds['ESC'] < -260])

    save_to_igor_itx(file_path=f'fig3_entropy_vs_gamma.itx', xs=[fit_cg_vals, int_cg_vals],
                     datas=[fit_entropies, integrated_entropies],
                     names=['fit_entropy_vs_coupling', 'integrated_entropy_vs_coupling'],
                     x_labels=['Coupling Gate /mV'] * 2,
                     y_labels=['Entropy /kB', 'Entropy /kB'])

    # Plot entropy_vs_coupling
    fig, ax = plt.subplots(1, 1)
    ax = entropy_vs_coupling(ax, int_coupling=int_cg_vals, int_entropy=integrated_entropies,
                             fit_coupling=fit_cg_vals, fit_entropy=fit_entropies)
    plt.tight_layout()
    fig.show()
