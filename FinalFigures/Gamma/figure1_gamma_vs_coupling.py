import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.colors as colors
import numpy as np
from typing import Union, List
from itertools import chain
import lmfit as lm

import src.UsefulFunctions as U

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
          'text.usetex': False,
          'font.family': 'sans-serif',
          'font.sans-serif': ['Helvetica'],
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


def getting_amplitude_and_dt(ax: plt.Axes, x: np.ndarray, cold: np.ndarray, hot: np.ndarray) -> plt.Axes:
    """Adds hot and cold trace to axes with a straight line before and after transition to emphasise amplitude etc"""

    ax.set_title("Calculating Scaling Factor")
    ax.set_xlabel('Sweep Gate /mV')
    ax.set_ylabel('Charge Sensor Current /nA')
    ax.legend()

    ax.plot(x, cold, color='blue')
    ax.plot(x, hot, color='red')

    # Add straight lines before and after transition to emphasise amplitude
    transition_width = 10
    before_transition_id = U.get_data_index(x, np.mean(x) - transition_width, is_sorted=True)
    after_transition_id = U.get_data_index(x, np.mean(x) + transition_width, is_sorted=True)

    line = lm.models.LinearModel()
    top_line = line.fit(cold[:before_transition_id], x=x[:before_transition_id], nan_policy='omit')
    bottom_line = line.fit(cold[after_transition_id:], x=x[after_transition_id:], nan_policy='omit')

    ax.plot(x, top_line.eval(x=x), linestyle=':', color='black')
    ax.plot(x, bottom_line.eval(x=x), linestyle=':', color='black')

    # Add vertical arrow between dashed lines
    x_val = (np.mean(x) + x[-1]) / 2  # 3/4 along
    y_bot = bottom_line.eval(x=x_val)
    y_top = top_line.eval(x=x_val)
    arrow = ax.annotate(s='sensitivity', xy=(x_val, y_bot), xytext=(x_val, y_top), arrowprops=dict(arrowstyle='<|-|>'))

    # Add horizontal lines to show thetas
    # TODO: should decide if I want to do this from fit results, or if it is just to give an idea theta..

    return ax


def dndt_signal(ax: plt.Axes, xs: List[np.ndarray], datas: List[np.ndarray], gamma_over_ts: list) -> plt.Axes:
    """Plots dN/dTs """

    ax.set_xlabel('Sweep Gate /mV')
    ax.set_ylabel('dN/dT Scaled')
    ax.set_title('dN/dT at various Gamma/T')

    for x, data, gt in zip(xs, datas, gamma_over_ts):
        scale = 1 / (np.nanmax(data))
        ax.plot(x, data * scale, label=f'{gt:.1f}')
    leg = ax.legend()
    leg.set_title('Gamma/T')
    return ax


if __name__ == '__main__':
    from src.DatObject.Make_Dat import get_dats, get_dat, DatHDF

    #############################################################################################

    # Data for single hot/cold plot
    fit_name = 'forced_theta_linear'
    dat = get_dat(2164)
    out = dat.SquareEntropy.get_Outputs(name=fit_name)
    sweep_x = out.x
    cold_transition = np.nanmean(out.averaged[(0, 2), :], axis=0)
    hot_transition = np.nanmean(out.averaged[(1, 3), :], axis=0)

    U.save_to_igor_itx(file_path=f'fig1_hot_cold.itx', xs=[sweep_x] * 2, datas=[cold_transition, hot_transition],
                       names=['cold', 'hot'], x_labels=['Sweep Gate /mV'] * 2, y_labels=['Current /nA'] * 2)

    # Plotting for Single hot/cold plot
    fig, ax = plt.subplots(1, 1)
    getting_amplitude_and_dt(ax, x=sweep_x, cold=cold_transition, hot=hot_transition)
    plt.tight_layout()
    fig.show()

    # Data for dN/dT
    fit_name = 'forced_theta_linear'
    dats = get_dats(range(2164, 2170 + 1, 3)) + [get_dat(2216)]
    tonly_dats = get_dats([dat.datnum + 1 for dat in dats])

    outs = [dat.SquareEntropy.get_Outputs(name=fit_name) for dat in dats]
    int_infos = [dat.Entropy.get_integration_info(name=fit_name) for dat in dats]

    xs = [out.x for out in outs]
    dndts = [out.average_entropy_signal for out in outs]
    gts = [dat.Transition.get_fit(name=fit_name).best_values.g / dat.Transition.get_fit(name=fit_name).best_values.theta
           for dat in tonly_dats]

    U.save_to_igor_itx(file_path=f'fig1_dndt.itx', xs=xs + [np.arange(4)], datas=dndts + [np.array(gts)],
                       names=[f'dndt_{i}' for i in range(len(dndts))] + ['gts_for_dndts'],
                       x_labels=['Sweep Gate /mV'] * len(dndts) + ['index'],
                       y_labels=['dN/dT /nA'] * len(dndts) + ['G/T'])

    # dNdT Plot
    fig, ax = plt.subplots(1, 1)
    dndt_signal(ax, xs=xs, datas=dndts, gamma_over_ts=gts)
    plt.tight_layout()
    fig.show()
