import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.colors as colors
import numpy as np
from typing import Union, List, Optional
from itertools import chain
import lmfit as lm
from matplotlib import pyplot as plt

import src.UsefulFunctions as U

import src.Plotting.Mpl.PlotUtil as PU
import src.Plotting.Mpl.AddCopyFig
from src import UsefulFunctions as U
from src.Plotting.Mpl.Plots import display_2d

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
          # 'font.family': 'sans-serif',
          # 'font.sans-serif': ['Helvetica'],
          }
pylab.rcParams.update(params)


def getting_amplitude_and_dt(ax: plt.Axes, x: np.ndarray, cold: np.ndarray, hot: np.ndarray) -> plt.Axes:
    """Adds hot and cold trace to axes with a straight line before and after transition to emphasise amplitude etc"""

    ax.set_title("Hot and cold part of transition")
    ax.set_xlabel('Sweep Gate /mV')
    ax.set_ylabel('Charge Sensor Current /nA')

    ax.plot(x, cold, color='blue', label='Cold')
    ax.plot(x, hot, color='red', label='Hot')

    # Add straight lines before and after transition to emphasise amplitude
    transition_width = 30
    before_transition_id = U.get_data_index(x, np.mean(x) - transition_width, is_sorted=True)
    after_transition_id = U.get_data_index(x, np.mean(x) + transition_width, is_sorted=True)

    line = lm.models.LinearModel()
    top_line = line.fit(cold[:before_transition_id], x=x[:before_transition_id], nan_policy='omit')
    bottom_line = line.fit(cold[after_transition_id:], x=x[after_transition_id:], nan_policy='omit')

    ax.plot(x, top_line.eval(x=x), linestyle=':', color='black')
    ax.plot(x, bottom_line.eval(x=x), linestyle=':', color='black')

    # Add vertical arrow between dashed lines
    # x_val = (np.mean(x) + x[-1]) / 2  # 3/4 along
    x_val = 20
    y_bot = bottom_line.eval(x=x_val)
    y_top = top_line.eval(x=x_val)
    arrow = ax.annotate(text='', xy=(x_val, y_bot), xytext=(x_val, y_top), arrowprops=dict(arrowstyle='<|-|>'))
    text = ax.text(x=x_val+2, y=(y_top + y_bot) / 2, s='dI/dN')

    ax.set_xlim(-50, 50)
    ax.legend(loc='center left')

    # Add horizontal lines to show thetas
    # TODO: should decide if I want to do this from fit results, or if it is just to give an idea theta..

    return ax


def dndt_signal(ax: plt.Axes, xs: List[np.ndarray], datas: List[np.ndarray], labels: Optional[list] = None,
                single: bool = True, scaled: bool = False) -> plt.Axes:
    """
    Plots dN/dTs

    Args:
        ax ():
        xs ():
        datas ():
        labels ():
        single (): If only plotting a single trace (doesn't add label or scale data)
        scaled (): Whether to scale data so that they all have height 1 when plotting multiple, has no effect on single.

    Returns:

    """
    ax.set_xlabel('Sweep Gate /mV')
    ax.set_ylabel('dN/dT Scaled')

    if single:
        xs, datas, labels = [xs], [datas], [labels]

    for x, data, label in zip(xs, datas, labels):
        if single:
            ax.plot(x, data)
        else:
            if scaled:
                scale = 1 / (np.nanmax(data))
            else:
                scale = 1
            ax.plot(x, data * scale, label=label)
    if not single:
        leg = ax.legend()
        leg.set_title('Gamma/T')
    return ax


def gamma_vs_coupling(ax: plt.Axes, coupling_gates: Union[list, np.ndarray],
                      gammas: Union[list, np.ndarray]) -> plt.Axes:
    """Adds Gamma vs Coupling plot to axes"""
    # ax.set_title('Gamma/T vs Coupling Gate')
    ax.set_xlabel('Coupling Gate /mV')
    ax.set_ylabel('Gamma/KbT')

    ax.plot(coupling_gates, gammas, marker='.', color='black')
    ax.set_yscale('log')
    return ax


def amp_theta_vs_coupling(ax: plt.Axes, amp_coupling: Union[list, np.ndarray], amps: Union[list, np.ndarray],
                          dt_coupling: Union[list, np.ndarray], dt: Union[list, np.ndarray]) -> plt.Axes:
    """Adds both amplitude vs coupling and theta vs coupling to axes"""
    ax.set_title('Amplitude and dT vs Coupling Gate')
    ax.set_xlabel('Coupling Gate /mV')
    ax.set_ylabel('dI/dN /nA')

    ax.plot(amp_coupling, amps, label='dI/dN', marker='+')

    ax2 = ax.twinx()
    ax2.set_ylabel('dT /mV')
    ax2.plot(dt_coupling, dt, label='dT', marker='x')
    ax.legend()
    return ax


def integrated_entropy(ax: plt.Axes, xs: List[np.ndarray], datas: List[np.ndarray], labels: list) -> plt.Axes:
    """Plots integrated entropy vs gate voltage in real mV"""

    ax.set_xlabel('Sweep gate /mV')
    ax.set_ylabel('Entropy /kB')

    for x, data, label in zip(xs, datas, labels):
        ax.plot(x, data, label=label)

    leg = ax.legend()
    ax.get_legend().set_title('Gamma/T')
    return ax


def entropy_vs_coupling(ax: plt.Axes,
                        int_coupling: Union[list, np.ndarray], int_entropy: Union[list, np.ndarray],
                        int_peaks: Optional[Union[list, np.ndarray]] = None,
                        fit_coupling: Union[list, np.ndarray] = None, fit_entropy: Union[list, np.ndarray] = None,
                        ) -> plt.Axes:
    """
    Plots fit and integrated entropy vs coupling gate

    Args:
        ax ():
        int_coupling ():
        int_entropy ():
        int_peaks (): Optional pass in to also plot the peak of integrated entropy as dotted line
        fit_coupling ():
        fit_entropy ():
    Returns:

    """
    ax.set_title('Entropy vs Coupling Gate')
    ax.set_xlabel('Coupling Gate /mV')
    ax.set_ylabel('Entropy /kB')

    line = ax.plot(int_coupling, int_entropy, marker='.', label='From integration')[0]
    if int_peaks is not None:
        color = line.get_color()
        ax.plot(int_coupling, int_peaks, marker='.', linestyle='--', color=color, label='Peak from integration')
        ax.axhline(y=np.log(3), color='black', linestyle=':')
    ax.plot(fit_coupling, fit_entropy, marker='+', label='From dN/dT fit')
    ax.axhline(y=np.log(2), color='black', linestyle=':')

    ax.set_ylim(0, None)
    ax.legend()
    return ax


def dndt_2d(ax: plt.Axes, x: Union[list, np.ndarray], y: Union[list, np.ndarray], data: Union[list, np.ndarray]) -> plt.Axes:
    """2D plot of dN/dTs into gamma broadened (i.e. showing change from peak dip to just a broadened peak"""
    ax = display_2d(x=x, y=y, data=data, ax=ax, x_label='Sweep Gate /mV', y_label='Gamma/Theta')

    return ax




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
