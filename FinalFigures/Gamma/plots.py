import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.colors as colors
import numpy as np
from typing import Union, List, Optional, Union
from itertools import chain
import lmfit as lm
from matplotlib import pyplot as plt

import dat_analysis.useful_functions as U

from dat_analysis.dat_analysis.characters import DELTA, ALPHA
import dat_analysis.plotting.mpl.PlotUtil as PU
import dat_analysis.plotting.mpl.AddCopyFig
from dat_analysis import useful_functions as U
from dat_analysis.plotting.mpl.Plots import display_2d

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
    ax.set_xlabel('Sweep Gate (mV)')
    ax.set_ylabel('I (nA)')

    ax.plot(x, cold, color='blue', label='Cold', linewidth=1)
    ax.plot(x, hot, color='red', label='Hot', linewidth=1)

    # Add straight lines before and after transition to emphasise amplitude
    transition_width = 0.30
    before_transition_id = U.get_data_index(x, np.mean(x) - transition_width, is_sorted=True)
    after_transition_id = U.get_data_index(x, np.mean(x) + transition_width, is_sorted=True)

    line = lm.models.LinearModel()
    top_line = line.fit(cold[:before_transition_id], x=x[:before_transition_id], nan_policy='omit')
    bottom_line = line.fit(cold[after_transition_id:], x=x[after_transition_id:], nan_policy='omit')

    ax.plot(x, top_line.eval(x=x), linestyle=':', color='black')
    ax.plot(x, bottom_line.eval(x=x), linestyle=':', color='black')

    # Add vertical arrow between dashed lines
    # x_val = (np.mean(x) + x[-1]) / 2  # 3/4 along
    x_val = 0.20
    y_bot = bottom_line.eval(x=x_val)
    y_top = top_line.eval(x=x_val)
    arrow = ax.annotate(text='', xy=(x_val, y_bot), xytext=(x_val, y_top), arrowprops=dict(arrowstyle='<|-|>',
                                                                                           lw=1))
    text = ax.text(x=x_val+0.02, y=(y_top + y_bot) / 2, s='dI/dN')

    ax.set_xlim(-0.5, 0.5)
    ax.legend(loc='center left')

    # Add horizontal lines to show thetas
    # TODO: should decide if I want to do this from fit results, or if it is just to give an idea theta..

    return ax


def dndt_signal(ax: plt.Axes, xs: Union[np.ndarray, List[np.ndarray]], datas: Union[np.ndarray, List[np.ndarray]],
                labels: Optional[list] = None,
                single: bool = True, scaled: bool = False,
                amp_sensitivity: Optional[float] = None) -> plt.Axes:
    """
    Plots entropy signal (as delta current)

    Args:
        ax ():
        xs ():
        datas ():
        labels ():
        single (): If only plotting a single trace (doesn't add label or scale data)
        scaled (): Whether to scale data so that they all have height 1 when plotting multiple, has no effect on single.
        amp_sensitivity (): Amplitude of transition to convert delta I to delta N for second y-axis

    Returns:

    """
    ax.set_xlabel('Sweep Gate (mV)')
    ax.set_ylabel(f'{DELTA}I (nA)', labelpad=-5)

    if amp_sensitivity:
        def convert_ax2(ax1):
            y1, y2 = ax1.get_ylim()
            ax2.set_ylim(y1 / amp_sensitivity, y2 / amp_sensitivity)

        ax2 = ax.twinx()
        ax2.set_ylabel(f'{DELTA}N', labelpad=0)
        # ax2.tick_params(axis='y', labelrotation=45)
        ax.callbacks.connect("ylim_changed", convert_ax2)

    if single:
        xs, datas, labels = [xs], [datas], [labels]

    for x, data, label in zip(xs, datas, labels):
        if single:
            ax.plot(x, data, marker='+')
        else:
            if scaled:
                scale = 1 / (np.nanmax(data))
            else:
                scale = 1
            ax.plot(x, data * scale, label=label, marker='+')
    if not single:
        leg = ax.legend()
        leg.set_title('Gamma/T')

    return ax


def gamma_vs_coupling(ax: plt.Axes, coupling_gates: Union[list, np.ndarray],
                      gammas: Union[list, np.ndarray]) -> plt.Axes:
    """Adds Gamma vs Coupling plot to axes"""
    ax.set_title('Gamma/T vs Coupling Gate')
    ax.set_xlabel('Coupling Gate (mV)')
    ax.set_ylabel('Gamma/KbT')

    ax.plot(coupling_gates, gammas, marker='.', color='black')
    ax.set_yscale('log')
    return ax


# def amp_theta_vs_coupling(ax: plt.Axes, amp_coupling: Union[list, np.ndarray], amps: Union[list, np.ndarray],
#                           dt_coupling: Union[list, np.ndarray], dt: Union[list, np.ndarray]) -> plt.Axes:
def amp_theta_vs_coupling(ax: plt.Axes, amp_coupling: Union[list, np.ndarray], amps: Union[list, np.ndarray],
                          lever_coupling: Union[list, np.ndarray], levers: Union[list, np.ndarray],
                          line_slope: float, line_intercept: float) -> plt.Axes:
    """Adds both amplitude vs coupling and theta vs coupling to axes"""
    # ax.set_title('Amplitude and dT vs Coupling Gate')
    # ax.set_xlabel('Coupling Gate (mV)')
    # ax.set_ylabel('dI/dN (nA)')
    #
    # ax.plot(amp_coupling, amps, label='dI/dN', marker='+')
    #
    # ax2 = ax.twinx()
    # ax2.set_ylabel('dT (mV)')
    # ax2.plot(dt_coupling, dt, color='C3', marker='x')
    # PU.add_legend_label(label='dT', ax=ax, color='C3', linestyle='-', marker='x')
    #
    # ax.legend(loc='center left')
    ax.set_title('Amplitude and lever arm vs Coupling Gate')
    ax.set_xlabel('Coupling Gate (mV)')
    ax.set_ylabel('dI (nA)')

    ax.plot(amp_coupling, amps, label='dI', marker='+')

    ax2 = ax.twinx()

    line = lm.models.LinearModel()
    pars = line.make_params()
    pars['slope'].value = line_slope
    pars['intercept'].value = line_intercept
    ax2.plot(amp_coupling, line.eval(x=amp_coupling, params=pars), color='C4')

    ax2.set_ylabel(f'{ALPHA} (units???)')
    ax2.scatter(lever_coupling, levers, color='C3', marker='x')
    PU.add_legend_label(label=f'{ALPHA}', ax=ax, color='C3', linestyle='-', marker='x')
    PU.add_legend_label(label='Linear Fit', ax=ax, color='C4', linestyle='-', marker='')
    ax.legend(loc='center left')
    return ax


def amp_sf_vs_coupling(ax: plt.Axes, amp_coupling: Union[list, np.ndarray], amps: Union[list, np.ndarray],
                          sf_coupling: Union[list, np.ndarray], sf: Union[list, np.ndarray]) -> plt.Axes:
    """Adds both amplitude vs coupling and theta vs coupling to axes"""
    ax.set_title('Amplitude and Scaling Factor vs Coupling Gate')
    ax.set_xlabel('Coupling Gate (mV)')
    ax.set_ylabel('dI/dN (nA)')

    ax.plot(amp_coupling, amps, label='dI/dN', marker='+')

    ax2 = ax.twinx()
    ax2.set_ylabel('dI/dN*dT')
    ax2.plot(sf_coupling, sf, color='C3', marker='x')
    PU.add_legend_label(label='dI/dN*dT', ax=ax, color='C3', linestyle='-', marker='x')

    ax.legend(loc='upper right')
    return ax


def integrated_entropy(ax: plt.Axes, xs: List[np.ndarray], datas: List[np.ndarray], labels: list) -> plt.Axes:
    """Plots integrated entropy vs gate voltage in real mV"""

    ax.set_xlabel('Sweep gate (mV)')
    ax.set_ylabel('Entropy (kB)')

    for x, data, label in zip(xs, datas, labels):
        ax.plot(x, data, label=label)

    for v in np.log(2), np.log(3):
        ax.axhline(v, linestyle=':', color='black')

    leg = ax.legend()
    ax.get_legend().set_title('Gamma/T')
    return ax


def entropy_vs_coupling(ax: plt.Axes,
                        int_coupling: Union[list, np.ndarray], int_entropy: Union[list, np.ndarray],
                        int_peaks: Optional[Union[list, np.ndarray]] = None,
                        fit_coupling: Union[list, np.ndarray] = None, fit_entropy: Union[list, np.ndarray] = None,
                        peak_diff_coupling: Optional[Union[list, np.ndarray]] = None,
                        peak_diff: Optional[Union[list, np.ndarray]] = None,
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
        peak_diff_coupling:  x axis for peak - final
        peak_diff:  peak entropy - final entropy
    Returns:

    """
    ax.set_title('Entropy vs Coupling Gate')
    ax.set_xlabel('Coupling Gate (mV)')
    ax.set_ylabel('Entropy (kB)')

    line = ax.plot(int_coupling, int_entropy, marker='.', label='From integration')[0]
    if int_peaks is not None:
        color = line.get_color()
        ax.plot(int_coupling, int_peaks, marker='.', linestyle='--', color=color, label='Peak from integration')
        ax.axhline(y=np.log(3), color='black', linestyle=':')
        if peak_diff_coupling is not None and peak_diff is not None:
            ax.plot(peak_diff_coupling, peak_diff, marker='x', linestyle=':', color='C3', label='Peak - Final')

    ax.plot(fit_coupling, fit_entropy, marker='+', label='From dN/dT fit')
    ax.axhline(y=np.log(2), color='black', linestyle=':')

    ax.set_ylim(0, 1.5)
    ax.legend()
    return ax


def dndt_2d(ax: plt.Axes, x: Union[list, np.ndarray], y: Union[list, np.ndarray], data: Union[list, np.ndarray]) -> plt.Axes:
    """2D plot of dN/dTs into gamma broadened (i.e. showing change from peak dip to just a broadened peak"""
    ax = display_2d(x=x, y=y, data=data, ax=ax, x_label='Sweep Gate (mV)', y_label='Gamma/Theta')

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
