import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.colors as colors
import numpy as np
from typing import List


def integrated_entropy(ax: plt.Axes, xs: List[np.ndarray], datas: List[np.ndarray], gamma_over_ts: list) -> plt.Axes:
    """Plots integrated entropy vs gate voltage in real mV"""

    ax.set_xlabel('Sweep gate /mV')
    ax.set_ylabel('Entropy /kB')
    ax.title = 'Entropy at various Gamma/T'
    leg = ax.legend()
    leg.title = 'Gamma/T'

    for x, data, gt in zip(xs, datas, gamma_over_ts):
        ax.plot(x, data, label=f'{gamma_over_ts:.1f}')
    return ax


def dndt_signal(ax: plt.Axes, xs: np.ndarray, datas: np.ndaarray, gamma_over_ts: list) -> plt.Axes:
    """Plots dN/dTs """

    ax.set_xlabel('Sweep Gate /mV')
    ax.set_ylabel('dN/dT /nA')
    ax.title = 'dN/dT at various Gamma/T'
    leg = ax.legend()
    leg.title = 'Gamma/T'

    for x, data, gt in zip(xs, datas, gamma_over_ts):
        ax.plot(x, data, label=f'{gamma_over_ts:.1f}')
    return ax


