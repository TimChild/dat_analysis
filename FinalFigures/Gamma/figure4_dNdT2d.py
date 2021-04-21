import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.colors as colors
import numpy as np
import lmfit as lm

import src.UsefulFunctions as U
from src.Plotting.Mpl.Plots import display_2d


def dndt_2d(ax: plt.Axes, x: np.ndarray, y:np.ndarray, data: np.ndarray) -> plt.Axes:
    """2D plot of dN/dTs into gamma broadened (i.e. showing change from peak dip to just a broadened peak"""
    ax = display_2d(x=x, y=y, data=data, ax=ax, x_label='Sweep Gate /mV', y_label='Coupling Gate /mV')
    ax.title = 'dN/dT vs Sweep gate and Coupling gate'

    return ax
