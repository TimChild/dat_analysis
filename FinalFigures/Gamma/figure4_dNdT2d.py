import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.colors as colors
import src.Plotting.Mpl.AddCopyFig
import numpy as np
import lmfit as lm
from typing import Union
from scipy.interpolate import interp1d

import src.UsefulFunctions as U
from src.Plotting.Mpl.Plots import display_2d


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


def dndt_2d(ax: plt.Axes, x: Union[list, np.ndarray], y: Union[list, np.ndarray], data: Union[list, np.ndarray]) -> plt.Axes:
    """2D plot of dN/dTs into gamma broadened (i.e. showing change from peak dip to just a broadened peak"""
    ax = display_2d(x=x, y=y, data=data, ax=ax, x_label='Sweep Gate /mV', y_label='Gamma/Theta')

    return ax


if __name__ == '__main__':
    from src.DatObject.Make_Dat import get_dats, get_dat, DatHDF

    fit_name = 'forced_theta_linear'
    dats = get_dats(range(2095, 2125 + 1, 2))  # Goes up to 2141 but the last few aren't great
    tonly_dats = get_dats(range(2096, 2126 + 1, 2))
    # Loading fitting done in Analysis.Feb2021.entropy_gamma_final

    gamma_cg_vals = [dat.Logs.fds['ESC'] for dat in tonly_dats]
    gammas_over_thetas = [dat.Transition.get_fit(name=fit_name).best_values.g /
                          dat.Transition.get_fit(name=fit_name).best_values.theta for dat in tonly_dats]

    full_x = np.linspace(-150, 150, 1000)
    data = []
    for dat in dats:
        out = dat.SquareEntropy.get_Outputs(name=fit_name)
        x = out.x
        dndt = out.average_entropy_signal
        interper = interp1d(x=x, y=dndt, bounds_error=False)
        data.append(interper(x=full_x))


    # Do Plotting
    # 2D dN/dT
    fig, ax = plt.subplots(1, 1)
    ax = dndt_2d(ax, x=full_x, y=gammas_over_thetas, data=data)
    plt.tight_layout()
    fig.show()
