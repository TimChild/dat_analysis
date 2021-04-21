import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.colors as colors
import numpy as np
import lmfit as lm

from typing import Union

import src.UsefulFunctions as U


def amp_theta_vs_coupling(ax: plt.Axes, amp_coupling: Union[list, np.ndarray], amps: Union[list, np.ndarray],
                          theta_coupling: Union[list, np.ndarray], thetas: Union[list, np.ndarray]) -> plt.Axes:
    """Adds both amplitude vs coupling and theta vs coupling to axes"""
    ax.set_title('Amplitude and dT vs Coupling Gate')
    ax.set_xlabel('Coupling Gate /mV')
    ax.set_ylabel('Amplitude /nA')

    ax.plot(amp_coupling, amps)

    ax2 = ax.twinx()
    ax2.set_ylabel('dT /mV')
    ax2.plot(theta_coupling, thetas)
    return ax


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
    before_transition_id = U.get_data_index(x, np.mean(x)-transition_width, is_sorted=True)
    after_transition_id = U.get_data_index(x, np.mean(x)+transition_width, is_sorted=True)

    line = lm.models.LinearModel()
    top_line = line.fit(cold[:before_transition_id], x=x[:before_transition_id], nan_policy='omit')
    bottom_line = line.fit(cold[after_transition_id:], x=x[after_transition_id:], nan_policy='omit')

    ax.plot(x, top_line.eval(x=x), linestyle=':', color='black')
    ax.plot(x, bottom_line.eval(x=x), linestyle=':', color='black')

    # Add vertical arrow between dashed lines
    x_val = (np.mean(x)+x[-1])/2   # 3/4 along
    y_bot = bottom_line.eval(x=x_val)
    y_top = top_line.eval(x=x_val)
    arrow = ax.annotate(s='sensitivity', xy=(x_val, y_bot), xytext=(x_val, y_top), arrowprops=dict(arrowstyle='<|-|>'))

    # Add horizontal lines to show thetas
    # TODO: should decide if I want to do this from fit results, or if it is just to give an idea theta..

    return ax



if __name__ == '__main__':
    from src.DatObject.Make_Dat import get_dats, get_dat, DatHDF

    # Get data to be plotted
    fit_name = 'forced_theta_linear'
    dats = get_dats(range(2095, 2125 + 1, 2))  # Goes up to 2141 but the last few aren't great
    tonly_dats = get_dats(range(2096, 2126 + 1, 2))
    # Loading fitting done in Analysis.Feb2021.entropy_gamma_final

    amp_cg_vals = [dat.Logs.fds['ESC'] for dat in tonly_dats]
    amps = [dat.Transition.get_fit(name=fit_name).best_values.amp for dat in tonly_dats]

    theta_cg_vals = [dat.Logs.fds['ESC'] for dat in tonly_dats]
    thetas = [dat.Transition.get_fit(name=fit_name).best_values.theta for dat in tonly_dats]

    # For the single hot/cold plot
    dat = get_dat(2164)
    out = dat.SquareEntropy.get_Outputs(name=fit_name)
    sweep_x = out.x
    cold_transition = np.nanmean(out.averaged[(0, 2), :], axis=0)
    hot_transition = np.nanmean(out.averaged[(1, 3), :], axis=0)


    # Do plotting
    fig, ax = plt.subplots(1, 1)
    getting_amplitude_and_dt(ax, x=sweep_x, cold=cold_transition, hot=hot_transition)
    plt.tight_layout()
    fig.show()


    fig, ax = plt.subplots(1, 1)
    amp_theta_vs_coupling(ax, amp_coupling=amp_cg_vals, amps=amps,
                          theta_coupling=theta_cg_vals, thetas=thetas)
    plt.tight_layout()
    fig.show()




