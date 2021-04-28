import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

import src.UsefulFunctions as U
from FinalFigures.Gamma.plots import gamma_vs_coupling, amp_theta_vs_coupling, dndt_signal


if __name__ == '__main__':
    from src.DatObject.Make_Dat import get_dats

    ####################################################
    # Data for gamma_vs_coupling
    fit_name = 'forced_theta_linear'
    # dats = get_dats(range(2095, 2125 + 1, 2))  # Goes up to 2141 but the last few aren't great
    tonly_dats = get_dats(range(2096, 2126 + 1, 2))
    # Loading fitting done in Analysis.Feb2021.entropy_gamma_final

    gamma_cg_vals = np.array([dat.Logs.fds['ESC'] for dat in tonly_dats])
    gammas = np.array([dat.Transition.get_fit(name=fit_name).best_values.g for dat in tonly_dats])

    U.save_to_igor_itx(file_path=f'fig2_gamma_vs_coupling.itx', xs=[gamma_cg_vals], datas=[gammas],
                       names=['gammas'],
                       x_labels=['Coupling Gate /mV'],
                       y_labels=['Gamma /mV'])

    # Plotting gamma_vs_coupling
    fig, ax = plt.subplots(1, 1)
    ax = gamma_vs_coupling(ax, coupling_gates=gamma_cg_vals, gammas=gammas)
    plt.tight_layout()
    fig.show()

    #####################################################
    # Data for amp_theta_vs_coupling
    fit_name = 'forced_theta_linear'
    dats = get_dats(range(2095, 2125 + 1, 2))  # Goes up to 2141 but the last few aren't great
    tonly_dats = get_dats(range(2096, 2126 + 1, 2))

    amp_cg_vals = np.array([dat.Logs.fds['ESC'] for dat in tonly_dats])
    amps = np.array([dat.Transition.get_fit(name=fit_name).best_values.amp for dat in tonly_dats])

    theta_cg_vals = np.array([dat.Logs.fds['ESC'] for dat in tonly_dats])
    thetas = np.array([dat.Transition.get_fit(name=fit_name).best_values.theta for dat in tonly_dats])

    U.save_to_igor_itx(file_path=f'fig2_amp_theta_vs_coupling.itx', xs=[amp_cg_vals, theta_cg_vals], datas=[amps, thetas],
                       names=['amplitudes', 'thetas'],
                       x_labels=['Coupling Gate /mV'] * 2,
                       y_labels=['dI/dN /nA', 'Theta /mV'])

    # Plotting amp_theta_vs_coupling
    fig, ax = plt.subplots(1, 1)
    amp_theta_vs_coupling(ax, amp_coupling=amp_cg_vals, amps=amps,
                          theta_coupling=theta_cg_vals, thetas=thetas)
    plt.tight_layout()
    fig.show()
