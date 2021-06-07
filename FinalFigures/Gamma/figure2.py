import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import lmfit as lm
from itertools import chain

import src.UsefulFunctions as U
from src.Characters import ALPHA
from Analysis.Feb2021.common import integrated_entropy_value
from src.UsefulFunctions import save_to_igor_itx
from src.Plotting.Mpl.PlotUtil import set_default_rcParams
from FinalFigures.Gamma.plots import gamma_vs_coupling, amp_theta_vs_coupling, dndt_signal, integrated_entropy, entropy_vs_coupling


if __name__ == '__main__':
    set_default_rcParams()
    from src.DatObject.Make_Dat import get_dats, get_dat

    # Data for weakly coupled dN/dTs
    fit_name = 'forced_theta_linear'
    # all_dats = get_dats(range(2095, 2111 + 1, 2))[::4]
    # all_dats = get_dats(range(7322, 7361 + 1, 2))[::4]
    all_dats = get_dats([2097, 2103, 2107])
    # all_dats = get_dats([2097, 2103, 2109])

    # all_dats = get_dats(chain(range(7322, 7361 + 1, 2), range(7378, 7399 + 1, 2), range(7400, 7421 + 1, 2)))
    # all_dats = [dat for dat in all_dats if dat.Logs.fds['ESC'] < -245]
                # and 0.74 < integrated_entropy_value(dat, fit_name) < 0.76]
    all_dats = U.order_list(all_dats, [dat.Logs.fds['ESC'] for dat in all_dats])

    outs = [dat.SquareEntropy.get_Outputs(name=fit_name) for dat in all_dats]

    xs = [out.x/100 for out in outs]  # /100 to convert to real mV
    dndts = [out.average_entropy_signal for out in outs]

    U.save_to_igor_itx(file_path=f'fig2_dndt.itx',
                       xs=xs+[np.array([dat.datnum for dat in all_dats])],
                       datas=dndts+[np.array([dat.Logs.dacs['ESC'] for dat in all_dats])],
                       names=[f'dndt_stacked_{i}' for i in range(len(dndts))] + ['stacked_coupling_gates'],
                       x_labels=['Sweep Gate (mV)'] * len(dndts) + ['Datnum'],
                       y_labels=['dN/dT (nA)'] * len(dndts) + ['ESC (mV)'])

    # Plotting dNdT for several weakly coupled
    fig, ax = plt.subplots(1, 1)
    ax = dndt_signal(ax, xs=xs, datas=dndts, labels=[f'{dat.Logs.fds["ESC"]:.1f}' for dat in all_dats], single=False)
    for line in ax.lines:
        line.set_marker('')
    ax.get_legend().set_title('Coupling Gate (mV)')
    ax.set_xlim(-1, 1)
    # ax.set_title('dN/dT for weakly coupled')
    plt.tight_layout()
    fig.show()

    ##########################################################################
    # Data for integrated_entropy
    fit_name = 'forced_theta_linear'
    # all_dats = get_dats(range(2095, 2111 + 1, 2))[::4]
    # all_dats = get_dats(range(7322, 7361 + 1, 2))[::4]
    # all_dats = get_dats([2097, 2103, 2107])
    all_dats = get_dats([2097, 2103, 2109])

    # all_dats = get_dats(chain(range(7322, 7361 + 1, 2), range(7378, 7399 + 1, 2), range(7400, 7421 + 1, 2)))
    all_dats = [dat for dat in all_dats if dat.Logs.fds['ESC'] < -245]
                # and 0.74 < integrated_entropy_value(dat, fit_name) < 0.76]
    all_dats = U.order_list(all_dats, [dat.Logs.fds['ESC'] for dat in all_dats])

    # tonly_dats = get_dats([dat.datnum + 1 for dat in dats])

    outs = [dat.SquareEntropy.get_Outputs(name=fit_name) for dat in all_dats]
    int_infos = [dat.Entropy.get_integration_info(name=fit_name) for dat in all_dats]

    xs = [out.x/100 for out in outs]  # /100 to convert to real mV
    int_entropies = [int_info.integrate(out.average_entropy_signal) for int_info, out in zip(int_infos, outs)]
    # gts = [dat.Transition.get_fit(name=fit_name).best_values.g / dat.Transition.get_fit(name=fit_name).best_values.theta
    #        for dat in tonly_dats]

    U.save_to_igor_itx(file_path=f'fig2_integrated.itx',
                       xs=xs+[np.array([dat.datnum for dat in all_dats])],
                       datas=int_entropies+[np.array([dat.Logs.dacs['ESC'] for dat in all_dats])],
                       names=[f'integrated_{i}' for i in range(len(int_entropies))] + ['integrated_coupling_gates'],
                       x_labels=['Sweep Gate (mV)'] * len(int_entropies) + ['Datnum'],
                       y_labels=['dN/dT (nA)'] * len(int_entropies) + ['ESC (mV)'])

    # Plot Integrated Entropy
    fig, ax = plt.subplots(1, 1)
    ax = integrated_entropy(ax, xs=xs, datas=int_entropies, labels=[f'{dat.Logs.fds["ESC"]:.1f}' for dat in all_dats])
    ax.get_legend().set_title('Coupling Gate (mV)')
    plt.tight_layout()
    fig.show()

    ##################################################################

    # Data for amp and dT scaling factors for weakly coupled
    fit_name = 'forced_theta_linear'
    entropy_dats = get_dats(range(2095, 2136 + 1, 2))
    # entropy_dats = get_dats(range(7322, 7361 + 1, 2))[::4]
    # dats = get_dats(range(2164, 2170 + 1, 3))
    # entropy_dats = get_dats(chain(range(7322, 7361 + 1, 2), range(7378, 7399 + 1, 2), range(7400, 7421 + 1, 2)))
    entropy_dats = [dat for dat in entropy_dats if dat.Logs.fds['ESC'] < -0]
    entropy_dats = U.order_list(entropy_dats, [dat.Logs.fds['ESC'] for dat in entropy_dats])
    t_dats = [get_dat(dat.datnum+1) for dat in entropy_dats]

    outs = [dat.SquareEntropy.get_Outputs(name=fit_name) for dat in entropy_dats]
    int_infos = [dat.Entropy.get_integration_info(name=fit_name) for dat in entropy_dats]
    t_fits = [dat.Transition.get_fit(name='forced_gamma_zero') for dat in t_dats if dat.Logs.dacs['ESC'] < -285]

    thetas = np.array([fit.best_values.theta/100 for fit in t_fits])  # /100 to convert to mV
    lever_coupling = np.array([dat.Logs.dacs['ESC'] for dat in t_dats if dat.Logs.dacs['ESC'] < -285])
    levers = thetas  # TODO: Conver to lever arm


    amps = np.array([int_info.amp for int_info in int_infos])
    # amps = np.array([fit.best_values.amp for fit in t_fits])
    # dts = np.array([int_info.dT for int_info in int_infos])
    # sfs = np.array([int_info.sf for int_info in int_infos])
    cg_vals = np.array([dat.Logs.fds['ESC'] for dat in entropy_dats])

    line_coefs = [3.3933e-5, 0.05073841]  # slope, intercept

    U.save_to_igor_itx(file_path=f'fig2_amp_lever_for_weakly_coupled.itx',
                       xs=[cg_vals, lever_coupling, np.array([0, 1])],
                       datas=[amps, levers, np.array(line_coefs)],
                       names=['amplitudes', 'lever_arms', 'lever_line_coefs'],
                       x_labels=['Coupling Gate (mV)'] * 2 + ['slope/intercept'],
                       y_labels=['dI/dN (nA)', f'{ALPHA} (units???)'] + [''])

    # Plotting amp and dT scaling factors for weakly coupled
    fig, ax = plt.subplots(1, 1)
    amp_theta_vs_coupling(ax, amp_coupling=cg_vals, amps=amps,
                          lever_coupling=lever_coupling, levers=levers,
                          line_slope=line_coefs[0], line_intercept=line_coefs[1])
    plt.tight_layout()
    fig.show()

    ############################################################################
    # Data for entropy_vs_coupling
    fit_name = 'forced_theta_linear'
    all_dats = get_dats(range(2095, 2111 + 1, 2))
    # all_dats = get_dats(chain(range(7322, 7361 + 1, 2), range(7378, 7399 + 1, 2), range(7400, 7421 + 1, 2)))
    all_dats = [dat for dat in all_dats if dat.Logs.fds['ESC'] < -245]
    all_dats = U.order_list(all_dats, [dat.Logs.fds['ESC'] for dat in all_dats])

    int_cg_vals = np.array([dat.Logs.fds['ESC'] for dat in all_dats])
    # TODO: Need to make sure all these integrated entropies are being calculated at good poitns
    #  (i.e. not including slopes)
    integrated_entropies = np.array([np.nanmean(
        dat.Entropy.get_integrated_entropy(name=fit_name,
                                           data=dat.SquareEntropy.get_Outputs(
                                               name=fit_name, check_exists=True).average_entropy_signal
                                           )[-10:]) for dat in all_dats])

    fit_cg_vals = np.array([dat.Logs.fds['ESC'] for dat in all_dats if dat.Logs.fds['ESC'] < -245])
    fit_entropies = np.array(
        [dat.Entropy.get_fit(name=fit_name).best_values.dS for dat in all_dats if dat.Logs.fds['ESC'] < -245])

    save_to_igor_itx(file_path=f'fig2_entropy_weak_only.itx', xs=[fit_cg_vals, int_cg_vals],
                     datas=[fit_entropies, integrated_entropies],
                     names=['fit_entropy_weak_only', 'integrated_entropy_weak_only'],
                     x_labels=['Coupling Gate (mV)'] * 2,
                     y_labels=['Entropy (kB)', 'Entropy (kB)'])

    # Plot entropy_vs_coupling
    fig, ax = plt.subplots(1, 1)
    ax = entropy_vs_coupling(ax, int_coupling=int_cg_vals, int_entropy=integrated_entropies,
                             fit_coupling=fit_cg_vals, fit_entropy=fit_entropies)
    plt.tight_layout()
    fig.show()

    # #####################################################
    # # Data for amp_theta_vs_coupling
    # fit_name = 'forced_theta_linear'
    # dats = get_dats(range(2095, 2125 + 1, 2))  # Goes up to 2141 but the last few aren't great
    # tonly_dats = get_dats(range(2096, 2126 + 1, 2))
    #
    # amp_cg_vals = np.array([dat.Logs.fds['ESC'] for dat in tonly_dats])
    # amps = np.array([dat.Transition.get_fit(name=fit_name).best_values.amp for dat in tonly_dats])
    #
    # theta_cg_vals = np.array([dat.Logs.fds['ESC'] for dat in tonly_dats])
    # thetas = np.array([dat.Transition.get_fit(name=fit_name).best_values.theta for dat in tonly_dats])
    #
    # U.save_to_igor_itx(file_path=f'fig2_amp_theta_vs_coupling.itx', xs=[amp_cg_vals, theta_cg_vals], datas=[amps, thetas],
    #                    names=['amplitudes', 'thetas'],
    #                    x_labels=['Coupling Gate /mV'] * 2,
    #                    y_labels=['dI/dN /nA', 'Theta /mV'])
    #
    # # Plotting amp_theta_vs_coupling
    # fig, ax = plt.subplots(1, 1)
    # amp_theta_vs_coupling(ax, amp_coupling=amp_cg_vals, amps=amps,
    #                       dt_coupling=theta_cg_vals, dt=thetas)
    # plt.tight_layout()
    # fig.show()
