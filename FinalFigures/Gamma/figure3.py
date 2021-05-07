import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

from FinalFigures.Gamma.plots import integrated_entropy, entropy_vs_coupling, gamma_vs_coupling, amp_theta_vs_coupling
from src.UsefulFunctions import save_to_igor_itx

if __name__ == '__main__':
    from src.DatObject.Make_Dat import get_dats, get_dat

    ####################################################
    # Data for gamma_vs_coupling
    fit_name = 'forced_theta_linear'
    # dats = get_dats(range(2095, 2125 + 1, 2))  # Goes up to 2141 but the last few aren't great
    # tonly_dats = get_dats(range(2096, 2126 + 1, 2))
    tonly_dats = get_dats(range(7323, 7357 + 1, 2))  # TODO: extend slightly (dats still coming in)
    # Loading fitting done in Analysis.Feb2021.entropy_gamma_final

    gamma_cg_vals = np.array([dat.Logs.fds['ESC'] for dat in tonly_dats])
    gammas = np.array([dat.Transition.get_fit(name=fit_name).best_values.g for dat in tonly_dats])

    save_to_igor_itx(file_path=f'fig3_gamma_vs_coupling.itx', xs=[gamma_cg_vals], datas=[gammas],
                     names=['gammas'],
                     x_labels=['Coupling Gate /mV'],
                     y_labels=['Gamma /mV'])

    # Plotting gamma_vs_coupling
    fig, ax = plt.subplots(1, 1)
    ax = gamma_vs_coupling(ax, coupling_gates=gamma_cg_vals, gammas=gammas)
    plt.tight_layout()
    fig.show()

    ##########################################################################
    # Data for integrated_entropy
    fit_name = 'forced_theta_linear'
    # all_dats = get_dats(range(2164, 2170 + 1, 3)) + [get_dat(2216)]
    all_dats = get_dats(range(7322, 7357 + 1, 2))  # TODO: extend slightly (dats still coming in)
    tonly_dats = get_dats([dat.datnum + 1 for dat in all_dats])

    outs = [dat.SquareEntropy.get_Outputs(name=fit_name) for dat in all_dats]
    int_infos = [dat.Entropy.get_integration_info(name=fit_name) for dat in all_dats]

    xs = [out.x for out in outs]
    int_entropies = [int_info.integrate(out.average_entropy_signal) for int_info, out in zip(int_infos, outs)]
    gts = [dat.Transition.get_fit(name=fit_name).best_values.g / dat.Transition.get_fit(name=fit_name).best_values.theta
           for dat in tonly_dats]

    save_to_igor_itx(file_path=f'fig3_integrated_entropy.itx',
                     xs=xs + [np.arange(len(gts))],
                     datas=int_entropies + [np.array(gts)],
                     names=[f'int_entropy_{i}' for i in range(len(int_entropies))] + ['gts_for_int_entropies'],
                     x_labels=['Sweep Gate /mV'] * len(int_entropies) + ['index'],
                     y_labels=['Entropy /kB'] * len(int_entropies) + ['G/T'])

    # Plot Integrated Entropy
    fig, ax = plt.subplots(1, 1)
    integrated_entropy(ax, xs=xs, datas=int_entropies, labels=[f'{gt:.1f}' for gt in gts])
    plt.tight_layout()
    fig.show()

    ##################################################################

    # Data for amp and dT scaling
    fit_name = 'forced_theta_linear'
    # all_dats = get_dats(range(2095, 2125 + 1, 2))  # Goes up to 2141 but the last few aren't great
    all_dats = get_dats(range(7322, 7357 + 1, 2))  # TODO: extend slightly (dats still coming in)
    # dats = get_dats(range(2164, 2170 + 1, 3))

    outs = [dat.SquareEntropy.get_Outputs(name=fit_name) for dat in all_dats]
    int_infos = [dat.Entropy.get_integration_info(name=fit_name) for dat in all_dats]

    amps = np.array([int_info.amp for int_info in int_infos])
    dts = np.array([int_info.dT for int_info in int_infos])
    sfs = np.array([int_info.sf for int_info in int_infos])
    cg_vals = np.array([dat.Logs.fds['ESC'] for dat in all_dats])

    save_to_igor_itx(file_path=f'fig3_amp_dt_for_weakly_coupled.itx', xs=[cg_vals, cg_vals], datas=[amps, dts],
                       names=['amplitudes', 'dts'],
                       x_labels=['Coupling Gate /mV'] * 2,
                       y_labels=['dI/dN /nA', 'dT /mV'])

    # Plotting amp and dT scaling factors for weakly coupled
    fig, ax = plt.subplots(1, 1)
    amp_theta_vs_coupling(ax, amp_coupling=cg_vals, amps=amps,
                          dt_coupling=cg_vals, dt=dts)
    plt.tight_layout()
    fig.show()

  ############################################################################
    # Data for entropy_vs_coupling
    fit_name = 'forced_theta_linear'
    # all_dats = get_dats(range(2095, 2125 + 1, 2))  # Goes up to 2141 but the last few aren't great
    all_dats = get_dats(list(range(7322, 7361 + 1, 2)) + list(range(7378, 7399 + 1, 2)) + list(range(7400, 7421 + 1, 2)))
    # all_dats = get_dats(range(7322, 7357 + 1, 2))  # TODO: extend slightly (dats still coming in)
    # dats = get_dats(range(2095, 2111 + 1, 2))

    int_cg_vals = np.array([dat.Logs.fds['ESC'] for dat in all_dats])
    # TODO: Need to make sure all these integrated entropies are being calculated at good poitns (i.e. not including slopes)
    integrated_data = np.array([
        dat.Entropy.get_integrated_entropy(name=fit_name,
                                           data=dat.SquareEntropy.get_Outputs(
                                               name=fit_name, check_exists=True).average_entropy_signal
                                           ) for dat in all_dats])
    integrated_entropies = [np.nanmean(data[-10:]) for data in integrated_data]
    integrated_peaks = [np.nanmax(data) for data in integrated_data]

    fit_cg_vals = np.array([dat.Logs.fds['ESC'] for dat in all_dats if dat.Logs.fds['ESC'] < -200])
    fit_entropies = np.array(
        [dat.Entropy.get_fit(name=fit_name).best_values.dS for dat in all_dats if dat.Logs.fds['ESC'] < -200])

    save_to_igor_itx(file_path=f'fig2_entropy_vs_gamma.itx', xs=[fit_cg_vals, int_cg_vals, int_cg_vals],
                     datas=[fit_entropies, integrated_entropies, integrated_peaks],
                     names=['fit_entropy_vs_coupling', 'integrated_entropy_vs_coupling', 'integrated_peaks_vs_coupling'],
                     x_labels=['Coupling Gate /mV'] * 3,
                     y_labels=['Entropy /kB', 'Entropy /kB', 'Entropy /kB'])

    # Plot entropy_vs_coupling
    fig, ax = plt.subplots(1, 1)
    ax = entropy_vs_coupling(ax, int_coupling=int_cg_vals, int_entropy=integrated_entropies, int_peaks=integrated_peaks,
                             fit_coupling=fit_cg_vals, fit_entropy=fit_entropies)
    plt.tight_layout()
    fig.show()

