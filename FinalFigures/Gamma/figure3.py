import matplotlib.pyplot as plt
import numpy as np
import json
from itertools import chain
import lmfit as lm

from FinalFigures.Gamma.plots import integrated_entropy, entropy_vs_coupling, gamma_vs_coupling, amp_theta_vs_coupling, \
    amp_sf_vs_coupling
from src.useful_functions import save_to_igor_itx, order_list, fig_to_data_json, data_to_json
from src.plotting.Mpl.PlotUtil import set_default_rcParams


def fit_line(x, data) -> lm.model.ModelResult:
    line = lm.models.LinearModel()
    pars = line.guess(data, x=x)
    fit = line.fit(data=data.astype(np.float32), x=x.astype(np.float32), params=pars)
    return fit


if __name__ == '__main__':
    from src.dat_object.make_dat import get_dats, get_dat

    set_default_rcParams()

    ####################################################
    # Data for gamma_vs_coupling
    # fit_name = 'forced_theta_linear'
    fit_name = 'forced_theta'
    dats = get_dats(range(2095, 2135 + 1, 2))
    dats = [dat for dat in dats if dat.datnum != 2127]
    tonly_dats = get_dats([dat.datnum + 1 for dat in dats if dat.Logs.dacs['ESC'] > -285])
    # tonly_dats = get_dats(chain(range(7323, 7361 + 1, 2), range(7379, 7399 + 1, 2), range(7401, 7421 + 1, 2)))
    # tonly_dats = order_list(tonly_dats, [dat.Logs.fds['ESC'] for dat in tonly_dats])
    # tonly_dats = [dat for dat in tonly_dats if dat.Logs.fds['ESC'] > -245
    #               and dat.datnum < 7362 and dat.datnum not in [7349, 7351]]  # So no duplicates
    # Loading fitting done in Analysis.Feb2021.entropy_gamma_final

    gamma_cg_vals = np.array([dat.Logs.fds['ESC'] for dat in tonly_dats])
    # gammas = np.array([dat.Transition.get_fit(name=fit_name).best_values.g for dat in tonly_dats])
    gammas = np.array([dat.NrgOcc.get_fit(name=fit_name).best_values.g for dat in tonly_dats])
    thetas = np.array([dat.NrgOcc.get_fit(name=fit_name).best_values.theta for dat in tonly_dats])
    gts = gammas/thetas

    fit = fit_line(gamma_cg_vals[np.where(gamma_cg_vals < -235)],
                   np.log10(gts[np.where(gamma_cg_vals < -235)]))
    print(f'Line fit to Log10(gamma/kT) vs coupling: \n'
          f'{fit.best_values}')

    save_to_igor_itx(file_path=f'fig3_gamma_vs_coupling.itx', xs=[gamma_cg_vals], datas=[gts],
                     names=['gamma_over_ts'],
                     x_labels=['Coupling Gate (mV)'],
                     y_labels=['Gamma/kT'])

    # plotting gamma_vs_coupling
    fig, ax = plt.subplots(1, 1)
    ax = gamma_vs_coupling(ax, coupling_gates=gamma_cg_vals, gammas=gts)
    ax.plot((x_ := np.linspace(-375, -235)), np.power(10, fit.eval(x=x_)), linestyle=':')
    plt.tight_layout()
    fig.show()

    ##########################################################################
    # Data for integrated_entropy
    fit_name = 'forced_theta_linear'
    temp_NRG_fit_name = 'forced_theta'  # Need to actually recalculate entropy scaling using this NRG fit as well
    # all_dats = get_dats(range(2164, 2170 + 1, 3)) + [get_dat(2121)]  # + [get_dat(2213)]
    all_dats = get_dats([2164, 2121, 2167, 2133])

    # all_dats = get_dats(chain(range(7322, 7361 + 1, 2), range(7378, 7399+1, 2), range(7400, 7421+1, 2)))
    # all_dats = order_list(all_dats, [dat.Logs.fds['ESC'] for dat in all_dats])
    # all_dats = [dat for dat in all_dats if
    #             0.74 < integrated_entropy_value(dat, fit_name) < 0.76]

    # all_dats = get_dats([7404, 7342, 7350, 7358])

    tonly_dats = get_dats([dat.datnum + 1 for dat in all_dats])

    outs = [dat.SquareEntropy.get_Outputs(name=fit_name) for dat in all_dats]
    int_infos = [dat.Entropy.get_integration_info(name=fit_name) for dat in all_dats]

    xs = [out.x / 100 for out in outs]  # /100 to convert to real mV
    int_entropies = [int_info.integrate(out.average_entropy_signal) for int_info, out in zip(int_infos, outs)]
    gts = [dat.NrgOcc.get_fit(name=temp_NRG_fit_name).best_values.g / dat.NrgOcc.get_fit(name=temp_NRG_fit_name).best_values.theta
           for dat in tonly_dats]

    save_to_igor_itx(file_path=f'fig3_integrated_entropy.itx',
                     xs=xs + [np.arange(len(gts))],
                     datas=int_entropies + [np.array(gts)],
                     names=[f'int_entropy_{k}' for k in ['weak', 'med', 'strong', 'similar']] + ['gts_for_int_vary_g'],
                     x_labels=['Sweep Gate (mV)'] * len(int_entropies) + ['index'],
                     y_labels=['Entropy (kB)'] * len(int_entropies) + ['G/T'])

    # data_to_json(datas=int_entropies + xs,
    #              names=[f'int_entropy_{i}' for i, _ in enumerate(int_entropies)] +
    #                    [f'int_entropy_{i}_x' for i, _ in enumerate(xs)],
    #              filepath=r'D:\GitHub\dat_analysis\dat_analysis\Analysis\Feb2021\figs/data/3IntegratedEntropies.json'),

    # Plot Integrated Entropy
    fig, ax = plt.subplots(1, 1)
    integrated_entropy(ax, xs=xs, datas=int_entropies, labels=[f'{gt:.1f}' for gt in gts])
    ax.set_xlim(-5, 5)
    plt.tight_layout()
    fig.show()

    ##################################################################

    # # Data for amp and dT scaling
    # fit_name = 'forced_theta_linear'
    # all_dats = get_dats(range(2095, 2125 + 1, 2))  # Goes up to 2141 but the last few aren't great
    # # all_dats = get_dats(chain(range(7322, 7361 + 1, 2), range(7378, 7399 + 1, 2), range(7400, 7421 + 1, 2)))
    # # all_dats = order_list(all_dats, [dat.Logs.fds['ESC'] for dat in all_dats])
    # # dats = get_dats(range(2164, 2170 + 1, 3))
    #
    # outs = [dat.SquareEntropy.get_Outputs(name=fit_name) for dat in all_dats]
    # int_infos = [dat.Entropy.get_integration_info(name=fit_name) for dat in all_dats]
    #
    # amps = np.array([int_info.amp for int_info in int_infos])
    # dts = np.array([int_info.dT for int_info in int_infos])
    # sfs = np.array([int_info.sf for int_info in int_infos])
    # cg_vals = np.array([dat.Logs.fds['ESC'] for dat in all_dats])
    #
    # save_to_igor_itx(file_path=f'fig3_amp_dt_for_weakly_coupled.itx', xs=[cg_vals, cg_vals, cg_vals],
    #                  datas=[amps, dts, sfs],
    #                  names=['amplitudes', 'dts', 'sfs'],
    #                  x_labels=['Coupling Gate (mV)'] * 3,
    #                  y_labels=['dI/dN (nA)', 'dT (mV)', 'dI/dN*dT'])
    #
    # # plotting amp and dT scaling factors for weakly coupled
    # fig, ax = plt.subplots(1, 1)
    # amp_theta_vs_coupling(ax, amp_coupling=cg_vals, amps=amps, dt_coupling=cg_vals, dt=dts)
    # plt.tight_layout()
    # fig.show()
    #
    # fig, ax = plt.subplots(1, 1)
    # amp_sf_vs_coupling(ax, amp_coupling=cg_vals, amps=amps, sf_coupling=cg_vals, sf=sfs)
    # plt.tight_layout()
    # fig.show()

    ############################################################################
    # Data for entropy_vs_coupling
    fit_name = 'forced_theta_linear'
    all_dats = get_dats(range(2095, 2136 + 1, 2))  # Goes up to 2141 but the last few aren't great
    # all_dats = get_dats(chain(range(7322, 7361 + 1, 2), range(7378, 7399 + 1, 2), range(7400, 7421 + 1, 2)))
    # all_dats = order_list(all_dats, [dat.Logs.fds['ESC'] for dat in all_dats])
    # dats = get_dats(range(2095, 2111 + 1, 2))
    all_dats = [dat for dat in all_dats if dat.datnum != 2127]
    all_dats = order_list(all_dats, [dat.Logs.dacs['ESC'] for dat in all_dats])

    int_cg_vals = np.array([dat.Logs.fds['ESC'] for dat in all_dats])
    # TODO: Need to make sure all these integrated entropies are being calculated at good poitns (i.e. not including slopes)
    integrated_data = np.array([
        dat.Entropy.get_integrated_entropy(name=fit_name,
                                           data=dat.SquareEntropy.get_Outputs(
                                               name=fit_name, check_exists=True).average_entropy_signal
                                           ) for dat in all_dats])
    integrated_entropies = [np.nanmean(data[-10:]) for data in integrated_data]
    integrated_peaks = [np.nanmax(data) for data in integrated_data]
    peak_cg_vals = int_cg_vals
    peak_diffs = [p - v for p, v in zip(integrated_peaks, integrated_entropies)]

    fit_cg_vals = np.array([dat.Logs.fds['ESC'] for dat in all_dats if dat.Logs.fds['ESC'] < -250])
    fit_entropies = np.array(
        [dat.Entropy.get_fit(name=fit_name).best_values.dS for dat in all_dats if dat.Logs.fds['ESC'] < -250])

    save_to_igor_itx(file_path=f'fig3_entropy_vs_gamma.itx', xs=[fit_cg_vals, int_cg_vals, int_cg_vals, peak_cg_vals],
                     datas=[fit_entropies, integrated_entropies, integrated_peaks, peak_diffs],
                     names=['fit_entropy_vs_coupling', 'integrated_entropy_vs_coupling',
                            'integrated_peaks_vs_coupling', 'integrated_peak_sub_end'],
                     x_labels=['Coupling Gate (mV)'] * 4,
                     y_labels=['Entropy (kB)', 'Entropy (kB)', 'Entropy (kB)', 'Entropy (kB)'])

    # Plot entropy_vs_coupling
    fig, ax = plt.subplots(1, 1)
    ax = entropy_vs_coupling(ax, int_coupling=int_cg_vals, int_entropy=integrated_entropies, int_peaks=integrated_peaks,
                             fit_coupling=fit_cg_vals, fit_entropy=fit_entropies,
                             peak_diff_coupling=peak_cg_vals,
                             peak_diff=peak_diffs
                             )
    plt.tight_layout()
    fig.show()
