from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import lmfit as lm
from itertools import chain
from typing import TYPE_CHECKING, List

import src.useful_functions as U
from src.characters import ALPHA
from src.constants import kb
from src.analysis_tools.entropy import integrated_entropy_value
from src.useful_functions import save_to_igor_itx
from src.plotting.Mpl.PlotUtil import set_default_rcParams
from temp import get_avg_i_sense_data, get_avg_entropy_data, _center_func, get_integrated_data, calc_int_info
from FinalFigures.Gamma.plots import gamma_vs_coupling, amp_theta_vs_coupling, dndt_signal, integrated_entropy, entropy_vs_coupling
from src.analysis_tools.general_fitting import FitInfo
from src.analysis_tools.nrg import NrgUtil, NRGParams
from src.plotting.plotly import OneD, Data1D


if TYPE_CHECKING:
    from src.dat_object.dat_hdf import DatHDF

p1d = OneD(dat=None)

def fit_line(x, data) -> FitInfo:
    line = lm.models.LinearModel()
    fit = FitInfo.from_fit(line.fit(data, x=x, nan_policy='omit'))
    return fit


def get_N_vs_sweepgate(dat: DatHDF, fit_name: str = 'gamma_small') -> Data1D:
    """Quick fn for Josh, to get Occupation vs sweep to see that peak is at 2/3"""
    fit = dat.NrgOcc.get_fit(name=fit_name)
    nrg = NrgUtil(NRGParams.from_lm_params(fit.params))
    x = dat.NrgOcc.avg_x
    occ = nrg.data_from_params(x=x, which_data='occupation')
    occ.x = occ.x/100  # Convert to real mV
    return occ


def save_data1ds_to_igor_itx(filepath: str, datas: List[Data1D], names: List[str]):
    U.save_to_igor_itx(file_path=filepath,
                       xs=[d.x for d in datas],
                       datas=[d.data for d in datas],
                       names=[n for n in names],
                       x_labels='V_D (mV)',
                       y_labels='Occupation')


if __name__ == '__main__':
    set_default_rcParams()
    from src.dat_object.make_dat import get_dats, get_dat

    # csq_datnum = 2197
    csq_datnum = None
    # Data for weakly coupled dN/dTs
    all_dats = get_dats([2097, 2103, 2107])

    all_dats = U.order_list(all_dats, [dat.Logs.fds['ESC'] for dat in all_dats])

    datas = [get_avg_entropy_data(dat, center_func=_center_func, csq_datnum=csq_datnum) for dat in all_dats]
    xs = [data.x/100 for data in datas]  # /100 to convert to real mV
    dndts = [data.data for data in datas]

    U.save_to_igor_itx(file_path=f'fig2_dndt.itx',
                       xs=xs+[np.array([dat.datnum for dat in all_dats])],
                       datas=dndts+[np.array([dat.Logs.dacs['ESC'] for dat in all_dats])],
                       names=[f'dndt_stacked_{i}' for i in range(len(dndts))] + ['stacked_coupling_gates'],
                       x_labels=['Sweep Gate (mV)'] * len(dndts) + ['Datnum'],
                       y_labels=['dN/dT (nA)'] * len(dndts) + ['ESC (mV)'])

    # plotting dNdT for several weakly coupled
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
    # fit_name = 'csq_gamma_small'
    fit_name = 'gamma_small'
    all_dats = get_dats([2097, 2103, 2109])

    # all_dats = [dat for dat in all_dats if dat.Logs.fds['ESC'] < -245]
    all_dats = U.order_list(all_dats, [dat.Logs.fds['ESC'] for dat in all_dats])

    integrated_datas = [get_integrated_data(dat, fit_name=fit_name,
                                            zero_point=-100,
                                            csq_datnum=csq_datnum,
                                            which_linear_theta_fit='csq mapped') for dat in all_dats]
    for data in integrated_datas:
        data.x = data.x/100  # Convert to real mV

    U.save_to_igor_itx(file_path=f'fig2_integrated.itx',
                       xs=[data.x for data in integrated_datas]+[np.array([dat.datnum for dat in all_dats])],
                       datas=[data.data for data in integrated_datas]+[np.array([dat.Logs.dacs['ESC'] for dat in all_dats])],
                       names=[f'integrated_{i}' for i in range(len(integrated_datas))] + ['integrated_coupling_gates'],
                       x_labels=['Sweep Gate (mV)'] * len(integrated_datas) + ['Datnum'],
                       y_labels=['Entropy (kB)'] * len(integrated_datas) + ['ESC (mV)'])

    # Plot Integrated Entropy
    fig, ax = plt.subplots(1, 1)
    ax = integrated_entropy(ax, xs=[data.x for data in integrated_datas],
                            datas=[data.data for data in integrated_datas],
                            labels=[f'{dat.Logs.fds["ESC"]:.1f}' for dat in all_dats])
    ax.get_legend().set_title('Coupling Gate (mV)')
    plt.tight_layout()
    fig.show()

    ##################################################################

    # Data for amp and dT scaling factors for weakly coupled
    # lever_fit_name = 'csq_gamma_small'  # Name of fit which was used to generate linear theta
    # which_linear_theta_fit = 'csq mapped'  # Name of linear theta fit params
    # weak_fit_name = 'csq_gamma_small'
    # strong_fit_name = 'csq_forced_theta'
    lever_fit_name = 'gamma_small'  # Name of fit which was used to generate linear theta
    which_linear_theta_fit = 'normal'  # Name of linear theta fit params
    weak_fit_name = 'gamma_small'
    strong_fit_name = 'forced_theta'
    strong_gamma_cutoff = 1  # If gamma of strong_fit_name fit > strong_gamma_cutoff then use strong_fit_name
    # otherwise weak_fit_name
    entropy_dats = get_dats(range(2095, 2136 + 1, 2))
    entropy_dats = [dat for dat in entropy_dats if dat.datnum not in [2125]]
    entropy_dats = U.order_list(entropy_dats, [dat.Logs.fds['ESC'] for dat in entropy_dats])
    t_dats = [get_dat(dat.datnum+1) for dat in entropy_dats]

    thetas, lever_coupling, levers = [], [], []
    # for dat in t_dats:
    for dat in entropy_dats+t_dats:
        fit = dat.NrgOcc.get_fit(name=lever_fit_name)
        theta = fit.best_values.theta/100
        thetas.append(theta)  # /100 to convert to mV
        if dat.Logs.dacs['ESC'] < -285:
            lever_coupling.append(dat.Logs.dacs['ESC'])
            levers.append(kb*0.1/theta)  # Convert to lever arm (kbT/Theta = alpha*e)

    amps, cg_vals, gammas = [], [], []
    for dat in entropy_dats:
        strong_fit = dat.NrgOcc.get_fit(name=strong_fit_name)
        g = strong_fit.best_values.g
        gammas.append(g)
        if g > strong_gamma_cutoff:
            fit_name = strong_fit_name
        else:
            fit_name = weak_fit_name
        int_info = calc_int_info(dat, fit_name=fit_name, which_linear_theta_fit=which_linear_theta_fit)
        amps.append(int_info.amp)
        cg_vals.append(dat.Logs.dacs['ESC'])

    thetas, lever_coupling, levers, amps, cg_vals = [np.array(arr) for arr in
                                                     [thetas, lever_coupling, levers, amps, cg_vals]]

    line_fit = fit_line(lever_coupling, levers)
    line_coefs = (line_fit.best_values.slope, line_fit.best_values.intercept)
    print(f'Lever fit (slope, intercept): {line_coefs[0]:.4g}, {line_coefs[1]:.4f}')
    # line_coefs = [3.3933e-5, 0.05073841]  # slope, intercept

    U.save_to_igor_itx(file_path=f'fig2_amp_lever_for_weakly_coupled.itx',
                       xs=[cg_vals, lever_coupling, np.array([0, 1])],
                       datas=[amps, levers, np.array(line_coefs)],
                       names=['amplitudes', 'lever_arms', 'lever_line_coefs'],
                       x_labels=['Coupling Gate (mV)'] * 2 + ['slope/intercept'],
                       y_labels=['dI/dN (nA)', f'{ALPHA} (unitless)'] + [''])

    # plotting amp and dT scaling factors for weakly coupled
    fig, ax = plt.subplots(1, 1)
    amp_theta_vs_coupling(ax, amp_coupling=cg_vals, amps=amps,
                          lever_coupling=lever_coupling, levers=levers,
                          line_slope=line_coefs[0], line_intercept=line_coefs[1])
    plt.tight_layout()
    fig.show()

    ############################################################################
    # Data for entropy_vs_coupling
    # weak_fit_name = 'csq_gamma_small'
    # strong_fit_name = 'csq_forced_theta'
    # which_linear_theta_fit = 'csq mapped'  # Name of linear theta fit params
    weak_fit_name = 'gamma_small'
    strong_fit_name = 'forced_theta'
    which_linear_theta_fit = 'normal'  # Name of linear theta fit params
    strong_gamma_cutoff = 1  # If gamma of strong_fit_name fit > strong_gamma_cutoff then use strong_fit_name

    all_dats = get_dats(range(2095, 2111 + 1, 2))
    # all_dats = get_dats(chain(range(7322, 7361 + 1, 2), range(7378, 7399 + 1, 2), range(7400, 7421 + 1, 2)))
    all_dats = [dat for dat in all_dats if dat.Logs.fds['ESC'] < -245]
    all_dats = U.order_list(all_dats, [dat.Logs.fds['ESC'] for dat in all_dats])

    cg_vals, int_entropies, fit_entropies = [], [], []
    for dat in all_dats:
        w_val = 100
        strong_fit = dat.NrgOcc.get_fit(name=strong_fit_name)
        if strong_fit.best_values.g > strong_gamma_cutoff:
            fit_name = strong_fit_name
        else:
            fit_name = weak_fit_name

        int_data = get_integrated_data(dat, fit_name=fit_name, zero_point=-w_val,
                                       csq_datnum=csq_datnum, which_linear_theta_fit=which_linear_theta_fit)

        idx = U.get_data_index(int_data.x, w_val)  # Measure entropy at x = xxx
        if idx >= int_data.x.shape[-1] - 1:  # or the end if it doesn't reach xxx
            idx -= 10
        int_entropy = np.nanmean(int_data.data[idx - 10:idx + 10])

        avg_dndt = get_avg_entropy_data(dat, center_func=_center_func, csq_datnum=csq_datnum)
        fit = dat.Entropy.get_fit(calculate_only=True, data=avg_dndt.data, x=avg_dndt.x)

        cg_vals.append(dat.Logs.dacs['ESC'])
        int_entropies.append(int_entropy)
        fit_entropies.append(fit.best_values.dS)

    cg_vals, int_entropies, fit_entropies = [np.array(arr) for arr in [cg_vals, int_entropies, fit_entropies]]

    save_to_igor_itx(file_path=f'fig2_entropy_weak_only.itx', xs=[cg_vals, cg_vals],
                     datas=[fit_entropies, int_entropies],
                     names=['fit_entropy_weak_only', 'integrated_entropy_weak_only'],
                     x_labels=['Coupling Gate (mV)'] * 2,
                     y_labels=['Entropy (kB)', 'Entropy (kB)'])

    # Plot entropy_vs_coupling
    fig, ax = plt.subplots(1, 1)
    ax = entropy_vs_coupling(ax, int_coupling=cg_vals, int_entropy=int_entropies,
                             fit_coupling=cg_vals, fit_entropy=fit_entropies)
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
    # # plotting amp_theta_vs_coupling
    # fig, ax = plt.subplots(1, 1)
    # amp_theta_vs_coupling(ax, amp_coupling=amp_cg_vals, amps=amps,
    #                       dt_coupling=theta_cg_vals, dt=thetas)
    # plt.tight_layout()
    # fig.show()
