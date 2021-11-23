from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import lmfit as lm
from typing import TYPE_CHECKING, Tuple
from itertools import chain

from FinalFigures.Gamma.plots import integrated_entropy, entropy_vs_coupling, gamma_vs_coupling
from dat_analysis.useful_functions import save_to_igor_itx, order_list, get_data_index, resample_data
from dat_analysis.plotting.mpl.PlotUtil import set_default_rcParams
from dat_analysis.plotting.plotly import OneD
from dat_analysis.core_util import Data1D
import dat_analysis.useful_functions as U

from temp import get_avg_entropy_data, get_integrated_data, _center_func

p1d = OneD(dat=None)

if TYPE_CHECKING:
    from dat_analysis.dat_object.dat_hdf import DatHDF


def fit_line(x, data) -> lm.model.ModelResult:
    line = lm.models.LinearModel()
    pars = line.guess(data, x=x)
    fit = line.fit(data=data.astype(np.float32), x=x.astype(np.float32), params=pars)
    return fit


def plot_integrated(dat: DatHDF, zero_point=-500):
    data = get_integrated_data(dat, fit_name='forced_theta', zero_point=zero_point)
    fig = p1d.plot(data=data.data, x=data.x, title=f'Dat{dat.datnum}: Integrated entropy with zero at {zero_point}',
                   trace_name=f'Dat{dat.datnum}')
    return fig


def get_nrg_integrated(dat: DatHDF) -> Data1D:
    from temp import get_avg_i_sense_data, get_centered_x_at_half_occ, get_centers, get_2d_data
    from dat_analysis.analysis_tools.nrg import NrgUtil, NRGParams
    # i_data = get_avg_i_sense_data(dat, None, _center_func, overwrite=False, hot_or_cold='hot')
    data2d = get_2d_data(dat, hot_or_cold='cold', csq_datnum=None)
    if _center_func(dat):
        centers = get_centers(dat, data2d, name=None, overwrite=False)
    else:
        centers = [0]*data2d.data.shape[0]
    hot_data2d = get_2d_data(dat, hot_or_cold='hot', csq_datnum=None)
    avg_data, avg_x = dat.NrgOcc.get_avg_data(x=data2d.x, data=hot_data2d.data, centers=centers, return_x=True,
                                     name='hot_cold_centers', check_exists=False, overwrite=False)
    i_data = Data1D(avg_x, avg_data)
    i_data.x = get_centered_x_at_half_occ(dat, None, fit_name='csq_forced_theta')
    init_fit = dat.NrgOcc.get_fit(name='csq_forced_theta')
    new_pars = U.edit_params(init_fit.params, ['theta', 'g'], [None, None], vary=[True, False])
    nrg = NrgUtil(None)
    new_fit = nrg.get_fit(i_data.x, i_data.data, initial_params=new_pars, which_data='i_sense')

    expected_data = nrg.data_from_params(NRGParams.from_lm_params(new_fit.params), x=i_data.x, which_data='dndt')
    expected_data.data = np.cumsum(expected_data.data)
    expected_data.data = expected_data.data/expected_data.data[-1]*np.log(2)
    expected_data.x = expected_data.x/100  # Convert to real mV
    return expected_data

def invert_nrg_fit_params(x: np.ndarray, data: np.ndarray, gamma, theta, mid, amp, lin, const, occ_lin,
                          data_type: str = 'i_sense') -> Tuple[np.ndarray, np.ndarray]:

    if data_type in ['i_sense', 'i_sense_cold', 'i_sense_hot']:
        # new_data = 1/(amp * (1 + occ_lin * (x - mid))) * data - lin * (x-mid) - const # - 1/2
        # new_data = 1/(amp * (1 + 0 * (x - mid))) * data - lin * (x-mid) - const # - 1/2
        new_data = (data - lin * (x - mid) - const + amp / 2) / (amp * (1 + occ_lin * (x - mid)))
    else:
        new_data = data
    # new_x = (x - mid - gamma*(-1.76567) - theta*(-1)) / gamma
    new_x = (x - mid) / gamma
    # new_x = (x - mid)
    return new_x, new_data


if __name__ == '__main__':
    from dat_analysis.dat_object.make_dat import get_dats

    set_default_rcParams()

    csq_datnum = 2197
    which_linear_theta_fit = 'csq mapped'
    # ####################################################
    # # Data for gamma_vs_coupling
    # # fit_name = 'forced_theta_linear'
    # fit_name = 'csq_forced_theta'
    # dats = get_dats(range(2095, 2135 + 1, 2))
    # dats = [dat for dat in dats if dat.datnum != 2127]
    # gamma_dats = [dat for dat in dats if dat.Logs.dacs['ESC'] > -285]
    #
    # gamma_cg_vals = np.array([dat.Logs.fds['ESC'] for dat in gamma_dats])
    # # gammas = np.array([dat.Transition.get_fit(name=fit_name).best_values.g for dat in tonly_gamma_dats])
    # gammas = np.array([dat.NrgOcc.get_fit(name=fit_name).best_values.g for dat in gamma_dats])
    # thetas = np.array([dat.NrgOcc.get_fit(name=fit_name).best_values.theta for dat in gamma_dats])
    # gts = gammas / thetas
    #
    # fit = fit_line(gamma_cg_vals[np.where(gamma_cg_vals < -235)],
    #                np.log10(gts[np.where(gamma_cg_vals < -235)]))
    # print(f'Line fit to Log10(gamma/kT) vs coupling: \n'
    #       f'{fit.best_values}')
    #
    # save_to_igor_itx(file_path=f'fig3_gamma_vs_coupling.itx', xs=[gamma_cg_vals], datas=[gts],
    #                  names=['gamma_over_ts'],
    #                  x_labels=['Coupling Gate (mV)'],
    #                  y_labels=['Gamma/kT'])
    #
    # # plotting gamma_vs_coupling
    # fig, ax = plt.subplots(1, 1)
    # ax = gamma_vs_coupling(ax, coupling_gates=gamma_cg_vals, gammas=gts)
    # ax.plot((x_ := np.linspace(-375, -235)), np.power(10, fit.eval(x=x_)), linestyle=':')
    # plt.tight_layout()
    # fig.show()
    #
    ##########################################################################
    # Data for integrated_entropy
    nrg_fit_name = 'csq_forced_theta'
    entropy_dats = get_dats([2164, 2121, 2167, 2133])
    # tonly_dats = get_dats([dat.datnum + 1 for dat in entropy_dats])
    tonly_dats = get_dats([dat.datnum for dat in entropy_dats])

    datas, gts = [], []
    fit_datas = []
    for dat, tonly_dat in zip(entropy_dats, tonly_dats):
        data = get_integrated_data(dat, fit_name=nrg_fit_name, zero_point=-350,
                                   csq_datnum=csq_datnum,
                                   which_linear_theta_fit=which_linear_theta_fit)
        fit = tonly_dat.NrgOcc.get_fit(name=nrg_fit_name)
        gt = fit.best_values.g / \
             fit.best_values.theta

        data.x = data.x / 100  # Convert to real mV
        datas.append(data)
        gts.append(gt)
        fit_datas.append(get_nrg_integrated(dat))

    save_to_igor_itx(file_path=f'fig3_integrated_entropy.itx',
                     xs=[data.x for data in datas] + [data.x for data in fit_datas] + [np.arange(len(gts))],
                     datas=[data.data for data in datas] + [data.data for data in fit_datas] + [np.array(gts)],
                     names=[f'int_entropy_{k}' for k in ['weak', 'similar', 'med', 'strong']] +
                           [f'int_entropy_{k}_fit' for k in ['weak', 'similar', 'med', 'strong']] +
                           ['gts_for_int_vary_g'],
                     x_labels=['Sweep Gate (mV)'] * len(datas)*2 + ['index'],
                     y_labels=['Entropy (kB)'] * len(datas)*2 + ['G/T'])

    # Plot Integrated Entropy
    fig, ax = plt.subplots(1, 1)
    integrated_entropy(ax, xs=[data.x for data in datas], datas=[data.data for data in datas],
                       labels=[f'{gt:.1f}' for gt in gts])
    for data in fit_datas:
        ax.plot(data.x, data.data, label='fit')
    ax.set_xlim(-5, 5)
    plt.tight_layout()
    fig.show()

    ################################################################
    # Data for occupation data
    from temp import get_avg_i_sense_data, get_centered_x_at_half_occ
    from dat_analysis.analysis_tools.nrg import NrgUtil, NRGParams
    nrg_fit_name = nrg_fit_name  # Use same as above
    entropy_dats = entropy_dats  # Use same as above

    nrg = NrgUtil()
    # which = 'occupation'  # i_sense or occupation
    which = 'i_sense'  # i_sense or occupation
    datas = []
    nrg_datas = []
    fits = []
    for dat in entropy_dats:
        data = get_avg_i_sense_data(dat, csq_datnum, None, hot_or_cold='hot')
        data.x = get_centered_x_at_half_occ(dat, csq_datnum=csq_datnum, fit_name=nrg_fit_name)
        data.data = U.decimate(data.data, measure_freq=dat.Logs.measure_freq, numpnts=200)
        data.x = U.get_matching_x(data.x, data.data)

        start_fit = dat.NrgOcc.get_fit(name=nrg_fit_name)
        fit = dat.NrgOcc.get_fit(initial_params=start_fit.params, data=data.data, x=data.x, calculate_only=True)
        expected_data = nrg.data_from_params(NRGParams.from_lm_params(fit.params), x=data.x, which_data=which)

        if which == 'occupation':
            bv = fit.best_values
            data.x, data.data = invert_nrg_fit_params(data.x, data.data, bv.g, bv.theta, bv.mid, bv.amp, bv.lin, bv.const, bv.occ_lin, data_type='i_sense')
            data.data = data.data * -1 + 1
            data.x = data.x * bv.g + bv.mid
            # data.x = data.x * -1
        else:
            const = fit.best_values.const
            data.data = data.data - const
            expected_data.data = expected_data.data - const
        datas.append(data)
        nrg_datas.append(expected_data)
        fits.append(fit)

    ylabel = 'I_sense (nA)' if which == 'i_sense' else 'Occupation'
    name = 'transition' if which == 'i_sense' else 'occupation'

    save_to_igor_itx(file_path=f'fig3_{which}.itx',
                     xs=[data.x/100 for data in datas] + [data.x for data in nrg_datas],
                     datas=[data.data for data in datas] + [data.data for data in nrg_datas],
                     names=list(chain.from_iterable([[f'{name}_{t}_{k}' for k in ['weak', 'similar', 'med', 'strong']] for t in ['data', 'nrg']])),
                     x_labels=['Sweep Gate (mV)']*4*2,
                     y_labels=[ylabel]*4*2)

    # Plot occupation data
    fig, ax = plt.subplots(1, 1)
    for data, nrg_data, name, fit in zip(datas, nrg_datas, ['weak', 'similar', 'med', 'strong'], fits):
        ax.plot(data.x/100, data.data, label=name)
        ax.plot(nrg_data.x/100, nrg_data.data, label=f'{name}_nrg', linestyle=":")
    ax.plot()


    ##################################################################

    # # Data for entropy_vs_coupling
    # fit_name = 'csq_forced_theta'
    # entropy_dats = get_dats(range(2095, 2136 + 1, 2))  # Goes up to 2141 but the last few aren't great
    # entropy_dats = [dat for dat in entropy_dats if dat.datnum != 2127]
    # entropy_dats = order_list(entropy_dats, [dat.Logs.dacs['ESC'] for dat in entropy_dats])
    #
    # int_cg_vals = np.array([dat.Logs.fds['ESC'] for dat in entropy_dats])
    # peak_cg_vals = int_cg_vals
    # fit_lim = -260
    # fit_cg_vals = np.array([dat.Logs.fds['ESC'] for dat in entropy_dats if dat.Logs.fds['ESC'] < fit_lim])
    #
    # integrated_data = []
    # integrated_entropies, integrated_peaks = [], []
    # fit_entropies = []
    #
    # for dat in entropy_dats:
    #     fit = dat.NrgOcc.get_fit(name=nrg_fit_name)
    #     w_val = max(100, 10*fit.best_values.g)  # Width for zeroing and measuring int entropy
    #     integrated = get_integrated_data(dat, fit_name=nrg_fit_name, zero_point=-w_val,
    #                                      csq_datnum=csq_datnum,
    #                                      which_linear_theta_fit=which_linear_theta_fit)
    #
    #     idx = get_data_index(integrated.x, w_val)  # Measure entropy at x = xxx
    #     if idx >= integrated.x.shape[-1] - 1:  # or the end if it doesn't reach xxx
    #         idx -= 10
    #     integrated_entropies.append(np.nanmean(integrated.data[idx - 10:idx + 10]))
    #     integrated_peaks.append(np.nanmax(resample_data(integrated.data, max_num_pnts=50)))
    #
    #     if dat.Logs.dacs['ESC'] < fit_lim:
    #         avg_dndt = get_avg_entropy_data(dat,
    #                                         center_func=_center_func,
    #                                         csq_datnum=csq_datnum)
    #         fit_entropies.append(
    #             dat.Entropy.get_fit(data=avg_dndt.data, x=avg_dndt.x, calculate_only=True).best_values.dS)
    #
    # integrated_entropies = np.array(integrated_entropies)
    # integrated_peaks = np.array(integrated_peaks)
    # fit_entropies = np.array(fit_entropies)
    #
    # save_to_igor_itx(file_path=f'fig3_entropy_vs_gamma.itx', xs=[fit_cg_vals, int_cg_vals, int_cg_vals],
    #                  datas=[fit_entropies, integrated_entropies, integrated_peaks],
    #                  names=['fit_entropy_vs_coupling', 'integrated_entropy_vs_coupling',
    #                         'integrated_peaks_vs_coupling'],
    #                  x_labels=['Coupling Gate (mV)'] * 3,
    #                  y_labels=['Entropy (kB)', 'Entropy (kB)', 'Entropy (kB)'])
    #
    # # Plot entropy_vs_coupling
    # fig, ax = plt.subplots(1, 1)
    # ax = entropy_vs_coupling(ax, int_coupling=int_cg_vals, int_entropy=integrated_entropies, int_peaks=integrated_peaks,
    #                          fit_coupling=fit_cg_vals, fit_entropy=fit_entropies,
    #                          peak_diff_coupling=peak_cg_vals,
    #                          )
    # plt.tight_layout()
    # fig.show()
