from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import json
from itertools import chain
import lmfit as lm
from typing import TYPE_CHECKING, Callable

from FinalFigures.Gamma.plots import integrated_entropy, entropy_vs_coupling, gamma_vs_coupling, amp_theta_vs_coupling, \
    amp_sf_vs_coupling
from src.useful_functions import save_to_igor_itx, order_list, fig_to_data_json, data_to_json, mean_data, get_data_index
from src.plotting.Mpl.PlotUtil import set_default_rcParams
from Analysis.Feb2021.entropy_gamma_final import dT_from_linear
from src.plotting.plotly import Data1D, Data2D

from temp import get_2d_data, get_centers

if TYPE_CHECKING:
    from src.dat_object.Attributes.Entropy import IntegrationInfo
    from src.dat_object.dat_hdf import DatHDF


def fit_line(x, data) -> lm.model.ModelResult:
    line = lm.models.LinearModel()
    pars = line.guess(data, x=x)
    fit = line.fit(data=data.astype(np.float32), x=x.astype(np.float32), params=pars)
    return fit


def dt_with_set_params(esc: float) -> float:
    """
    Just wraps dT_from_linear with some fixed params for generating new dT (currently based on fitting both entropy and
    transition thetas with new NRG fits see Tim Child/Gamma Paper/Data/Determining Amplitude)
    Args:
        esc (): New ESC value to get dT for

    Returns:

    """
    # dt = dT_from_linear(base_dT=1.158, base_esc=-309.45, lever_slope=0.00304267, lever_intercept=5.12555961, new_esc=esc)

    # Using New NRG fits to Dats 2148, 2149, 2143 (cold)
    dt = dT_from_linear(base_dT=1.15566, base_esc=-339.36, lever_slope=0.00322268, lever_intercept=5.160500, new_esc=esc)
    return dt


def calc_int_info(dat: DatHDF) -> IntegrationInfo:
    """Calculates integration info and saves in Dat, and returns the integration info"""
    fit_name = 'forced_theta'
    fit = dat.NrgOcc.get_fit(name=fit_name)
    # if dat.Logs.dacs['ESC'] > -225:
    if dat.Logs.dacs['ESC'] > -0:
        amp = get_linear_gamma_amplitude(dat)
    else:
        amp = fit.best_values.amp
    dt = dt_with_set_params(dat.Logs.dacs['ESC'])
    int_info = dat.Entropy.set_integration_info(dT=dt, amp=amp, name='forced_theta_nrg', overwrite=True)
    return int_info


def get_avg_entropy_data(dat, center_func: Callable, overwrite: bool = False) -> Data1D:
    """Get avg entropy data (including setpoint start thing)"""
    data2d = get_2d_data(dat, 'entropy')
    if center_func(dat):
        centers = get_centers(dat, data2d, name=None, overwrite=overwrite)
    else:
        centers = [0]*data2d.data.shape[0]
    data, x = mean_data(data2d.x, data2d.data, centers, return_x=True)
    return Data1D(x=x, data=data)


def get_linear_gamma_amplitude(dat: DatHDF) -> float:
    if (v := dat.Logs.dacs['ESC']) < -270:
        raise ValueError(f'dat{dat.datnum} has ESC = {v}mV which is weakly coupled. This is only for extrapolating'
                         f'amplitude into very gamma broadened ')
    line = lm.models.LinearModel()
    pars = line.make_params()

    # Based on fit to dats 2117 -> 2126 New NRG fits with forced theta linear
    pars['slope'].value = -0.00435440
    pars['intercept'].value = -0.25988496

    return float(line.eval(params=pars, x=dat.Logs.dacs['ESC']))





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
    nrg_fit_name = 'forced_theta'  # Need to actually recalculate entropy scaling using this NRG fit as well
    entropy_dats = get_dats([2164, 2121, 2167, 2133])
    tonly_dats = get_dats([dat.datnum + 1 for dat in entropy_dats])

    datas, gts = [], []
    for dat, tonly_dat in zip(entropy_dats, tonly_dats):
        data = get_avg_entropy_data(dat,
                             center_func=lambda dat: True if dat.Logs.dacs['ESC'] < -260 else False,
                             overwrite=False)
        int_info = calc_int_info(dat)
        data.data = int_info.integrate(data.data)

        offset_x = dat.NrgOcc.get_x_of_half_occ(nrg_fit_name)
        data.x = data.x-offset_x
        if data.x[0] < -200:
            offset_y = np.mean(data.data[np.where(np.logical_and(data.x > -400, data.x < -300))])
            data.data -= offset_y

        gt = tonly_dat.NrgOcc.get_fit(name=nrg_fit_name).best_values.g / \
              tonly_dat.NrgOcc.get_fit(name=nrg_fit_name).best_values.theta

        data.x = data.x/100  # Convert to real mV
        datas.append(data)
        gts.append(gt)

    save_to_igor_itx(file_path=f'fig3_integrated_entropy.itx',
                     xs=[data.x for data in datas] + [np.arange(len(gts))],
                     datas=[data.data for data in datas] + [np.array(gts)],
                     names=[f'int_entropy_{k}' for k in ['weak', 'med', 'strong', 'similar']] + ['gts_for_int_vary_g'],
                     x_labels=['Sweep Gate (mV)'] * len(datas) + ['index'],
                     y_labels=['Entropy (kB)'] * len(datas) + ['G/T'])

    # Plot Integrated Entropy
    fig, ax = plt.subplots(1, 1)
    integrated_entropy(ax, xs=[data.x for data in datas], datas=[data.data for data in datas], labels=[f'{gt:.1f}' for gt in gts])
    ax.set_xlim(-5, 5)
    plt.tight_layout()
    fig.show()

    ##################################################################

    # Data for entropy_vs_coupling
    fit_name = 'forced_theta_linear'
    entropy_dats = get_dats(range(2095, 2136 + 1, 2))  # Goes up to 2141 but the last few aren't great
    # all_dats = get_dats(chain(range(7322, 7361 + 1, 2), range(7378, 7399 + 1, 2), range(7400, 7421 + 1, 2)))
    # all_dats = order_list(all_dats, [dat.Logs.fds['ESC'] for dat in all_dats])
    # dats = get_dats(range(2095, 2111 + 1, 2))
    entropy_dats = [dat for dat in entropy_dats if dat.datnum != 2127]
    entropy_dats = order_list(entropy_dats, [dat.Logs.dacs['ESC'] for dat in entropy_dats])

    int_cg_vals = np.array([dat.Logs.fds['ESC'] for dat in entropy_dats])
    # TODO: Need to make sure all these integrated entropies are being calculated at good poitns (i.e. not including slopes)
    integrated_data = np.array([
        dat.Entropy.get_integrated_entropy(name=fit_name,
                                           data=dat.SquareEntropy.get_Outputs(
                                               name=fit_name, check_exists=True).average_entropy_signal
                                           ) for dat in entropy_dats])
    integrated_entropies = [np.nanmean(data[-10:]) for data in integrated_data]
    integrated_peaks = [np.nanmax(data) for data in integrated_data]
    peak_cg_vals = int_cg_vals
    peak_diffs = [p - v for p, v in zip(integrated_peaks, integrated_entropies)]

    fit_cg_vals = np.array([dat.Logs.fds['ESC'] for dat in entropy_dats if dat.Logs.fds['ESC'] < -250])
    fit_entropies = np.array(
        [dat.Entropy.get_fit(name=fit_name).best_values.dS for dat in entropy_dats if dat.Logs.fds['ESC'] < -250])

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
