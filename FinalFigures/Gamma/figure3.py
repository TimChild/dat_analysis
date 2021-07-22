from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import lmfit as lm
from typing import TYPE_CHECKING

from FinalFigures.Gamma.plots import integrated_entropy, entropy_vs_coupling, gamma_vs_coupling
from src.useful_functions import save_to_igor_itx, order_list, get_data_index, resample_data
from src.plotting.Mpl.PlotUtil import set_default_rcParams
from src.plotting.plotly import OneD

from temp import get_avg_entropy_data, get_integrated_data, _center_func

p1d = OneD(dat=None)

if TYPE_CHECKING:
    from src.dat_object.dat_hdf import DatHDF


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


if __name__ == '__main__':
    from src.dat_object.make_dat import get_dats

    set_default_rcParams()

    ####################################################
    # Data for gamma_vs_coupling
    # fit_name = 'forced_theta_linear'
    fit_name = 'forced_theta'
    dats = get_dats(range(2095, 2135 + 1, 2))
    dats = [dat for dat in dats if dat.datnum != 2127]
    gamma_dats = [dat for dat in dats if dat.Logs.dacs['ESC'] > -285]
    # tonly_dats = get_dats([dat.datnum + 1 for dat in dats if dat.Logs.dacs['ESC'] > -285])
    # tonly_dats = get_dats(chain(range(7323, 7361 + 1, 2), range(7379, 7399 + 1, 2), range(7401, 7421 + 1, 2)))
    # tonly_dats = order_list(tonly_dats, [dat.Logs.fds['ESC'] for dat in tonly_dats])
    # tonly_dats = [dat for dat in tonly_dats if dat.Logs.fds['ESC'] > -245
    #               and dat.datnum < 7362 and dat.datnum not in [7349, 7351]]  # So no duplicates
    # Loading fitting done in Analysis.Feb2021.entropy_gamma_final

    gamma_cg_vals = np.array([dat.Logs.fds['ESC'] for dat in gamma_dats])
    # gammas = np.array([dat.Transition.get_fit(name=fit_name).best_values.g for dat in tonly_gamma_dats])
    gammas = np.array([dat.NrgOcc.get_fit(name=fit_name).best_values.g for dat in gamma_dats])
    thetas = np.array([dat.NrgOcc.get_fit(name=fit_name).best_values.theta for dat in gamma_dats])
    gts = gammas / thetas

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
    # tonly_dats = get_dats([dat.datnum + 1 for dat in entropy_dats])
    tonly_dats = get_dats([dat.datnum for dat in entropy_dats])

    datas, gts = [], []
    for dat, tonly_dat in zip(entropy_dats, tonly_dats):
        data = get_integrated_data(dat, fit_name=nrg_fit_name, zero_point=-350)
        gt = tonly_dat.NrgOcc.get_fit(name=nrg_fit_name).best_values.g / \
             tonly_dat.NrgOcc.get_fit(name=nrg_fit_name).best_values.theta

        data.x = data.x / 100  # Convert to real mV
        datas.append(data)
        gts.append(gt)

    save_to_igor_itx(file_path=f'fig3_integrated_entropy.itx',
                     xs=[data.x for data in datas] + [np.arange(len(gts))],
                     datas=[data.data for data in datas] + [np.array(gts)],
                     names=[f'int_entropy_{k}' for k in ['weak', 'similar', 'med', 'strong']] + ['gts_for_int_vary_g'],
                     x_labels=['Sweep Gate (mV)'] * len(datas) + ['index'],
                     y_labels=['Entropy (kB)'] * len(datas) + ['G/T'])

    # Plot Integrated Entropy
    fig, ax = plt.subplots(1, 1)
    integrated_entropy(ax, xs=[data.x for data in datas], datas=[data.data for data in datas],
                       labels=[f'{gt:.1f}' for gt in gts])
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
    peak_cg_vals = int_cg_vals
    fit_lim = -250
    fit_cg_vals = np.array([dat.Logs.fds['ESC'] for dat in entropy_dats if dat.Logs.fds['ESC'] < fit_lim])

    integrated_data = []
    integrated_entropies, integrated_peaks = [], []
    fit_entropies = []

    for dat in entropy_dats:
        fit = dat.NrgOcc.get_fit(name=nrg_fit_name)
        w_val = max(100, 10*fit.best_values.g)  # Width for zeroing and measuring int entropy
        integrated = get_integrated_data(dat, fit_name=nrg_fit_name, zero_point=-w_val)

        idx = get_data_index(integrated.x, w_val)  # Measure entropy at x = xxx
        if idx >= integrated.x.shape[-1] - 1:  # or the end if it doesn't reach xxx
            idx -= 10
        integrated_entropies.append(np.nanmean(integrated.data[idx - 10:idx + 10]))
        integrated_peaks.append(np.nanmax(resample_data(integrated.data, max_num_pnts=100)))

        if dat.Logs.dacs['ESC'] < fit_lim:
            avg_dndt = get_avg_entropy_data(dat,
                                            center_func=_center_func,
                                            overwrite=False)
            fit_entropies.append(
                dat.Entropy.get_fit(data=avg_dndt.data, x=avg_dndt.x, calculate_only=True).best_values.dS)

    integrated_entropies = np.array(integrated_entropies)
    integrated_peaks = np.array(integrated_peaks)
    fit_entropies = np.array(fit_entropies)

    save_to_igor_itx(file_path=f'fig3_entropy_vs_gamma.itx', xs=[fit_cg_vals, int_cg_vals, int_cg_vals],
                     datas=[fit_entropies, integrated_entropies, integrated_peaks],
                     names=['fit_entropy_vs_coupling', 'integrated_entropy_vs_coupling',
                            'integrated_peaks_vs_coupling'],
                     x_labels=['Coupling Gate (mV)'] * 3,
                     y_labels=['Entropy (kB)', 'Entropy (kB)', 'Entropy (kB)'])

    # Plot entropy_vs_coupling
    fig, ax = plt.subplots(1, 1)
    ax = entropy_vs_coupling(ax, int_coupling=int_cg_vals, int_entropy=integrated_entropies, int_peaks=integrated_peaks,
                             fit_coupling=fit_cg_vals, fit_entropy=fit_entropies,
                             peak_diff_coupling=peak_cg_vals,
                             )
    plt.tight_layout()
    fig.show()
