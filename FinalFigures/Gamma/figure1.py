import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
from typing import Optional
from dataclasses import dataclass

import dat_analysis.useful_functions as U
from dat_analysis.plotting.mpl.PlotUtil import set_default_rcParams
from dat_analysis.plotting.plotly import OneD
from FinalFigures.Gamma.plots import getting_amplitude_and_dt, dndt_signal
from dat_analysis.analysis_tools.nrg import NRG_func_generator, NrgUtil, NRGParams
from temp import get_avg_entropy_data, _center_func, get_avg_i_sense_data

p1d = OneD(dat=None)


# @dataclass
# class NRGParams:
#     gamma: float
#     theta: float
#     center: float
#     amp: float
#     lin: float
#     const: float
#     lin_occ: float
#     vary_theta: bool = False
#     vary_gamma: bool = False
#     datnum: Optional[int] = None


if __name__ == '__main__':
    set_default_rcParams()
    from dat_analysis.dat_object.make_dat import get_dats, get_dat

    # csq_datnum = 2197
    csq_datnum = None
    #############################################################################################

    # Data for dN/dT
    # all_dats = get_dats([2164, 2170])
    all_dats = get_dats([2164, 2167])
    # fit_names = ['csq_gamma_small', 'csq_forced_theta']
    fit_names = ['gamma_small', 'forced_theta']
    # all_dats = get_dats([2164, 2216])  # Weak, Strong coupling
    # all_dats = get_dats([7334, 7356])  # Weak, Strong coupling
    # all_dats = get_dats([7334, 7360])  # Weak, Strong coupling
    tonly_dats = get_dats([dat.datnum + 1 for dat in all_dats])

    dndt_datas, gts, nrg_fits, amps = [], [], [], []
    for dat, fit_name in zip(all_dats, fit_names):
        dndt = get_avg_entropy_data(dat, center_func=_center_func, csq_datnum=csq_datnum)

        init_fit = dat.NrgOcc.get_fit(name=fit_name)
        # params = NRGParams.from_lm_params(init_fit.params)
        params = NRGParams.from_lm_params(U.edit_params(init_fit.params,
                                                        ['theta', 'g'],
                                                        [init_fit.best_values.theta, init_fit.best_values.g],
                                                        [False, False]))
        nrg_fit = NrgUtil(inital_params=params).get_fit(
            x=dndt.x, data=dndt.data, which_data='dndt'
        )

        dndt.x = dndt.x/100  # Convert to real mV
        gt = init_fit.best_values.g/init_fit.best_values.theta

        dndt_datas.append(dndt)
        gts.append(gt)
        nrg_fits.append(nrg_fit)
        amps.append(init_fit.best_values.amp)

    U.save_to_igor_itx(file_path=f'fig1_dndt.itx',
                       xs=[data.x for data in dndt_datas] + [np.linspace(-3, 3, 1000)] + [np.arange(4)],
                       datas=[data.data for data in dndt_datas] +
                             [nrg_fits[1].eval_fit(x=np.linspace(-3, 3, 1000)*100)] + [np.array(gts)],
                       names=[f'dndt_{i}' for i in range(len(dndt_datas))] + ['dndt_1_nrg_fit'] + ['gts_for_dndts'],
                       x_labels=['Sweep Gate (mV)'] * (len(dndt_datas)+1) + ['index'],
                       y_labels=['dN/dT (nA)'] * (len(dndt_datas)+1) + ['G/T'])

    # dNdT Plots (one for weakly coupled only, one for strongly coupled only)
    weak_fig, ax = plt.subplots(1, 1)
    ax = dndt_signal(ax, xs=dndt_datas[0].x, datas=dndt_datas[0].data, amp_sensitivity=amps[0])
    ax.set_xlim(-0.6, 0.6)
    ax.set_title('')
    plt.tight_layout()
    weak_fig.show()

    strong_fig, ax = plt.subplots(1, 1)
    dndt_ = dndt_datas[1].data
    # dndt_, freq = U.decimate(dndts[1], measure_freq=all_dats[1].Logs.measure_freq, numpnts=200, return_freq=True)
    x_ = U.get_matching_x(dndt_datas[1].x, dndt_)
    # dndt_signal(ax, xs=xs[1], datas=dndts[1])
    dndt_signal(ax, xs=x_, datas=dndt_, amp_sensitivity=amps[1])
    ax.set_xlim(-3, 3)
    # ax.set_title('dN/dT for gamma broadened')

    ax.plot(x_, nrg_fits[1].eval_fit(x=x_*100), label='NRG Fit')

    for ax in strong_fig.axes:
        ax.set_ylabel(ax.get_ylabel(), labelpad=5)
    plt.tight_layout()
    strong_fig.show()



    # Data for single hot/cold plot
    dat = get_dat(2164)
    # dat = get_dat(7334)

    _, avg_x = dat.NrgOcc.get_avg_data(check_exists=True, return_x=True)
    sweep_x = avg_x/100  # Convert to real mV
    cold_data = get_avg_i_sense_data(dat, None, _center_func, False, hot_or_cold='cold')
    hot_data = get_avg_i_sense_data(dat, None, _center_func, False, hot_or_cold='hot')

    U.save_to_igor_itx(file_path=f'fig1_hot_cold.itx', xs=[sweep_x] * 2, datas=[cold_data.data, hot_data.data],
                       names=['cold', 'hot'], x_labels=['Sweep Gate (mV)'] * 2, y_labels=['Current (nA)'] * 2)

    # plotting for Single hot/cold plot
    # fig, ax = plt.subplots(1, 1)
    ax = weak_fig.add_axes([0.5, 0.5, 0.30, 0.4])
    ax: plt.Axes
    getting_amplitude_and_dt(ax, x=sweep_x, cold=cold_data.data, hot=hot_data.data, )
    ax.set_title('')
    ax.get_legend().remove()
    ax.set_xlabel('')
    ax.set_ylabel('')

    plt.tight_layout()
    weak_fig.show()
