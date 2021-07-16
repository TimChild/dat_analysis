import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
from typing import Optional
from dataclasses import dataclass

import src.useful_functions as U
from src.plotting.Mpl.PlotUtil import set_default_rcParams
from src.plotting.plotly import OneD
from FinalFigures.Gamma.plots import getting_amplitude_and_dt, dndt_signal
from src.analysis_tools.nrg import NRG_func_generator, NrgUtil, NRGParams

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
    from src.dat_object.make_dat import get_dats, get_dat

    #############################################################################################

    # Data for dN/dT
    fit_name = 'forced_theta_linear'
    # all_dats = get_dats([2164, 2170])
    all_dats = get_dats([2164, 2167])
    # all_dats = get_dats([2164, 2216])  # Weak, Strong coupling
    # all_dats = get_dats([7334, 7356])  # Weak, Strong coupling
    # all_dats = get_dats([7334, 7360])  # Weak, Strong coupling
    tonly_dats = get_dats([dat.datnum + 1 for dat in all_dats])

    params_2164 = NRGParams(   # This is done for hot trace
        gamma=0.4732,
        theta=4.672,
        center=7.514,
        amp=0.939,
        lin=0.00152,
        const=7.205,
        lin_occ=-0.0000358,
        # vary_theta=True,
        # vary_gamma=False,
        # datnum=2164
    )

    params_2167 = NRGParams(
        gamma=23.4352,
        theta=4.5,
        center=78.4,
        amp=0.675,
        lin=0.00121,
        const=7.367,
        lin_occ=0.0001453,
        # vary_theta=False,
        # vary_gamma=True,
        # datnum=2167
    )

    outs = [dat.SquareEntropy.get_Outputs(name=fit_name) for dat in all_dats]
    int_infos = [dat.Entropy.get_integration_info(name=fit_name) for dat in all_dats]

    nrg_func = NRG_func_generator('dndt')

    nrg_params = [U.edit_params(pars.to_lm_params(), 'theta', dat.NrgOcc.get_fit(name='forced_theta').best_values.theta)
                  for pars, dat in zip([params_2164, params_2167], all_dats)]
    nrg_fits = [NrgUtil(inital_params=NRGParams.from_lm_params(params)).get_fit(x=x, data=data, which_data='dndt') for params, x, data in
                zip(nrg_params, [out.x for out in outs], [out.average_entropy_signal for out in outs])]
    # nrg_dndts = [nrg_func(out.x, param.center, param.gamma, param.theta) for out, param in
    #              zip(outs, [params_2164, params_2167])]

    xs = [out.x / 100 for out in outs]  # /100 to convert to real mV
    dndts = [out.average_entropy_signal for out in outs]
    gts = [dat.Transition.get_fit(name=fit_name).best_values.g / dat.Transition.get_fit(name=fit_name).best_values.theta
           for dat in tonly_dats]

    U.save_to_igor_itx(file_path=f'fig1_dndt.itx',
                       xs=xs + [np.linspace(-3, 3, 1000)] + [np.arange(4)],
                       datas=dndts + [nrg_fits[1].eval_fit(x=np.linspace(-3, 3, 1000)*100)] + [np.array(gts)],
                       names=[f'dndt_{i}' for i in range(len(dndts))] + ['dndt_1_nrg_fit'] + ['gts_for_dndts'],
                       x_labels=['Sweep Gate (mV)'] * (len(dndts)+1) + ['index'],
                       y_labels=['dN/dT (nA)'] * (len(dndts)+1) + ['G/T'])

    # dNdT Plots (one for weakly coupled only, one for strongly coupled only)
    weak_fig, ax = plt.subplots(1, 1)
    ax = dndt_signal(ax, xs=xs[0], datas=dndts[0], amp_sensitivity=int_infos[0].amp)
    ax.set_xlim(-0.6, 1)
    # ax.set_title('dN/dT for weakly coupled')
    ax.set_title('')
    plt.tight_layout()
    weak_fig.show()

    strong_fig, ax = plt.subplots(1, 1)
    dndt_ = dndts[1]
    # dndt_, freq = U.decimate(dndts[1], measure_freq=all_dats[1].Logs.measure_freq, numpnts=200, return_freq=True)
    x_ = U.get_matching_x(xs[1], dndt_)
    # dndt_signal(ax, xs=xs[1], datas=dndts[1])
    dndt_signal(ax, xs=x_, datas=dndt_, amp_sensitivity=int_infos[1].amp)
    ax.set_xlim(-3, 3)
    # ax.set_title('dN/dT for gamma broadened')

    ax.plot(x_, nrg_fits[1].eval_fit(x=x_*100), label='NRG Fit')

    for ax in strong_fig.axes:
        ax.set_ylabel(ax.get_ylabel(), labelpad=5)
    plt.tight_layout()
    strong_fig.show()

    # Data for single hot/cold plot
    fit_name = 'forced_theta_linear'
    dat = get_dat(2164)
    # dat = get_dat(7334)
    out = dat.SquareEntropy.get_Outputs(name=fit_name)
    sweep_x = out.x / 100  # /100 to make real mV
    cold_transition = np.nanmean(out.averaged[(0, 2), :], axis=0)
    hot_transition = np.nanmean(out.averaged[(1, 3), :], axis=0)

    U.save_to_igor_itx(file_path=f'fig1_hot_cold.itx', xs=[sweep_x] * 2, datas=[cold_transition, hot_transition],
                       names=['cold', 'hot'], x_labels=['Sweep Gate (mV)'] * 2, y_labels=['Current (nA)'] * 2)

    # plotting for Single hot/cold plot
    # fig, ax = plt.subplots(1, 1)
    ax = weak_fig.add_axes([0.5, 0.5, 0.30, 0.4])
    ax: plt.Axes
    getting_amplitude_and_dt(ax, x=sweep_x, cold=cold_transition, hot=hot_transition, )
    ax.set_title('')
    ax.get_legend().remove()
    ax.set_xlabel('')
    ax.set_ylabel('')

    plt.tight_layout()
    weak_fig.show()
