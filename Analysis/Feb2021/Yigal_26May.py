"""
Sep 21 -- Lots of interesting stuff, but no longer useful. Was using this to replot data downloaded from dash app which
already had graphs that were close to what we wanted at the time, so all of that is no longer useful.
Then the functions which generate new data/fits directly are out of date and it's now easier to use the functions in
nrg.py
No useful functions worth taking from here.
"""
from typing import Optional, List
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from scipy.interpolate import interp1d
import lmfit as lm
from deprecation import deprecated

import dat_analysis.useful_functions as U
from dat_analysis.dat_analysis.characters import DELTA
from dat_analysis.dat_object.make_dat import get_dat
from dat_analysis.plotting.plotly.dat_plotting import OneD, TwoD
from dat_analysis.analysis_tools.general_fitting import calculate_fit
from dat_analysis.analysis_tools.nrg import NRG_func_generator
from OLD.new_dash.pages import invert_nrg_fit_params

import plotly.io as pio

pio.renderers.default = 'browser'

p1d = OneD(dat=None)
p2d = TwoD(dat=None)


@dataclass
class Params:
    gamma: float
    theta: float
    center: float
    amp: float
    lin: float
    const: float
    lin_occ: float
    vary_theta: bool = False
    vary_gamma: bool = False
    datnum: Optional[int] = None


@deprecated(details='Using NRGUtil.get_fit instead')
def nrg_fit(x, data,
            init_params: Params,
            ):
    fit_params = lm.Parameters()
    fit_params.add_many(
        ('mid', init_params.center, True, np.nanmin(x), np.nanmax(x), None, None),
        ('theta', init_params.theta, init_params.vary_theta, 0.5, 200, None, None),
        ('amp', init_params.amp, True, 0.1, 3, None, None),
        ('lin', init_params.lin, True, 0, 0.005, None, None),
        ('occ_lin', init_params.lin_occ, True, -0.0003, 0.0003, None, None),
        ('const', init_params.const, True, np.nanmin(data), np.nanmax(data), None, None),
        ('g', init_params.gamma, init_params.vary_gamma, 0.2, 4000, None, None),
    )
    # Note: Theta or Gamma MUST be fixed (and makes sense to fix theta usually)
    fit = calculate_fit(x, data, params=fit_params, func=NRG_func_generator(which='i_sense'), method='powell')
    return fit


@deprecated(deprecated_in="09-2021",
            details="This is overcomplicated now and should use functions from nrg.py (although this exact plot might"
                    "not exist anywhere else yet)")
def plot_dndt_without_zero_offset(datnum: int,
                                  params: Params,
                                  fit_to_hot: bool
                                  ) -> go.Figure:
    """
    Re-plot the scaled dNdTs from Dash with data starting at zero instead of being somewhere else
    Args:

    Returns:

    """
    dat = get_dat(datnum)

    out = dat.SquareEntropy.get_Outputs(name='forced_theta_linear_non_csq')
    sweep_x = out.x
    data_dndt = out.average_entropy_signal
    if fit_to_hot:
        data_isense = np.nanmean(out.averaged[(1, 3), :], axis=0)
    else:
        data_isense = np.nanmean(out.averaged[(0, 2), :], axis=0)

    fit = nrg_fit(sweep_x, data_isense, init_params=params)
    params = Params(
        gamma=fit.best_values.g,
        theta=fit.best_values.theta,
        center=fit.best_values.mid,
        amp=fit.best_values.amp,
        lin=fit.best_values.lin,
        const=fit.best_values.const,
        lin_occ=fit.best_values.occ_lin,
    )

    _, data_occ = invert_nrg_fit_params(x=sweep_x, data=data_isense,
                                        gamma=params.gamma,
                                        theta=params.theta,
                                        mid=params.center,
                                        amp=params.amp,
                                        lin=params.lin,
                                        const=params.const,
                                        occ_lin=params.lin_occ,
                                        data_type='i_sense')
    data_occ = (data_occ - 0.5) * -1 + 0.5

    nrg_dndt_func = NRG_func_generator('dndt')
    nrg_occ_func = NRG_func_generator('occupation')
    nrg_dndt = nrg_dndt_func(sweep_x, params.center, params.gamma, params.theta)  # , amp=1, lin=0, const=0, occ_lin=0)
    nrg_dndt = nrg_dndt / nrg_dndt.max() * np.nanmax(data_dndt)
    nrg_occ = nrg_occ_func(sweep_x, params.center, params.gamma, params.theta)

    # fig = p1d.figure(xlabel='Sweep Gate (mV)', ylabel=f'{DELTA}I (nA)')
    layout = go.Layout(
        yaxis=dict(title=f'{DELTA}I (nA)'),
        yaxis2=dict(title='Occupation',
                    overlaying='y',
                    side='right'))
    fig = go.Figure(data=None, layout=layout)
    fig.update_xaxes(title='Sweep Gate (mV)')
    sweep_x = sweep_x / 100  # Convert to real mV

    # Plot data with offset subtracted
    fig.add_trace(p1d.trace(data=data_dndt, x=sweep_x, mode='lines', name='Data'))
    trace = p1d.trace(data=data_occ, x=sweep_x, mode='markers', name='Data Occupation')
    trace.update(yaxis='y2')
    fig.add_trace(trace)

    # Plot nrg with offset subtracted and scaled to data
    fig.add_trace(p1d.trace(data=nrg_dndt, x=sweep_x, mode='lines', name='NRG'))
    trace = p1d.trace(data=nrg_occ, x=sweep_x, mode='lines', name='NRG Occupation')
    trace.update(yaxis='y2')
    fig.add_trace(trace)

    fig.update_layout(template='simple_white')
    return fig


def plot_dndt_vs_N_correct_scale(data_json_path: str,
                                 max_dndt: float = 1) -> go.Figure:
    """
    Re-plot the dN/dT vs N from Dash with data and NRG scaled the same and y-axis in units of nA (if max_dndt provided)
    Args:
        data_json_path ():
        max_dndt (): To rescale the dN/dT data back into nA instead of 0 -> 1

    Returns:

    """

    def rescale(d, max_):
        d = d / np.nanmax(d) * max_
        return d

    all_data = U.data_from_json(data_json_path)
    x = all_data['Data dN/dT_x']
    data_dndt = all_data['Data dN/dT']
    nrg_dndt = all_data['NRG dN/dT']

    fig = p1d.figure(xlabel='Occupation', ylabel=f'{DELTA}I (nA)')

    # Plot data with offset subtracted
    data = rescale(data_dndt, max_dndt)
    fig.add_trace(p1d.trace(data=data, x=x, mode='lines+markers', name='Data'))

    # Plot nrg with offset subtracted and scaled to data
    data = rescale(nrg_dndt, max_dndt)
    fig.add_trace(p1d.trace(data=data, x=x, mode='lines', name='NRG'))

    fig.update_layout(template='simple_white')
    return fig


def plot_integrated(datnum: int,
                    params: Params,
                    ) -> go.Figure:
    """

    Args:
        datnum (): To get the integration info/dndt from

    Returns:

    """
    dat = get_dat(datnum)

    out = dat.SquareEntropy.get_Outputs(name='forced_theta_linear_non_csq')
    sweep_x = out.x
    data_dndt = out.average_entropy_signal
    int_info = dat.Entropy.get_integration_info(name='forced_theta_linear_non_csq')
    data_int = int_info.integrate(data_dndt)
    data_isense_cold = np.nanmean(out.averaged[(0, 2), :], axis=0)

    fit = nrg_fit(sweep_x, data_isense_cold, init_params=params)
    params = Params(
        gamma=fit.best_values.g,
        theta=fit.best_values.theta,
        center=fit.best_values.mid,
        amp=fit.best_values.amp,
        lin=fit.best_values.lin,
        const=fit.best_values.const,
        lin_occ=fit.best_values.occ_lin,
    )

    nrg = NRG_func_generator('int_dndt')
    nrg_integrated = nrg(sweep_x, params.center, params.gamma, params.theta)  # , amp=1, lin=0, const=0, occ_lin=0)

    fig = p1d.figure(xlabel='Sweep Gate (mV)', ylabel='Entropy (kB)')

    for data, label in zip([data_int, nrg_integrated], ['Data', 'NRG']):
        fig.add_trace(p1d.trace(x=sweep_x / 100, data=data, mode='lines', name=label))  # /100 to convert to real mV

    for v in [np.log(2), np.log(3)]:
        p1d.add_line(fig, v, mode='horizontal', color='black', linewidth=1, linetype='dash')

    fig.update_layout(template='simple_white')
    return fig


def plot_integrated_vs_N(datnum: int,
                         params: Params,
                         ) -> go.Figure:
    """
    Plot integrated entropy vs occupation instead of vs sweepgate
    Args:
        datnum (): use dat to get integrated data directly because it is easier to do

    Returns:

    """
    dat = get_dat(datnum)
    occ_func = NRG_func_generator('occupation')
    int_func = NRG_func_generator('int_dndt')

    out = dat.SquareEntropy.get_Outputs(name='forced_theta_linear_non_csq')
    sweep_x = out.x
    data_dndt = out.average_entropy_signal
    int_info = dat.Entropy.get_integration_info(name='forced_theta_linear_non_csq')
    data_int = int_info.integrate(data_dndt)
    data_isense_cold = np.nanmean(out.averaged[(0, 2), :], axis=0)

    fit = nrg_fit(sweep_x, data_isense_cold, init_params=params)
    params = Params(
        gamma=fit.best_values.g,
        theta=fit.best_values.theta,
        center=fit.best_values.mid,
        amp=fit.best_values.amp,
        lin=fit.best_values.lin,
        const=fit.best_values.const,
        lin_occ=fit.best_values.occ_lin,
    )

    _, data_occ = invert_nrg_fit_params(x=sweep_x, data=data_isense_cold,
                                        gamma=params.gamma,
                                        theta=params.theta,
                                        mid=params.center,
                                        amp=params.amp,
                                        lin=params.lin,
                                        const=params.const,
                                        occ_lin=params.lin_occ,
                                        data_type='i_sense')
    data_occ = (data_occ - 0.5) * -1 + 0.5
    data_occ_x = get_occupation_x_axis(sweep_x, data_occ)

    nrg_occ = occ_func(sweep_x, params.center, params.gamma, params.theta)
    nrg_int = int_func(sweep_x, params.center, params.gamma, params.theta)
    nrg_occ_x = get_occupation_x_axis(sweep_x, nrg_occ)

    fig = p1d.figure(xlabel='Occupation', ylabel='Entropy (kB)')
    for x, data, label in zip([data_occ_x, nrg_occ_x], [data_int, nrg_int], ['Data', 'NRG']):
        fig.add_trace(p1d.trace(x=x, data=data, mode='lines+markers', name=label))

    for v in [np.log(2), np.log(3)]:
        p1d.add_line(fig, v, mode='horizontal', color='black', linewidth=1, linetype='dash')

    fig.update_layout(template='simple_white')
    return fig


def plot_multiple_dndt(
        params: List[Params]) -> go.Figure:
    fig = p1d.figure(xlabel='Sweep Gate/max(Gamma, T)', ylabel=f'{DELTA}I*max(Gamma, T)',
                     title=f'{DELTA}I vs Sweep gate for various Gamma/T')
    for param in params:
        dat = get_dat(param.datnum)

        out = dat.SquareEntropy.get_Outputs(name='forced_theta_linear_non_csq')
        sweep_x = out.x / 100  # /100 to convert to real mV
        data_dndt = out.average_entropy_signal
        # transition_fit = tdat.Transition.get_fit(name='forced_theta_linear_non_csq')
        # gamma_over_t = transition_fit.best_values.g/transition_fit.best_values.theta
        gamma_over_t = param.gamma / param.theta

        rescale = max(param.gamma, param.theta)

        fig.add_trace(
            p1d.trace(x=sweep_x / rescale, data=data_dndt * rescale, mode='lines', name=f'{gamma_over_t:.2f}'))
    fig.update_layout(legend_title='Gamma/Theta')
    fig.update_xaxes(range=[-0.2, 0.2])
    fig.update_layout(template='simple_white')
    return fig


def plot_and_save_multiple_dndt(
        params: List[Params],
        title: str, save_name: str, ):
    fig = plot_multiple_dndt(

        params=params)
    fig.update_layout(title=title)
    U.fig_to_data_json(fig, f'figs/data/{save_name}')
    fig.write_image(f'figs/{save_name}.pdf')
    fig.show()
    return fig


def get_occupation_x_axis(x, occupation) -> np.ndarray:
    interp_range = np.where(np.logical_and(occupation < 0.99, occupation > 0.01))
    interp_data = occupation[interp_range]
    interp_x = x[interp_range]

    interper = interp1d(x=interp_x, y=interp_data, assume_sorted=True, bounds_error=False)
    occ_x = interper(x)
    return occ_x


def plot_and_save_dndt_vs_sweepgate(
        datnum: int,
        params: Params,
        fit_to_hot: bool,
        title: str,
        save_name: str) -> go.Figure:
    fig = plot_dndt_without_zero_offset(datnum=datnum,
                                        params=params,
                                        fit_to_hot=fit_to_hot)

    fig.update_layout(title=title)
    U.fig_to_data_json(fig, f'figs/data/{save_name}')
    fig.write_image(f'figs/{save_name}.pdf')
    fig.show()
    return fig


def plot_and_save_dndt_vs_N(
        data_json_path: str,
        max_dndt: float,
        title: str,
        save_name: str) -> go.Figure:
    fig = plot_dndt_vs_N_correct_scale(data_json_path=data_json_path,
                                       max_dndt=max_dndt)
    fig.update_layout(title=title)
    U.fig_to_data_json(fig, f'figs/data/{save_name}')
    fig.write_image(f'figs/{save_name}.pdf')
    fig.show()
    return fig


def plot_and_save_integrated_vs_sweepgate(
        datnum: int,
        params: Params,
        title: str,
        save_name: str) -> go.Figure:
    fig = plot_integrated(datnum=datnum,
                          params=params
                          )
    fig.update_layout(title=title)
    U.fig_to_data_json(fig, f'figs/data/{save_name}')
    fig.write_image(f'figs/{save_name}.pdf')
    fig.show()
    return fig


def plot_and_save_integrated_vs_N(
        datnum: int,
        params: Params,
        title: str,
        save_name: str) -> go.Figure:
    fig = plot_integrated_vs_N(datnum=datnum, params=params)
    fig.update_layout(title=title)
    U.fig_to_data_json(fig, f'figs/data/{save_name}')
    fig.write_image(f'figs/{save_name}.pdf')
    fig.show()
    return fig


if __name__ == '__main__':
    gamma_expected_theta_params = Params(
        gamma=23.4352,
        theta=4.5,
        center=78.4,
        amp=0.675,
        lin=0.00121,
        const=7.367,
        lin_occ=0.0001453,
        vary_theta=False,
        vary_gamma=True,
        datnum=2167
    )

    more_gamma_expected_theta_params = Params(
        gamma=59.0125,
        theta=4.5,
        center=21.538,
        amp=0.580,
        lin=0.00109,
        const=7.207,
        lin_occ=0.0001884,
        vary_theta=False,
        vary_gamma=True,
        datnum=2170
    )

    most_gamma_expected_theta_params = Params(
        gamma=109.6544,
        theta=4.5,
        center=-56.723,
        amp=0.481,
        lin=0.00097,
        const=7.214,
        lin_occ=0.00097,
        vary_theta=False,
        vary_gamma=True,
        datnum=2213
    )

    equal_gamma_theta_params = Params(
        gamma=5.7764,
        theta=4.2,
        center=8.825,
        amp=0.784,
        lin=0.00137,
        const=7.101,
        lin_occ=0.0000485,
        vary_theta=False,
        vary_gamma=True,
        datnum=2121
    )

    gamma_force_fit_params = Params(
        gamma=21.5358,
        theta=9.597,
        center=90.782,
        amp=0.667,
        lin=0.00121,
        const=7.378,
        lin_occ=0.0001217,
        vary_theta=True,
        vary_gamma=True,
        datnum=2167
    )

    thermal_hot_fit_params = Params(
        gamma=0.4732,
        theta=4.672,
        center=7.514,
        amp=0.939,
        lin=0.00152,
        const=7.205,
        lin_occ=-0.0000358,
        vary_theta=True,
        vary_gamma=False,
        datnum=2164
    )
    plot_and_save_dndt_vs_sweepgate(datnum=2167,
                                    params=gamma_expected_theta_params,
                                    fit_to_hot=False,
                                    title='Gamma Broadened - Expected Gamma/Theta',
                                    save_name='BroadenedExpectedTheta_dI')

    plot_and_save_dndt_vs_sweepgate(datnum=2167,
                                    params=gamma_force_fit_params,
                                    fit_to_hot=False,
                                    title='Gamma Broadened - Best fit to NRG',
                                    save_name='BroadenedForcedFit_dI', )

    plot_and_save_dndt_vs_sweepgate(datnum=2164,
                                    params=thermal_hot_fit_params,
                                    fit_to_hot=True,
                                    title='Thermally Broadened - Hot fit',
                                    save_name='WeaklyCoupled_dI', )
    #
    # # plot_and_save_dndt_vs_sweepgate(data_json_path='downloaded_data_jsons/dat2164_weakly_coupled_cold_fit_nrg.json',
    # #                                 max_dndt=0.14628,
    # #                                 title='Thermally Broadened - Cold fit',
    # #                                 save_name='WeakColdFit.json', )

    # ############### vs N graphs
    plot_and_save_dndt_vs_N(data_json_path='downloaded_data_jsons/dat2167_gamma_broadened_expected_theta_vs_N.json',
                            max_dndt=0.02555,
                            title='Gamma Broadened - Expected Gamma/Theta',
                            save_name='BroadenedExpectedTheta_dIvsN')

    plot_and_save_dndt_vs_N(data_json_path='downloaded_data_jsons/dat2167_gamma_broadened_best_fit_nrg_vs_N.json',
                            max_dndt=0.02555,
                            title='Gamma Broadened - Best fit to NRG',
                            save_name='BroadenedForcedFit_dIvsN', )

    plot_and_save_dndt_vs_N(data_json_path='downloaded_data_jsons/dat2164_weakly_coupled_hot_fit_nrg_vs_N.json',
                            max_dndt=0.14628,
                            title='Thermally Broadened - Hot fit',
                            save_name='WeaklyCoupled_dIvsN', )
    #
    # plot_and_save_dndt_vs_N(data_json_path='downloaded_data_jsons/dat2164_weakly_coupled_cold_fit_nrg_vs_N.json',
    #                         max_dndt=0.14628,
    #                         title='Thermally Broadened - Cold fit',
    #                         save_name='WeakColdFit_vs_N.json', )

    # ############## Integrated vs sweepgate graphs
    #
    plot_and_save_integrated_vs_sweepgate(datnum=2167,
                                          params=gamma_expected_theta_params,
                                          title='Gamma Broadened Entropy vs Sweepgate',
                                          save_name='BroadenedExpectedTheta_Entropy')

    plot_and_save_integrated_vs_sweepgate(datnum=2167,
                                          params=gamma_force_fit_params,
                                          title='Gamma Broadened Entropy vs Sweepgate',
                                          save_name='BroadenedForcedFit_Entropy', )

    plot_and_save_integrated_vs_sweepgate(datnum=2164,
                                          params=thermal_hot_fit_params,
                                          title='Thermally Broadened Entropy vs Sweepgate',
                                          save_name='WeaklyCoupled_Entropy', )

    #
    # # ############## Integrated vs Occupation graphs
    #
    plot_and_save_integrated_vs_N(datnum=2167,
                                  params=gamma_expected_theta_params,
                                  title='Gamma Broadened Entropy vs Sweepgate',
                                  save_name='BroadenedExpectedTheta_EntropyVsN')

    plot_and_save_integrated_vs_N(datnum=2167,
                                  params=gamma_force_fit_params,
                                  title='Gamma Broadened Entropy vs Sweepgate',
                                  save_name='BroadenedForcedFit_EntropyVsN', )

    plot_and_save_integrated_vs_N(datnum=2164,
                                  params=thermal_hot_fit_params,
                                  title='Thermally Broadened Entropy vs Sweepgate',
                                  save_name='WeaklyCoupled_EntropyVsN', )

    # ################ Multiple dN/dT and integrated on one plot

    plot_and_save_multiple_dndt(
        params=[thermal_hot_fit_params,
                equal_gamma_theta_params,
                gamma_expected_theta_params,
                more_gamma_expected_theta_params,
                most_gamma_expected_theta_params,
                ],
        title=f'{DELTA}I for several Gamma/T ratios',
        save_name='MultipledNdT')
