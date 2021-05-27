from scipy.interpolate import interp1d
import numpy as np
import plotly.graph_objects as go

import src.UsefulFunctions as U
from src.Characters import DELTA
from src.DatObject.Make_Dat import get_dats, get_dat
from src.Dash.DatPlotting import OneD, TwoD

import plotly.io as pio

pio.renderers.default = 'browser'

p1d = OneD(dat=None)
p2d = TwoD(dat=None)


def plot_dndt_without_zero_offset(data_json_path: str,
                                  max_dndt: float = 1) -> go.Figure:
    """
    Re-plot the scaled dNdTs from Dash with data starting at zero instead of being somewhere else
    Args:
        data_json_path ():
        max_dndt (): To rescale the dN/dT data back into something like nA instead of 0 -> 1

    Returns:

    """

    def get_offset(d):
        return np.nanmean(d[:round(d.shape[-1] / 10)])  # Use first 10% to find 'zero'

    def rescale(d, offset_, max_):
        d = d - offset_
        d = d / np.nanmax(d) * max_
        return d

    all_data = U.data_from_json(data_json_path)
    x = all_data['Data - i_sense_cold_x'] if 'Data - i_sense_cold_x' in all_data else all_data['Data - i_sense_hot_x']
    data_dndt = all_data['Scaled Data - dndt']
    nrg_dndt = all_data['Scaled NRG dndt']
    # occupation = all_data['Scaled NRG i_sense']

    fig = p1d.figure(xlabel='Sweep Gate (mV)', ylabel=f'{DELTA}I (nA)')
    x = x / 100  # Convert to real mV

    # Plot data with offset subtracted
    offset = get_offset(data_dndt)
    data = rescale(data_dndt, offset, max_dndt)
    fig.add_trace(p1d.trace(data=data, x=x, mode='lines', name='Data'))

    # Plot nrg with offset subtracted and scaled to data
    offset = get_offset(nrg_dndt)
    data = rescale(nrg_dndt, offset, max_dndt)
    fig.add_trace(p1d.trace(data=data, x=x, mode='lines', name='NRG'))

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
    return fig


def plot_integrated(datnum: int):
    dat = get_dat(datnum)
    x = dat.SquareEntropy.avg_x / 100
    dndt = dat.SquareEntropy.get_Outputs(name='forced_theta_linear').average_entropy_signal
    int_info = dat.Entropy.get_integration_info(name='forced_theta_linear')
    int_dndt = int_info.integrate(dndt)

    fig = p1d.figure(xlabel='Sweep Gate (mV)', ylabel='Entropy (kB)')
    fig.add_trace(p1d.trace(x=x, data=int_dndt, mode='lines'))
    return fig


def plot_integrated_vs_N(datnum: int,
                         data_json_path: str,
                         ) -> go.Figure:
    """
    Plot integrated entropy vs occupation instead of vs sweepgate
    Args:
        datnum (): use dat to get integrated data directly because it is easier to do
        data_json_path (): Need this to get the occupation data from the NRG Fit to same dat as datnum (the vs N plot)

    Returns:

    """
    dat = get_dat(datnum)
    all_data = U.data_from_json(data_json_path)
    print(all_data.keys())
    x = all_data['Data dN/dT_x']

    dndt = dat.SquareEntropy.get_Outputs(name='forced_theta_linear').average_entropy_signal
    int_info = dat.Entropy.get_integration_info(name='forced_theta_linear')
    int_dndt = int_info.integrate(dndt)

    fig = p1d.figure(xlabel='Occupation', ylabel='Entropy (kB)')
    fig.add_trace(p1d.trace(x=x, data=int_dndt, mode='lines+markers'))

    return fig


def plot_and_save_dndt_vs_sweepgate(
        data_json_path: str,
        max_dndt: float,
        title: str,
        save_name: str) -> go.Figure:
    fig = plot_dndt_without_zero_offset(data_json_path=data_json_path,
                                        max_dndt=max_dndt)
    fig.update_layout(title=title)
    U.fig_to_data_json(fig, f'figs/data/{save_name}')
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
    fig.show()
    return fig


def plot_and_save_integrated_vs_sweepgate(
        datnum: int,
        title: str,
        save_name: str) -> go.Figure:
    fig = plot_integrated(datnum=datnum)
    fig.update_layout(title=title)
    U.fig_to_data_json(fig, f'figs/data/{save_name}')
    fig.show()
    return fig


def plot_and_save_integrated_vs_N(
        datnum: int,
        data_json_path: str,
        title: str,
        save_name: str) -> go.Figure:
    fig = plot_integrated_vs_N(datnum=datnum, data_json_path=data_json_path)
    fig.update_layout(title=title)
    U.fig_to_data_json(fig, f'figs/data/{save_name}')
    fig.show()
    return fig


if __name__ == '__main__':
    # plot_and_save_dndt_vs_sweepgate(data_json_path='downloaded_data_jsons/dat2167_gamma_broadened_expected_theta.json',
    #                                 max_dndt=0.02555,
    #                                 title='Gamma Broadened - Expected Gamma/Theta',
    #                                 save_name='BroadenedExpectedTheta.json')
    #
    # plot_and_save_dndt_vs_sweepgate(data_json_path='downloaded_data_jsons/dat2167_gamma_broadened_best_fit_nrg.json',
    #                                 max_dndt=0.02555,
    #                                 title='Gamma Broadened - Best fit to NRG',
    #                                 save_name='BroadenedBestFit.json',)
    #
    # plot_and_save_dndt_vs_sweepgate(data_json_path='downloaded_data_jsons/dat2164_weakly_coupled_hot_fit_nrg.json',
    #                                 max_dndt=0.14628,
    #                                 title='Thermally Broadened - Hot fit',
    #                                 save_name='WeakHotFit.json', )
    #
    # plot_and_save_dndt_vs_sweepgate(data_json_path='downloaded_data_jsons/dat2164_weakly_coupled_cold_fit_nrg.json',
    #                                 max_dndt=0.14628,
    #                                 title='Thermally Broadened - Cold fit',
    #                                 save_name='WeakColdFit.json', )

    # ############### vs N graphs
    # plot_and_save_dndt_vs_N(data_json_path='downloaded_data_jsons/dat2167_gamma_broadened_expected_theta_vs_N.json',
    #                         max_dndt=0.02555,
    #                         title='Gamma Broadened - Expected Gamma/Theta',
    #                         save_name='BroadenedExpectedTheta_vs_N.json')
    #
    # plot_and_save_dndt_vs_N(data_json_path='downloaded_data_jsons/dat2167_gamma_broadened_best_fit_nrg_vs_N.json',
    #                         max_dndt=0.02555,
    #                         title='Gamma Broadened - Best fit to NRG',
    #                         save_name='BroadenedBestFit_vs_N.json', )
    #
    # plot_and_save_dndt_vs_N(data_json_path='downloaded_data_jsons/dat2164_weakly_coupled_hot_fit_nrg_vs_N.json',
    #                         max_dndt=0.14628,
    #                         title='Thermally Broadened - Hot fit',
    #                         save_name='WeakHotFit_vs_N.json', )
    #
    # plot_and_save_dndt_vs_N(data_json_path='downloaded_data_jsons/dat2164_weakly_coupled_cold_fit_nrg_vs_N.json',
    #                         max_dndt=0.14628,
    #                         title='Thermally Broadened - Cold fit',
    #                         save_name='WeakColdFit_vs_N.json', )

    # ############## Integrated vs sweepgate graphs

    plot_and_save_integrated_vs_sweepgate(datnum=2164, title='Thermally Broadened Entropy vs Sweepgate',
                                          save_name='WeakEntropyVsSweepgate.json')

    plot_and_save_integrated_vs_sweepgate(datnum=2167, title='Gamma Broadened Entropy vs Sweepgate',
                                          save_name='GammaEntropyVsSweepgate.json')

    # ############## Integrated vs Occupation graphs

    plot_and_save_integrated_vs_N(datnum=2164,
                                  data_json_path='downloaded_data_jsons/dat2164_weakly_coupled_hot_fit_nrg_vs_N.json',
                                  title='Thermally Broadened Entropy vs Sweepgate',
                                  save_name='WeakEntropyVsN.json')

    plot_and_save_integrated_vs_N(datnum=2167,
                                  data_json_path='downloaded_data_jsons/dat2167_gamma_broadened_best_fit_nrg_vs_N.json',
                                  title='Gamma Broadened Entropy vs Sweepgate',
                                  save_name='GammaEntropyVsN.json')
