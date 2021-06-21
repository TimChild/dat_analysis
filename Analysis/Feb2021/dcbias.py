from typing import List, Callable
from progressbar import progressbar
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import copy

from src.dat_object.make_dat import DatHDF, get_dat, get_dats
from src.plotting.plotly.dat_plotting import OneD, TwoD
from Analysis.Feb2021.common import _get_transition_fit_func_params, sort_by_temps
from Analysis.Feb2021.common_plotting import common_dat_hover_infos
from src.plotting.plotly.hover_info import HoverInfo, HoverInfoGroup
from src.core_util import order_list

pio.renderers.default = 'browser'


def dcbias_multi_dat(dats: List[DatHDF]):
    plotter = OneD(dats=dats)

    fig = plotter.figure(xlabel='Heater Current Bias /nA', ylabel='Theta /mV',
                         title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: DCbias')

    thetas = []
    biases = []
    for dat in dats:
        fit = dat.Transition.get_fit(which='avg', name='default', check_exists=False)
        theta = fit.best_values.theta
        thetas.append(theta)
        biases.append(dat.Logs.fds['HO1/10M']/10)

    hover_infos = common_dat_hover_infos(datnum=True)
    hover_infos.append(HoverInfo(name='Bias', func=lambda dat: dat.Logs.fds['HO1/10M']/10, precision='.2f', units='nA'))
    hover_infos.append(HoverInfo(name='Theta', func=lambda dat: dat.Transition.get_fit().best_values.theta, precision='.2f', units='mV'))
    hover_group = HoverInfoGroup(hover_infos=hover_infos)

    fig.add_trace(plotter.trace(x=biases, data=thetas, mode='markers+lines', hover_template=hover_group.template,
                                hover_data=hover_group.customdata(dats)))
    return fig


def dcbias_single_dat(dat: DatHDF, fig_x_func: Callable, x_label: str):
    plotter = OneD(dat=dat)
    x = dat.Transition.x
    data = dat.Transition.data
    func, params = _get_transition_fit_func_params(x, data[0], 'i_sense', theta=None, gamma=0)
    fits = dat.Transition.get_row_fits(name='i_sense', fit_func=func, initial_params=None, check_exists=False,
                                       overwrite=False)
    thetas = [fit.best_values.theta for fit in fits]
    # sweepgates_y = dat.Data.get_data('sweepgates_y')
    # y = np.linspace(sweepgates_y[0][1]/10, sweepgates_y[0][2]/10, dat.Data.get_data('y').shape[0])
    fig_x = fig_x_func(dat)
    fig = plotter.plot(data=thetas, x=fig_x, xlabel=x_label, ylabel='Theta /mV', mode='markers+lines',
                       title=f'Dat{dat.datnum}: MC temp={dat.Logs.temps.mc * 1000:.1f}mK DCBias thetas')
    return fig


def temp_calc(datnum):
    """Just so I can pass into a multiprocessor"""
    get_dat(datnum).Transition.get_fit(check_exists=False)
    return datnum


if __name__ == '__main__':
    # Single Dat DCBias
    # # dats = get_dats((5352, 5357+1))
    # # dats = get_dats([5726, 5727, 5731])
    # # dats = get_dats([6432, 6434, 6435, 6436, 6437, 6439, 6440])
    # dats = get_dats([6447])
    # fig_x_func = lambda dat: np.linspace(dat.Data.get_data('sweepgates_y')[0][1]/10,
    #                                  dat.Data.get_data('sweepgates_y')[0][2]/10,
    #                                  dat.Data.get_data('y').shape[0])
    # x_label = "HQPC bias /nA"
    #
    # # fig_x_func = lambda dat: dat.Data.get_data('y')
    # # x_label = dats[-1].Logs.ylabel
    #
    # for dat in dats[-1:]:
    #     fig = dcbias_single_dat(dat, fig_x_func=fig_x_func, x_label=x_label)
    #     fig.show()

    # Multi Dat DCbias
    # all_dats = get_dats((6449, 6456 + 1))
    # all_dats = get_dats((6912, 6963 + 1))
    # all_dats = get_dats((6960, 6963 + 1))
    all_dats = get_dats((7437, 7844 + 1))
    with ProcessPoolExecutor() as pool:
        dones = pool.map(temp_calc,  [dat.datnum for dat in all_dats])
        for num in dones:
            print(num)

    dats_by_temp = sort_by_temps(all_dats)
    figs = []
    for temp, dats in progressbar(dats_by_temp.items()):
        dats = order_list(dats, [dat.Logs.fds['HO1/10M'] for dat in dats])
        fig = dcbias_multi_dat(dats)
        fig.update_layout(title=f'Dats{min([dat.datnum for dat in dats])}-{max([dat.datnum for dat in dats])}: DC bias '
                                f'at {np.nanmean([dat.Logs.temps.mc*1000 for dat in dats]):.0f}mK')
        fig.data[0].update(name=f'{temp:.0f}mK')
        figs.append(fig)

    for fig in figs:
        fig.show()


    multi_fig = copy.copy(figs[0])
    for fig in figs[1:]:
        multi_fig.add_trace(fig.data[0])

    multi_fig.show()

