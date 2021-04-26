from src.DatObject.Make_Dat import DatHDF, get_dat, get_dats
from src.Dash.DatPlotting import OneD, TwoD
from Analysis.Feb2021.common import _get_transition_fit_func_params
from src.CoreUtil import order_list
from typing import List, Callable

import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

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

    fig.add_trace(plotter.trace(x=biases, data=thetas, mode='markers+lines'))
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
    dats = get_dats((6449, 6456+1))
    dats = order_list(dats, [dat.Logs.fds['HO1/10M'] for dat in dats])
    fig = dcbias_multi_dat(dats)
    fig.show()
