from typing import List, Callable

from Analysis.Feb2021.common_plotting import common_dat_hover_infos
from src.dat_object.dat_hdf import DatHDF
from src.plotting.plotly import OneD
from src.plotting.plotly.hover_info import HoverInfo, HoverInfoGroup
from src.analysis_tools.transition import _get_transition_fit_func_params


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
    hover_infos.append(HoverInfo(name='Bias', func=lambda dat: dat.Logs.dacs['HO1/10M']/10, precision='.2f', units='nA'))
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
    fig_x = fig_x_func(dat)
    fig = plotter.plot(data=thetas, x=fig_x, xlabel=x_label, ylabel='Theta /mV', mode='markers+lines',
                       title=f'Dat{dat.datnum}: MC temp={dat.Logs.temps.mc * 1000:.1f}mK DCBias thetas')
    return fig