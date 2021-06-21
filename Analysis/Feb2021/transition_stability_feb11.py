from __future__ import annotations
import plotly.graph_objects as go
from typing import TYPE_CHECKING, Iterable
import numpy as np
import plotly.io as pio

from src.dat_object.make_dat import get_dats
import src.useful_functions as U
from src.plotting.plotly.dat_plotting import OneD, TwoD

if TYPE_CHECKING:
    from src.dat_object.make_dat import DatHDF


pio.renderers.default = 'browser'


def plot_avg_thetas(dats: Iterable[DatHDF]) -> go.Figure:
    dats = list(dats)
    thetas = [dat.Transition.avg_fit.best_values.theta for dat in dats]
    x = [dat.Logs.fds['ESS'] for dat in dats]

    plotter = OneD(dats=dats)
    fig = plotter.figure(xlabel='ESS /mV', ylabel='Theta /mV',
                         title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: Avg Theta')
    fig.add_trace(plotter.trace(data=thetas, x=x, mode='markers', name='Avg Fit Theta'))
    return fig


def plot_per_row_of_transition_param(dats: Iterable[DatHDF], param_name: str, x: U.ARRAY_LIKE,
                                     xlabel: str, stdev_only=False) -> go.Figure:
    """
    Fit values or stdev per row of Transition data

    Args:
        dats ():
        param_name ():
        x ():
        xlabel ():
        stdev_only (): Whether to plot the stdev of fits only rather than fit values

    Returns:

    """

    def get_ylabel(name: str) -> str:
        t = get_full_name(name)
        u = get_units(name)
        if stdev_only:
            return f'{U.Characters.SIG} {t}/{u}'
        else:
            return f'{t}/{u}'

    def get_full_name(name: str) -> str:
        if name == 'mid':
            t = 'Center'
        elif name == 'theta':
            t = 'Theta'
        elif name == 'amp':
            t = 'Amplitude'
        elif name == 'lin':
            t = 'Linear Component'
        elif name == 'const':
            t = 'Constant Offset'
        elif name == 'g':
            t = 'Gamma'
        else:
            raise KeyError(f'{name} not a recognized key')
        return t

    def get_units(name: str) -> str:
        if name == 'mid':
            u = 'mV'
        elif name == 'theta':
            u = 'mV'
        elif name == 'amp':
            u = 'nA'
        elif name == 'lin':
            u = 'nA/mV'
        elif name == 'const':
            u = 'nA'
        elif name == 'g':
            u = 'mV'
        else:
            raise KeyError(f'{name} not a recognized key')
        return u

    def title():
        a = f'Dats{dats[0].datnum}-{dats[-1].datnum}: '
        b = f'Standard Deviation of '
        c = f'{get_full_name(param_name)}'
        if stdev_only:
            return a + b + c
        else:
            return a + c

    dats = list(dats)
    if param_name not in dats[0].Transition.avg_fit.best_values.keys:
        raise KeyError(f'{param_name} not in {dats[0].Transition.avg_fit.best_values.keys}')

    fit_vals = [[fit.best_values.get(param_name, default=np.nan) for fit in dat.Transition.row_fits] for
                dat in dats]
    errs = [np.nanstd(row) for row in fit_vals]
    if stdev_only:
        fit_vals = errs
        errs = None
    else:
        fit_vals = [np.mean(row) for row in fit_vals]

    plotter = OneD(dats=dats)
    fig = plotter.plot(data=fit_vals, data_err=errs, x=x, text=[f'{dat.datnum}' for dat in dats],
                       xlabel=xlabel, ylabel=get_ylabel(param_name),
                       title=title(),
                       mode='markers')
    return fig


def plot_stdev_of_avg(dat: DatHDF) -> go.Figure:
    """Plot the stdev of averaging the 2D data (i.e. looking for whether more uncertainty near transition)"""
    plotter = OneD(dat=dat)

    fig = plotter.figure(
        ylabel=f'{U.Characters.SIG}I_sense /nA',
        title=f'Dat{dat.datnum}: Standard deviation of averaged I_sense data after centering',
    )
    fig.add_trace(trace_stdev_of_avg(dat))
    return fig


def trace_stdev_of_avg(dat: DatHDF) -> go.Scatter:
    stdev = dat.Transition.avg_data_std
    x = dat.Transition.x

    plotter = OneD(dat=dat)
    trace = plotter.trace(data=stdev, x=x,
                          name=f'Dat{dat.datnum}',
                          mode='lines')
    return trace


def waterfall_stdev_of_avg(dats: Iterable[DatHDF]) -> go.Figure:
    dats = list(dats)

    data = np.array([dat.Transition.avg_data_std for dat in dats])
    xs = [dat.Transition.avg_x for dat in dats]

    data, xs = [U.resample_data(data=arr, max_num_pnts=500, resample_method='bin') for arr in [data, xs]]

    plotter = TwoD(dats=dats)
    fig = plotter.figure(
        title=f'Dat{dats[0].datnum}-{dats[-1].datnum}: '
              f'Standard deviation of averaged I_sense data after centering')
    fig.add_traces([go.Scatter3d(mode='lines', x=x,
                                 y=[dat.Logs.fds['ESS']] * len(dat.Transition.avg_x),
                                 z=row, name=f'Dat{dat.datnum}') for x, row, dat in zip(xs, data, dats)])
    fig.update_layout(
        scene=dict(
            xaxis_title=dats[0].Logs.xlabel,
            yaxis_title=f'ESS /mV',
            zaxis_title=f'{U.Characters.SIG}I_sense /nA',
        )
    )
    return fig


if __name__ == '__main__':
    # datnums = range(649, 664 + 1)
    # dats = get_dats(datnums, exp2hdf=Feb21Exp2HDF)

    # fig = plot_avg_thetas(dats)
    # fig = plot_stdev_of_avg(dats[2])

    # for param in ['mid', 'const', 'lin', 'theta', 'amp']:
    #     fig = plot_per_row_of_transition_param(dats, param_name=param,
    #                                            x=[dat.Logs.fds['ESS'] for dat in dats], xlabel='ESS /mV',
    #                                            stdev_only=False)
    #     fig.show(renderer='browser')

    # fig = plot_stdev_of_avg(dats[0])
    # for dat in dats[1:]:
    #     fig.add_trace(trace_stdev_of_avg(dat))
    # fig = waterfall_stdev_of_avg(dats)
    # fig.show(renderer='browser')

    # datnums = [702, 703, 707, 708]
    datnums = [7436, 7435]
    all_dats = get_dats(datnums)

    plotter = OneD(dats=all_dats)
    fig = plotter.figure(xlabel='Time /s', ylabel='Current /Arbitrary',
                         title=f'Dats{all_dats[0].datnum}-{all_dats[-1].datnum}: Transition ReadVsTime<br>Decimated to 10Hz',
                         )
    for dat, name, bias in zip(all_dats,
                               ['On Transition 300uV',
                                'Off Transition 300uV',
                                'On Transition 500uV',
                                'Off Transition 500uV'],
                               [300, 300, 500,  500]):
        data = dat.Data.get_data('i_sense')
        numpts = data.shape[-1]
        time_elapsed = numpts / dat.Logs.measure_freq
        x = np.linspace(0, time_elapsed, numpts)

        data = data - np.mean(data)
        data = data / bias
        data = U.decimate(data, dat.Logs.measure_freq, desired_freq=10)
        x = U.get_matching_x(x, data)

        fig.add_trace(plotter.trace(data, x=x, mode='lines', name=name))

    fig.show(renderer='browser')
