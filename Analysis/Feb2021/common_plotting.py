"""
Sep 21 -- A few of the plots used in analysis, very far from a complete list, and probably most are too specific to be
useful again.
TODO: Some functions in here are probably worth extracting. Especially the common_dat_info stuff, although that likely
needs to be improved to be very useful
"""

from __future__ import annotations
from typing import List, Callable, Optional, Union, TYPE_CHECKING

import numpy as np
from plotly import graph_objects as go
import re

from Analysis.Feb2021.common import dat_integrated_sub_lin
from src import useful_functions as U
from src.plotting.plotly.dat_plotting import OneD
from src.dat_object.dat_hdf import DatHDF
from src.dat_object.make_dat import get_dats
from src.plotting.plotly.hover_info import HoverInfo, HoverInfoGroup, _additional_data_dict_converter

if TYPE_CHECKING:
    from Analysis.Feb2021.entropy_gamma_final import AnalysisGeneral


def plot_fit_integrated_comparison(dats: List[DatHDF], x_func: Callable, x_label: str, title_append: Optional[str] = '',
                                   int_info_name: Optional[str] = None, fit_name: str = 'SPS.0045',
                                   plot=True) -> go.Figure():
    if int_info_name is None:
        int_info_name = 'default'
    plotter = OneD(dats=dats)
    fig = plotter.figure(
        title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: Fit and Integrated (\'{int_info_name}\') Entropy{title_append}',
        xlabel=x_label,
        ylabel='Entropy /kB')

    hover_infos = [
        HoverInfo(name='Dat', func=lambda dat: dat.datnum, precision='.d', units=''),
        HoverInfo(name='Temperature', func=lambda dat: dat.Logs.temps.mc * 1000, precision='.1f', units='mK'),
        HoverInfo(name='Bias', func=lambda dat: dat.AWG.max(0) / 10, precision='.1f', units='nA'),
        # HoverInfo(name='Fit Entropy', func=lambda dat: dat.SquareEntropy.get_fit(which_fit='entropy', fit_name=fit_name).best_values.dS,
        #           precision='.2f', units='kB'),
        HoverInfo(name='Integrated Entropy',
                  func=lambda dat: np.nanmean(dat.Entropy.get_integrated_entropy(name=int_info_name,
                                                                                 data=dat.SquareEntropy.get_Outputs(
                                                                                     name=fit_name).average_entropy_signal)[
                                              -10:]),
                  # TODO: Change to using proper output (with setpoints)
                  precision='.2f', units='kB'),
    ]

    funcs, template = _additional_data_dict_converter(hover_infos)
    x = [x_func(dat) for dat in dats]
    hover_data = [[func(dat) for func in funcs] for dat in dats]

    entropies = [dat.Entropy.get_fit(name=fit_name).best_values.dS for dat in dats]
    # entropy_errs = [np.nanstd([
    #     f.best_values.dS if f.best_values.dS is not None else np.nan
    #     for f in dat.Entropy.get_row_fits(name=fit_name) for dat in dats
    # ]) / np.sqrt(dat.Data.y_array.shape[0]) for dat in dats]
    entropy_errs = None
    fig.add_trace(plotter.trace(
        data=entropies, data_err=entropy_errs, x=x, name=f'Fit Entropy',
        mode='markers',
        trace_kwargs={'customdata': hover_data, 'hovertemplate': template})
    )
    integrated_entropies = [np.nanmean(
        dat.Entropy.get_integrated_entropy(name=int_info_name,
                                           data=dat.SquareEntropy.get_Outputs(name=fit_name).average_entropy_signal)[
        -10:]) for dat in dats]

    ### For plotting the impurity dot scans (i.e. isolate only the integrated entropy due to the main dot transition)
    # integrated_entropies = []
    # for dat in dats:
    #     out = dat.SquareEntropy.get_Outputs(name=fit_name)
    #     x1, x2 = U.get_data_index(out.x, [-40, 40], is_sorted=True)
    #     integrated = dat.Entropy.get_integrated_entropy(name=int_info_name,
    #                                                     data=out.average_entropy_signal)
    #     integrated_entropies.append(integrated[x2]-integrated[x1])

    fig.add_trace(plotter.trace(
        data=integrated_entropies, x=x, name=f'Integrated',
        mode='markers+lines',
        trace_kwargs={'customdata': hover_data, 'hovertemplate': template})
    )

    if plot:
        fig.show(renderer='browser')
    return fig


def entropy_vs_time_trace(dats: List[DatHDF], trace_name=None, integrated=False, fit_name: str = 'SPS.005',
                          integrated_name: str = 'first'):
    plotter = OneD(dats=dats)
    if integrated is False:
        entropies = [dat.Entropy.get_fit(which='avg', name=fit_name).best_values.dS for dat in dats]
        entropies_err = [np.nanstd(
            [fit.best_values.dS if fit.best_values.dS is not None else np.nan for fit in
             dat.Entropy.get_row_fits(name=fit_name)]
        ) / np.sqrt(len(dats)) for dat in dats]
    else:
        entropies = [np.nanmean(
            dat.Entropy.get_integrated_entropy(
                name=integrated_name,
                data=dat.SquareEntropy.get_Outputs(name=fit_name).average_entropy_signal)[-10:]) for dat in dats]
        entropies_err = [0.0] * len(dats)

    times = [str(dat.Logs.time_completed) for dat in dats]

    trace = plotter.trace(data=entropies, data_err=entropies_err, x=times, text=[dat.datnum for dat in dats],
                          mode='lines', name=trace_name)
    return trace


def entropy_vs_time_fig(title: Optional[str] = None):
    plotter = OneD()
    if title is None:
        title = 'Entropy vs Time'
    fig = plotter.figure(xlabel='Time', ylabel='Entropy /kB', title=title)
    fig.update_xaxes(tickformat="%H:%M\n%a")
    return fig


def get_integrated_trace(dats: List[DatHDF], x_func: Callable, x_label: str,
                         trace_name: str,
                         save_name: str,
                         int_info_name: Optional[str] = None, SE_output_name: Optional[str] = None,
                         sub_linear: bool = False, signal_width: Optional[Union[float, Callable]] = None
                         ) -> go.Scatter:
    """
    Returns a trace Integrated Entropy vs x_func
    Args:
        dats ():
        x_func ():
        x_label ():
        trace_name ():
        save_name ():
        int_info_name ():
        SE_output_name ():
        sub_linear (): Whether to subtract linear entropy term from both integrated trace (note: requires signal_width)
        signal_width (): How wide the actual entropy signal is so that a slope can be fit to the sides around it,
            Can be a callable which takes 'dat' as the argument

    Returns:
        go.Scatter: The trace
    """
    if int_info_name is None:
        int_info_name = save_name
    if SE_output_name is None:
        SE_output_name = save_name

    plotter = OneD(dats=dats)
    dats = U.order_list(dats, [x_func(dat) for dat in dats])

    standard_hover_infos = common_dat_hover_infos(datnum=True,
                                                  heater_bias=True,
                                                  fit_entropy_name=save_name,
                                                  fit_entropy=True,
                                                  int_info_name=int_info_name,
                                                  output_name=SE_output_name,
                                                  integrated_entropy=True, sub_lin=sub_linear,
                                                  sub_lin_width=signal_width,
                                                  int_info=True,
                                                  )
    standard_hover_infos.append(HoverInfo(name=x_label, func=lambda dat: x_func(dat), precision='.1f', units='mV',
                                          position=1))
    hover_infos = HoverInfoGroup(standard_hover_infos)

    x = [x_func(dat) for dat in dats]
    if not sub_linear:
        integrated_entropies = [np.nanmean(
            dat.Entropy.get_integrated_entropy(name=int_info_name,
                                               data=dat.SquareEntropy.get_Outputs(
                                                   name=SE_output_name, check_exists=True).average_entropy_signal
                                               )[-10:]) for dat in dats]
    else:
        integrated_entropies = [np.nanmean(dat_integrated_sub_lin(dat, signal_width=signal_width(dat),
                                                                  int_info_name=int_info_name,
                                                                  output_name=SE_output_name)[-10:]) for dat in dats]
    trace = plotter.trace(
        data=integrated_entropies, x=x, name=trace_name,
        mode='markers+lines',
        trace_kwargs=dict(customdata=hover_infos.customdata(dats), hovertemplate=hover_infos.template)
    )
    return trace


def get_integrated_fig(dats=None, title_append: str = '') -> go.Figure:
    plotter = OneD()
    if dats:
        title_prepend = f'Dats{dats[0].datnum}-{dats[-1].datnum}: '
    else:
        title_prepend = ''
    fig = plotter.figure(xlabel='ESC /mV', ylabel='Entropy /kB',
                         title=f'{title_prepend}Integrated Entropy {title_append}')
    return fig


def transition_trace(dats: List[DatHDF], x_func: Callable,
                     from_square_entropy: bool = True, fit_name: str = 'default',
                     param: str = 'amp', label: str = '',
                     **kwargs) -> go.Scatter:
    divide_const, divide_acc_divider = False, False
    if param == 'amp/const':
        divide_const = True
        param = 'amp'
    elif param == 'theta real':
        divide_acc_divider = True
        param = 'theta'

    plotter = OneD(dats=dats)
    if from_square_entropy:
        vals = [dat.SquareEntropy.get_fit(which_fit='transition', fit_name=fit_name, check_exists=True).best_values.get(
            param) for dat in dats]
    else:
        vals = [dat.Transition.get_fit(name=fit_name).best_values.get(param) for dat in dats]
        if divide_const:
            vals = [amp / dat.Transition.get_fit(name=fit_name).best_values.const for amp, dat in zip(vals, dats)]
        elif divide_acc_divider:
            divider_vals = [int(re.search(r'\d+', dat.Logs.xlabel)[0]) for dat in dats]  # e.g. get the 1000 part of ACC*1000 /mV
            vals = [val/divider for val, divider in zip(vals, divider_vals)]
    x = [x_func(dat) for dat in dats]
    trace = plotter.trace(x=x, data=vals, name=label, text=[dat.datnum for dat in dats], **kwargs)
    return trace


def single_transition_trace(dat: DatHDF, label: Optional[str] = None, subtract_fit=False,
                            fit_only=False, fit_name: str = 'narrow', transition_only=True,
                            se_output_name: str = 'SPS.005',
                            csq_mapped=False) -> go.Scatter():
    plotter = OneD(dat=dat)
    if transition_only:
        if csq_mapped:
            x = dat.Data.get_data('csq_x_avg')
        else:
            x = dat.Transition.avg_x
    else:
        if csq_mapped:
            raise NotImplementedError
        x = dat.SquareEntropy.avg_x

    if not fit_only:
        if transition_only:
            if csq_mapped:
                data = dat.Data.get_data('csq_mapped_avg')
            else:
                data = dat.Transition.avg_data
        else:
            data = dat.SquareEntropy.get_transition_part(name=se_output_name, part='cold')
    else:
        data = None  # Set below

    if fit_only or subtract_fit:
        if transition_only:
            fit = dat.Transition.get_fit(name=fit_name)
        else:
            fit = dat.SquareEntropy.get_fit(which_fit='transition', fit_name=fit_name)
        if fit_only:
            data = fit.eval_fit(x=x)
        elif subtract_fit:
            data = data - fit.eval_fit(x=x)

    trace = plotter.trace(x=x, data=data, name=label, mode='lines')
    return trace


def transition_fig(dats: Optional[List[DatHDF]] = None, xlabel: str = '/mV', title_append: str = '',
                   param: str = 'amp') -> go.Figure:
    plotter = OneD(dats=dats)
    titles = {
        'amp': 'Amplitude',
        'theta': 'Theta',
        'theta real': 'Theta',
        'g': 'Gamma',
        'amp/const': 'Amplitude/Const (sensitivity)',
    }
    ylabels = {
        'amp': 'Amplitude /nA',
        'theta': 'Theta /mV',
        'theta real': 'Theta /mV (real)',
        'g': 'Gamma /mV',
        'amp/const': 'Amplitude/Const'
    }

    fig = plotter.figure(xlabel=xlabel, ylabel=ylabels[param],
                         title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: {titles[param]}{title_append}')
    return fig


def common_dat_hover_infos(datnum=True,
                           heater_bias=False,
                           fit_entropy_name: Optional[str] = None,
                           fit_entropy=False,
                           int_info_name: Optional[str] = None,
                           output_name: Optional[str] = None,
                           integrated_entropy=False,
                           sub_lin: bool = False,
                           sub_lin_width: Optional[Union[float, Callable]] = None,
                           int_info=False,
                           amplitude=False,
                           theta=False,
                           gamma=False,
                           ) -> List[HoverInfo]:
    """
    Returns a list of HoverInfos for the specified parameters. To do more complex things, append specific
    HoverInfos before/after this.

    Examples:
        hover_infos = common_dat_hover_infos(datnum=True, amplitude=True, theta=True)
        hover_group = HoverInfoGroup(hover_infos)

    Args:
        datnum ():
        heater_bias ():
        fit_entropy_name (): Name of saved fit_entropy if wanting fit_entropy
        fit_entropy ():
        int_info_name (): Name of int_info if wanting int_info or integrated_entropy
        output_name (): Name of SE output to integrate (defaults to int_info_name)
        integrated_entropy ():
        sub_lin (): Whether to subtract linear term from integrated_info first
        sub_lin_width (): Width of transition to avoid in determining linear terms
        int_info (): amp/dT/sf from int_info

    Returns:
        List[HoverInfo]:
    """

    hover_infos = []
    if datnum:
        hover_infos.append(HoverInfo(name='Dat', func=lambda dat: dat.datnum, precision='.d', units=''))
    if heater_bias:
        hover_infos.append(HoverInfo(name='Bias', func=lambda dat: dat.AWG.max(0) / 10, precision='.1f', units='nA'))
    if fit_entropy:
        hover_infos.append(HoverInfo(name='Fit Entropy',
                                     func=lambda dat: dat.Entropy.get_fit(name=fit_entropy_name,
                                                                          check_exists=True).best_values.dS,
                                     precision='.2f', units='kB'), )
    if integrated_entropy:
        if output_name is None:
            output_name = int_info_name
        if sub_lin:
            if sub_lin_width is None:
                raise ValueError(f'Must specify sub_lin_width if subtrating linear term from integrated entropy')
            elif not isinstance(sub_lin_width, Callable):
                sub_lin_width = lambda _: sub_lin_width  # make a value into a function so so that can assume function
            data = lambda dat: dat_integrated_sub_lin(dat, signal_width=sub_lin_width(dat), int_info_name=int_info_name,
                                                      output_name=output_name)
            hover_infos.append(HoverInfo(name='Sub lin width', func=sub_lin_width, precision='.1f', units='mV'))
        else:
            data = lambda dat: dat.Entropy.get_integrated_entropy(
                name=int_info_name,
                data=dat.SquareEntropy.get_Outputs(
                    name=output_name).average_entropy_signal)
        hover_infos.append(HoverInfo(name='Integrated Entropy',
                                     func=lambda dat: np.nanmean(data(dat)[-10:]),
                                     precision='.2f', units='kB'))

    if int_info:
        info = lambda dat: dat.Entropy.get_integration_info(name=int_info_name)
        hover_infos.append(HoverInfo(name='SF amp',
                                     func=lambda dat: info(dat).amp,
                                     precision='.3f',
                                     units='nA'))
        hover_infos.append(HoverInfo(name='SF dT',
                                     func=lambda dat: info(dat).dT,
                                     precision='.3f',
                                     units='mV'))
        hover_infos.append(HoverInfo(name='SF',
                                     func=lambda dat: info(dat).sf,
                                     precision='.3f',
                                     units=''))

    return hover_infos


def plot_transition_values(datnums: List[int], save_name: str, general: AnalysisGeneral, param_name: str = 'theta',
                           transition_only: bool = True, show=True):
    param = param_name
    if transition_only:
        all_dats = get_dats(datnums)
        fig = transition_fig(dats=all_dats, xlabel='ESC /mV', title_append=' vs ESC for Transition Only scans',
                             param=param)
        dats = get_dats(datnums)
        fig.add_trace(transition_trace(dats, x_func=general.x_func, from_square_entropy=False,
                                       fit_name=save_name, param=param, label='Data'))
        # print(
        #     f'Avg weakly coupled cold theta = '
        #     f'{np.mean([dat.Transition.get_fit(name=fit_name).best_values.theta for dat in dats if dat.Logs.fds["ESC"] <= -330])}')
    else:
        all_dats = get_dats(datnums)
        fig = transition_fig(dats=all_dats, xlabel='ESC /mV', title_append=' vs ESC for Entropy scans', param=param)
        dats = get_dats(datnums)
        fig.add_trace(transition_trace(dats, x_func=general.x_func, from_square_entropy=True,
                                       fit_name=save_name, param=param, label='Data'))
        # print(
        #     f'Avg weakly coupled cold theta = {np.mean([dat.SquareEntropy.get_fit(which_fit="transition", fit_name=fit_name).best_values.theta for dat in dats if dat.Logs.fds["ESC"] <= -330])}')
    if show:
        fig.show()
    return fig