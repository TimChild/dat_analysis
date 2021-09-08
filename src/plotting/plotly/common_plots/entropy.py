from typing import List, Callable, Optional, Union

import numpy as np
from plotly import graph_objs as go

from src.analysis_tools.entropy import dat_integrated_sub_lin
from Analysis.Feb2021.common_plotting import common_dat_hover_infos
from src import useful_functions as U
from src.dat_object.dat_hdf import DatHDF
from src.plotting.plotly import OneD
from src.plotting.plotly.hover_info import HoverInfo, _additional_data_dict_converter, HoverInfoGroup


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