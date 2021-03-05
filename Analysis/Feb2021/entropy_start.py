import src.UsefulFunctions as U
from src.UsefulFunctions import run_multiprocessed
from src.DatObject.Make_Dat import get_dat, get_dats, DatHDF
from src.DatObject.Attributes.SquareEntropy import square_wave_time_array
from src.DatObject.Attributes.Transition import i_sense, i_sense_digamma
from src.Plotting.Plotly.PlotlyUtil import additional_data_dict_converter, HoverInfo, add_horizontal
from src.Dash.DatPlotting import OneD, TwoD
from src.HDF_Util import NotFoundInHdfError

import logging
import lmfit as lm
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import List, Optional, Callable
from progressbar import progressbar
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)

pool = ProcessPoolExecutor()


def do_calc(datnum):
    """Just a function which can be passed to a process pool for faster calculation"""
    save_name = 'SPS.0045'

    dat = get_dat(datnum)

    setpoints = [0.0045, None]

    # Get other inputs
    setpoint_times = square_wave_time_array(dat.SquareEntropy.square_awg)
    sp_start, sp_fin = [U.get_data_index(setpoint_times, sp) for sp in setpoints]
    logger.debug(f'Setpoint times: {setpoints}, Setpoint indexs: {sp_start, sp_fin}')

    # Run Fits
    pp = dat.SquareEntropy.get_ProcessParams(name=None,  # Start from default and modify from there
                                             setpoint_start=sp_start, setpoint_fin=sp_fin,
                                             transition_fit_func=i_sense,
                                             save_name=save_name)
    out = dat.SquareEntropy.get_Outputs(name=save_name, inputs=None, process_params=pp, overwrite=False)
    dat.Entropy.get_fit(which='avg', name=save_name, data=out.average_entropy_signal, x=out.x, check_exists=False)
    [dat.Entropy.get_fit(which='row', row=i, name=save_name,
                         data=row, x=out.x, check_exists=False) for i, row in enumerate(out.entropy_signal)]


def entropy_vs_time_trace(dats: List[DatHDF], trace_name=None, integrated=False):
    fit_name = 'SPS.0045'
    plotter = OneD(dats=dats)
    if integrated is False:
        entropies = [dat.Entropy.get_fit(which='avg', name=fit_name).best_values.dS for dat in dats]
    else:
        entropies = [dat.Entropy.integrated_entropy[-1] for dat in dats]
    # entropies_err = [np.nanstd(
    #     [dat.Entropy.get_fit(which='row', row=i, name=fit_name).best_values.dS for i in range(len(dat.Data.y_array))]
    # ) for dat in dats]

    times = [str(dat.Logs.time_completed) for dat in dats]

    trace = plotter.trace(data=entropies, x=times, text=[dat.datnum for dat in dats], mode='lines', name=trace_name)
    return trace


def entropy_vs_time_fig():
    plotter = OneD()
    fig = plotter.figure(xlabel='Time', ylabel='Entropy /kB', title=f'Entropy vs Time')
    fig.update_xaxes(tickformat="%H:%M\n%a")
    return fig


def entropy_vs_gate_fig(dats: Optional[List[DatHDF]] = None, x_gate: str = None):
    plotter = OneD(dats=dats)
    if dats is not None:
        title_pre = f'Dats{dats[0].datnum}->{dats[-1].datnum}: '
    else:
        title_pre = ''
    title = title_pre + f'Entropy vs {x_gate}'
    fig = plotter.figure(xlabel=f'{x_gate}/mV', ylabel="Entropy /kB", title=title)
    return fig


def entropy_vs_gate_trace(dats: List[DatHDF], x_gate, y_gate=None):
    fit_name = "SPS.0045"
    plotter = OneD(dats=dats)
    entropy = [dat.Entropy.get_fit(which='avg', name=fit_name).best_values.dS for dat in dats]
    entropy_errs = [np.nanstd([f.best_values.dS if f.best_values.dS is not None else np.nan
                               for f in dat.Entropy.get_row_fits(name=fit_name)]) for dat in dats]

    x = [dat.Logs.fds[x_gate] for dat in dats]
    trace = plotter.trace(data=entropy, data_err=entropy_errs, x=x, mode='markers+lines',
                          name=f'Dats{dats[0].datnum}->{dats[-1].datnum}')

    hover_infos = [
        HoverInfo(name='Dat', func=lambda dat: dat.datnum, precision='.d', units=''),
        HoverInfo(name=x_gate, func=lambda dat: dat.Logs.fds[x_gate], precision='.1f', units='mV'),
        # HoverInfo(name='Time', func=lambda dat: dat.Logs.time_completed.strftime('%H:%M'), precision='', units=''),
    ]
    if y_gate:
        hover_infos.append(HoverInfo(name=y_gate, func=lambda dat: dat.Logs.fds[y_gate], precision='.2f', units='mV'))

    funcs, hover_template = additional_data_dict_converter(info=hover_infos)
    hover_data = [[f(dat) for f in funcs] for dat in dats]
    trace.update(hovertemplate=hover_template,
                 customdata=hover_data)
    return trace


def get_deltaT(dat):
    """Returns deltaT of a given dat in mV"""
    ho1 = dat.AWG.max(0)  # 'HO1/10M' gives nA * 10
    t = dat.Logs.temps.mc

    # Datnums to search through (only thing that should be changed)
    datnums = set(range(1312, 1451 + 1)) - set(range(1312, 1451 + 1, 4))

    dats = get_dats(datnums)

    dats = [d for d in dats if
            np.isclose(d.Logs.temps.mc, dat.Logs.temps.mc, rtol=0.1)]  # Get all dats where MC temp is within 10%
    bias_lookup = np.array([d.Logs.fds['HO1/10M'] for d in dats])

    indp = int(np.argmin(abs(bias_lookup - ho1)))
    indm = int(np.argmin(abs(bias_lookup + ho1)))
    theta_z = np.nanmean([d.Transition.avg_fit.best_values.theta for d in dats if d.Logs.fds['HO1/10M'] == 0])

    theta_p = dats[indp].Transition.avg_fit.best_values.theta
    theta_m = dats[indm].Transition.avg_fit.best_values.theta
    # theta_z = dats[indz].Transition.avg_fit.best_values.theta
    return (theta_p + theta_m) / 2 - theta_z


def get_SE_dt(dat: DatHDF) -> float:
    fit_func = i_sense
    fits = [dat.SquareEntropy.get_fit(which='avg', which_fit='transition', transition_part=part,
                                      fit_func=fit_func,
                                      data=dat.SquareEntropy.get_Outputs(name='SPS.0045', existing_only=True).averaged,
                                      check_exists=False)
            for part in ['cold', 'hot']]
    thetas = [fit.best_values.theta for fit in fits]
    return thetas[1] - thetas[0]


def compare_dTs(dat: DatHDF, verbose=True) -> float:
    dc_dt = get_deltaT(dat)
    se_dt = get_SE_dt(dat)
    percent_diff = (se_dt - dc_dt) / dc_dt * 100
    if verbose:
        print(f'Dat{dat.datnum}:\n'
              f'dT calculated from DC bias = {dc_dt:.2f}mV\n'
              f'dT calculated from Square Wave = {se_dt:.2f}mV\n'
              f'Difference = {percent_diff:.1f}%\n')
    return percent_diff


def plot_dT_comparison(dats: List[DatHDF], plot=True):
    plotter = OneD(dats=dats)
    fig = plotter.figure(xlabel='Fridge Temp /mK',
                         ylabel='% Difference between calculated dTs',
                         title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: Difference between DC bias calculated dT and '
                               f'Square Entropy Calculated dT')

    hover_infos = [
        HoverInfo(name='Dat', func=lambda dat: dat.datnum, precision='.d', units=''),
        HoverInfo(name='Temperature', func=lambda dat: dat.Logs.temps.mc * 1000, precision='.1f', units='mK'),
        HoverInfo(name='Bias', func=lambda dat: dat.AWG.max(0) / 10, precision='.1f', units='nA'),
    ]
    funcs, template = additional_data_dict_converter(hover_infos)

    for bias in sorted(list(set([dat.AWG.max(0) for dat in dats]))):
        ds = [dat for dat in dats if dat.AWG.max(0) == bias]
        diffs = [compare_dTs(dat, verbose=False) for dat in ds]
        hover_data = [[func(dat) for func in funcs] for dat in ds]
        fig.add_trace(plotter.trace(data=diffs, x=[dat.Logs.temps.mc * 1000 for dat in ds],
                                    name=f'Bias={bias / 10:.0f}nA',
                                    mode='markers+lines',
                                    trace_kwargs={'customdata': hover_data, 'hovertemplate': template}))
    if plot:
        fig.show(renderer='browser')
    return fig


def plot_entropy_vs_temp(dats: List[DatHDF], integrated=False, plot=True):
    fit_name = 'SPS.0045'
    plotter = OneD(dats=dats)
    _tname = 'Integrated' if integrated else 'Fit'
    fig = plotter.figure(
        title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: {_tname} Entropy',
        xlabel='Heating Bias /nA',
        ylabel='Entropy /kB')
    temps = list(range(0, 300, 10))

    hover_infos = [
        HoverInfo(name='Dat', func=lambda dat: dat.datnum, precision='.d', units=''),
        HoverInfo(name='Temperature', func=lambda dat: dat.Logs.temps.mc * 1000, precision='.1f', units='mK'),
        HoverInfo(name='Bias', func=lambda dat: dat.AWG.max(0) / 10, precision='.1f', units='nA'),
    ]
    funcs, template = additional_data_dict_converter(hover_infos)

    for temp in temps:
        ds = [dat for dat in dats if np.isclose(dat.Logs.temps.mc * 1000, temp, atol=5)]
        if len(ds) > 0:
            x = [dat.AWG.max(0) / 10 for dat in ds]
            hover_data = [[func(dat) for func in funcs] for dat in ds]

            if integrated is False:
                entropies = [dat.Entropy.get_fit(name=fit_name).best_values.dS for dat in ds]
                entropy_errs = [np.nanstd([
                    f.best_values.dS if f.best_values.dS is not None else np.nan
                    for f in dat.Entropy.get_row_fits(name=fit_name) for dat in ds
                ]) / np.sqrt(dat.Data.y_array.shape[0]) for dat in ds]
                fig.add_trace(plotter.trace(
                    data=entropies, data_err=entropy_errs, x=x, name=f'{temp:.0f}mK',
                    mode='markers+lines',
                    trace_kwargs={'customdata': hover_data, 'hovertemplate': template})
                )
            else:
                integrated_entropies = [np.nanmean(dat.Entropy.integrated_entropy[-10:]) for dat in ds]
                fig.add_trace(plotter.trace(
                    data=integrated_entropies, x=x, name=f'{temp:.0f}mK',
                    mode='markers+lines',
                    trace_kwargs={'customdata': hover_data, 'hovertemplate': template})
                )
    if plot:
        fig.show(renderer='browser')
    return fig


def plot_fit_integrated_comparison(dats: List[DatHDF], x_func: Callable, x_label: str, title_append: Optional[str] = '',
                                   int_info_name: Optional[str] = None, plot=True) -> go.Figure():
    fit_name = 'SPS.0045'
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
        HoverInfo(name='Fit Entropy', func=lambda dat: dat.Entropy.get_fit(name=fit_name).best_values.dS,
                  precision='.2f', units='kB'),
        HoverInfo(name='Integrated Entropy',
                  func=lambda dat: np.nanmean(dat.Entropy.get_integrated_entropy(name=int_info_name)[-10:]),
                  precision='.2f', units='kB'),
    ]

    funcs, template = additional_data_dict_converter(hover_infos)
    x = [x_func(dat) for dat in dats]
    hover_data = [[func(dat) for func in funcs] for dat in dats]

    entropies = [dat.Entropy.get_fit(name=fit_name).best_values.dS for dat in dats]
    entropy_errs = [np.nanstd([
        f.best_values.dS if f.best_values.dS is not None else np.nan
        for f in dat.Entropy.get_row_fits(name=fit_name) for dat in dats
    ]) / np.sqrt(dat.Data.y_array.shape[0]) for dat in dats]
    fig.add_trace(plotter.trace(
        data=entropies, data_err=entropy_errs, x=x, name=f'Fit Entropy',
        mode='markers+lines',
        trace_kwargs={'customdata': hover_data, 'hovertemplate': template})
    )
    integrated_entropies = [np.nanmean(dat.Entropy.get_integrated_entropy(name=int_info_name)[-10:]) for dat in dats]
    fig.add_trace(plotter.trace(
        data=integrated_entropies, x=x, name=f'Integrated',
        mode='markers+lines',
        trace_kwargs={'customdata': hover_data, 'hovertemplate': template})
    )

    if plot:
        fig.show(renderer='browser')
    return fig


if __name__ == '__main__':
    # datnums = list(range(1097, 1139 + 1))
    # datnums.remove(1118)
    #
    # for num in progressbar(datnums):
    #     print(f'Dat{num}')
    #     do_calc(num)
    #
    # dats = get_dats(datnums)
    #
    # fig = entropy_vs_time_fig()
    #
    # for dn_start, pos_num in zip(range(1097, 1113, 3), range(1, 5)):
    #     datnums = [dn_start, dn_start + 1, dn_start + 2]
    #     datnums.extend([d + 3 * 6 for d in datnums])
    #     print(datnums)
    #     if 1118 in datnums:
    #         datnums.remove(1118)
    #     dats = get_dats(datnums)
    #     fig.add_trace(entropy_vs_time_trace(dats, trace_name=f'Position: {pos_num}'))
    #
    # fig.show(renderer='browser')

    # datnums = list(range(1146, 1246))

    # datnums = list(range(1270, 1278+1))
    # dats = get_dats(datnums)
    # fig = entropy_vs_gate_fig(dats, x_gate="ESS")
    # fig.add_trace(entropy_vs_gate_trace(dats, x_gate='ESS', y_gate='ESP'))
    # add_horizontal(fig, np.log(2))
    # fig.show(renderer='browser')

    datnums = list(range(1312, 1449, 4))
    # for num in progressbar(datnums):
    #     print(num)
    #     do_calc(num)
    # dats = get_dats(datnums)
    # fig = entropy_vs_time_fig()
    # fig.add_trace(entropy_vs_time_trace(dats))
    # fig.add_trace(entropy_vs_time_trace(dats, trace_name='Integrated', integrated=True))
    # fig.show(renderer='browser')
    # for dat in dats:
    #     print(f'Dat{dat.datnum}: {get_deltaT(dat)}')
    #     dat.Entropy.set_integration_info(dT=get_deltaT(dat),
    #                                      amp=dat.SquareEntropy.get_fit(which='avg', which_fit='transition',
    #                                                                    transition_part='cold', check_exists=False).best_values.amp)

    # datnums = list(range(1312, 1449, 4))
    # dats = get_dats(datnums)
    #
    # # fit_fig = plot_entropy_vs_temp(dats, integrated=False, plot=False)
    # # int_fit = plot_entropy_vs_temp(dats, integrated=True, plot=False)
    #
    # temps = list(range(0, 300, 50))
    # figs = []
    # for temp in temps:
    #     ds = [dat for dat in dats if np.isclose(dat.Logs.temps.mc*1000, temp, atol=10)]
    #     if len(ds) > 0:
    #         figs.append(plot_fit_integrated_comparison(ds, x_func=lambda dat: dat.AWG.max(0)/10,
    #                                                    x_label='Heating Bias /nA',
    #                                                    title_append=f' at {temp}mK',
    #                                                    plot=True))


    # datnums = list(range(1497, 1520))
    # for num in range(1501, 1508+1):
    #     datnums.remove(num)
    # x_gate = 'ESS'
    # dT_dict = {
    #     100: 6.5,
    #     50: 2.85
    # }  # Heating mV: dT

    datnums = list(range(1530, 1553 + 1))
    x_gate = 'ESC'
    dT_dict = {
        100: 6.49,
        50: 2.8
    }  # Heating mV: dT
    lin_amp = lm.models.LinearModel()
    lin_pars = lin_amp.make_params(slope=-0.00512387, intercept=-0.3237704)

    overwrite = True
    x_gate_label = x_gate + '/mV'

    dats = get_dats(datnums)
    for dat in progressbar(dats):
        dat: DatHDF
        do_calc(dat.datnum)
        dT = dT_dict[dat.AWG.max(0)]
        if dat.Entropy.integration_info is None or overwrite is True:
            dat.Entropy.set_integration_info(dT=dT,
                                             amp=dat.SquareEntropy.get_fit(which_fit='transition').best_values.amp,
                                             overwrite=overwrite
                                             )
        try:
            dat.Entropy.set_integration_info(dT=dT, amp=float(lin_amp.eval(params=lin_pars, x=dat.Logs.fds[x_gate])),
                                             name='linear amp', overwrite=overwrite)
        except FileExistsError:
            pass


    # for dat in dats:
    #     print(f'Dat{dat.datnum}:\n'
    #           f'Bias = {dat.AWG.max(0) / 10}nA\n'
    #           f'{x_gate} = {dat.Logs.fds[x_gate]:.1f}mV\n'
    #           f'SE dT = {get_SE_dt(dat):.2f}mV\n'
    #           f'DC dT = {get_deltaT(dat):.2f}mV\n'
    #           )

    biases = set([dat.AWG.max(0) for dat in dats])
    figs = []
    for bias in biases:
        ds = [dat for dat in dats if dat.AWG.max(0) == bias]
        figs.append(plot_fit_integrated_comparison(ds, x_func=lambda dat: dat.Logs.fds[x_gate], x_label=x_gate_label,
                                                   title_append=f' with {ds[0].AWG.max(0) / 10}nA Heating Current',
                                                   int_info_name='linear amp'))

    for i, fig in enumerate(figs):
        fig.write_html(f'temp{i}.html')
