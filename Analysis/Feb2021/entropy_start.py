import src.UsefulFunctions as U
from src.UsefulFunctions import run_multiprocessed
from src.DatObject.Make_Dat import get_dat, get_dats, DatHDF
from src.DatObject.Attributes.SquareEntropy import square_wave_time_array
from src.DatObject.Attributes.Transition import i_sense, i_sense_digamma
from src.Plotting.Plotly.PlotlyUtil import additional_data_dict_converter, HoverInfo, add_horizontal
from src.Dash.DatPlotting import OneD, TwoD

import logging
import numpy as np
import pandas as pd
from typing import List, Optional
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
    title = title_pre+f'Entropy vs {x_gate}'
    fig = plotter.figure(xlabel=f'{x_gate}/mV', ylabel="Entropy /kB", title=title)
    return fig


def entropy_vs_gate_trace(dats: List[DatHDF], x_gate, y_gate=None):
    fit_name = "SPS.0045"
    plotter = OneD(dats=dats)
    entropy = [dat.Entropy.get_fit(which='avg', name=fit_name).best_values.dS for dat in dats]
    entropy_errs = [np.nanstd([f.best_values.dS if f.best_values.dS is not None else np.nan
                               for f in dat.Entropy.get_row_fits(name=fit_name)]) for dat in dats]

    x = [dat.Logs.fds[x_gate] for dat in dats]
    trace = plotter.trace(data=entropy, data_err=entropy_errs, x=x, mode='markers+lines', name=f'Dats{dats[0].datnum}->{dats[-1].datnum}')

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
    datnums = set(range(1312, 1451+1)) - set(range(1312, 1451+1, 4))
    # datnums = set()
    # for j in range(5):
    #     datnums = datnums.union(set(range(28 * j + 1312, 28 * j + 1312 + 4 * 7 + 1)) - set([28 * j + 1312 + 4 * i for i in range(8)]))
    # datnums = list(datnums)

    dats = get_dats(datnums)

    dats = [d for d in dats if np.isclose(d.Logs.temps.mc, dat.Logs.temps.mc, rtol=0.1)]  # Get all dats where MC temp is within 10%
    bias_lookup = np.array([d.Logs.fds['HO1/10M'] for d in dats])

    indp = np.argmin(abs(bias_lookup - ho1))
    indm = np.argmin(abs(bias_lookup + ho1))
    theta_z = np.nanmean([d.Transition.avg_fit.best_values.theta for d in dats if d.Logs.fds['HO1/10M'] == 0])

    # temp_lookup = np.array([d.Logs.temps.mc for d in dats])
    # bias_lookup = np.array([d.Logs.fds['HO1/10M'] for d in dats])
    #
    # indp = np.argmin(temp_lookup - t + bias_lookup - ho1)
    # indm = np.argmin(temp_lookup - t + bias_lookup + ho1)
    # indz = np.argmin(temp_lookup - t + bias_lookup)

    theta_p = dats[indp].Transition.avg_fit.best_values.theta
    theta_m = dats[indm].Transition.avg_fit.best_values.theta
    # theta_z = dats[indz].Transition.avg_fit.best_values.theta
    return (theta_p + theta_m) / 2 - theta_z


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

    do_calc(1416)
    print(get_deltaT(get_dat(1416)))

    datnums = list(range(1312, 1449, 4))
    # for num in progressbar(datnums):
    #     print(num)
    #     do_calc(num)
    dats = get_dats(datnums)
    fig = entropy_vs_time_fig()
    fig.add_trace(entropy_vs_time_trace(dats))
    fig.add_trace(entropy_vs_time_trace(dats, trace_name='Integrated', integrated=True))
    fig.show(renderer='browser')
    # for dat in dats:
    #     print(f'Dat{dat.datnum}: {get_deltaT(dat)}')
    #     dat.Entropy.set_integration_info(dT=get_deltaT(dat),
    #                                      amp=dat.SquareEntropy.get_fit(which='avg', which_fit='transition',
    #                                                                    transition_part='cold', check_exists=False).best_values.amp)