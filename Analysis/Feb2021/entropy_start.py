import src.UsefulFunctions as U
from src.UsefulFunctions import run_multiprocessed
from src.DatObject.Make_Dat import get_dat, get_dats, DatHDF
from src.DatObject.Attributes.SquareEntropy import square_wave_time_array
from src.DatObject.Attributes.Transition import i_sense, i_sense_digamma

from src.Dash.DatPlotting import OneD, TwoD

import logging
import numpy as np
import pandas as pd
from typing import List
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


def entropy_vs_time_trace(dats: List[DatHDF], trace_name=None):
    fit_name = 'SPS.0045'
    plotter = OneD(dats=dats)
    entropies = [dat.Entropy.get_fit(which='avg', name=fit_name).best_values.dS for dat in dats]
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


if __name__ == '__main__':
    dat = get_dat(1103)
    do_calc(1103)
    datnums = list(range(1097, 1139 + 1))
    datnums.remove(1118)

    # for num in progressbar(datnums):
    #     print(f'Dat{num}')
    #     do_calc(num)

    dats = get_dats(datnums)

    fig = entropy_vs_time_fig()

    for dn_start, pos_num in zip(range(1097, 1113, 3), range(1, 5)):
        datnums = [dn_start, dn_start + 1, dn_start + 2]
        datnums.extend([d + 3 * 6 for d in datnums])
        print(datnums)
        if 1118 in datnums:
            datnums.remove(1118)
        dats = get_dats(datnums)
        fig.add_trace(entropy_vs_time_trace(dats, trace_name=f'Position: {pos_num}'))

    fig.show(renderer='browser')
