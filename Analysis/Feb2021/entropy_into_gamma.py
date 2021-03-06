import src.UsefulFunctions as U
from src.UsefulFunctions import run_multiprocessed
from src.DatObject.Make_Dat import get_dat, get_dats, DatHDF
from src.DatObject.Attributes.SquareEntropy import square_wave_time_array
from src.DatObject.Attributes.Transition import i_sense, i_sense_digamma
from src.Plotting.Plotly.PlotlyUtil import additional_data_dict_converter, HoverInfo, add_horizontal
from src.Dash.DatPlotting import OneD, TwoD
from src.HDF_Util import NotFoundInHdfError
from Analysis.Feb2021.common import get_deltaT

import logging
import lmfit as lm
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import List, Optional, Callable
from progressbar import progressbar
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

logger = logging.getLogger(__name__)

pool = ProcessPoolExecutor()
thread_pool = ThreadPoolExecutor()


def do_calc(datnum):
    """Just a function which can be passed to a process pool for faster calculation"""
    save_name = 'SPS.002'

    dat = get_dat(datnum)

    setpoints = [0.002, None]

    # Get other inputs
    setpoint_times = square_wave_time_array(dat.SquareEntropy.square_awg)
    sp_start, sp_fin = [U.get_data_index(setpoint_times, sp) for sp in setpoints]
    logger.debug(f'Setpoint times: {setpoints}, Setpoint indexs: {sp_start, sp_fin}')

    # Run Fits
    pp = dat.SquareEntropy.get_ProcessParams(name=None,  # Load default and modify from there
                                             setpoint_start=sp_start, setpoint_fin=sp_fin,
                                             transition_fit_func=i_sense,
                                             save_name=save_name)
    out = dat.SquareEntropy.get_Outputs(name=save_name, inputs=None, process_params=pp, overwrite=False)
    dat.Entropy.get_fit(which='avg', name=save_name, data=out.average_entropy_signal, x=out.x, check_exists=False)
    [dat.Entropy.get_fit(which='row', row=i, name=save_name,
                         data=row, x=out.x, check_exists=False) for i, row in enumerate(out.entropy_signal)]


def do_amp_calc(datnum: int, weak_cutoff: float = -260, couple_gate: str = 'ESC', overwrite=False):
    # TODO: Do more careful row fits and use those to center (probably makes no difference)
    dat = get_dat(datnum)
    params = dat.Transition.get_default_params()
    if dat.Logs.fds[couple_gate] <= weak_cutoff:
        fit_func = i_sense
    else:
        fit_func = i_sense_digamma
        params.add('g', 0, min=-50, max=1000, vary=True)

    fit = dat.Transition.get_fit(which='avg', name='careful',
                                 fit_func=fit_func, initial_params=params,
                                 check_exists=False, overwrite=overwrite)
    return fit


def get_amplitude(dat: DatHDF, transition_fit_name: str = 'default', gate: str = 'ESC'):
    """Returns amplitude of a given dat in mV based on a fit through amplitude determined from long transition
    specific scans """

    def get_fit(d: DatHDF):
        fit = d.Transition.get_fit(which='avg', name=transition_fit_name, check_exists=True)
        return fit

    # Datnums to search through (only thing that should be changed)
    datnums = list(range(1604, 1635, 2))
    dats = get_dats(datnums)

    fits = list(thread_pool.map(get_fit, dats))
    amps = [fit.best_values.amp for fit in fits]
    x = [d.Logs.fds[gate] for d in dats]

    quad = lm.models.QuadraticModel()
    fit = quad.fit(x=x, data=amps)

    return fit.eval(x=dat.Logs.fds[gate])


TRANSITION_DATNUMS = list(range(1604, 1635, 2))

DATNUMS1 = list(range(1637, 1652 + 1))  # First set all the way from weakly coupled to gamma broadened

if __name__ == '__main__':
    dats = get_dats(DATNUMS1)
    tdats = get_dats(TRANSITION_DATNUMS)

    fits = list(pool.map(do_amp_calc, [d.datnum for d in tdats]))
    # fit = do_amp_calc(tdats[4].datnum)
