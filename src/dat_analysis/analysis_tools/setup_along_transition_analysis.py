"""
Collection of functions useful for setting up analysis of along transition scans.
General process is:
1. Figure out which dats are weakly coupled. i.e. Transition fits to transition_only OR cold part of entropy data for weakly coupled (i.e. gamma = 0)
2. Linear fit thetas in weakly coupled over a region which makes sense (i.e. excluding anomalous points)
3. Determine dT from DCbias scans OR from dT of Square Entropy directly
4. Fit transitions with digamma and/or NRG function forcing theta to follow linear fit
5. Fit entropy dN/dT
6. Set integration info:
    dT proportionally changed based on linear theta
    amp from transition only OR entropy transition


Sep 21 -- This is all particularly useful for scans into gamma broadened where the "along transition" covers a wide
range of gate space. Probably this is not as useful for future scans, but some of it might still be useful again.

If this file doesn't get used at all, it should be deleted. (8/9/21)
"""
# External general imports
from __future__ import annotations
from typing import List, TYPE_CHECKING
from functools import partial
from progressbar import progressbar
import plotly.graph_objs as go
import numpy as np
import lmfit as lm
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# My general useful imports
import Analysis.Feb2021.common as common
import Analysis.Feb2021.common_plotting as cp
from .. import useful_functions as U
from .square_wave import set_transition_data_to_cold_only
from ..hdf_util import NotFoundInHdfError
from ..dat_object.make_dat import get_dat, get_dats
from .general_fitting import FitInfo
from ..plotting.plotly.dat_plotting import OneD, TwoD

# Imports Specifically useful in this module
from .transition import do_transition_only_calc, linear_fit_thetas
from .entropy import calculate_new_sf_only
from ..plotting.plotly.common_plots.dcbias import dcbias_multi_dat
from Analysis.Feb2021.entropy_gamma_final import make_vs_gamma_analysis_params, run_processing, AnalysisGeneral

# For type checking only
if TYPE_CHECKING:
    pass

# General setup stuff
p1d = OneD(dat=None)
p2d = TwoD(dat=None)

USE_MULTIPROCESSING = True
WEAK_ONLY_NAME = 'i_sense_only'


# ########### Which fits are weakly coupled
def _setup_dats(datnums: List[int],
                setpoint_start: float):
    """
    Gets Square Processed data and saves in dat.Transition if it exists
    Args:
        datnums ():
        setpoint_start ():

    Returns:
    """

    dats = get_dats(datnums)

    entropy_datnums = []
    logging.info(f'Checking if any datnums are entropy dats')
    for dat in dats:
        try:
            _ = dat.Logs.awg
            entropy_datnums.append(dat.datnum)
        except NotFoundInHdfError:
            pass

    # For any entropy dats, copy the cold part of transition into dat.Transition.data (overwrites dat.Transition.data)
    if entropy_datnums:
        logging.info(f'Copying Cold part of Square Processed transition data into dat.Transition for entropy dats')
        if USE_MULTIPROCESSING:
            with ThreadPoolExecutor() as pool:
                done = list(pool.map(set_transition_data_to_cold_only, entropy_datnums))
        else:
            done = [set_transition_data_to_cold_only(datnum, setpoint_start) for datnum in progressbar(entropy_datnums)]


def _get_weak_fits(datnums: List[int], overwrite: bool) -> List[FitInfo]:
    """
    Fits dat.Transition.data with weak fit centering all dats
    Args:
        datnums ():
        overwrite ():

    Returns:

    """
    pool_func = partial(do_transition_only_calc, save_name=WEAK_ONLY_NAME,
                        theta=None, gamma=0,
                        width=None,
                        t_func_name='i_sense',
                        center_func='i_sense',
                        csq_mapped=False,
                        centering_threshold=1000,  # Center all because assuming they may be weakly coupled
                        overwrite=overwrite
                        )
    if USE_MULTIPROCESSING:
        with ProcessPoolExecutor() as pool:
            fits = list(pool.map(pool_func,
                                 datnums))
    else:
        fits = []
        for datnum in progressbar(datnums):
            fits.append(pool_func(datnum))
    return fits


def complete_fit_assuming_weakly_coupled(datnums: List[int],
                                         setpoint_start: float = 0.005,
                                         overwrite: bool = False,
                                         add_temp_to_title=True,
                                         ):
    """
    Weak coupling fit to each dat

    Args:
        datnums ():
        setpoint_start ():

    Returns:

    """
    from ..plotting.plotly.common_plots.transition import transition_fig, transition_trace
    # Ensure all dats have transition data in dat.Transition.data
    _setup_dats(datnums, setpoint_start=setpoint_start)

    weak_fits = _get_weak_fits(datnums, overwrite=overwrite)

    dats = get_dats(datnums)
    dat = dats[0]
    fig = transition_fig(dats, xlabel='ESC /mV', title_append=' I_sense only', param='theta real')
    fig.add_trace(
        transition_trace(dats, x_func=lambda dat: dat.Logs.fds['ESC'],
                                                                     from_square_entropy=False, fit_name=WEAK_ONLY_NAME, param='theta real'))
    if add_temp_to_title:
        fig.update_layout(title=dict(text=fig.layout.title.text + f' at {dat.Logs.temps.mc * 1000:.1f}mK'))
    return fig


def _get_general_analysis_params_for_linear_theta(datnums,
                                                  save_name: str,
                                                  linear_theta_slope: float,
                                                  linear_theta_intercept: float,
                                                  base_dt: float,
                                                  esc_of_dt: float,
                                                  overwrite: bool = False) -> AnalysisGeneral:
    entropy_datnums = []
    transition_datnums = []
    for datnum in datnums:
        dat = get_dat(datnum)
        try:
            _ = dat.Logs.awg
            entropy_datnums.append(datnum)
        except NotFoundInHdfError:
            transition_datnums.append(datnum)

    if len(entropy_datnums) == 0:
        raise RuntimeError(f'This only works with entropy dats')
    elif 0 < len(transition_datnums) != len(entropy_datnums):
        raise RuntimeError(f'Different number of transition datnums/entropy datnums:'
                           f' ({len(transition_datnums), len(entropy_datnums)})')
    elif len(transition_datnums) == 0:
        transition_datnums = entropy_datnums

    all_params = make_vs_gamma_analysis_params(entropy_datnums, transition_datnums, save_name=save_name,
                                               force_theta=-1, force_gamma=None,
                                               sf_from_square_transition=True, width=None)
    line = lm.models.LinearModel()
    line_pars = line.make_params()
    line_pars['slope'].value = linear_theta_slope * 1000
    line_pars['intercept'].value = linear_theta_intercept * 1000
    theta_for_dt = line.eval(x=esc_of_dt, params=line_pars)  # dT is calculated at -270mV ESC
    base_dt = base_dt * 1000

    for par in all_params:
        dat = get_dat(par.transition_only_datnum)
        theta = line.eval(params=line_pars, x=dat.Logs.fds['ESC'])
        par.force_theta = theta
        par.force_dt = base_dt * theta / theta_for_dt  # Scale dT with same proportion as theta

    general = AnalysisGeneral(params_list=all_params, calculate=True,
                              overwrite_entropy=overwrite, overwrite_transition=overwrite)
    return general


def _set_scaling_factors(general_analysis_params: AnalysisGeneral):
    all_params = general_analysis_params.params_list
    for par in all_params:
        calculate_new_sf_only(entropy_datnum=par.entropy_datnum, save_name=par.save_name,
                              dt=par.force_dt, amp=par.force_amp,
                              from_square_transition=par.sf_from_square_transition,
                              transition_datnum=par.transition_only_datnum,
                              fit_name=par.save_name)


def do_gamma_analysis_fitting(general_analysis_params: AnalysisGeneral):
    """Fit dats with digamma function using linear theta"""
    run_processing(general_analysis_params, multiprocessed=USE_MULTIPROCESSING)
    _set_scaling_factors(general_analysis_params)


# ############ DC bias stuff
def dc_bias_plot(fridge_temp: float) -> go.Figure:
    all_dc_dats = get_dats((7437, 7844 + 1))
    dc_dats_by_temp = common.sort_by_temps(all_dc_dats)

    temps = np.array(list(dc_dats_by_temp.keys()))
    closest_temp = temps[np.argmin(np.abs(temps - fridge_temp))]
    dc_dats = dc_dats_by_temp[closest_temp]

    dc_dats = U.order_list(dc_dats, [dat.Logs.fds['HO1/10M'] for dat in dc_dats])
    fig = dcbias_multi_dat(dc_dats)
    fig.update_layout(
        title=f'Dats{min([dat.datnum for dat in dc_dats])}-{max([dat.datnum for dat in dc_dats])}: DC bias '
              f'at {np.nanmean([dat.Logs.temps.mc * 1000 for dat in dc_dats]):.0f}mK')
    return fig

