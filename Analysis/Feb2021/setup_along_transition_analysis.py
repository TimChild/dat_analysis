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
"""
# External general imports
from __future__ import annotations
from typing import List, Union, Optional, Dict, Tuple, Callable, TYPE_CHECKING
from functools import partial
from progressbar import progressbar
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np
import lmfit as lm
import pandas as pd
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# My general useful imports
import Analysis.Feb2021.common as common
import Analysis.Feb2021.common_plotting as cp
import src.AnalysisTools.entropy
import src.UsefulFunctions as U
from src.UsefulFunctions import NotFoundInHdfError
from src.DatObject.Make_Dat import get_dat, get_dats, DatHDF
from src.AnalysisTools.general_fitting import FitInfo, calculate_fit
from src.Dash.DatPlotting import OneD, TwoD

# Imports Specifically useful in this module
from Analysis.Feb2021.common import linear_fit_thetas
from src.AnalysisTools.transition import do_transition_only_calc
from src.AnalysisTools.entropy import do_entropy_calc, calculate_new_sf_only
from Analysis.Feb2021.dcbias import dcbias_multi_dat
from Analysis.Feb2021.entropy_gamma_final import make_vs_gamma_analysis_params, run_processing, AnalysisGeneral

# For type checking only
if TYPE_CHECKING:
    pass

# General setup stuff
pio.renderers.default = 'browser'
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

    def copy_cold_part_of_transition(datnum) -> int:
        dat = get_dat(datnum)
        sp_start, _ = src.AnalysisTools.entropy.get_setpoint_indexes(dat, start_time=setpoint_start, end_time=None)
        pp = dat.SquareEntropy.get_ProcessParams(setpoint_start=sp_start)
        out = dat.SquareEntropy.get_row_only_output(name='default', process_params=pp, calculate_only=True)

        # This overwrites the data in dat.Transition
        dat.Transition.data = dat.SquareEntropy.get_transition_part(which='row', row=None,  # Get all rows
                                                                    part='cold', data=out.cycled)
        dat.Transition.x = U.get_matching_x(dat.Transition.x, dat.Transition.data)
        return dat.datnum  # Just a simple return to make it easier to see which ran successfully

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
                done = list(pool.map(copy_cold_part_of_transition, entropy_datnums))
        else:
            done = [copy_cold_part_of_transition(datnum) for datnum in progressbar(entropy_datnums)]


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
    # Ensure all dats have transition data in dat.Transition.data
    _setup_dats(datnums, setpoint_start=setpoint_start)

    weak_fits = _get_weak_fits(datnums, overwrite=overwrite)

    dats = get_dats(datnums)
    dat = dats[0]
    fig = cp.transition_fig(dats, xlabel='ESC /mV', title_append=' I_sense only', param='theta real')
    fig.add_trace(cp.transition_trace(dats, x_func=lambda dat: dat.Logs.fds['ESC'],
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


if __name__ == '__main__':
    # ALL_DATNUMS = list(range(7995, 8038 + 1))  # 200mK
    # ALL_DATNUMS = list(range(8039, 8089 + 1))  # 100mK
    # ALL_DATNUMS = list(range(8090, 8126 + 1))  # 50mK
    ALL_DATNUMS = list(range(8134, 8156 + 1))   # 300mK
    dats = get_dats(ALL_DATNUMS)

    do_weak_fits = False
    do_linear_theta = False
    do_dcbias_stuff = False
    do_processing = True
    plot_transition_values = True
    plot_integrated_entropy = True

    # Look at what data is weakly coupled
    if do_weak_fits:
        fig = complete_fit_assuming_weakly_coupled(ALL_DATNUMS)
        fig.show()

    if do_linear_theta:
        # Fit good weakly coupled data to get linear change in theta
        # 300mK
        filter = lambda dat: True if 8135 <= dat.datnum <= 8142 else False

        # 200mK
        # filter = lambda dat: True if dat.Logs.fds['ESC'] < -184 and dat.Transition.get_fit(
        #                                                                          name=WEAK_ONLY_NAME).best_values.theta > 70 else False

        # 100mK
        # filter = lambda dat: True if dat.Logs.fds['ESC'] < -205 and dat.datnum not in [8079] and dat.Transition.get_fit(
        #                                                                          name=WEAK_ONLY_NAME).best_values.theta > 30 else False

        # 50mK
        # filter = lambda dat: True if dat.Logs.fds['ESC'] < -200 and dat.datnum not in [8202, 8110] else False

        linear_theta_fit = linear_fit_thetas(dats, fit_name=WEAK_ONLY_NAME,
                                             filter_func=filter,
                                             show_plots=True)
        linear_theta_slope = linear_theta_fit.best_values.slope
        linear_theta_intercept = linear_theta_fit.best_values.intercept
        print(f'linear_theta_slope = {linear_theta_slope}\n'
              f'linear_theta_intercept = {linear_theta_intercept}')
    else:
        # 300mK
        linear_theta_slope = 7.317751559157747e-05
        linear_theta_intercept = 0.1377091260077624

        # 200mK
        # linear_theta_slope = 0.00011614418401069156
        # linear_theta_intercept = 0.10797192554472015

        # 100mK
        # linear_theta_slope = 3.053322566667311e-05
        # linear_theta_intercept = 0.04584982272383697

        # 50mK
        # linear_theta_slope = 4.730226501957049e-05
        # linear_theta_intercept = 0.03224312969412328


    # Get dT from DCbias (for now getting a rough idea from looking at plots)
    if do_dcbias_stuff:
        dat = dats[0]
        fig = dc_bias_plot(dat.Logs.temps.mc * 1000)
        [p1d.add_line(fig, value=sign * dat.AWG.max(num=0) / 10, mode='vertical', color='black', linetype='dash') for
         sign
         in [1, -1]]
        fig.show()
    # All must be in REAL mV
    # For 300mK
    dT = 0.040
    esc_of_dcbias = -270

    # For 200mK
    # dT = 0.0236
    # esc_of_dcbias = -270

    # For 100mK
    # dT = 0.013
    # esc_of_dcbias = -270

    # For 50mK
    # dT = 0.00766
    # esc_of_dcbias = -270

    # Fit with digamma/NRG with theta forced to be linear
    broadened_linear_theta_name = 'forced_theta_linear'
    general = _get_general_analysis_params_for_linear_theta(ALL_DATNUMS,
                                                            save_name=broadened_linear_theta_name,
                                                            linear_theta_slope=linear_theta_slope,
                                                            linear_theta_intercept=linear_theta_intercept,
                                                            base_dt=dT,
                                                            esc_of_dt=esc_of_dcbias,
                                                            overwrite=False,
                                                            )
    if do_processing:
        do_gamma_analysis_fitting(general)

    if plot_transition_values:
        figs = []
        figs.append(cp.plot_transition_values(general.transition_datnums, save_name=broadened_linear_theta_name,
                                              general=general, param_name='theta',
                                              transition_only=True, show=False))
        figs.append(cp.plot_transition_values(general.transition_datnums, save_name=broadened_linear_theta_name,
                                              general=general, param_name='g',
                                              transition_only=True, show=False))
        figs.append(cp.plot_transition_values(general.transition_datnums, save_name=broadened_linear_theta_name,
                                              general=general, param_name='amp',
                                              transition_only=True, show=False))

        for fig in figs:
            fig.show()

    if plot_integrated_entropy:
        fig = cp.get_integrated_fig(get_dats(general.entropy_datnums), title_append=f'comparing scaling factors')
        dat_chunks = [get_dats(ALL_DATNUMS)]
        for dats in dat_chunks:
            if len(dats) > 0:
                fig.add_trace(cp.get_integrated_trace(dats=dats,
                                                      x_func=general.x_func, x_label=general.x_label,
                                                      trace_name=f'Dats{min([dat.datnum for dat in dats])}-'
                                                                 f'{max([dat.datnum for dat in dats])}',
                                                      save_name=broadened_linear_theta_name,
                                                      int_info_name=broadened_linear_theta_name,
                                                      SE_output_name=broadened_linear_theta_name))
        fig.show()
