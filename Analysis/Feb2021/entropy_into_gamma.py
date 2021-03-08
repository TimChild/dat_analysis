import src.UsefulFunctions as U
from src.UsefulFunctions import run_multiprocessed
from src.DatObject.Make_Dat import get_dat, get_dats, DatHDF
from src.DatObject.Attributes.SquareEntropy import square_wave_time_array
from src.DatObject.Attributes.Transition import i_sense, i_sense_digamma
from src.Plotting.Plotly.PlotlyUtil import additional_data_dict_converter, HoverInfo, add_horizontal
from src.Dash.DatPlotting import OneD, TwoD
from src.HDF_Util import NotFoundInHdfError
from Analysis.Feb2021.common import get_deltaT, plot_fit_integrated_comparison, entropy_vs_time_trace, \
    entropy_vs_time_fig, do_narrow_fits

import logging
import lmfit as lm
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import pandas as pd
from typing import List, Optional, Callable
from progressbar import progressbar
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

pio.renderers.default = 'browser'
logger = logging.getLogger(__name__)

pool = ProcessPoolExecutor()
thread_pool = ThreadPoolExecutor()


def do_calc(datnum, overwrite=True):
    """Just a function which can be passed to a process pool for faster calculation"""
    save_name = 'SPS.005'

    dat = get_dat(datnum)

    setpoints = [0.005, None]

    # Get other inputs
    setpoint_times = square_wave_time_array(dat.SquareEntropy.square_awg)
    sp_start, sp_fin = [U.get_data_index(setpoint_times, sp) for sp in setpoints]
    logger.debug(f'Setpoint times: {setpoints}, Setpoint indexs: {sp_start, sp_fin}')

    # Run Fits
    pp = dat.SquareEntropy.get_ProcessParams(name=None,  # Load default and modify from there
                                             setpoint_start=sp_start, setpoint_fin=sp_fin,
                                             transition_fit_func=i_sense,
                                             save_name=save_name)
    out = dat.SquareEntropy.get_Outputs(name=save_name, inputs=None, process_params=pp, overwrite=overwrite)
    dat.Entropy.get_fit(which='avg', name=save_name, data=out.average_entropy_signal, x=out.x, check_exists=False,
                        overwrite=overwrite)
    [dat.Entropy.get_fit(which='row', row=i, name=save_name,
                         data=row, x=out.x, check_exists=False,
                         overwrite=overwrite) for i, row in enumerate(out.entropy_signal)]

    if 'first' not in dat.Entropy.get_integration_info_names() or overwrite:
        dat.Entropy.set_integration_info(dT=get_deltaT(dat),
                                         amp=get_amplitude(dat, transition_fit_name=None, gate='ESC'),
                                         name='first', overwrite=overwrite)
    return True


def do_amp_calc(datnum: int, weak_cutoff: float = -260, couple_gate: str = 'ESC', overwrite=False):
    # TODO: Do more careful row fits and use those to center (probably makes no difference)
    dat = get_dat(datnum)
    params = dat.Transition.get_default_params(x=dat.Transition.avg_x, data=dat.Transition.avg_data)
    if dat.Logs.fds[couple_gate] <= weak_cutoff:
        fit_func = i_sense
    else:
        fit_func = i_sense_digamma
        params.add('g', 0, min=-50, max=1000, vary=True)
        params = U.edit_params(params, 'theta', 4.02, False)

    fit = dat.Transition.get_fit(which='avg', name='careful',
                                 fit_func=fit_func, initial_params=params,
                                 check_exists=False, overwrite=overwrite)
    return fit


def get_amplitude(dat: DatHDF, transition_fit_name: str = 'default', gate: str = 'ESC'):
    """Returns amplitude of a given dat in mV based on a fit through amplitude determined from long transition
    specific scans """

    def get_amp(d: DatHDF):
        amp = d.SquareEntropy.get_fit(which_fit='transition', name=transition_fit_name, check_exists=True).best_values.amp
        return amp

    if transition_fit_name is None:
        logger.info(f'Dat{dat.datnum}: Using Quad to get amplitude estimate')
        quad = lm.models.QuadraticModel()
        # pars = quad.make_params(a=-5.33813705e-05, b=-3.22415423e-02, c=-3.76905669)
        # pars = quad.make_params(a=-2.85721853e-05, b=-1.80534391e-02, c=-1.75449395)
        pars = quad.make_params(a=-8.57934297e-06, b=-8.99777901e-03, c=-7.54523798e-01)
        amp = quad.eval(params=pars, x=dat.Logs.fds[gate])
    else:
        amp = get_amp(dat)
    return amp


def do_amplitude_only_stuff():
    tdats = get_dats(TRANSITION_DATNUMS)

    fits = list(pool.map(partial(do_amp_calc, overwrite=True), [d.datnum for d in tdats]))

    weak_dats = [dat for dat in tdats if dat.Logs.fds['ESC'] <= -260]
    for dat in weak_dats:
        print(f'Dat{dat.datnum}: theta = {dat.Transition.get_fit(name="careful").best_values.theta:.2f}mV')

    print(
        f'Average Theta: {np.nanmean([dat.Transition.get_fit(name="careful").best_values.theta for dat in weak_dats]):.2f}mV')


def get_integrated_trace(dats: List[DatHDF], x_func: Callable,
                         trace_name: str,
                         int_info_name: Optional[str] = None, SE_output_name: Optional[str] = None,
                         ) -> go.Scatter:
    if int_info_name is None:
        int_info_name = 'first'
    if SE_output_name is None:
        SE_output_name = 'SPS.005'
    fit_name = 'SPS.005'

    plotter = OneD(dats=dats)

    hover_infos = [
        HoverInfo(name='Dat', func=lambda dat: dat.datnum, precision='.d', units=''),
        HoverInfo(name='ESC', func=lambda dat: x_func(dat), precision='.1f', units='mV'),
        HoverInfo(name='Bias', func=lambda dat: dat.AWG.max(0) / 10, precision='.1f', units='nA'),
        HoverInfo(name='Fit Entropy', func=lambda dat: dat.Entropy.get_fit(name=fit_name).best_values.dS,
                  precision='.2f', units='kB'),
        HoverInfo(name='Integrated Entropy',
                  func=lambda dat: np.nanmean(
                      dat.Entropy.get_integrated_entropy(
                          name=int_info_name,
                          data=dat.SquareEntropy.get_Outputs(
                              name=SE_output_name).average_entropy_signal)[-10:]
                  ),
                  precision='.2f', units='kB'),
        HoverInfo(name='Amp for sf', func=lambda dat: dat.Entropy.get_integration_info(name=int_info_name).amp,
                  units='nA'),
        HoverInfo(name='dT for sf', func=lambda dat: dat.Entropy.get_integration_info(name=int_info_name).dT,
                  units='mV'),
        HoverInfo(name='sf', func=lambda dat: dat.Entropy.get_integration_info(name=int_info_name).sf,
                  units=''),
    ]

    funcs, template = additional_data_dict_converter(hover_infos)
    hover_data = [[func(dat) for func in funcs] for dat in dats]

    x = [x_func(dat) for dat in dats]
    integrated_entropies = [np.nanmean(
        dat.Entropy.get_integrated_entropy(name=int_info_name,
                                           data=dat.SquareEntropy.get_Outputs(
                                               name=SE_output_name).average_entropy_signal
                                           )[-10:]) for dat in dats]
    trace = plotter.trace(
        data=integrated_entropies, x=x, name=trace_name,
        mode='markers+lines',
        trace_kwargs=dict(customdata=hover_data, hovertemplate=template)
    )
    return trace


def get_integrated_fig(title_append: str = '') -> go.Figure:
    plotter = OneD()
    fig = plotter.figure(xlabel='ESC /mV', ylabel='Entropy /kB',
                         title=f'Integrated Entropy {title_append}')
    return fig


def transition_trace(dats: List[DatHDF], x_func: Callable,
               from_square_entropy: bool=True, fit_name: str = 'default',
                     param: str = 'amp') -> go.Scatter:
    plotter = OneD(dats=dats)
    if from_square_entropy:
        amps = [dat.SquareEntropy.get_fit(which_fit='transition', name=fit_name, check_exists=True).best_values.get(param) for dat in dats]
    else:
        amps = [dat.Transition.get_fit(name=fit_name).best_values.get(param) for dat in dats]

    x = [x_func(dat) for dat in dats]
    trace = plotter.trace(x=x, data=amps)
    return trace


def transition_fig(dats: Optional[List[DatHDF]] = None, xlabel: str = '/mV', title_append: str = '',
                   param: str = 'amp') -> go.Figure:
    plotter = OneD(dats=dats)
    titles = {
        'amp': 'Amplitude',
        'theta': 'Theta',
        'g': 'Gamma',
    }
    ylabels = {
        'amp': 'Amplitude /nA',
        'theta': 'Theta /mV',
        'g': 'Gamma /mV',
    }

    fig = plotter.figure(xlabel=xlabel, ylabel=ylabels[param],
                         title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: {titles[param]}{title_append}')
    return fig



TRANSITION_DATNUMS = list(range(1604, 1635, 2))

DATNUMS1 = list(range(1637, 1652 + 1))  # First set all the way from weakly coupled to gamma broadened
DATNUMS2 = list(range(1653, 1668 + 1))  # Second set all the way from weakly coupled to gamma broadened
DATNUMS3 = list(range(1669, 1684 + 1))  # Second set all the way from weakly coupled to gamma broadened

POS2 = list(range(1778, 1794 + 1))  # First set at different ESS (-350mV instead of -375mV)

VS_TIME = list(range(1685, 1772 + 1))

if __name__ == '__main__':
    # list(pool.map(do_calc, DATNUMS1+DATNUMS2+DATNUMS3))
    # list(pool.map(do_calc, DATNUMS2))
    # list(pool.map(do_calc, POS2))
    # a = list(pool.map(do_calc, DATNUMS3))
    # dats = get_dats(DATNUMS1)
    dats = get_dats(POS2)

    plot_amp = False
    plot_vs_gamma = False
    plot_vs_time = True

    if plot_amp:
        do_narrow_fits(dats, theta=3.9756, output_name='SPS.005', overwrite=True)
        param = 'amp'
        fig = transition_fig(dats=dats, xlabel='ESC /mV', title_append=' vs ESC for Entropy scans', param=param)
        fig.add_trace(transition_trace(dats, x_func=lambda dat: dat.Logs.fds['ESC'], from_square_entropy=True,
                                       fit_name='narrow', param=param))
        print(f'Avg weakly coupled cold theta = {np.mean([dat.SquareEntropy.get_fit(which_fit="transition", name="narrow").best_values.theta for dat in dats if dat.Logs.fds["ESC"] <= -300])}')
        fig.show()


    if plot_vs_gamma:
        fig = get_integrated_fig(title_append=f' at ESS = {dats[0].Logs.fds["ESS"]}mV')
        # for datnums, label in zip([DATNUMS1, DATNUMS2, DATNUMS3], ['Set 1', 'Set 2', 'Set 3']):
        for datnums, label in zip([POS2], ['Set 1', 'Set 2', 'Set 3']):
            dats = get_dats(datnums)
            for dat in dats:
                dat.Entropy.set_integration_info(dT=get_deltaT(dat),
                                                 amp=get_amplitude(dat, transition_fit_name='narrow', gate='ESC'),
                                                 name='first', overwrite=True)

            fig.add_trace(get_integrated_trace(dats=dats, x_func=lambda dat: dat.Logs.fds['ESC'],
                                               trace_name=label,
                                               int_info_name='first', SE_output_name='SPS.005'))

            fig2 = plot_fit_integrated_comparison(dats, x_func=lambda dat: dat.Logs.fds['ESC'], x_label='ESC /mV',
                                                  int_info_name='first', fit_name='SPS.005',
                                                  plot=True)
        fig.show()

    if plot_vs_time:
        # list(pool.map(do_calc, VS_TIME))
        dats = get_dats(VS_TIME)
        fit_fig = entropy_vs_time_fig(title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: Fit Entropy vs Time')
        int_fig = entropy_vs_time_fig(title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: Integrated Entropy vs Time')
        for esc in set([dat.Logs.fds['ESC'] for dat in dats]):
            ds = [dat for dat in dats if dat.Logs.fds['ESC'] == esc]
            fit_fig.add_trace(entropy_vs_time_trace(dats=ds, integrated=False, trace_name=f'Fit Entropy {esc}mV',
                                                    fit_name='SPS.005', integrated_name='first'))
            int_fig.add_trace(entropy_vs_time_trace(dats=ds, integrated=True, trace_name=f'Integrated Entropy {esc}mV',
                                                    fit_name='SPS.005', integrated_name='first'))

        fit_fig.show()
        int_fig.show()

pool.shutdown()
thread_pool.shutdown()
