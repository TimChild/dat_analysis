import src.UsefulFunctions as U
import src.Characters as C
from src.UsefulFunctions import run_multiprocessed
from src.DatObject.Make_Dat import get_dat, get_dats, DatHDF
from src.DatObject.Attributes.SquareEntropy import square_wave_time_array
from src.DatObject.Attributes.Transition import i_sense, i_sense_digamma
from src.Plotting.Plotly.PlotlyUtil import additional_data_dict_converter, HoverInfo, add_horizontal
from src.Dash.DatPlotting import OneD, TwoD
from src.HDF_Util import NotFoundInHdfError
from Analysis.Feb2021.common import get_deltaT, plot_fit_integrated_comparison, entropy_vs_time_trace, \
    entropy_vs_time_fig, do_narrow_fits

from deprecation import deprecated
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

    do_narrow_fits([dat], theta=3.896, output_name='SPS.005', overwrite=overwrite)

    if 'first' not in dat.Entropy.get_integration_info_names() or overwrite:
        dat.Entropy.set_integration_info(dT=get_deltaT(dat),
                                         amp=get_amplitude(dat, transition_fit_name='narrow', gate='ESC'),
                                         name='first', overwrite=overwrite)
    return True


@deprecated
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

    fit = dat.Transition.get_fit(which='avg', fit_name='careful',
                                 fit_func=fit_func, initial_params=params,
                                 check_exists=False, overwrite=overwrite)
    return fit


def get_amplitude(dat: DatHDF, transition_fit_name: str = 'default', gate: str = 'ESC'):
    """Returns amplitude of a given dat in mV based on a fit through amplitude determined from long transition
    specific scans """

    def get_amp(d: DatHDF):
        amp = d.SquareEntropy.get_fit(which_fit='transition', fit_name=transition_fit_name,
                                      check_exists=True).best_values.amp
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
        print(f'Dat{dat.datnum}: theta = {dat.Transition.get_fit(fit_name="careful").best_values.theta:.2f}mV')

    print(
        f'Average Theta: {np.nanmean([dat.Transition.get_fit(fit_name="careful").best_values.theta for dat in weak_dats]):.2f}mV')


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
        # HoverInfo(name='Fit Entropy', func=lambda dat: dat.SquareEntropy.get_fit(which_fit='entropy', fit_name=fit_name).best_values.dS,
        #           precision='.2f', units='kB'),
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
    plotter = OneD(dats=dats)
    if from_square_entropy:
        amps = [dat.SquareEntropy.get_fit(which_fit='transition', fit_name=fit_name, check_exists=True).best_values.get(
            param) for dat in dats]
    else:
        amps = [dat.Transition.get_fit(name=fit_name).best_values.get(param) for dat in dats]

    x = [x_func(dat) for dat in dats]
    trace = plotter.trace(x=x, data=amps, name=label, text=[dat.datnum for dat in dats], **kwargs)
    return trace


def single_transition_trace(dat: DatHDF, label: Optional[str] = None, subtract_fit=False,
                            fit_only=False) -> go.Scatter():
    plotter = OneD(dat=dat)
    x = dat.SquareEntropy.avg_x

    if not fit_only:
        data = dat.SquareEntropy.get_transition_part(name='SPS.005', part='cold')
    else:
        data = None  # Set below

    if fit_only or subtract_fit:
        fit = dat.SquareEntropy.get_fit(which_fit='transition', fit_name='narrow')
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


def set_sf_from_transition(entropy_datnums, transition_datnums):
    for enum, tnum in progressbar(zip(entropy_datnums, transition_datnums)):
        edat = get_dat(enum)
        tdat = get_dat(tnum)
        set_amplitude_from_transition_only(edat, tdat)


def set_amplitude_from_transition_only(entropy_dat: DatHDF, transition_dat: DatHDF):
    ed = entropy_dat
    td = transition_dat
    for k in ['ESC', 'ESS', 'ESP']:
        if ed.Logs.fds[k] != td.Logs.fds[k]:
            raise ValueError(f'Non matching FDS for entropy_dat {ed.datnum} and transition_dat {td.datnum}: \n'
                             f'entropy_dat fds = {ed.Logs.fds}\n'
                             f'transition_dat fds = {td.Logs.fds}')
    amp = td.Transition.get_fit(name='narrow').best_values.amp
    ed.Entropy.set_integration_info(dT=get_deltaT(ed),
                                    amp=amp,
                                    name='amp from transition',
                                    overwrite=True)
    return True


def calculate_transition_only(datnums, theta=None, vary_theta=False):
    with ProcessPoolExecutor() as pool:
        if vary_theta:
            fits = list(pool.map(
                partial(do_narrow_fits, theta=None, gamma=0, width=500, overwrite=True, transition_only=True),
                datnums))
        else:
            fits = list(pool.map(partial(do_narrow_fits, theta=theta, gamma=None, width=500, overwrite=True,
                                         transition_only=True), datnums))


TRANSITION_DATNUMS = list(range(1604, 1635, 2))
TRANSITION_DATNUMS_2 = list(range(1833, 1866, 2))  # Taken at ESS = -340mV

DATNUMS1 = list(range(1637, 1652 + 1))  # First set all the way from weakly coupled to gamma broadened
DATNUMS2 = list(range(1653, 1668 + 1))  # Second set all the way from weakly coupled to gamma broadened
DATNUMS3 = list(range(1669, 1684 + 1))  # Second set all the way from weakly coupled to gamma broadened

POS2 = list(range(1778, 1794 + 1))  # First set at different ESS (-350mV instead of -375mV)
POS3 = list(range(1798, 1814 + 1))  # ESS at -340mV
POS3_2 = list(range(1815, 1831 + 1))  # ESS at -340mV
POS3_3 = list(range(1869, 1918 + 1, 2))  # ESS at -340mV (alternates with Transition Only)
POS3_3.remove(1909)

POS3_3_Tonly = list(range(1870, 1918 + 1, 2))  # Same as above but transition only scans
POS3_3_Tonly.remove(1910)

POS3_100 = list(range(1919, 1926+1, 2))
POS3_100_Tonly = list(range(1920, 1926+1, 2))

VS_TIME = list(range(1685, 1772 + 1))

if __name__ == '__main__':
    entropy_datnums = POS3_100
    transition_datnums = POS3_100_Tonly
    # Calculations

    recalculate = True
    if recalculate:
        with ProcessPoolExecutor() as pool:
            calculate_transition_only(transition_datnums, theta=3.896, vary_theta=False)
            list(pool.map(partial(do_calc, overwrite=True), entropy_datnums))
            pass
        set_sf_from_transition(entropy_datnums, transition_datnums)


    plot_transition_fitting = False
    plot_transition_values = True
    plot_entropy_vs_gamma = True
    plot_entropy_vs_time = False
    plot_amp_comparison = True

    if plot_transition_fitting:
        dat = get_dat(1831)
        for fit_width in [200, 300, 400, 500, 600, 700]:
            do_narrow_fits([dat], theta=3.9756, width=fit_width, output_name='SPS.005', overwrite=True)
            plotter = OneD(dat=dat)
            fig_fit = plotter.figure(title=f'Dat{dat.datnum}: Transition Data with Fit (width={fit_width})',
                                     ylabel=f'Current /nA')
            fig_fit.add_trace(single_transition_trace(dat, label='Data'))
            fig_fit.add_trace(single_transition_trace(dat, label='Fit', fit_only=True))

            fig_minus = plotter.figure(title=f'Dat{dat.datnum}: Transition Data minus Fit (width={fit_width})',
                                       ylabel=f'{C.DELTA}Current /nA')
            fig_minus.add_trace(single_transition_trace(dat, label=None, subtract_fit=True))

            for fig in [fig_minus, fig_fit]:
                plotter.add_line(fig, value=fit_width, mode='vertical')
                plotter.add_line(fig, value=-fit_width, mode='vertical')

            fit = dat.SquareEntropy.get_fit(which_fit='transition', fit_name='narrow')
            print(f'Dat{dat.datnum}:\n'
                  f'\tWidth: {C.PM}{fit_width}mV\n'
                  f'\tAmp: {fit.best_values.amp:.3f}nA\n'
                  f'\tGamma: {fit.best_values.g:.1f}mV\n'
                  f'\tTheta: {fit.best_values.theta:.2f}mV\n'
                  f'\tLin: {fit.best_values.lin:.3g}nA/mV\n'
                  f'\tCenter: {fit.best_values.mid:.1f}mV\n'
                  )

            fig_fit.show()
            fig_minus.show()

    if plot_transition_values:
        transition_only = True
        param = 'g'
        if transition_only:
            all_dats = get_dats(transition_datnums)
            fig = transition_fig(dats=all_dats, xlabel='ESC /mV', title_append=' vs ESC for Transition Only scans',
                                 param=param)
            for dnums, label in zip([transition_datnums], ['Set 1', 'Set 2']):
                dats = get_dats(dnums)
                fig.add_trace(transition_trace(dats, x_func=lambda dat: dat.Logs.fds['ESC'], from_square_entropy=False,
                                               fit_name='narrow', param=param, label=label))
                # print(
                #     f'Avg weakly coupled cold theta = '
                #     f'{np.mean([dat.Transition.get_fit(name="narrow").best_values.theta for dat in dats if dat.Logs.fds["ESC"] <= -300])}')
        else:
            all_dats = get_dats(entropy_datnums)
            param = 'amp'
            fig = transition_fig(dats=all_dats, xlabel='ESC /mV', title_append=' vs ESC for Entropy scans', param=param)
            for datnums, label in zip([entropy_datnums], ['Set 1', 'Set 2']):
                dats = get_dats(datnums)
                fig.add_trace(transition_trace(dats, x_func=lambda dat: dat.Logs.fds['ESC'], from_square_entropy=True,
                                               fit_name='narrow', param=param, label=label))
                print(
                    f'Avg weakly coupled cold theta = {np.mean([dat.SquareEntropy.get_fit(which_fit="transition", fit_name="narrow").best_values.theta for dat in dats if dat.Logs.fds["ESC"] <= -300])}')
        fig.show()

    if plot_entropy_vs_gamma:
        integration_info_name = 'amp from transition'
        dats = get_dats(entropy_datnums)
        fig = get_integrated_fig(dats, title_append=f' at ESS = {dats[0].Logs.fds["ESS"]}mV')
        # for datnums, label in zip([DATNUMS1, DATNUMS2, DATNUMS3], ['Set 1', 'Set 2', 'Set 3']):
        for datnums, label in zip([entropy_datnums, POS3_3], ['100% Heating', '30% Heating', 'Set 3']):
            dats = get_dats(datnums)
            fig.add_trace(get_integrated_trace(dats=dats, x_func=lambda dat: dat.Logs.fds['ESC'],
                                               trace_name=label,
                                               int_info_name=integration_info_name, SE_output_name='SPS.005'))

            # fig2 = plot_fit_integrated_comparison(dats, x_func=lambda dat: dat.Logs.fds['ESC'], x_label='ESC /mV',
            #                                       int_info_name=integration_info_name, fit_name='SPS.005',
            #                                       plot=True)
        fig.show()

    if plot_amp_comparison:
        compare_amps = True
        compare_integrated = True

        entropy_dats = get_dats(entropy_datnums)
        transition_dats = get_dats(transition_datnums)

        # for datnums, label in zip([DATNUMS1, DATNUMS2, DATNUMS3], ['Set 1', 'Set 2', 'Set 3']):
        if compare_amps:
            fig = transition_fig(entropy_dats + transition_dats, xlabel='ESC /mV',
                                 title_append=' Comparison of amp from Entropy vs Transition only', param='amp')
            fig.add_trace(transition_trace(entropy_dats, x_func=lambda dat: dat.Logs.fds['ESC'],
                                           from_square_entropy=True, fit_name='narrow', param='amp',
                                           label='Cold part of Entropy',
                                           mode='markers+lines'))
            fig.add_trace(transition_trace(transition_dats, x_func=lambda dat: dat.Logs.fds['ESC'],
                                           from_square_entropy=False, fit_name='narrow', param='amp',
                                           label='Transition Only',
                                           mode='markers+lines'))
            fig.show()
        if compare_integrated:
            fig = get_integrated_fig(title_append=f' at ESS = {entropy_dats[0].Logs.fds["ESS"]}mV<br>'
                                                  f'Comparison of amp from Entropy vs Transition')
            for integration_info_name, label in zip(['first', 'amp from transition'],
                                                    ['Amp from Entropy', 'Amp from Transition']):
                fig.add_trace(get_integrated_trace(dats=entropy_dats, x_func=lambda dat: dat.Logs.fds['ESC'],
                                                   trace_name=label,
                                                   int_info_name=integration_info_name, SE_output_name='SPS.005'))
            fig.show()

    if plot_entropy_vs_time:
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

