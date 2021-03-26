"""Hopefully all the final analysis of Gamma Broadened Entropy measurements"""
import src.UsefulFunctions as U
import src.Characters as C
from src.DatObject.Make_Dat import get_dat, get_dats, DatHDF
from src.DatObject.Attributes.SquareEntropy import square_wave_time_array
from src.DatObject.Attributes.Transition import i_sense, i_sense_digamma
from src.DatObject.Attributes.DatAttribute import DatDataclassTemplate
from src.Plotting.Plotly.PlotlyUtil import additional_data_dict_converter, HoverInfo
from src.Dash.DatPlotting import OneD, TwoD
from Analysis.Feb2021.common import get_deltaT, plot_fit_integrated_comparison, entropy_vs_time_trace, \
    entropy_vs_time_fig, do_narrow_fits, do_entropy_calc, do_transition_only_calc, set_sf_from_transition, \
    calculate_csq_map, setup_csq_dat, NRG_fitter, get_integrated_trace, get_integrated_fig, transition_trace, \
    transition_fig, single_transition_trace

from typing import List, Optional, Callable, Union, Tuple
import h5py
from progressbar import progressbar
import logging
import lmfit as lm
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from functools import partial
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

pio.renderers.default = 'browser'
logger = logging.getLogger(__name__)


def reset_dats(*args: Union[list, int, None]):
    """Fully overwrites DatHDF of any datnums/lists of datnums passed in"""
    if reset_dats:
        all_datnums = []
        for datnums in args:
            if isinstance(datnums, list):
                all_datnums.extend(datnums)
            elif isinstance(datnums, (int, np.int32)):
                all_datnums.append(datnums)
        for datnum in all_datnums:
            get_dat(datnum, overwrite=True)


@dataclass
class GammaAnalysisParams(DatDataclassTemplate):
    """All the various things that go into calculating Entropy etc"""
    # To save in HDF
    save_name: str

    # For Entropy calculation (and can also determine transition info from entropy dat)
    entropy_datnum: int
    setpoint_start: Optional[float] = 0.005  # How much data to throw away after each heating setpoint change in secs
    entropy_transition_func_name: str = 'i_sense'  # For centering data only if Transition specific dat supplied,
    # Otherwise this also is used to calculate gamma, amplitude and optionally dT
    entropy_fit_width: Optional[float] = None
    entropy_data_rows: tuple = (None, None)  # Only fit between rows specified (None means beginning or end)

    # Integrated Entropy
    force_dt: Optional[float] = None  # 1.11  #  dT for scaling
    # (to determine from Entropy set dt_from_square_transition = True)
    force_amp: Optional[float] = None  # Otherwise determined from Transition Only, or from Cold part of Entropy
    sf_from_square_transition: bool = False  # Set True to determine dT from Hot - Cold theta

    # For Transition only fitting (determining amplitude and gamma)
    transition_only_datnum: Optional[int] = None  # For determining amplitude and gamma
    transition_func_name: str = 'i_sense_digamma'
    transition_center_func_name: Optional[str] = None  # Which fit func to use for centering data
    # (defaults to same as transition_center_func_name)
    transition_fit_width: Optional[float] = None
    force_theta: Optional[float] = None  # 3.9  # Theta must be forced in order to get an accurate gamma for broadened
    force_gamma: Optional[float] = None  # Gamma must be forced zero to get an accurate theta for weakly coupled
    transition_data_rows: Tuple[Optional[int], Optional[int]] = (
        None, None)  # Only fit between rows specified (None means beginning or end)

    # For CSQ mapping, applies to both Transition and Entropy (since they should always be taken in pairs anyway)
    csq_mapped: bool = False  # Whether to use CSQ mapping
    csq_datnum: Optional[int] = None  # CSQ trace to use for CSQ mapping

    def __str__(self):
        return f'Dat{self.entropy_datnum}:\n' \
               f'\tEntropy Params:\n' \
               f'entropy datnum = {self.entropy_datnum}\n' \
               f'setpoint start = {self.setpoint_start}\n' \
               f'transition func = {self.entropy_transition_func_name}\n' \
               f'force theta = {self.force_theta}\n' \
               f'force gamma = {self.force_gamma}\n' \
               f'fit width = {self.entropy_fit_width}\n' \
               f'data rows = {self.entropy_data_rows}\n' \
               f'\tIntegrated Entropy Params:\n' \
               f'from Entropy directly = {self.sf_from_square_transition}\n' \
               f'forced dT = {self.force_dt}\n' \
               f'forced amp = {self.force_amp}\n' \
               f'\tTransition Params:\n' \
               f'transition datnum = {self.transition_only_datnum}\n' \
               f'fit func = {self.transition_func_name}\n' \
               f'center func = {self.transition_center_func_name}\n' \
               f'fit width = {self.transition_fit_width}\n' \
               f'force theta = {self.force_theta}\n' \
               f'force gamma = {self.force_gamma}\n' \
               f'data rows = {self.transition_data_rows}\n' \
               f'\tCSQ Mapping Params:\n' \
               f'Mapping used: {self.csq_mapped}\n' \
               f'csq datnum: {self.csq_datnum}\n'


@dataclass
class AnalysisGeneral:
    params_list: List[GammaAnalysisParams]

    calculate: bool = True
    overwrite_csq: bool = False
    overwrite_transition: bool = False
    overwrite_entropy: bool = False

    x_func: Callable = lambda dat: dat.Logs.fds['ESC']
    x_label: str = 'ESC /mV'

    def __post_init__(self):
        self.entropy_datnums = [params.entropy_datnum for params in self.params_list if
                                params.entropy_datnum is not None]
        self.transition_datnums = [params.transition_only_datnum for params in self.params_list if
                                   params.transition_only_datnum is not None]
        self.csq_datnums = [params.csq_datnum for params in self.params_list if params.csq_datnum is not None]


def save_gamma_analysis_params_to_dat(dat: DatHDF, analysis_params: GammaAnalysisParams,
                                      name: str):
    """Save GammaAnalysisParams to suitable place in DatHDF"""
    with h5py.File(dat.hdf.hdf_path, 'r+') as hdf:
        analysis_group = hdf.require_group('Gamma Analysis')
        analysis_params.save_to_hdf(analysis_group, name=name)


def load_gamma_analysis_params(dat: DatHDF, name: str) -> GammaAnalysisParams:
    """Load GammaAnalysisParams from DatHDF"""
    with h5py.File(dat.hdf.hdf_path, 'r') as hdf:
        analysis_group = hdf.get('Gamma Analysis')
        analysis_params = GammaAnalysisParams.from_hdf(analysis_group, name=name)
    return analysis_params


def process_single(params: GammaAnalysisParams, overwrite_transition=False, overwrite_entropy=False):
    """Does all the processing necessary for a single GammaAnalysisParams (i.e. csq mapping, transition fitting,
    entropy fitting and setting integration info"""
    if params.csq_mapped:
        calculate_csq_map(params.entropy_datnum, csq_datnum=params.csq_datnum, overwrite=overwrite_entropy)
        if params.transition_only_datnum:
            calculate_csq_map(params.transition_only_datnum, csq_datnum=params.csq_datnum,
                              overwrite=overwrite_transition)

    if params.transition_only_datnum is not None:
        if params.save_name + '_cold' not in get_dat(params.transition_only_datnum).Transition.fit_names \
                or overwrite_transition:
            do_transition_only_calc(datnum=params.transition_only_datnum, save_name=params.save_name,
                                    theta=params.force_theta, gamma=params.force_gamma,
                                    center_func=params.transition_center_func_name,
                                    width=params.transition_fit_width, t_func_name=params.transition_func_name,
                                    csq_mapped=params.csq_mapped, data_rows=params.transition_data_rows,
                                    overwrite=overwrite_transition)
    if params.save_name not in get_dat(params.entropy_datnum).Entropy.fit_names or overwrite_entropy:
        do_entropy_calc(params.entropy_datnum, save_name=params.save_name,
                        setpoint_start=params.setpoint_start,
                        t_func_name=params.entropy_transition_func_name,
                        csq_mapped=params.csq_mapped,
                        theta=params.force_theta, gamma=params.force_gamma, width=params.entropy_fit_width,
                        data_rows=params.transition_data_rows,
                        overwrite=overwrite_entropy)

    calculate_new_sf_only(params.entropy_datnum, params.save_name,
                          dt=params.force_dt, amp=params.force_amp,
                          from_square_transition=params.sf_from_square_transition,
                          transition_datnum=params.transition_only_datnum,
                          fit_name=params.save_name)

    # Save to HDF
    save_gamma_analysis_params_to_dat(dat, analysis_params=params, name=params.save_name)


def calculate_new_sf_only(entropy_datnum: int, save_name: str,
                          dt: Optional[float] = None, amp: Optional[float] = None,
                          from_square_transition: bool = False, transition_datnum: Optional[int] = None,
                          fit_name: Optional[str] = None,
                          ):
    """
    Calculate a scaling factor for integrated entropy either from provided dT/amp, entropy dat directly, or with help
    of a transition only dat
    Args:
        entropy_datnum ():
        save_name ():  Name the integration info will be saved under in the entropy dat
        dt ():
        amp ():
        from_square_transition ():
        transition_datnum ():
        fit_name (): Fit names to look for in Entropy and Transition (not the name which the sf will be saved under)

    Returns:

    """
    # Set integration info
    if from_square_transition:  # Only sets dt and amp if not forced
        dat = get_dat(entropy_datnum)
        cold_fit = dat.SquareEntropy.get_fit(fit_name=fit_name + '_cold')
        hot_fit = dat.SquareEntropy.get_fit(fit_name=fit_name + '_hot')
        if dt is None:
            if any([fit.best_values.theta is None for fit in [cold_fit, hot_fit]]):
                raise RuntimeError(f'Dat{dat.datnum}: Failed to fit for hot or cold...\n{cold_fit}\n\n{hot_fit}')
            dt = hot_fit.best_values.theta - cold_fit.best_values.theta
        if amp is None:
            amp = cold_fit.best_values.amp
    else:
        if dt is None:
            raise ValueError(f"Dat{entropy_datnum}: dT must be provided if not calculating from square entropy")
        if amp is None:
            dat = get_dat(transition_datnum)
            amp = dat.Transition.get_fit(name=fit_name).best_values.amp

    dat = get_dat(entropy_datnum)
    dat.Entropy.set_integration_info(dT=dt, amp=amp,
                                     name=save_name, overwrite=True)  # Fast to write


def run_processing(analysis_general: AnalysisGeneral):
    a = analysis_general
    if a.calculate:
        with ProcessPoolExecutor() as pool:
            if a.csq_datnums:
                list(pool.map(partial(setup_csq_dat, overwrite=a.overwrite_csq), a.csq_datnums))
                print(f'Done Setting up CSQonly dats')

            list(pool.map(partial(process_single, overwrite_transition=a.overwrite_transition,
                                  overwrite_entropy=a.overwrite_entropy), a.params_list))

        print(f'Done Processing')


@dataclass
class PlotInfo:
    title_append: str
    ylabel: str
    data_func: Callable
    x_func: Callable
    trace_name: Callable


def plot_stacked_square_heated(datnums: List[int], save_name: str):
    dats = get_dats(datnums)

    # Plot Integrated
    integrated_plot_info = PlotInfo(
        title_append='Integrated Entropy',
        ylabel='Entropy /kB',
        data_func=lambda dat: dat.Entropy.get_integrated_entropy(name=save_name, data=
        dat.SquareEntropy.get_Outputs(name=save_name, check_exists=True).average_entropy_signal),
        x_func=lambda dat: dat.SquareEntropy.get_Outputs(name=save_name, check_exists=True).x,
        trace_name=lambda dat: f'Dat{dat.datnum}'
    )

    fit_plot_info = PlotInfo(
        title_append='Fit Entropy',
        ylabel='Entropy /kB',
        data_func=lambda dat: dat.SquareEntropy.get_Outputs(name=save_name,
                                                            check_exists=True).average_entropy_signal,
        x_func=lambda dat: dat.SquareEntropy.get_Outputs(name=save_name, check_exists=True).x,
        trace_name=lambda dat: f'Dat{dat.datnum}'
    )

    for plot_info in [integrated_plot_info]:
        plotter = OneD(dats=dats)
        dat = dats[0]
        fig = plotter.figure(xlabel=dat.Logs.xlabel, ylabel=plot_info.ylabel,
                             title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: {plot_info.title_append}')
        for dat in dats:
            data = plot_info.data_func(dat)
            x = plot_info.x_func(dat)

            hover_infos = [
                HoverInfo(name='Datnum', func=lambda dat: dat.datnum, precision='d', units=''),
                HoverInfo(name=dat.Logs.xlabel, func=lambda dat: plot_info.x_func(dat), precision='.2f',
                          units='/mV'),
                HoverInfo(name=plot_info.ylabel, func=lambda dat: dat.datnum, precision='d', units=''),
                HoverInfo(name='Datnum', func=lambda dat: dat.datnum, precision='d', units=''),

            ]
            hover_funcs, template = additional_data_dict_converter(hover_infos)

            hover_data = []
            for func in hover_funcs:
                v = func(dat)
                if not hasattr(v, '__len__') or len(v) == 1:  # Make sure a hover info for each x_coord
                    v = [v] * len(x)
                hover_data.append(v)

            fig.add_trace(plotter.trace(x=x, data=data,
                                        name=plot_info.trace_name(dat),
                                        hover_data=hover_data, hover_template=template,
                                        mode='lines'))
        fig.show()


def plot_vs_gamma(datnums: List[int], save_name: str, general: AnalysisGeneral, sf_name: Optional[str] = None,
                  show=True):
    dats = get_dats(datnums)
    fig = get_integrated_fig(dats, title_append=f' at ESS = {dats[0].Logs.fds["ESS"]}mV')

    int_info_name = save_name if sf_name is None else sf_name

    fig.add_trace(get_integrated_trace(dats=dats, x_func=general.x_func, x_label=general.x_label,
                                       trace_name='Data',
                                       fit_name=save_name,
                                       int_info_name=int_info_name, SE_output_name=save_name))

    fig2 = plot_fit_integrated_comparison(dats, x_func=general.x_func, x_label=general.x_label,
                                          int_info_name=int_info_name, fit_name=save_name,
                                          plot=True)
    if show:
        fig.show()
    return fig


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


# def plot_gamma_dcbias(datnums: List[int], save_name: str):
#     if calculate:
#         with ProcessPoolExecutor() as pool:
#             list(pool.map(partial(do_transition_only_calc, save_name=save_name, theta=theta, gamma=None, width=600,
#                                   t_func_name='i_sense_digamma', overwrite=False), GAMMA_DCbias))
#     dats = get_dats(GAMMA_DCbias)
#     plotter = OneD(dats=dats)
#     # fig = plotter.figure(ylabel='Current /nA',
#     #                      title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: DCbias in Gamma broadened' )
#     # dat_pairs = np.array(dats).reshape((-1, 2))
#     # line = lm.models.LinearModel()
#     # params = line.make_params()
#     # for ds in dat_pairs:
#     #     for dat, color in zip(ds, ['blue', 'red']):
#     #         params['slope'].value = dat.Transition.avg_fit.best_values.lin
#     #         params['intercept'].value = dat.Transition.avg_fit.best_values.const
#     #         fig.add_trace(plotter.trace(x=dat.Transition.avg_x, data=dat.Transition.avg_data-line.eval(params=params, x=dat.Transition.avg_x),
#     #                                     name=f'Dat{dat.datnum}: Bias={dat.Logs.fds["HO1/10M"]/10:.1f}nA',
#     #                                     mode='lines',
#     #                                     trace_kwargs=dict(line=dict(color=color)),
#     #                                     ))
#
#     fig = plotter.figure(ylabel='Current /nA',
#                          title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: DCbias in Gamma broadened' )
#     line = lm.models.LinearModel()
#     params = line.make_params()
#     for dat in dats[1::2]:
#         params['slope'].value = dat.Transition.avg_fit.best_values.lin
#         params['intercept'].value = dat.Transition.avg_fit.best_values.const
#         fig.add_trace(plotter.trace(x=dat.Transition.avg_x, data=dat.Transition.avg_data-line.eval(params=params, x=dat.Transition.avg_x),
#                                     name=f'Dat{dat.datnum}: Bias={dat.Logs.fds["HO1/10M"]/10:.1f}nA',
#                                     mode='lines',
#                                     ))
#     fig.show()


def make_long_analysis_params(entropy_datnums, transition_datnums, csq_datnums,
                              save_name: str,
                              entropy_data_rows: Tuple[Optional[int], Optional[int]] = (None, None),
                              transition_data_rows: Tuple[Optional[int], Optional[int]] = (None, None),
                              ) -> List[GammaAnalysisParams]:
    all_params = []
    for ed, td, cs in zip(entropy_datnums, transition_datnums, csq_datnums):
        all_params.append(
            GammaAnalysisParams(
                save_name=save_name,
                entropy_datnum=ed, transition_only_datnum=td, csq_datnum=cs,
                setpoint_start=0.005,
                entropy_transition_func_name='i_sense', entropy_fit_width=None, entropy_data_rows=entropy_data_rows,
                force_dt=None, force_amp=None,
                sf_from_square_transition=True,
                force_theta=None, force_gamma=None,  # Applies to entropy and transition only
                transition_func_name='i_sense_digamma', transition_fit_width=None,
                transition_center_func_name='i_sense_digamma', transition_data_rows=transition_data_rows,
                csq_mapped=True
            )
        )
    return all_params


def make_vs_gamma_analysis_params(entropy_datnums, transition_datnums,
                                  save_name: str,
                                  entropy_data_rows: Tuple[Optional[int], Optional[int]] = (None, None),
                                  transition_data_rows: Tuple[Optional[int], Optional[int]] = (None, None),
                                  force_theta=None, force_gamma: Optional[int] = None,
                                  sf_from_square_transition: bool = True,
                                  width: Optional[int] = None,
                                  ) -> List[GammaAnalysisParams]:
    all_params = []
    for ed, td in zip(entropy_datnums, transition_datnums):
        all_params.append(
            GammaAnalysisParams(
                csq_mapped=False,
                save_name=save_name,
                entropy_datnum=ed, transition_only_datnum=td,
                setpoint_start=0.005,
                entropy_transition_func_name='i_sense_digamma', entropy_fit_width=width,
                entropy_data_rows=entropy_data_rows,
                force_dt=None, force_amp=None,
                sf_from_square_transition=sf_from_square_transition,
                force_theta=force_theta, force_gamma=force_gamma,  # Applies to entropy and transition only
                transition_func_name='i_sense_digamma', transition_fit_width=width,
                transition_center_func_name='i_sense_digamma', transition_data_rows=transition_data_rows,
            )
        )
    return all_params


############## Relevant Dats ##################
# TODO: Should combine the 2213/2216 (both just long 50kBT scans)
# TODO: Should combine the GAMMA_25 with 2170 (all are 25kBT)
DCbias = list(range(2143, 2155 + 1))

VS_GAMMA = list(range(2095, 2142, 2))
VS_GAMMA_Tonly = list(range(2096, 2142 + 1, 2))

LONG_GAMMA = [2164, 2167, 2170, 2213, 2216]
LONG_GAMMA_Tonly = [2165, 2168, 2171, 2214, 2217]
LONG_GAMMA_csq = [2173, 2174, 2175, 2215, 2218]

GAMMA_25 = [2131, 2160, 2170, 2176, 2182]

DC_GAMMA = list(range(2219, 2230))  # DCbias scans in gamma broadened (~25kBT) to make sure that heating not
# broadening transition more than it should
#################################################


if __name__ == '__main__':
    # all_params = make_long_analysis_params(LONG_GAMMA, LONG_GAMMA_Tonly, LONG_GAMMA_csq, save_name='test',
    #                                   entropy_data_rows=(0, 10), transition_data_rows=(0, 10))
    name = 'forced_theta_linear'
    # sf_name = 'fixed_dT'
    sf_name = 'scaled_dT'
    all_params = make_vs_gamma_analysis_params(VS_GAMMA, VS_GAMMA_Tonly, save_name=name,
                                               force_theta=None, force_gamma=None,
                                               sf_from_square_transition=False, width=600)

    # Setting theta according to linear fit in weakly coupled regime with gamma = 0
    line = lm.models.LinearModel()
    line_pars = line.make_params()
    line_pars['slope'].value = 0.00348026
    line_pars['intercept'].value = 5.09205057
    for par in all_params:
        dat = get_dat(par.transition_only_datnum)
        theta = line.eval(params=line_pars, x=dat.Logs.fds['ESC'])
        par.force_theta = theta
        par.force_dt = 1.111 * (theta) / 3.9  # Scale dT with same proportion as theta
        # par.force_dt = 1.111
        if par.transition_only_datnum == 2136:
            par.force_amp = 0.52

    # Do processing
    general = AnalysisGeneral(params_list=all_params, calculate=False, overwrite_entropy=True, overwrite_transition=True)
    run_processing(general)
    for par in all_params:
        calculate_new_sf_only(entropy_datnum=par.entropy_datnum, save_name=sf_name,
                              dt=par.force_dt, amp=par.force_amp,
                              from_square_transition=par.sf_from_square_transition,
                              transition_datnum=par.transition_only_datnum,
                              fit_name=name)

    # Plotting
    # plot_stacked_square_heated(LONG_GAMMA, save_name='test')

    # fig = get_integrated_fig(get_dats(general.entropy_datnums), title_append=f'comparing scaling factors')
    # for int_name in ['scaled_dT', 'fixed_dT']:
    #     fig.add_trace(get_integrated_trace(dats=get_dats(general.entropy_datnums),
    #                                    x_func=general.x_func, x_label=general.x_label,
    #                                    trace_name=int_name,
    #                                    fit_name=name,
    #                                    int_info_name=int_name, SE_output_name=name))
    # fig.show()


    # fig = plot_transition_values(general.transition_datnums, save_name=name, general=general, param_name='theta',
    #                        transition_only=True)
    # fig = plot_transition_values(general.transition_datnums, save_name=name, general=general, param_name='g',
    #                        transition_only=True)
    # fig = plot_transition_values(general.transition_datnums, save_name=name, general=general, param_name='amp',
    #                        transition_only=True)

    # fig = plot_transition_values(general.transition_datnums, save_name=name, general=general, param_name='theta',
    #                              transition_only=True, show=False)
    # line = lm.models.LinearModel()
    # dats = get_dats(general.transition_datnums)
    # dats = [dat for dat in dats if dat.Logs.fds['ESC'] < -290]
    # x = np.array([dat.Logs.fds['ESC'] for dat in dats])
    # thetas = np.array([dat.Transition.get_fit(name=name).best_values.theta for dat in dats])
    # fit = line.fit(data=thetas, x=x)
    # plotter = OneD(dats=dats)
    # fig.add_trace(plotter.trace(fit.eval(x=(x := np.linspace(-380, -180, 101))), x=x, mode='lines', name='Fit'))
    # print(fit.fit_report())
    # fig.show()
