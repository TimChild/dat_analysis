"""Hopefully all the final analysis of Gamma Broadened Entropy measurements"""
from src.DatObject.Make_Dat import get_dat, get_dats, DatHDF
from src.Plotting.Plotly.PlotlyUtil import additional_data_dict_converter, HoverInfo
from src.Dash.DatPlotting import OneD

from Analysis.Feb2021.common import plot_fit_integrated_comparison, do_entropy_calc, do_transition_only_calc, \
    calculate_csq_map, setup_csq_dat, get_integrated_trace, get_integrated_fig, transition_trace, \
    transition_fig
from src.AnalysisTools.gamma_entropy import GammaAnalysisParams, save_gamma_analysis_params_to_dat
import src.UsefulFunctions as U
from src.Plotting.Plotly.PlotlyUtil import get_slider_figure
from src.DatObject.Attributes.SquareEntropy import entropy_signal
from src.Characters import DELTA

from typing import List, Optional, Callable, Union, Tuple, Iterable
import logging
import lmfit as lm
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


def process_single(pars: GammaAnalysisParams, overwrite_transition=False, overwrite_entropy=False):
    """Does all the processing necessary for a single GammaAnalysisParams (i.e. csq mapping, transition fitting,
    entropy fitting and setting integration info"""
    if pars.csq_mapped:
        calculate_csq_map(pars.entropy_datnum, csq_datnum=pars.csq_datnum, overwrite=overwrite_entropy)
        if pars.transition_only_datnum:
            calculate_csq_map(pars.transition_only_datnum, csq_datnum=pars.csq_datnum,
                              overwrite=overwrite_transition)

    if pars.transition_only_datnum is not None:
        print(f'Dat{pars.transition_only_datnum}')
        if pars.save_name + '_cold' not in get_dat(pars.transition_only_datnum).Transition.fit_names \
                or overwrite_transition:
            do_transition_only_calc(datnum=pars.transition_only_datnum, save_name=pars.save_name,
                                    theta=pars.force_theta, gamma=pars.force_gamma,
                                    center_func=pars.transition_center_func_name,
                                    width=pars.transition_fit_width, t_func_name=pars.transition_func_name,
                                    csq_mapped=pars.csq_mapped, data_rows=pars.transition_data_rows,
                                    overwrite=overwrite_transition)
    if pars.save_name not in get_dat(pars.entropy_datnum).Entropy.fit_names or overwrite_entropy:
        print(f'Dat{pars.entropy_datnum}')
        do_entropy_calc(pars.entropy_datnum, save_name=pars.save_name,
                        setpoint_start=pars.setpoint_start,
                        t_func_name=pars.entropy_transition_func_name,
                        csq_mapped=pars.csq_mapped,
                        theta=pars.force_theta, gamma=pars.force_gamma, width=pars.entropy_fit_width,
                        data_rows=pars.transition_data_rows,
                        overwrite=overwrite_entropy)

    calculate_new_sf_only(pars.entropy_datnum, pars.save_name,
                          dt=pars.force_dt, amp=pars.force_amp,
                          from_square_transition=pars.sf_from_square_transition,
                          transition_datnum=pars.transition_only_datnum,
                          fit_name=pars.save_name)

    # Save to HDF
    d = get_dat(pars.entropy_datnum)
    save_gamma_analysis_params_to_dat(d, analysis_params=pars, name=pars.save_name)
    return None


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


def plot_stacked_square_heated(datnums: List[int], save_name: str, plot=True):
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

    figs = []
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
        if plot:
            fig.show()
        figs.append(fig)
    return figs


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


def binned_y_fig(dat: DatHDF, save_name: str, num_chunks: int, which='entropy',
                 integrated_info_name: Optional[str] = None,  # Only if not the same as save_name and want integrated
                 ):
    """Plotting data binned in y axis (after centering)"""

    out = dat.SquareEntropy.get_Outputs(name=save_name, check_exists=True)
    x = out.x
    all_centers = out.centers_used
    all_data = out.cycled  # Per row data
    y = dat.Data.get_data('y')

    binned = centered_y_bin(x, all_data, all_centers, num_bins=num_chunks,
                            y=y)
    binned_data = binned.binned_data
    bin_centers = binned.y_centers

    if which == 'entropy':
        datas = [entropy_signal(d) for d in binned_data]
        ylabel = f'{DELTA}I/nA'
    elif which == 'transition':
        datas = binned_data
        ylabel = f'Current /nA'
    elif which == 'integrated':
        int_name = integrated_info_name if integrated_info_name else save_name
        int_info = dat.Entropy.get_integration_info(int_name)
        datas = [int_info.integrate(entropy_signal(d)) for d in binned_data]
        ylabel = f'Entropy /kB'
    else:
        raise NotImplementedError

    fig = get_slider_figure(datas=datas, xs=x,
                            ids=[v for v in bin_centers],
                            titles=[f'Dat{dat.datnum}: Bin centered at {v:.2f}' for v in bin_centers],
                            xlabel=dat.Logs.xlabel, ylabel=ylabel)
    return fig


@dataclass
class BinnedY:
    """For storing return of center_y_bin"""
    binned_data: List[np.ndarray]
    avg_xs: List[np.ndarray]
    y_centers: List[float]  # The center of the binned data in y-axis (or row num if y data not passed)


def centered_y_bin(x: np.ndarray, data: np.ndarray,
                   centers: Union[np.ndarray, Iterable], num_bins: int,
                   y: Optional[np.ndarray] = None) -> BinnedY:
    """
    Bin 2D data in y axis after centering using centers.
    Args:
        x (): X array for all
        data (): All data first
        centers ():
        num_bins ():
        y ():

    Returns:

    """
    if y is None:
        y = range(data.shape[0])
    bin_size = int(np.floor(data.shape[0] / num_bins))
    binned_data = []
    avg_xs = []
    bin_centers = []
    for i in range(int(data.shape[0] / bin_size)):
        s = np.s_[i * bin_size:i * bin_size + bin_size]
        d = data[s]
        cs = centers[s]
        bin_centers.append(float(np.nanmean(y[s])))
        binned, x = U.mean_data(x, d, centers=cs, return_x=True)
        binned_data.append(binned)
        avg_xs.append(x)

    return BinnedY(binned_data=binned_data, avg_xs=avg_xs, y_centers=bin_centers)


def plot_gamma_dcbias(datnums: List[int], save_name: str, show_each_data = True):
    """
    Makes a figure for Theta vs DCbias with option to show the data which is being used to obtain thetas
    Args:
        datnums (): Datnums that form DCbias measurement (i.e. repeats at fixed Biases)
        save_name (): Name of fits etc to be loaded (must already exist)
        show_each_data (): Whether to show the fit for each dataset (i.e. to check everything looks good)

    Returns:
        go.Figure: A plotly figure of Theta vs DCbias
    """
    # if calculate:
    #     with ProcessPoolExecutor() as pool:
    #         list(pool.map(partial(do_transition_only_calc, save_name=save_name, theta=theta, gamma=None, width=600,
    #                               t_func_name='i_sense_digamma', overwrite=False), GAMMA_DCbias))
    dats = get_dats(datnums)
    plotter = OneD(dats=dats)
    # fig = plotter.figure(ylabel='Current /nA',
    #                      title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: DCbias in Gamma broadened' )
    # dat_pairs = np.array(dats).reshape((-1, 2))
    # line = lm.models.LinearModel()
    # params = line.make_params()
    # for ds in dat_pairs:
    #     for dat, color in zip(ds, ['blue', 'red']):
    #         params['slope'].value = dat.Transition.avg_fit.best_values.lin
    #         params['intercept'].value = dat.Transition.avg_fit.best_values.const
    #         fig.add_trace(plotter.trace(x=dat.Transition.avg_x, data=dat.Transition.avg_data-line.eval(params=params, x=dat.Transition.avg_x),
    #                                     name=f'Dat{dat.datnum}: Bias={dat.Logs.fds["HO1/10M"]/10:.1f}nA',
    #                                     mode='lines',
    #                                     trace_kwargs=dict(line=dict(color=color)),
    #                                     ))

    fig = plotter.figure(ylabel='Current /nA',
                         title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: DCbias in Gamma broadened' )
    line = lm.models.LinearModel()
    params = line.make_params()
    for dat in dats[1::2]:
        params['slope'].value = dat.Transition.avg_fit.best_values.lin
        params['intercept'].value = dat.Transition.avg_fit.best_values.const
        fig.add_trace(plotter.trace(x=dat.Transition.avg_x, data=dat.Transition.avg_data-line.eval(params=params, x=dat.Transition.avg_x),
                                    name=f'Dat{dat.datnum}: Bias={dat.Logs.fds["HO1/10M"]/10:.1f}nA',
                                    mode='lines',
                                    ))
    fig.show()


def make_long_analysis_params(entropy_datnums, transition_datnums, csq_datnums,
                              save_name: str,
                              entropy_data_rows: Tuple[Optional[int], Optional[int]] = (None, None),
                              transition_data_rows: Tuple[Optional[int], Optional[int]] = (None, None),
                              transition_fit_width: Optional[float] = None,
                              force_theta=None, force_gamma: Optional[int] = None,
                              force_dt: Optional[float] = None, force_amp: Optional[float] = None,
                              sf_from_square_transition: bool = True,
                              ) -> List[GammaAnalysisParams]:
    all_params = []
    for ed, td, cs in zip(entropy_datnums, transition_datnums, csq_datnums):
        all_params.append(
            GammaAnalysisParams(
                save_name=save_name,
                entropy_datnum=ed, transition_only_datnum=td, csq_datnum=cs,
                setpoint_start=0.008,
                entropy_transition_func_name='i_sense', entropy_fit_width=None, entropy_data_rows=entropy_data_rows,
                force_dt=force_dt, force_amp=force_amp,
                sf_from_square_transition=sf_from_square_transition,
                force_theta=force_theta, force_gamma=force_gamma,  # Applies to entropy and transition only
                transition_func_name='i_sense_digamma', transition_fit_width=transition_fit_width,
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
                                               force_theta=-1, force_gamma=None,
                                               sf_from_square_transition=False, width=600)
    # all_params = make_long_analysis_params(LONG_GAMMA, LONG_GAMMA_Tonly, LONG_GAMMA_csq, save_name=name,
    #                                        force_theta=None, force_gamma=None,  # Theta set below
    #                                        transition_fit_width=500,
    #                                        force_dt=1.11,
    #                                        sf_from_square_transition=False,
    #                                        )

    # Setting theta according to linear fit in weakly coupled regime with gamma = 0
    line = lm.models.LinearModel()
    line_pars = line.make_params()
    line_pars['slope'].value = 0.00348026
    line_pars['intercept'].value = 5.09205057
    theta_for_dt = line.eval(x=-339.36, params=line_pars)  # Dat2101 is the setting where DCbias was done and dT is defined
    base_dt = 1.111
    for par in all_params:
        dat = get_dat(par.transition_only_datnum)
        theta = line.eval(params=line_pars, x=dat.Logs.fds['ESC'])
        par.force_theta = theta
        par.force_dt = base_dt * (theta) / theta_for_dt  # Scale dT with same proportion as theta
        # par.force_dt = base_dt
        if par.transition_only_datnum == 2136:
            par.force_amp = 0.52

    # Do processing
    general = AnalysisGeneral(params_list=all_params, calculate=True, overwrite_entropy=False,
                              overwrite_transition=False)
    run_processing(general)
    for par in all_params:
        calculate_new_sf_only(entropy_datnum=par.entropy_datnum, save_name=sf_name,
                              dt=par.force_dt, amp=par.force_amp,
                              from_square_transition=par.sf_from_square_transition,
                              transition_datnum=par.transition_only_datnum,
                              fit_name=name)
    #
    # name = 'temp'
    # params = make_long_analysis_params([2164], transition_datnums=[2165],csq_datnums=[2166], save_name=name,
    #                                        force_theta=3.9, force_gamma=None,
    #                                        sf_from_square_transition=False,
    #                                        transition_fit_width=500)[0]
    # params.force_dt = 1.11
    #
    # setup_csq_dat(2166)
    # process_single(params, overwrite_transition=False, overwrite_entropy=False)

    # dat = get_dat(2213)
    #
    # out = dat.SquareEntropy.get_Outputs(name=name, check_exists=True)
    # x = out.x
    # all_centers = out.centers_used
    # all_data = out.cycled  # Per row data
    # y = dat.Data.get_data('y')
    #
    # binned = centered_y_bin(x, all_data, all_centers, num_bins=10,
    #                         y=y)
    #
    # int_info = dat.Entropy.get_integration_info(name=name)
    # plotter = OneD(dat=dat)
    # integrated_entropies = [int_info.integrate(entropy_signal(d))[-1] for d in binned.binned_data]
    # x = binned.y_centers
    # fig = plotter.plot(data=integrated_entropies, x=x, xlabel=dat.Logs.ylabel, ylabel='Entropy /kB', title=f'Dat{dat.datnum}: Integrated entropy for binned data')
    # fig.show()
    # # fits = dat.SquareEntropy.get_row_fits(name='temp')
    #
    # # for n in ['entropy', 'transition', 'integrated']:
    # for n in ['integrated']:
    #     fig = binned_y_fig(dat, save_name=name, num_chunks=10, which=n, integrated_info_name=name)
    #     fig.show()
    #     if n == 'integrated':
    #         fig.write_html(f'figs/Dat{dat.datnum}-slider_integrated.html')
    #
    # # Plotting
    # figs = plot_stacked_square_heated(LONG_GAMMA, save_name=name)
    # for i, fig in enumerate(figs):
    #     fig.write_html(f'temp{i}.html')

    fig = get_integrated_fig(get_dats(general.entropy_datnums), title_append=f'comparing scaling factors')
    # for int_name in ['scaled_dT', 'fixed_dT']:
    for int_name in ['scaled_dT']:
        fig.add_trace(get_integrated_trace(dats=get_dats(general.entropy_datnums),
                                   x_func=general.x_func, x_label=general.x_label,
                                   trace_name=int_name,
                                   fit_name=name,
                                   int_info_name=int_name, SE_output_name=name))
    fig.show()

    fig = plot_transition_values(general.transition_datnums, save_name=name, general=general, param_name='theta',
                           transition_only=True)
    fig = plot_transition_values(general.transition_datnums, save_name=name, general=general, param_name='g',
                           transition_only=True)
    fig = plot_transition_values(general.transition_datnums, save_name=name, general=general, param_name='amp',
                           transition_only=True)

    fig = plot_transition_values(general.transition_datnums, save_name=name, general=general, param_name='theta',
                                 transition_only=True, show=False)
    line = lm.models.LinearModel()
    dats = get_dats(general.transition_datnums)
    dats = [dat for dat in dats if dat.Logs.fds['ESC'] < -290]
    x = np.array([dat.Logs.fds['ESC'] for dat in dats])
    thetas = np.array([dat.Transition.get_fit(name=name).best_values.theta for dat in dats])
    print('done')
    # fit = line.fit(data=thetas, x=x)
    # plotter = OneD(dats=dats)
    # fig.add_trace(plotter.trace(fit.eval(x=(x := np.linspace(-380, -180, 101))), x=x, mode='lines', name='Fit'))
    # print(fit.fit_report())
    # fig.show()

# import h5py
# def test(f: h5py.File, shape=(1000,), dtype='S10'):
#     if 'test' in f.keys():
#         del f['test']
#     f.create_dataset('test', shape, dtype=dtype)
#     store = [b'blablablablabla']*70
#     f['test'][:70] = store
#     read = f['test'][:]
#     return read
#     # del f['test']
#

