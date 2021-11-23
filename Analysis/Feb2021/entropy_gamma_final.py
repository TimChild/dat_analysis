"""
Hopefully all the final analysis of Gamma Broadened Entropy measurements

Sep 21 -- How naive ^^ -- Definitely used for a lot of final analysis, but a super messy file now given how long final
analysis dragged on.
Functions here got too complicated to be useful in the future.
"""
from dat_analysis.dat_object.make_dat import get_dat, get_dats, DatHDF
from dat_analysis.plotting.plotly.hover_info import HoverInfo, _additional_data_dict_converter
from dat_analysis.plotting.plotly.dat_plotting import OneD

from dat_analysis.analysis_tools.transition import do_transition_only_calc, linear_fit_thetas
from dat_analysis.analysis_tools.csq_mapping import setup_csq_dat, calculate_csq_map
from dat_analysis.plotting.plotly.common_plots.transition import plot_transition_values
from dat_analysis.plotting.plotly.common_plots.entropy import plot_fit_integrated_comparison, get_integrated_trace, \
    get_integrated_fig
from dat_analysis.analysis_tools.entropy import GammaAnalysisParams, save_gamma_analysis_params_to_dat, do_entropy_calc, \
    calculate_new_sf_only
import dat_analysis.useful_functions as U
from dat_analysis.plotting.plotly.plotly_util import make_slider_figure
from dat_analysis.dat_object.attributes.SquareEntropy import entropy_signal
from dat_analysis.dat_analysis.characters import DELTA

from typing import List, Optional, Callable, Union, Tuple, Iterable
from progressbar import progressbar
import logging
import lmfit as lm
import plotly.io as pio
import numpy as np
from functools import partial
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

pio.renderers.default = 'browser'
logger = logging.getLogger(__name__)


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


def run_processing(analysis_general: AnalysisGeneral, multiprocessed=True):
    a = analysis_general
    if a.calculate:
        if multiprocessed:
            with ProcessPoolExecutor() as pool:
                if a.csq_datnums:
                    list(pool.map(partial(setup_csq_dat, overwrite=a.overwrite_csq), a.csq_datnums))
                    print(f'Done Setting up CSQonly dats')

                list(pool.map(partial(process_single, overwrite_transition=a.overwrite_transition,
                                      overwrite_entropy=a.overwrite_entropy), a.params_list))
        else:
            if a.csq_datnums:
                for datnum in progressbar(a.csq_datnums):
                    setup_csq_dat(csq_datnum=datnum, overwrite=a.overwrite_csq)
                print(f'Done Setting up CSQonly dats')
            for param in progressbar(a.params_list):
                process_single(param, overwrite_transition=a.overwrite_transition,
                               overwrite_entropy=a.overwrite_entropy)

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

            ]
            hover_funcs, template = _additional_data_dict_converter(hover_infos)

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
                                       save_name=save_name,
                                       int_info_name=int_info_name, SE_output_name=save_name))

    fig2 = plot_fit_integrated_comparison(dats, x_func=general.x_func, x_label=general.x_label,
                                          int_info_name=int_info_name, fit_name=save_name,
                                          plot=True)
    if show:
        fig.show()
    return fig


def binned_y_fig(dat: DatHDF, save_name: str, num_chunks: int, which='entropy',
                 integrated_info_name: Optional[str] = None,  # Only if not the same as save_name and want integrated
                 ):
    """plotting data binned in y axis (after centering)"""

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

    fig = make_slider_figure(datas=datas, xs=x,
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


def plot_gamma_dcbias(datnums: List[int], save_name: str, show_each_data=True):
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
                         title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: DCbias in Gamma broadened')
    line = lm.models.LinearModel()
    params = line.make_params()
    for dat in dats[1::2]:
        params['slope'].value = dat.Transition.avg_fit.best_values.lin
        params['intercept'].value = dat.Transition.avg_fit.best_values.const
        fig.add_trace(plotter.trace(x=dat.Transition.avg_x,
                                    data=dat.Transition.avg_data - line.eval(params=params, x=dat.Transition.avg_x),
                                    name=f'Dat{dat.datnum}: Bias={dat.Logs.fds["HO1/10M"] / 10:.1f}nA',
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
                              csq_map: bool = True,
                              ) -> List[GammaAnalysisParams]:
    all_params = []
    for ed, td, cs in zip(entropy_datnums, transition_datnums, csq_datnums):
        all_params.append(
            GammaAnalysisParams(
                experiment_name='febmar21',
                save_name=save_name,
                entropy_datnum=ed, transition_only_datnum=td, csq_datnum=cs,
                setpoint_start=0.008,
                entropy_transition_func_name='i_sense', entropy_fit_width=None, entropy_data_rows=entropy_data_rows,
                force_dt=force_dt, force_amp=force_amp,
                sf_from_square_transition=sf_from_square_transition,
                force_theta=force_theta, force_gamma=force_gamma,  # Applies to entropy and transition only
                transition_func_name='i_sense_digamma', transition_fit_width=transition_fit_width,
                transition_center_func_name='i_sense_digamma', transition_data_rows=transition_data_rows,
                csq_mapped=csq_map,
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
                experiment_name='febmar21',
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


def _temp_calculate_from_non_csq():
    """Used this to recalculate the csq mapped output (used same centers because they are not really going to change
    for gamma broadened anyway)"""
    dat = get_dat(2170)
    from dat_analysis.analysis_tools.csq_mapping import calculate_csq_map
    calculate_csq_map(2170, None, 2172)
    inps = dat.SquareEntropy.get_Inputs(x_array=dat.Data.get_data('x'), i_sense=dat.Data.get_data('csq_mapped'),
                                        save_name='forced_theta_linear')
    pp = dat.SquareEntropy.get_ProcessParams(name='forced_theta_linear_non_csq', save_name='forced_theta_linear')

    out = dat.SquareEntropy.get_Outputs(name='forced_theta_linear', inputs=inps, process_params=pp)


def dT_from_linear(base_dT, base_esc, new_esc, lever_slope, lever_intercept) -> float:
    """
    Calculated expected dT for given ESC using a single well determined base dT at an ESC and then scaled based on the
    lever arm slope

    base_dT=1.158, base_esc=-309.45, lever_slope=0.00304267, lever_intercept=5.12555961
    Args:
        base_dT (): Single well determined dT
        base_esc (): The ESC of that well determined dT
        new_esc (): The ESC to predict the new dT at
        lever_slope (): Lever arm (or theta) slope
        lever_intercept (): Lever arm (or theta) intercept

    Returns:

    """
    line = lm.models.LinearModel()
    line_pars = line.make_params()
    line_pars['slope'].value = lever_slope
    line_pars['intercept'].value = lever_intercept
    theta_for_base_dt = line.eval(x=base_esc,
                                  params=line_pars)
    theta = line.eval(params=line_pars, x=new_esc)
    dT = base_dT * theta / theta_for_base_dt  # Scale dT with same proportion as theta (or lever arm)
    return dT


############## Relevant Dats ##################
# TODO: Should combine the 2213/2216 (both just long 50kBT scans)
# TODO: Should combine the GAMMA_25 with 2170 (all are 25kBT)
DCbias = list(range(2143, 2155 + 1))

VS_GAMMA = list(range(2095, 2142, 2))
VS_GAMMA_Tonly = list(range(2096, 2142 + 1, 2))
# VS_GAMMA_CSQ = list(range(2185, 2208 + 1))
VS_GAMMA_CSQ = [2189, 2190, 2191, 2192, 2193, 2194, 2195, 2196, 2197, 2198, 2199, 2200, 2201, 2202, 2203, 2205, 2205,
                2206, 2207, 2208, 2187, 2201, 2202, 2203]

LONG_GAMMA = [2164, 2167, 2170, 2213, 2216]
LONG_GAMMA_Tonly = [2165, 2168, 2171, 2214, 2217]
LONG_GAMMA_csq = [2173, 2174, 2175, 2215, 2218]

GAMMA_25 = [2131, 2160, 2170, 2176, 2182]

DC_GAMMA = list(range(2219, 2230))  # DCbias scans in gamma broadened (~25kBT) to make sure that heating not
# broadening transition more than it should
#################################################

MORE_SYMMETRIC_LONG2 = list(range(7322, 7361 + 1, 2)) + list(range(7378, 7399 + 1, 2)) + list(range(7400, 7421 + 1, 2))
MORE_SYMMETRIC_LONG_Tonly2 = list(range(7323, 7361 + 1, 2)) + list(range(7379, 7399 + 1, 2)) + list(
    range(7401, 7421 + 1, 2))

MORE_SYMMETRIC_LONG3 = list(range(7847, 7866 + 1, 2)) + list(range(7867, 7900 + 1, 2))
MORE_SYMMETRIC_LONG_Tonly3 = list(range(7848, 7866 + 1, 2)) + list(range(7868, 7900 + 1, 2))

# All entropy dats in more symmetric setting of May CD of FebMar21 notebook, before and after ACC Zap
ALL_DATS = list(range(7322, 7361 + 1, 2)) + list(range(7378, 7399 + 1, 2)) + list(range(7400, 7421 + 1, 2)) + \
           list(range(7845, 7866 + 1, 2)) + list(range(7867, 7900 + 1, 2)) + list(range(7901, 7934 + 1, 2))
ALL_DATS_Tonly = [x + 1 for x in ALL_DATS]
# ALL_DATS = list(range(7847, 7866 + 1, 2)) + list(range(7867, 7900 + 1, 2)) + list(range(7901, 7934 + 1, 2))
# ALL_DATS_Tonly = [x + 1 for x in ALL_DATS]
if __name__ == '__main__':
    # all_params = make_long_analysis_params(LONG_GAMMA, LONG_GAMMA_Tonly, LONG_GAMMA_csq, save_name='test',
    #                                   entropy_data_rows=(0, 10), transition_data_rows=(0, 10))
    name = 'forced_theta_linear'
    # name = 'forced_gamma_zero'
    # name = 'forced_theta_linear_non_csq'
    # name = 'ftlncsq'
    # name = 'forced_gamma_zero_non_csq'
    # sf_name = 'fixed_dT'
    # sf_name = 'scaled_dT'
    sf_name = name
    # all_params = make_vs_gamma_analysis_params(LONG_GAMMA, LONG_GAMMA_Tonly, save_name=name,
    #                                            force_theta=-1, force_gamma=None,
    #                                            sf_from_square_transition=True, width=None)

    # edats, csqdats = [get_dats(datnums) for datnums in [VS_GAMMA, VS_GAMMA_CSQ]]
    # csq_dict = {c.Logs.dacs['ESC']: c for c in csqdats}
    # csqs_in_entropy_order = [csq_dict[n] if (n := dat.Logs.dacs['ESC']) in csq_dict else csq_dict[
    #     min(csq_dict.keys(), key=lambda k: abs(k - n))] for dat in edats]
    # all_params = make_long_analysis_params(VS_GAMMA, VS_GAMMA_Tonly,
    #                                        VS_GAMMA_CSQ, save_name=name,
    # all_params = make_long_analysis_params(LONG_GAMMA[:3], LONG_GAMMA_Tonly[:3],
    #                                        LONG_GAMMA_csq[:3], save_name=name,
    #                                        force_theta=-1, force_gamma=None,  # Theta set below
    #                                        transition_fit_width=500,
    #                                        force_dt=None,
    #                                        sf_from_square_transition=False,
    #                                        csq_map=False
    #                                        )
    all_params = make_long_analysis_params([2164, 2167], [2165, 2168],
                                           [2166, 2169], save_name=name,
                                           force_theta=-1, force_gamma=None,  # Theta set below
                                           transition_fit_width=500,
                                           force_dt=None,
                                           sf_from_square_transition=False,
                                           csq_map=False
                                           )
    # all_params = make_long_analysis_params(VS_GAMMA, VS_GAMMA_Tonly,
    #                                        VS_GAMMA_CSQ, save_name=name,
    #                                        force_theta=-1, force_gamma=None,
    #                                        transition_fit_width=500,
    #                                        force_dt=None,
    #                                        sf_from_square_transition=False,
    #                                        csq_map=False,
    #                                        )

    # Setting theta according to linear fit in weakly coupled regime with gamma = 0
    line = lm.models.LinearModel()
    line_pars = line.make_params()
    # line_pars['slope'].value = 0.00348026
    # line_pars['intercept'].value = 5.09205057

    ########## 100mK (dats 2095 - 2216 csq mapped)
    line_pars['slope'].value = 3.3933e-5 * 100
    line_pars['intercept'].value = 0.05073841 * 100
    theta_for_dt = line.eval(x=-309.45,
                             params=line_pars)  # Dat2101 is the setting where DCbias was done and dT is defined
    base_dt = 1.158

    # ########## 100mK (dats 2095 - 2216 NON-csq mapped)
    # line_pars['slope'].value = 3.4648e-5 * 100
    # line_pars['intercept'].value = 0.05087656 * 100
    # theta_for_dt = line.eval(x=-309.45,
    #                          params=line_pars)  # Dat2101 is the setting where DCbias was done and dT is defined
    # base_dt = 1.149
    # ######### 100mK
    # line_pars['slope'].value = 0.08866821
    # line_pars['intercept'].value = 64.3754
    # theta_for_dt = line.eval(x=-265, params=line_pars)  # Dat2101 is the setting where DCbias was done and dT is defined
    # base_dt = 0.0127 * 1000
    #####

    ######## 50mK
    # line_pars['slope'].value = 1.086e-4*1000
    # line_pars['intercept'].value = 0.05184*1000
    # theta_for_dt = line.eval(x=-270, params=line_pars)  # dT is calculated at -270mV ESC
    # base_dt = 22.8

    for par in all_params:
        dat = get_dat(par.transition_only_datnum)
        theta = line.eval(params=line_pars, x=dat.Logs.fds['ESC'])
        par.force_theta = theta
        par.force_dt = base_dt * theta / theta_for_dt  # Scale dT with same proportion as theta

        # par.force_dt = base_dt
        # if par.transition_only_datnum == 2136:
        #     par.force_amp = 0.52

    # Do processing
    general = AnalysisGeneral(params_list=all_params, calculate=True, overwrite_entropy=True,
                              overwrite_transition=True)
    run_processing(general, multiprocessed=True)
    for par in all_params:
        calculate_new_sf_only(entropy_datnum=par.entropy_datnum, save_name=sf_name,
                              dt=par.force_dt, amp=par.force_amp,
                              from_square_transition=par.sf_from_square_transition,
                              transition_datnum=par.transition_only_datnum,
                              fit_name=name)

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
    # # plotting
    # figs = plot_stacked_square_heated(LONG_GAMMA, save_name=name)
    # for i, fig in enumerate(figs):
    #     fig.write_html(f'temp{i}.html')

    # fig = get_integrated_fig(get_dats(general.entropy_datnums), title_append=f'comparing scaling factors')
    # # for int_name in ['scaled_dT', 'fixed_dT']:
    # for int_name in ['scaled_dT']:
    #     fig.add_trace(get_integrated_trace(dats=get_dats(general.entropy_datnums),
    #                                        x_func=general.x_func, x_label=general.x_label,
    #                                        trace_name=int_name,
    #                                        save_name=name,
    #                                        int_info_name=int_name, SE_output_name=name))

    # all_dats = get_dats(ALL_DATS)
    # dat_chunks = [[dat for dat in all_dats if lower <= dat.datnum <= upper]
    #               for lower, upper in [(7322, 7399), (7400, 7421), (7847, 7866), (7867, 7900), (7901, 7934)]]
    # fig = get_integrated_fig(get_dats(general.entropy_datnums), title_append=f'comparing scaling factors')
    # for dats in dat_chunks:
    #     if len(dats) > 0:
    #         fig.add_trace(get_integrated_trace(dats=dats,
    #                                            x_func=general.x_func, x_label=general.x_label,
    #                                            trace_name=f'Dats{min([dat.datnum for dat in dats])}-'
    #                                                       f'{max([dat.datnum for dat in dats])}',
    #                                            save_name=name,
    #                                            int_info_name='scaled_dT', SE_output_name=name))
    # fig.show()
    #
    figs = []
    figs.append(plot_transition_values(general.transition_datnums, save_name=name, general=general, param_name='theta',
                                       transition_only=True))
    figs.append(plot_transition_values(general.transition_datnums, save_name=name, general=general, param_name='g',
                                       transition_only=True))
    figs.append(plot_transition_values(general.transition_datnums, save_name=name, general=general, param_name='amp',
                                       transition_only=True))

    # for i, fig in enumerate(figs):
    #     fig.write_html(f'figs/temp{i}.html')

    # transition_dats = get_dats(ALL_DATS_Tonly)
    # fits = []
    # for esc_range in [(-300, -230), (-280, -230), (-280, -250), (-290, -257)]:
    #     fits.append(
    #         linear_fit_thetas(dats=transition_dats, fit_name='forced_gamma_zero',
    #                           filter_func=lambda dat: True if esc_range[0] < dat.Logs.fds['ESC'] < esc_range[
    #                               1] else False,
    #                           show_plots=False)
    #     )
    # single_fit = linear_fit_thetas(dats=transition_dats, fit_name='forced_gamma_zero',
    #                                filter_func=lambda dat: True if (-282 < dat.logs.fds['esc'] < -265) or (-255 < dat.logs.fds['esc'] < -235) else False,
    #                                show_plots=False)

    from dat_analysis.analysis_tools.general_fitting import calculate_fit

    tdats = get_dats(VS_GAMMA_Tonly)
    linear_fit_thetas(dats=tdats, fit_name='forced_gamma_zero',
                      filter_func=lambda dat: True if dat.Logs.dacs["ESC"] < -285 else False,
                      show_plots=True,
                      sweep_gate_divider=100)
    # print('done')

    p1d = OneD(dat=None)
    fig = p1d.figure(xlabel='ESC /mV', ylabel='dT')

    dats = get_dats(VS_GAMMA)
    escs = [dat.Logs.dacs['ESC'] for dat in dats]
    # dts = [dat.SquareEntropy.get_fit(fit_name='forced_gamma_zero_non_csq_hot').best_values.theta -
    #        dat.SquareEntropy.get_fit(fit_name='forced_gamma_zero_non_csq_cold').best_values.theta for dat in dats]

    dts = [dat.Entropy.get_integration_info(name='forced_theta_linear_non_csq').dT for dat in dats]

    line = lm.models.LinearModel()
    fit = calculate_fit(x=np.array(escs[:11]), data=np.array(dts[:11]), params=line.make_params(), func=line.func)

    fig.add_trace(p1d.trace(x=escs, data=dts, text=[dat.datnum for dat in dats]))
    fig.add_trace(p1d.trace(x=escs, data=fit.eval_fit(x=np.array(escs)), mode='lines', name='fit'))
    fig.show()
