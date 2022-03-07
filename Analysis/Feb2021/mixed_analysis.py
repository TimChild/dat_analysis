"""
Sep 21 -- This file is a monster and definitely contains some useful functions. Need to go back through this one earlier
in the day
TODO: Extract useful functions from here!
"""
from __future__ import annotations
import plotly.io as pio
import plotly.graph_objects as go
import copy
import lmfit as lm
import numpy as np
from progressbar import progressbar
from typing import List, Optional, Union, Callable, Tuple, TYPE_CHECKING
import logging
from deprecation import deprecated

from dat_analysis.core_util import get_data_index, Data1D, Data2D
from dat_analysis.characters import PM
from dat_analysis.dat_object.attributes.entropy import IntegrationInfo
from dat_analysis.dat_object.make_dat import get_dat, DatHDF, get_dats
from dat_analysis.analysis_tools import NrgUtil, NRGParams, setup_csq_dat, calculate_csq_map, csq_map_data
from dat_analysis.analysis_tools.general_fitting import calculate_fit, FitInfo
from dat_analysis.plotting.plotly import OneD, TwoD
from dat_analysis.characters import DELTA
from dat_analysis.useful_functions import mean_data
from dat_analysis.hdf_util import NotFoundInHdfError
from dat_analysis.analysis_tools.transition import linear_fit_thetas
from Analysis.Feb2021.entropy_gamma_final import dT_from_linear

if TYPE_CHECKING:
    from dat_analysis.dat_object.attributes.square_entropy import Output

pio.renderers.default = 'browser'

p1d = OneD(dat=None)
p2d = TwoD(dat=None)


def _center_func(dat: DatHDF) -> bool:
    return True if dat.Logs.dacs['ESC'] < -230 else False


def figure_1_add_NRG_fit_to_gamma_dndt() -> go.Figure:
    fit_name = 'forced_theta_linear_non_csq'
    dat = get_dat(2170)

    init_params = NRGParams(
        gamma=23.4352,
        theta=4.396,
        center=78.4,
        amp=0.675,
        lin=0.00121,
        const=7.367,
        lin_occ=0.0001453,
    )

    out = dat.SquareEntropy.get_Outputs(name=fit_name)
    x = out.x
    z = out.average_entropy_signal

    fig = p1d.figure(xlabel='V_P', ylabel=f'{DELTA}I (nA)')
    fig.add_trace(p1d.trace(x=x, data=z, name='Data', mode='lines'))

    dndt_init_params = copy.copy(init_params)
    dndt_init_params.amp = 0.0001  # To rescale the arbitrary NRG dndt scale
    dndt_init_params.theta = dat.SquareEntropy.get_fit(fit_name=fit_name).best_values.theta
    nrg_fitter = NrgUtil(inital_params=dndt_init_params)
    fit = nrg_fitter.get_fit(x=x, data=z, which_data='dndt')

    fig.add_trace(p1d.trace(x=x, data=fit.eval_fit(x=x), mode='lines', name='Fit'))
    fig.add_trace(p1d.trace(x=x, data=fit.eval_init(x=x), mode='lines', name='Init'))

    fig.show()
    return fig


def check_nrg_fit(datnum, exisiting_fit='forced_theta_linear'):
    """

    Args:
        datnum ():
        exisiting_fit (): To get the Theta value from

    Returns:

    """
    dat = get_dat(datnum)
    x = dat.Data.x
    data = dat.Data.get_data('csq_mapped_avg')
    theta = dat.Transition.get_fit(name=exisiting_fit).best_values.theta
    nrg_fitter = NrgUtil(NRGParams(gamma=1, theta=theta, center=0, amp=1.5, lin=0.003, const=0, lin_occ=0))
    fit = nrg_fitter.get_fit(x=x, data=data)
    fig = p1d.plot(data=data, x=x, trace_name='data')
    fig.add_trace(p1d.trace(x=x, data=fit.eval_fit(x=x), name='fit'))
    fig.add_trace(p1d.trace(x=x, data=fit.eval_init(x=x), name='init'))
    fig.show()
    print(f'Dat{dat.datnum}: G/T = {fit.best_values.g / fit.best_values.theta:.2f}')


def get_2d_data(dat: DatHDF, data_name='i_sense', hot_or_cold: str = 'cold',
                csq_datnum: Optional[int] = None) -> Data2D:
    """
    Get the 2D data from the dat which is directly saved from measurement (in nA)

    For entropy data it returns only the cold part of data

    Args:
        dat (): Dat object which interacts with HDF file
        data_name: Which data to load (i_sense, entropy)
        hot_or_cold: For i_sense data of entropy scans only, can choose heated or cold
        csq_datnum: Optional dat to use for CSQ mapping data

    Returns:

    """
    is_entropy = _is_entropy_dat(dat)

    x = dat.Data.get_data('x')
    y = dat.Data.get_data('y')
    data = dat.Data.get_data('i_sense')
    if csq_datnum is not None:
        data = csq_map_data(data, csq_datnum)

    if is_entropy:
        if data_name == 'i_sense':
            x, data = _get_part_of_square_wave(dat, x, data, hot_or_cold)
        elif data_name == 'entropy':
            out = _get_row_only_out(dat, x, data)
            x, data = out.x, out.entropy_signal
        else:
            raise NotImplementedError
    elif not is_entropy and data_name != 'i_sense':
        raise NotImplementedError

    return Data2D(x=x, y=y, data=data)


@deprecated(details="Use get_2d_data instead")
def get_2d_i_sense_csq_mapped(dat: DatHDF, csq_dat: DatHDF, overwrite: bool = False,
                              hot_or_cold: str = 'cold') -> Data2D:
    """
    Get the 2D data from the dat, and then map to csq using csq_dat
    Args:
        dat ():
        csq_dat():
        overwrite: Whether to overwrite the csq mapping in Dat (not overwriting the setup of the CSQ dat itself)
        hot_or_cold:

    Returns:

    """
    setup_csq_dat(csq_dat.datnum, experiment_name=None, overwrite=False)
    calculate_csq_map(dat.datnum, experiment_name=None, csq_datnum=csq_dat.datnum, overwrite=overwrite)
    x = dat.Data.get_data('x')
    y = dat.Data.get_data('y')
    i_sense = dat.Data.get_data('csq_mapped')

    if _is_entropy_dat(dat):
        x, i_sense = _get_part_of_square_wave(dat, x, i_sense, hot_or_cold=hot_or_cold)

    data = Data2D(x=x, y=y, data=i_sense)
    return data


def _is_entropy_dat(dat: DatHDF) -> bool:
    """Checks if dat is an entropy dat"""
    try:
        awg = dat.Logs.awg
        is_entropy = True
    except NotFoundInHdfError:
        is_entropy = False
    return is_entropy


def _get_row_only_out(dat: DatHDF, x: np.ndarray, i_sense: np.ndarray) -> Output:
    """
    Convert 2D data into entropy parts (using setpoint to ignore some data after dac steps)
    Args:
        dat ():
        x ():
        i_sense ():

    Returns:

    """
    is_entropy = _is_entropy_dat(dat)
    if is_entropy is False:
        raise RuntimeError(f'Dat{dat.datnum} is not an entropy dat')

    out = dat.SquareEntropy.get_row_only_output(calculate_only=True,
                                                inputs=dat.SquareEntropy.get_Inputs(
                                                    x_array=x,
                                                    i_sense=i_sense
                                                ),
                                                process_params=dat.SquareEntropy.get_ProcessParams(
                                                    setpoint_start=13,  # ~0.005s (1/4 setpoint) at 2.5kHz Acq
                                                ),
                                                )
    return out


def _get_part_of_square_wave(dat: DatHDF, x: np.ndarray, i_sense: np.ndarray, hot_or_cold: str = 'cold') -> \
        Tuple[np.ndarray, np.ndarray]:
    """Convert from just raw I_sense data to the cold or hot part of i_sense data only (for entropy dats)"""
    out = _get_row_only_out(dat, x, i_sense)
    x = out.x
    if hot_or_cold == 'cold':
        slice = (0, 2)
    elif hot_or_cold == 'hot':
        slice = (1, 3)
    else:
        raise NotImplementedError
    i_sense = np.nanmean(out.cycled[:, slice, :], axis=1)  # Only cold or hot part
    return x, i_sense


def plot_2d_i_sense(data: Data2D, title_prepend: str = '', trace_type='heatmap', using_csq: bool = False) -> go.Figure:
    """
    Plot 2D i_sense data
    Args:
        data ():
        trace_type: heatmap or waterfall
        using_csq: If using csq mapping then change units etc

    Returns:

    """
    units = 'nA' if not using_csq else 'mV'
    fig = p2d.figure(xlabel='Sweepgate (mV)', ylabel=f'Current ({units})',
                     title=f'{title_prepend}2D I_sense in {units}')
    fig.add_trace(p2d.trace(data=data.data, x=data.x, y=data.y, trace_type=trace_type))
    return fig


def get_initial_params(data: Data1D, which='i_sense') -> lm.Parameters:
    """Get initial transition fit lm.Parameters for either simple i_sense or NRG fit"""
    from dat_analysis.dat_object.attributes.transition import get_param_estimates
    initial_params = get_param_estimates(x=data.x, data=data.data)
    if which == 'nrg':
        theta = initial_params['theta'].value
        initial_params.add_many(
            ('g', 1, True, theta / 1000, theta * 50),
            ('occ_lin', 0, True, -0.001, 0.001),
        )
        if initial_params['amp'].value > 1.5:
            initial_params['amp'].value = 1.5
    return initial_params


def fit_single_transition(data: Data1D, fit_with: str = 'i_sense',
                          initial_params: Optional[lm.Parameters] = None) -> FitInfo:
    """
    Fit 1d_transition using either 'i_sense' function or 'nrg'
    Args:
        data (): 1D data to fit centers for
        fit_with (): Which function to use for fitting (i_sense or nrg)
        initial_params (): Optional initial params for fitting

    Returns:
        The fit results of fitting
    """
    if fit_with == 'i_sense':
        from dat_analysis.dat_object.attributes.transition import i_sense
        func = i_sense
        method = 'leastsq'
    elif fit_with == 'nrg':
        from dat_analysis.analysis_tools.nrg import NRG_func_generator
        func = NRG_func_generator(which='i_sense')
        method = 'powell'
    else:
        raise NotImplementedError

    if initial_params is None:
        initial_params = get_initial_params(data, which=fit_with)

    fit = calculate_fit(x=data.x, data=data.data, params=initial_params, func=func, method=method)
    return fit


def fit_2d_transition_data(data: Data2D, fit_with: str = 'i_sense',
                           initial_params: Optional[lm.Parameters] = None) -> List[FitInfo]:
    """
    Fit 2d_transition using either 'i_sense' function or 'nrg'
    Args:
        data (): 2D data to fit centers for
        fit_with (): Which function to use for fitting (i_sense or nrg)
        initial_params (): Optional initial params for fitting

    Returns:
        The fit results of fitting
    """

    fits = [fit_single_transition(data=Data1D(x=data.x, data=d), fit_with=fit_with, initial_params=initial_params)
            for d in data.data]
    return fits


def add_centers_to_plot(fig: go.Figure, centers: Union[list, np.ndarray], ys: np.ndarray,
                        color: str = 'white') -> go.Figure:
    """Adds center to 2D heatmap"""
    fig.add_trace(p1d.trace(x=centers, data=ys, mode='markers', name='Centers',
                            trace_kwargs=dict(marker=dict(
                                color=color, size=3, symbol='circle',
                            ))))
    return fig


def run_and_plot_center_comparsion(fig: go.Figure, data: Data2D) -> go.Figure:
    """Runs both I_sense and NRG fits to each row of data and adds the center values of each to the figure provided
    Expects a 2D heatmap fig to be passed in"""
    i_sense_fits = fit_2d_transition_data(data, fit_with='i_sense', initial_params=None)
    nrg_params = get_initial_params(Data1D(data.x, data.data[0]), 'nrg')
    nrg_params['g'].value = 0.001
    nrg_params['g'].vary = False
    nrg_fits = fit_2d_transition_data(data, fit_with='nrg', initial_params=nrg_params)
    for fits, color in zip([i_sense_fits, nrg_fits], ['white', 'green']):
        centers = [f.best_values.mid for f in fits]
        fig = add_centers_to_plot(fig, centers, data.y, color=color)
    return fig


def run_and_plot_single_fit_comparison(fig: go.Figure, data: Data1D) -> go.Figure:
    """
    Runs both
    Args:
        fig ():
        data ():

    Returns:

    """
    i_sense_fit = fit_single_transition(data, fit_with='i_sense', initial_params=None)
    nrg_params = get_initial_params(data, 'nrg')
    nrg_params['g'].value = 0.001
    nrg_params['g'].vary = False
    nrg_params['occ_lin'].vary = False
    nrg_fit = fit_single_transition(data, fit_with='nrg', initial_params=nrg_params)
    for fit, name in zip([i_sense_fit, nrg_fit], ['i_sense', 'NRG']):
        fig.add_trace(p1d.trace(x=data.x, data=fit.eval_fit(x=data.x), name=name, mode='lines'))
    print(i_sense_fit)
    print(nrg_fit)
    return fig


def average_data(data: Data2D, centers) -> Data1D:
    """Averages 2D data using centers and returns Data1D"""
    avg_data, avg_x = mean_data(data.x, data.data, centers=centers, return_x=True)
    return Data1D(x=avg_x, data=avg_data)


def plot_single_transition(data: Data1D, title_prepend: str = '', using_csq: bool = False) -> go.Figure:
    """
    Plots single trace of transition
    Args:
        data ():
        title_prepend ():
        using_csq: Change units if using csq mapped data

    Returns:

    """
    units = 'nA' if not using_csq else 'mV'
    fig = p1d.figure(xlabel='Sweepgate (mV)', ylabel=f'Current ({units})',
                     title=f'{title_prepend}1D I_sense ({units})')
    fig.add_trace(p1d.trace(x=data.x, data=data.data, name='Data', mode='lines'))
    return fig


def compare_nrg_with_i_sense_for_single_dat(datnum: int,
                                            csq_map_datnum: Optional[int] = None,
                                            show_2d_centering_comparsion=False,
                                            show_1d_fit_comparison=True):
    """
    Runs and can plot the comparsion of 2D centering, and 1D fitting of averaged data for NRG vs regular I_sense, either
    csq mapped or not
    Args:
        datnum ():
        csq_map_datnum ():  Optional CSQ datnum for csq mapping
        show_2d_centering_comparsion (): Whether to show the 2D plot with centers
        show_1d_fit_comparison (): Whether to show the 1D fit and fit info

    Returns:

    """
    dat = get_dat(datnum)
    if csq_map_datnum is not None:
        csq_dat = get_dat(csq_map_datnum)
        data = get_2d_i_sense_csq_mapped(dat, csq_dat)
        using_csq = True
    else:
        data = get_2d_data(dat)
        using_csq = False

    if show_2d_centering_comparsion:
        fig = plot_2d_i_sense(data, title_prepend=f'Dat{dat.datnum}: ', trace_type='heatmap', using_csq=using_csq)
        run_and_plot_center_comparsion(fig, data).show()

    fits = fit_2d_transition_data(data, fit_with='i_sense', initial_params=None)
    centers = [f.best_values.mid for f in fits]
    avg_data = average_data(data, centers)

    if show_1d_fit_comparison:
        fig = plot_single_transition(avg_data, title_prepend=f'Dat{dat.datnum}: ', using_csq=using_csq)
        run_and_plot_single_fit_comparison(fig, avg_data).show()


def run_weakly_coupled_nrg_fit(datnum: int, csq_datnum: Optional[int],
                               center_func: Optional[Callable[[DatHDF], bool]] = None,
                               overwrite: bool = False,
                               ) -> FitInfo:
    """
    Runs
    Args:
        datnum (): Dat to calculate for
        csq_datnum (): Num of dat to use for CSQ mapping  (will only calculate if necessary)
        center_func: Whether data should be centered first for dat
            (e.g. lambda dat: True if dat.Logs.dacs['ESC'] > -250 else False)
        overwrite (): NOTE: Only overwrites final Avg fit.

    Returns:

    """
    fit_name = 'csq_gamma_small' if csq_datnum is not None else 'gamma_small'
    dat = get_dat(datnum)
    avg_data = get_avg_i_sense_data(dat, csq_datnum, center_func=center_func)

    pars = get_initial_params(avg_data, which='nrg')
    pars['g'].value = 0.005
    pars['g'].vary = False
    pars['occ_lin'].vary = False
    fit = dat.NrgOcc.get_fit(which='avg', name=fit_name,
                             initial_params=pars,
                             data=avg_data.data, x=avg_data.x,
                             calculate_only=False, check_exists=False, overwrite=overwrite)
    return fit


def run_forced_theta_nrg_fit(datnum: int, csq_datnum: Optional[int],
                             center_func: Optional[Callable[[DatHDF], bool]] = None,
                             which_linear_theta_params: str = 'normal',
                             overwrite: bool = False,
                             ) -> FitInfo:
    """
    Runs
    Args:
        datnum (): Dat to calculate for
        csq_datnum (): Num of dat to use for CSQ mapping  (will only calculate if necessary)
        center_func: Whether data should be centered first for dat
            (e.g. lambda dat: True if dat.Logs.dacs['ESC'] > -250 else False)
        which_linear_theta_params: str =
        overwrite (): NOTE: Only overwrites final Avg fit.

    Returns:

    """
    if csq_datnum is not None:
        fit_name = 'csq_forced_theta'
    else:
        fit_name = 'forced_theta'

    if center_func is None:
        center_func = lambda dat: False  # Default to no centering for gamma broadened
    dat = get_dat(datnum)
    avg_data = get_avg_i_sense_data(dat, csq_datnum, center_func=center_func)

    pars = get_initial_params(avg_data, which='nrg')
    theta = get_linear_theta(dat, which_params=which_linear_theta_params)
    pars['theta'].value = theta
    pars['theta'].vary = False
    pars['g'].value = 5
    pars['g'].max = theta * 50  # limit of NRG data
    pars['g'].min = theta / 10000  # limit of NRG data
    pars['occ_lin'].vary = True

    if abs((x := avg_data.x)[-1] - x[0]) > 1500:  # If it's a wider scan, only fit over middle 1500
        cond = np.where(np.logical_and(x > -750, x < 750))
        avg_data.x, avg_data.data = x[cond], avg_data.data[cond]

    fit = dat.NrgOcc.get_fit(which='avg', name=fit_name,
                             initial_params=pars,
                             data=avg_data.data, x=avg_data.x,
                             calculate_only=False, check_exists=False, overwrite=overwrite)
    return fit


def get_linear_slope_intercept(which: str):
    """
    Returns the slope and intercept of fit through weakly coupled thetas
    Args:
        which ():

    Returns:

    """
    if which == 'normal':
        slope, intercept = 0.00322268, 5.16050029  # 7.4%, 1.52% error respectively
    elif which == 'csq mapped':  # csq dat2197 only
        slope, intercept = 0.00319, 5.17  # 0.0003, 0.08 error respectively
    elif which == 'entropy':
        logging.warning(f'Linear theta from entropy is out of date (still from when fit was to averaged not cold data)')
        slope, intercept = 0.00314035, 5.69446  # 17%, 3% error respectively
    elif which == 'transition':
        logging.warning(f'Linear theta from transition is out of dat (from csq mapped data where many different csqs'
                        f'were used)')
        slope, intercept = 0.00355797, 5.28561545  # 9%, 2% error respectively
    elif which == 'dats7000+':
        slope, intercept = 0.088668, 64.3754  # 2021-05-03
    elif which == 'dats5000+':
        slope, intercept = 0.005473, 6.478  # Gamma Paper>Data>Supplementary info
    else:
        raise NotImplementedError
    return slope, intercept


def get_linear_theta(dat, which_params: str = 'normal') -> float:
    """
    Calculates the expected theta based on linear fit parameters (either from fit to entropy dat transition thetas or
    transition only thetas)

    Args:
        dat (): Dat to return the expected theta for
        which_params (): Whether to use the linear theta params from fitting normal data or
            CSQ mapped (to dat2197 only)  -

    Returns:
        Expected theta value
    """
    slope, intercept = get_linear_slope_intercept(which_params)
    line = lm.models.LinearModel()
    pars = line.make_params()
    pars['slope'].value = slope
    pars['intercept'].value = intercept
    return float(line.eval(pars, x=dat.Logs.dacs['ESC']))


def get_avg_i_sense_data(dat: DatHDF,
                         csq_datnum: Optional[int] = None,
                         center_func: Optional[Callable[[DatHDF], bool]] = None,
                         overwrite: bool = False,
                         hot_or_cold: str = 'cold',
                         ) -> Data1D:
    """
    Get avg_data with/without centering based on center_func. Get's weakly coupled part of Entropy data (after
    removing first part of each step).

    Args:
        dat (): Dat to get data from
        csq_datnum (): CSQ dat for mapping
        center_func (): Callable which takes 'dat' as an argument and returns True or False
        overwrite: Whether to overwrite avg_data
        hot_or_cold: Whether to get hot or cold data for entropy dats

    Returns:

    """
    if csq_datnum is not None:
        name = 'csq_mapped'
    else:
        name = None
    data = get_2d_data(dat, hot_or_cold=hot_or_cold, csq_datnum=csq_datnum)

    if hot_or_cold != 'cold':
        name = f'{hot_or_cold}_{name}' if name else hot_or_cold

    # If already exists, just load and return, else carry on
    if overwrite is False:
        try:
            avg_data, avg_x = dat.NrgOcc.get_avg_data(return_x=True, name=name, check_exists=True)
            return Data1D(x=avg_x, data=avg_data)
        except NotFoundInHdfError:
            pass

    if center_func is None or center_func(dat):
        centers = get_centers(dat, data, name, overwrite)
    else:
        centers = [0] * data.data.shape[0]
    avg_data, avg_x = dat.NrgOcc.get_avg_data(x=data.x, data=data.data, centers=centers, return_x=True,
                                              name=name,
                                              overwrite=overwrite)
    avg_data = Data1D(avg_x, avg_data)
    return avg_data


def get_centers(dat: DatHDF, data: Optional[Data2D] = None,
                name: Optional[str] = None,
                overwrite: bool = False):
    """Calculate (or load) centers from 2D data"""
    check = False if overwrite or name is None else True
    if data is not None:
        fits = dat.Transition.get_row_fits(name=name, data=data.data, x=data.x, check_exists=check,
                                           overwrite=overwrite)
    else:
        fits = dat.Transition.get_row_fits(name=name, check_exists=True)
    centers = [f.best_values.mid for f in fits]
    return centers


def run_nrg_fit(dat: DatHDF, csq_datnum: Optional[DatHDF] = None,
                forced_theta=True,
                which_linear_theta_params: str = 'normal',
                overwrite: bool = False) -> FitInfo:
    """Run either a weakly coupled or strongly coupled NRG fit on dat"""
    if forced_theta:
        fit = run_forced_theta_nrg_fit(dat.datnum, csq_datnum, center_func=_center_func,
                                       which_linear_theta_params=which_linear_theta_params,
                                       overwrite=overwrite)
    else:
        fit = run_weakly_coupled_nrg_fit(dat.datnum, csq_datnum, center_func=_center_func,
                                         overwrite=overwrite)
    return fit


def run_multiple_nrg_fits(dats: List[DatHDF], csq_dats: Optional[List[DatHDF]] = None, forced_theta=True,
                          which_linear_theta_params: str = 'normal',
                          overwrite: bool = False) -> List[FitInfo]:
    """

    Args:
        dats (): Dats to fit transition of
        csq_dats ():  Optional csq_dats to use for csq_mapping, otherwise will just use regular i_sense
            Note: (can provide any number of csq_dats, they are sorted to best match the entropy dats anyway)
            Note: This will not have any effect if a csq_mapping already exists!
        forced_theta: Whether theta should be forced to linear theta (and gamma/occ_lin allowed to vary)
        which_linear_theta_params: Which linear theta fit params to use
        overwrite: Whether to overwrite the avg fit (does not overwrite anything else)

    Returns:

    """
    # Get best CSQ dats in order of dats
    if csq_dats is not None:
        csq_dict = {c.Logs.dacs['ESC']: c for c in csq_dats}
        csqs_in_entropy_order = [csq_dict[n] if (n := dat.Logs.dacs['ESC']) in csq_dict else csq_dict[
            min(csq_dict.keys(), key=lambda k: abs(k - n))] for dat in dats]
    else:
        csqs_in_entropy_order = [None] * len(dats)

    fits = []
    for dat, csq_dat in progressbar(zip(dats, csqs_in_entropy_order)):
        csq_datnum = csq_dat.datnum if csq_dat else None
        fits.append(run_nrg_fit(dat, csq_datnum, forced_theta, which_linear_theta_params, overwrite))
    return fits


def plot_linear_theta_nrg_fit(dats: List[DatHDF], show_plots=True, csq_mapped=False) -> FitInfo:
    """
    Plots thetas and runs a linear fit on weakly coupled ones. If fits aren't already saved, use "run_multiple_nrg_fits"
    first.

    Linear fit is returned
    Args:
        dats (): Dats to look for saved fits in
        show_plots (): Whether to display the plot of linear theta
        csq_mapped: Whether to use csq mapped fits or not

    Returns:
        linear_fit result
    """
    fit_name = 'csq_gamma_small' if csq_mapped else 'gamma_small'
    linear_fit = linear_fit_thetas(dats, fit_name=fit_name,
                                   filter_func=lambda dat: True if dat.Logs.dacs['ESC'] < -280 else False,
                                   show_plots=show_plots, sweep_gate_divider=1,
                                   dat_attr_saved_in='nrg')
    return linear_fit


def plot_amplitudes(dats: List[DatHDF],
                    csq_mapped: bool = True) -> go.Figure:
    if csq_mapped:
        title_append = 'CSQ mapped NRG fits'
        fit_name = 'csq_forced_theta'
    else:
        title_append = 'i_sense NRG fits'
        fit_name = 'forced_theta'

    fig = p1d.figure(xlabel='ESC (mV)', ylabel='Amplitude (mV)', title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: '
                                                                       f'Amplitudes from {title_append}')
    fits = [dat.NrgOcc.get_fit(name=fit_name) for dat in dats]
    amps = [f.best_values.amp for f in fits]
    xs = [dat.Logs.dacs['ESC'] for dat in dats]
    fig.add_trace(p1d.trace(x=xs, data=amps, text=[dat.datnum for dat in dats]))
    return fig


def plot_linear_theta_comparison(e_dats, t_dats, a_dats, fit_name='gamma_small') -> Tuple[List[FitInfo], go.Figure]:
    csq_mapped = True if 'csq' in fit_name else False
    efit = plot_linear_theta_nrg_fit(e_dats, show_plots=False, csq_mapped=csq_mapped)
    tfit = plot_linear_theta_nrg_fit(t_dats, show_plots=False, csq_mapped=csq_mapped)
    fit = plot_linear_theta_nrg_fit(a_dats, show_plots=False, csq_mapped=csq_mapped)

    plotter = OneD(dats=all_dats)
    fig = plotter.figure(xlabel='ESC /mV', ylabel='Theta /mV (real)',
                         title=f'Dats{all_dats[0].datnum}-{all_dats[-1].datnum}: Theta vs ESC')
    for dats, name in zip((e_dats, t_dats), ['entropy', 'transition']):
        xs = [dat.Logs.dacs['ESC'] for dat in dats]
        thetas = [dat.NrgOcc.get_fit(name=fit_name).best_values.theta for dat in dats]
        datnums = [dat.datnum for dat in dats]
        fig.add_trace(plotter.trace(data=thetas, x=xs, name=name, mode='markers', text=datnums))

    xs = [dat.Logs.dacs['ESC'] for dat in t_dats]
    for f, name in zip((efit, tfit, fit), ['entropy fit', 'transition fit', 'all fit']):
        fig.add_trace(plotter.trace(x=xs, data=f.eval_fit(np.array(xs)), name=name, mode='lines'))
    return [efit, tfit, fit], fig


def get_avg_entropy_data(dat, center_func: Callable,
                         csq_datnum: Optional[int] = None) -> Data1D:
    """Get avg entropy data (including setpoint start thing)"""
    data2d = get_2d_data(dat, 'entropy', csq_datnum=csq_datnum)
    transition_center_name = None
    if csq_datnum:
        transition_center_name = 'csq_mapped'
    if center_func(dat):
        centers = get_centers(dat, data=None, name=transition_center_name)
    else:
        centers = [0] * data2d.data.shape[0]
    data, x = mean_data(data2d.x, data2d.data, centers, return_x=True)
    return Data1D(x=x, data=data)


# if __name__ == '__main__':
#     dat = get_dat(2128)
#     fit = run_multiple_nrg_fits([dat], None, overwrite=False)
#
#     data = get_avg_i_sense_data(dat, center_func=_center_func)
#     fig = p1d.plot(data=data.data, x=data.x, trace_name='Data')
#     fit = dat.NrgOcc.get_fit(name='forced_theta')
#     fig.add_trace(p1d.trace(data=fit.eval_fit(x=data.x), x=data.x, name='Fit', mode='lines'))
#     fig.show()
#


def dt_with_set_params(esc: float, which_linear_fit: str) -> float:
    """
    Just wraps dT_from_linear with some fixed params for generating new dT (currently based on fitting both entropy and
    transition thetas with new NRG fits see Tim Child/Gamma Paper/Data/Determining Amplitude)
    Args:
        esc (): New ESC value to get dT for

    Returns:

    """
    slope, intercept = get_linear_slope_intercept(which=which_linear_fit)

    # Using New NRG fits to Dats 2148, 2149, 2143 (cold)
    dt = dT_from_linear(base_dT=1.15566, base_esc=-339.36, lever_slope=slope, lever_intercept=intercept,
                        new_esc=esc)
    return dt


def calc_int_info(dat: DatHDF, fit_name: str = 'forced_theta',
                  which_linear_theta_fit: str = 'normal') -> IntegrationInfo:
    """Calculates integration info and saves in Dat, and returns the integration info"""
    fit = dat.NrgOcc.get_fit(name=fit_name)
    amp = fit.best_values.amp
    # if abs(dat.Data.x[-1] - dat.Data.x[0]) > 1200:  # If a wider scan than this, then fit to whole
    #     amp = get_linear_gamma_amplitude(dat)
    #     # amp = get_restricted_fit_amp(dat, orig_fit_name=fit_name)
    # else:
    #     amp = fit.best_values.amp
    dt = dt_with_set_params(dat.Logs.dacs['ESC'], which_linear_fit=which_linear_theta_fit)
    int_info = dat.Entropy.set_integration_info(dT=dt, amp=amp, name=fit_name, overwrite=True)
    return int_info


def get_restricted_fit_amp(dat: DatHDF, orig_fit_name: str) -> float:
    """Calculates amplitude from fitting only over narrower range"""
    new_fit = _get_restricted_fit(dat, orig_fit_name)
    return new_fit.best_values.amp


def _get_restricted_fit(dat: DatHDF, orig_fit_name: str, force_amp: Optional[float] = None) -> FitInfo:
    orig_fit = dat.NrgOcc.get_fit(name=orig_fit_name)
    data = get_avg_i_sense_data(dat, None, center_func=lambda dat: True if dat.Logs.dacs['ESC'] < -230 else False,
                                overwrite=False)
    ids = get_data_index(data.x, [orig_fit.best_values.mid - 600, orig_fit.best_values.mid + 600])
    range = np.s_[ids[0]:ids[1]]
    params = copy.copy(orig_fit.params)
    params['theta'].vary = False  # Ensure theta doesn't vary
    if force_amp:
        params['amp'].value = force_amp
        params['amp'].vary = False
    new_fit = dat.NrgOcc.get_fit(data=data.data[range], x=data.x[range], calculate_only=True,
                                 initial_params=params)
    return new_fit


def plot_standard_and_restricted_fit(dat: DatHDF):
    def sub_lin(x, data, fit):
        line = lm.models.LinearModel()
        pars = line.make_params()
        pars['slope'].value = fit.best_values.lin
        pars['intercept'].value = fit.best_values.const
        return data - line.eval(params=pars, x=x)

    orig_fit_name = 'forced_theta'
    orig_fit = dat.NrgOcc.get_fit(name=orig_fit_name)
    new_fit = _get_restricted_fit(dat, orig_fit_name)
    forced_amp_fit = _get_restricted_fit(dat, orig_fit_name, force_amp=get_linear_gamma_amplitude(dat))
    data = get_avg_i_sense_data(dat, None, center_func=lambda dat: True if dat.Logs.dacs['ESC'] < -230 else False,
                                overwrite=False)
    fig = p1d.figure(xlabel='Sweep Gate', ylabel='Current /nA', title=f'Dat{dat.datnum}: Comparing wide vs narrow fit')
    fig.add_trace(p1d.trace(data=sub_lin(data.x, data.data, orig_fit), x=data.x, mode='markers', name='Data'))
    for fit, name in zip([orig_fit, new_fit, forced_amp_fit], ['Wide', 'Narrow', 'Amp Forced']):
        fig.add_trace(p1d.trace(data=sub_lin(data.x, fit.eval_fit(x=data.x), fit=orig_fit),
                                x=data.x, name=f'Fit {name}', mode='lines'))
    return fig


def plot_residual_standard_and_restricted_fit(dat: DatHDF):
    orig_fit_name = 'forced_theta'
    orig_fit = dat.NrgOcc.get_fit(name=orig_fit_name)
    new_fit = _get_restricted_fit(dat, orig_fit_name)
    forced_amp_fit = _get_restricted_fit(dat, orig_fit_name, force_amp=get_linear_gamma_amplitude(dat))
    data = get_avg_i_sense_data(dat, None, center_func=lambda dat: True if dat.Logs.dacs['ESC'] < -230 else False,
                                overwrite=False)
    fig = p1d.figure(xlabel='Sweep Gate', ylabel='Current /nA', title=f'Dat{dat.datnum}: Comparing wide vs narrow fit')
    for fit, name in zip([orig_fit, new_fit, forced_amp_fit], ['Wide', 'Narrow', 'Amp Forced']):
        fig.add_trace(p1d.trace(data=data.data - fit.eval_fit(x=data.x),
                                x=data.x, name=f'Fit {name}', mode='lines'))
    return fig


def get_linear_gamma_amplitude(dat: DatHDF) -> float:
    if (v := dat.Logs.dacs['ESC']) < -270:
        raise ValueError(f'dat{dat.datnum} has ESC = {v}mV which is weakly coupled. This is only for extrapolating'
                         f'amplitude into very gamma broadened ')
    line = lm.models.LinearModel()
    pars = line.make_params()

    # Based on fit to dats 2117 -> 2126 New NRG fits with forced theta linear
    pars['slope'].value = -0.00435440
    pars['intercept'].value = -0.25988496

    return float(line.eval(params=pars, x=dat.Logs.dacs['ESC']))


def get_x_offset(dat, fit_name: str = 'forced_theta') -> float:
    """get offset of 0.5 occupation from 0"""
    offset_x = dat.NrgOcc.get_x_of_half_occ(fit_name=fit_name)
    return offset_x


def get_centered_x_at_half_occ(dat: DatHDF, csq_datnum: Optional[int] = None,
                               fit_name: str = 'forced_theta') -> np.ndarray:
    """get x-array centered where occupation = 0.5"""
    occ_data = get_avg_i_sense_data(dat, csq_datnum, _center_func, overwrite=False)
    x = occ_data.x
    return x - get_x_offset(dat, fit_name)


def get_integrated_data(dat: DatHDF, fit_name: str = 'forced_theta', zero_point: float = -500,
                        csq_datnum: Optional[int] = None,
                        which_linear_theta_fit: str = 'normal') -> Data1D:
    """get integrated data for dat"""
    int_info = calc_int_info(dat, fit_name=fit_name, which_linear_theta_fit=which_linear_theta_fit)
    data = get_avg_entropy_data(dat,
                                center_func=_center_func,
                                csq_datnum=csq_datnum)

    data.data = int_info.integrate(data.data)
    data.x = get_centered_x_at_half_occ(dat, csq_datnum, fit_name=fit_name)
    w = abs(data.x[-1] - data.x[0])
    if data.x[0] < zero_point - w / 20:
        offset_y = np.mean(
            data.data[np.where(np.logical_and(data.x > zero_point - w / 20, data.x < zero_point + w / 20))])
        data.data -= offset_y
    return data



if __name__ == '__main__':
    # compare_nrg_with_i_sense_for_single_dat(datnum=2164, csq_map_datnum=2166,
    #                                         show_2d_centering_comparsion=False,
    #                                         show_1d_fit_comparison=True)

    # run_weakly_coupled_csq_mapped_nrg_fit(2164, 2166)

    # entropy_dats = get_dats(range(2095, 2142 + 1, 2))
    # transition_dats = get_dats(range(2096, 2142 + 1, 2))
    #
    # entropy_dats = get_dats([2164, 2167, 2170, 2121, 2213])
    # transition_dats = get_dats([dat.datnum + 1 for dat in entropy_dats])

    entropy_dats = get_dats(range(5303, 5307+1))
    transition_dats = []

    all_dats = entropy_dats + transition_dats
    # csq_dats = get_dats((2185, 2208 + 1))  # CSQ dats, NOT correctly ordered
    single_csq = get_dat(2197)

    for dat in progressbar(entropy_dats+transition_dats):
        # data_2d = get_2d_i_sense_csq_mapped(dat, single_csq, overwrite=True)
        # get_avg_i_sense_data(dat, csq_datnum=single_csq.datnum,
        #                      center_func=_center_func,
        #                      overwrite=True)
        # get_avg_i_sense_data(dat, csq_datnum=None,
        #                      center_func=_center_func,
        #                      overwrite=True)
        pass

    # run_multiple_nrg_fits(transition_dats, csq_dats, forced_theta=True, which_linear_theta_params='normal', overwrite=False)
    # run_multiple_nrg_fits(entropy_dats, csq_dats, forced_theta=True, which_linear_theta_params='normal', overwrite=False)
    # run_multiple_nrg_fits(transition_dats, [single_csq], forced_theta=False, overwrite=True)
    # run_multiple_nrg_fits(entropy_dats, [single_csq], forced_theta=False, overwrite=True)
    # run_multiple_nrg_fits(transition_dats, [single_csq], forced_theta=True, which_linear_theta_params='csq mapped',
    #                       overwrite=False)
    # run_multiple_nrg_fits(entropy_dats, [single_csq], forced_theta=True, which_linear_theta_params='csq mapped',
    #                       overwrite=False)
    # run_multiple_nrg_fits(transition_dats, None, forced_theta=False, overwrite=False)
    # run_multiple_nrg_fits(entropy_dats, None, forced_theta=False, overwrite=False)
    # run_multiple_nrg_fits(transition_dats, None, forced_theta=True, which_linear_theta_params='normal', overwrite=False)
    # run_multiple_nrg_fits(entropy_dats, None, forced_theta=True, which_linear_theta_params='normal', overwrite=False)

    # for dat in progressbar(transition_dats + entropy_dats):
    #     if -260 < dat.Logs.dacs['ESC'] < -230:
    #         get_avg_i_sense_data(dat, None, _center_func, overwrite=True)
    #         run_forced_theta_nrg_fit(dat.datnum, None, center_func=_center_func,
    #                                  which_linear_theta_params='normal', overwrite=True)

    # fits, fig = plot_linear_theta_comparison(entropy_dats, transition_dats, all_dats, 'gamma_small')
    fits, fig = plot_linear_theta_comparison(entropy_dats, entropy_dats, all_dats, 'gamma_small')
    fig.show()
    for fit, name in zip(fits, ['Entropy', 'Transition', 'All']):
        print(f'Linear theta fit to {name} dats:')
        print(f'Slope = {fit.best_values.slope:.4g}{PM}{fit.params["slope"].stderr:.2g}')
        print(f'Intercept = {fit.best_values.intercept:.4g}{PM}{fit.params["intercept"].stderr:.2g}')
        print()

    # plot_amplitudes(all_dats, csq_mapped=True).show()
