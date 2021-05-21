from typing import List, Callable, Union, Tuple, Optional, Dict

import numpy as np
from deprecation import deprecated
from progressbar import progressbar
from scipy.interpolate import interp1d
import lmfit as lm
import logging

from src import UsefulFunctions as U
from src.AnalysisTools.fitting import FitInfo, calculate_transition_only_fit, _get_transition_fit_func_params, \
    calculate_se_transition, calculate_se_entropy_fit, calculate_fit
from src.CoreUtil import get_data_index
from src.DatObject.Attributes.SquareEntropy import square_wave_time_array, Output

from src.UsefulFunctions import edit_params
from src.Dash.DatPlotting import OneD
from src.DatObject.Attributes.Transition import i_sense, i_sense_digamma, i_sense_digamma_amplin
from src.DatObject.DatHDF import DatHDF

from src.DatObject.Make_Dat import get_dats, get_dat

logger = logging.getLogger(__name__)


def get_deltaT(dat: DatHDF, from_self=False, fit_name: str = None, default_dt=0.947):
    """Returns deltaT of a given dat in mV"""
    if from_self is False:
        ho1 = dat.AWG.max(0)  # 'HO1/10M' gives nA * 10
        t = dat.Logs.temps.mc

        # Datnums to search through (only thing that should be changed)
        # datnums = set(range(1312, 1451 + 1)) - set(range(1312, 1451 + 1, 4))
        datnums = list(range(2143, 2156))

        dats = get_dats(datnums)

        dats = [d for d in dats if
                np.isclose(d.Logs.temps.mc, dat.Logs.temps.mc, rtol=0.1)]  # Get all dats where MC temp is within 10%
        bias_lookup = np.array([d.Logs.fds['HO1/10M'] for d in dats])

        indp = int(np.argmin(abs(bias_lookup - ho1)))
        indm = int(np.argmin(abs(bias_lookup + ho1)))
        theta_z = np.nanmean([d.Transition.avg_fit.best_values.theta for d in dats if d.Logs.fds['HO1/10M'] == 0])

        theta_p = dats[indp].Transition.avg_fit.best_values.theta
        theta_m = dats[indm].Transition.avg_fit.best_values.theta
        # theta_z = dats[indz].Transition.avg_fit.best_values.theta
        return (theta_p + theta_m) / 2 - theta_z
    else:
        cold_fit = dat.SquareEntropy.get_fit(fit_name=fit_name)
        hot_fit = dat.SquareEntropy.get_fit(which_fit='transition', transition_part='hot',
                                            initial_params=cold_fit.params, check_exists=False, output_name=fit_name,
                                            fit_name=fit_name + '_hot')
        if all([fit.best_values.theta is not None for fit in [cold_fit, hot_fit]]):
            dt = hot_fit.best_values.theta - cold_fit.best_values.theta
            return dt
        else:
            return default_dt


def narrow_fit(dat: DatHDF, width, initial_params, fit_func=i_sense, check_exists=False, save_name='narrow',
               output_name: str = 'default', transition_only: bool = False,
               overwrite=False, csq_map: bool = False):
    """
    Get a fit only including +/- width in dat.x around center of transition
    kwargs is the stuff to pass to get_fit
    Return a fit
    """
    if transition_only is False:
        x = np.copy(dat.SquareEntropy.avg_x)
        y = np.copy(dat.SquareEntropy.get_Outputs(name=output_name, check_exists=True).averaged)
        y = np.mean(y[(0, 2), :], axis=0)  # Average Cold parts
    else:
        x = np.copy(dat.Transition.avg_x)
        if csq_map:
            y = np.copy(dat.Data.get_data('csq_mapped_avg'))
        else:
            y = np.copy(dat.Transition.avg_data)

    start_ind = np.nanargmin(np.abs(np.add(x, width)))
    end_ind = np.nanargmin(np.abs(np.subtract(x, width)))

    x[:start_ind] = [np.nan] * start_ind
    x[end_ind:] = [np.nan] * (len(x) - end_ind)

    y[:start_ind] = [np.nan] * start_ind
    y[end_ind:] = [np.nan] * (len(y) - end_ind)

    if transition_only is False:
        fit = dat.SquareEntropy.get_fit(
            x=x,
            data=y,
            initial_params=initial_params,
            fit_func=fit_func,
            check_exists=check_exists, fit_name=save_name,
            overwrite=overwrite)
    else:
        fit = dat.Transition.get_fit(
            x=x, data=y,
            initial_params=initial_params,
            fit_func=fit_func,
            check_exists=check_exists, name=save_name,
            overwrite=overwrite
        )
    return fit


def do_narrow_fits(dats: Union[List[DatHDF], int],
                   theta=None, gamma=None, width=500,
                   output_name: str = 'default', overwrite=False,
                   transition_only=False,
                   fit_func: str = 'i_sense_digamma',
                   fit_name: str = 'narrow',
                   ):
    if isinstance(dats, int):  # To allow multiprocessing
        dats = [get_dat(dats)]

    if transition_only is False:
        fit = dats[0].SquareEntropy.get_fit(which='avg', which_fit='transition', transition_part='cold',
                                            check_exists=False)
    else:
        fit = dats[0].Transition.avg_fit
    params = fit.params
    if gamma is None:
        params.add('g', value=0, vary=True, min=-50, max=1000)
    else:
        params.add('g', value=gamma, vary=False, min=-50, max=1000)
    if theta is None:
        theta = fit.best_values.theta
        theta_vary = True
    else:
        theta_vary = False

    if fit_func == 'i_sense_digamma_amplin':
        params.add('amplin', value=0, vary=True)
        func = i_sense_digamma_amplin
    else:
        func = i_sense_digamma

    new_pars = edit_params(params, param_name='theta', value=theta, vary=theta_vary)

    amp_fits = [narrow_fit(
        dat,
        width,
        initial_params=new_pars,
        fit_func=func,
        output_name=output_name,
        check_exists=False, save_name=fit_name,
        transition_only=transition_only,
        overwrite=overwrite,
    )
        for dat in progressbar(dats)]

    return amp_fits


def get_setpoint_indexes(dat: DatHDF, start_time: Optional[float] = None, end_time: Optional[float] = None) -> Tuple[Union[int, None], Union[int, None]]:
    """
    Gets the indexes of setpoint_start/fin from a start and end time in seconds
    Args:
        dat (): Dat for which this is being applied
        start_time (): Time after setpoint change in seconds to start averaging setpoint
        end_time (): Time after setpoint change in seconds to finish averaging setpoint
    Returns:
        start, end indexes
    """
    setpoints = [start_time, end_time]
    setpoint_times = square_wave_time_array(dat.SquareEntropy.square_awg)
    sp_start, sp_fin = [U.get_data_index(setpoint_times, sp) for sp in setpoints]
    return sp_start, sp_fin


def do_entropy_calc(datnum, save_name: str,
                    setpoint_start: float = 0.005,
                    t_func_name: str = 'i_sense', csq_mapped=False,
                    center_for_avg: Union[bool, float] = True,
                    theta=None, gamma=None, width=None,
                    data_rows: Tuple[Optional[int], Optional[int]] = (None, None),
                    overwrite=False):
    """
    Mostly for calculating entropy signal and entropy fits.

    Args:
        datnum ():
        save_name ():
        setpoint_start ():
        t_func_name (): Transition function to fit to each row of data in order to calculate centers
        center_for_avg (): Whether to do any centering of SE data before averaging
        csq_mapped (): Whether to use i_sense data mapped back to CSQ gate instead
        theta ():
        gamma ():
        width ():
        overwrite ():

    Returns:

    """
    dat = get_dat(datnum)
    print(f'Working on {datnum}')

    if isinstance(center_for_avg, float):
        center_for_avg = True if dat.Logs.fds['ESC'] < center_for_avg else False

    sp_start, sp_fin = get_setpoint_indexes(dat, start_time=setpoint_start, end_time=None)

    s, f = data_rows

    x = dat.Data.get_data('x')
    if csq_mapped:
        data = dat.Data.get_data('csq_mapped')[s:f]
    else:
        data = dat.SquareEntropy.get_data('i_sense')[s:f]  # Changed from .Transition to .SquareEntropy 11/5/21
    x = U.get_matching_x(x, data)

    # Run Fits
    pp = dat.SquareEntropy.get_ProcessParams(name=None,  # Load default and modify from there
                                             setpoint_start=sp_start, setpoint_fin=sp_fin,
                                             save_name=save_name)
    inps = dat.SquareEntropy.get_Inputs(x_array=x, i_sense=data, save_name=save_name)
    if center_for_avg:
        t_func, params = _get_transition_fit_func_params(x=x, data=np.mean(data, axis=0),
                                                         t_func_name=t_func_name,
                                                         theta=theta, gamma=gamma)
        pp.transition_fit_func = t_func
        pp.transition_fit_params = params
    else:
        inps.centers = np.array([0] * data.shape[0])  # I.e. do not do any centering!
    out = dat.SquareEntropy.get_Outputs(name=save_name, inputs=inps, process_params=pp, overwrite=overwrite)

    if center_for_avg is False and width is not None:
        # Get an estimate of where the center is (I hope this works for very gamma broadened)
        # Note: May want a try except here with center = 0 otherwise
        center = dat.Transition.get_fit(x=out.x,
                                        data=np.nanmean(out.averaged[(0, 2), :],
                                                        axis=0), calculate_only=True).best_values.mid
    else:
        center = float(np.nanmean(out.centers_used))

    try:
        ent = calculate_se_entropy_fit(datnum, save_name=save_name, se_output_name=save_name, width=width, center=center,
                                   overwrite=overwrite)

        for t in ['cold', 'hot']:
            calculate_se_transition(datnum, save_name=save_name + f'_{t}', se_output_name=save_name,
                                    t_func_name=t_func_name,
                                    theta=theta, gamma=gamma,
                                    transition_part=t, width=width, center=center, overwrite=overwrite)

    except (TypeError, ValueError):
        print(f'Dat{dat.datnum}: Failed to calculate entropy or transition fit')
        return False
    return True


def do_transition_only_calc(datnum, save_name: str,
                            theta=None, gamma=None, width=None, t_func_name='i_sense_digamma',
                            center_func: Optional[str] = None,
                            csq_mapped=False, data_rows: Tuple[Optional[int], Optional[int]] = (None, None),
                            centering_threshold: float = 1000,
                            overwrite=False) -> FitInfo:
    """
    Do calculations on Transition only measurements
    Args:
        datnum ():
        save_name ():
        theta ():
        gamma ():
        width ():
        t_func_name ():
        center_func ():
        csq_mapped ():
        data_rows ():
        centering_threshold (): If dat.Logs.fds['ESC'] is below this value, centering will happen
        overwrite ():

    Returns:

    """
    dat = get_dat(datnum)
    print(f'Working on {datnum}')

    if csq_mapped:
        name = 'csq_mapped'
        data_group_name = 'Data'
    else:
        name = 'i_sense'
        data_group_name = 'Transition'

    data = dat.Data.get_data(f'{name}_avg{data_row_name_append(data_rows)}', data_group_name=data_group_name,
                             default=None)
    if data is None or overwrite:

        s, f = data_rows
        rows = range(s if s else 0, f if f else dat.Data.get_data('y').shape[0])  # For saving correct row fit

        x = dat.Data.get_data('x', data_group_name=data_group_name)
        data = dat.Data.get_data(name, data_group_name=data_group_name)[s:f]

        # For centering if data does not already exist or overwrite is True
        if dat.Logs.fds['ESC'] < centering_threshold:
            func_name = center_func if center_func is not None else t_func_name
            func, params = _get_transition_fit_func_params(x=x, data=np.mean(data, axis=0),
                                                           t_func_name=func_name,
                                                           theta=theta, gamma=gamma)

            center_fits = [dat.Transition.get_fit(which='row', row=row, name=f'{name}:{func_name}',
                                                  fit_func=func, initial_params=params,
                                                  data=d, x=x,
                                                  check_exists=False,
                                                  overwrite=overwrite) for row, d in zip(rows, data)]

            centers = [fit.best_values.mid for fit in center_fits]
        else:
            centers = [0]*len(dat.Data.get_data('y'))
        data_avg, x_avg = U.mean_data(x=x, data=data, centers=centers, method='linear', return_x=True)
        for d, n in zip([data_avg, x_avg], [name, 'x']):
            dat.Data.set_data(data=d, name=f'{n}_avg{data_row_name_append(data_rows)}',
                              data_group_name=data_group_name)

    x = dat.Data.get_data(f'x_avg{data_row_name_append(data_rows)}', data_group_name=data_group_name)
    data = dat.Data.get_data(f'{name}_avg{data_row_name_append(data_rows)}', data_group_name=data_group_name)

    try:
        fit = calculate_transition_only_fit(datnum, save_name=save_name, t_func_name=t_func_name, theta=theta,
                                        gamma=gamma, x=x, data=data, width=width,
                                        overwrite=overwrite)
    except (TypeError, ValueError):
        print(f'Dat{dat.datnum}: Fit Failed. Returning None')
        fit = None
    return fit


def set_sf_from_transition(entropy_datnums, transition_datnums, fit_name, integration_info_name, dt_from_self=False,
                           fixed_dt=None, fixed_amp=None):
    for enum, tnum in progressbar(zip(entropy_datnums, transition_datnums)):
        edat = get_dat(enum)
        tdat = get_dat(tnum)
        try:
            _set_amplitude_from_transition_only(edat, tdat, fit_name, integration_info_name, dt_from_self=dt_from_self,
                                            fixed_dt=fixed_dt, fixed_amp=fixed_amp)
        except (TypeError, U.NotFoundInHdfError):
            print(f'Failed to set scaling factor for dat{enum} using dat{tnum}')


def _set_amplitude_from_transition_only(entropy_dat: DatHDF, transition_dat: DatHDF, fit_name, integration_info_name,
                                        dt_from_self,
                                        fixed_dt=None, fixed_amp=None):
    ed = entropy_dat
    td = transition_dat
    # for k in ['ESC', 'ESS', 'ESP']:
    if fixed_dt is None:
        dt = get_deltaT(ed, from_self=dt_from_self, fit_name=fit_name)
    else:
        dt = fixed_dt

    if fixed_amp is None:
        for k in ['ESP']:
            if ed.Logs.fds[k] != td.Logs.fds[k]:
                raise ValueError(f'Non matching FDS for entropy_dat {ed.datnum} and transition_dat {td.datnum}: \n'
                                 f'entropy_dat fds = {ed.Logs.fds}\n'
                                 f'transition_dat fds = {td.Logs.fds}')
        amp = td.Transition.get_fit(name=fit_name).best_values.amp
    else:
        amp = fixed_amp
    ed.Entropy.set_integration_info(dT=dt,
                                    amp=amp if amp is not None else np.nan,
                                    name=integration_info_name,
                                    overwrite=True)
    return True


@deprecated
def calculate_csq_mapped_se_output(datnum: int, csq_datnum: Optional[int] = None, save_name: str = 'csq_mapped',
                                   process_params: Optional = None, centers: Optional[List[float]] = None,
                                   overwrite=False):
    dat = get_dat(datnum)
    if save_name not in dat.SquareEntropy.Output_names() or overwrite:
        if 'csq_mapped' not in dat.Data.keys:
            calculate_csq_map(datnum, csq_datnum=csq_datnum, overwrite=overwrite)

        x = dat.Data.get_data('csq_x_avg')
        data = dat.Data.get_data('csq_mapped')

        if process_params is None:
            process_params = dat.SquareEntropy.get_ProcessParams()

        inp = dat.SquareEntropy.get_Inputs(x_array=x, i_sense=data, centers=centers,
                                           save_name=save_name)

        dat.SquareEntropy.get_Outputs(name=save_name, inputs=inp, process_params=process_params,
                                      overwrite=overwrite, check_exists=False)
    out = dat.SquareEntropy.get_Outputs(name=save_name, check_exists=True)
    return out


def setup_csq_dat(csq_datnum: int, overwrite=False):
    """Run this on the CSQ dat once to set up the interpolating datasets"""
    csq_dat = get_dat(csq_datnum)
    if any([name not in csq_dat.Data.keys for name in ['csq_x', 'csq_data']]) or overwrite:
        csq_data = csq_dat.Data.get_data('cscurrent')
        csq_x = csq_dat.Data.get_data('x')

        in_range = np.where(np.logical_and(csq_data < 16, csq_data > 1))
        cdata = U.decimate(csq_data[in_range], measure_freq=csq_dat.Logs.measure_freq, numpnts=100)
        cx = U.get_matching_x(csq_x[in_range], cdata)  # Note: cx is the target y axis for cscurrent

        # Remove a few nans from beginning and end (from decimating)
        cx = cx[np.where(~np.isnan(cdata))]
        cdata = cdata[np.where(~np.isnan(cdata))]

        cx = cx - cx[U.get_data_index(cdata, 7.25,
                                      is_sorted=True)]  # Make 7.25nA be the zero point since that's where I try center the CS

        csq_dat.Data.set_data(cx, name='csq_x')
        csq_dat.Data.set_data(cdata, name='csq_data')


def calculate_csq_map(datnum: int, csq_datnum: Optional[int] = None, overwrite=False):
    """Do calculations to generate data in csq gate from i_sense using csq trace from csq_dat"""
    if csq_datnum is None:
        csq_datnum = 1619
    dat = get_dat(datnum)
    csq_dat = get_dat(csq_datnum)

    if 'csq_mapped' not in dat.Data.keys or overwrite:
        if any([name not in csq_dat.Data.keys for name in ['csq_x', 'csq_data']]):
            raise RuntimeError(f'CSQ_Dat{csq_datnum}: Has not been initialized, run setup_csq_dat({csq_datnum}) first')
        cx = csq_dat.Data.get_data('csq_x')
        cdata = csq_dat.Data.get_data('csq_data')

        interper = interp1d(cdata, cx, kind='linear', bounds_error=False, fill_value=np.nan)
        odata = dat.Data.get_data('i_sense')

        ndata = interper(odata)

        dat.Data.set_data(ndata, name='csq_mapped')
    return dat.Data.get_data('csq_mapped')


def data_row_name_append(data_rows: Optional[Tuple[Optional[int], Optional[int]]]) -> str:
    if data_rows is not None and not all(v is None for v in data_rows):
        return f':Rows[{data_rows[0]}:{data_rows[1]}]'
    else:
        return ''


def _calculate_csq_avg(datnum: int, centers=None,
                       data_rows: Optional[Tuple[Optional[int], Optional[int]]] = None) -> Tuple[
    np.ndarray, np.ndarray]:
    dat = get_dat(datnum)
    if centers is None:
        logger.warning(f'Dat{dat.datnum}: No centers passed for averaging CSQ mapped data')
        raise ValueError('Need centers')

    x = dat.Data.get_data('x')
    data = dat.Data.get_data('csq_mapped')[data_rows[0]: data_rows[1]]

    avg_data, csq_x_avg = U.mean_data(x, data, centers, method='linear', return_x=True)

    dat.Data.set_data(avg_data, f'csq_mapped_avg{data_row_name_append(data_rows)}')
    dat.Data.set_data(csq_x_avg, f'csq_x_avg{data_row_name_append(data_rows)}')
    return avg_data, csq_x_avg


def calculate_csq_mapped_avg(datnum: int, csq_datnum: Optional[int] = None,
                             centers: Optional[List[float]] = None,
                             data_rows: Tuple[Optional[int], Optional[int]] = (None, None),
                             overwrite=False):
    """Calculates CSQ mapped data, and averaaged data and saves in dat.Data....
    Note: Not really necessary to have avg data calculated for square entropy, because running SE will average and
    center data anyway
    """
    dat = get_dat(datnum)
    if 'csq_mapped' not in dat.Data.keys or overwrite:
        calculate_csq_map(datnum, csq_datnum=csq_datnum, overwrite=overwrite)

    if f'csq_mapped_avg{data_row_name_append(data_rows)}' not in dat.Data.keys or overwrite:
        _calculate_csq_avg(datnum, centers=centers, data_rows=data_rows)

    return dat.Data.get_data(f'csq_mapped_avg{data_row_name_append(data_rows)}'), \
           dat.Data.get_data(f'csq_x_avg{data_row_name_append(data_rows)}')


def data_from_output(o: Output, w: str):
    if w == 'i_sense_cold':
        return np.nanmean(o.averaged[(0, 2,), :], axis=0)
    elif w == 'i_sense_hot':
        return np.nanmean(o.averaged[(1, 3,), :], axis=0)
    elif w == 'entropy':
        return o.average_entropy_signal
    elif w == 'dndt':
        return o.average_entropy_signal
    elif w == 'integrated':
        d = np.nancumsum(o.average_entropy_signal)
        return d / np.nanmax(d)
    else:
        return None


def dat_integrated_sub_lin(dat: DatHDF, signal_width: float, int_info_name: str,
                           output_name: Optional[str] = None) -> np.ndarray:
    """
    Returns integrated entropy signal subtract average linear term from both sides outside of 'signal_width' from center
    of transition
    Args:
        dat ():
        signal_width ():
        int_info_name (): Name of integrated info to use
        output_name (): Optional name of SE output to use (defaults to int_info_name)

    Returns:
        np.ndarray: Integrated Entropy subtract average linear term
    """
    if output_name is None:
        output_name = int_info_name
    out = dat.SquareEntropy.get_Outputs(name=output_name)
    x = out.x
    data = dat.Entropy.get_integrated_entropy(name=int_info_name, data=out.average_entropy_signal)
    tdata = np.nanmean(out.averaged[(0, 2), :], axis=0)
    center = center_from_diff_i_sense(x, tdata, measure_freq=dat.Logs.measure_freq)
    return integrated_data_sub_lin(x=x, data=data, center=center, width=signal_width)


def center_from_diff_i_sense(x, data, measure_freq: Optional[float] = None) -> float:
    if measure_freq:
        smoothed = U.decimate(data, measure_freq=measure_freq, numpnts=20)
        x = U.get_matching_x(x, smoothed)
    else:
        smoothed = data
    return x[np.nanargmin(np.diff(smoothed))]


def integrated_data_sub_lin(x: np.ndarray, data: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Calculates linear term outside of center+-width and subtracts that from data

    Args:
        x ():
        data ():
        center (): Center of transition
        width (): Width of signal (i.e. will ignore center+-width when calculating linear terms)

    Returns:

    """
    line = lm.models.LinearModel()

    lower, upper = U.get_data_index(x, [center - width, center + width])
    l1_x, l1_data = x[:lower], data[:lower]
    l2_x, l2_data = x[upper:], data[upper:]

    line_fits = []
    for x_, data_ in zip([l1_x, l2_x], [l1_data, l2_data]):
        pars = line.make_params()
        pars['slope'].value = 0
        pars['intercept'].value = 0
        fit = calculate_fit(x_, data_, params=pars, func=line.func)
        line_fits.append(fit)

    avg_slope = np.mean([fit.best_values.slope for fit in line_fits if fit is not None])
    avg_intercept = np.mean([fit.best_values.intercept for fit in line_fits if fit is not None])

    pars = line.make_params()
    pars['slope'].value = avg_slope
    pars['intercept'].value = avg_intercept

    data_sub_lin = data - line.eval(x=x, params=pars)
    data_sub_lin = data_sub_lin - np.nanmean(data_sub_lin[:lower])
    return data_sub_lin


def sort_by_temps(dats: List[DatHDF]) -> Dict[float, List[DatHDF]]:
    d = {
        temp: [dat for dat in dats if np.isclose(dat.Logs.temps.mc * 1000, temp, atol=25)]
        for temp in [500, 400, 300, 200, 100, 50, 10]}
    for k in list(d.keys()):
        if len(d[k]) == 0:
            d.pop(k)
    return d


def sort_by_coupling(dats: List[DatHDF]) -> Dict[float, List[DatHDF]]:
    d = {
        gate: [dat for dat in dats if np.isclose(dat.Logs.fds['ESC'], gate, atol=5)]
        for gate in set([U.my_round(dat.Logs.fds['ESC'], base=10) for dat in dats])}
    for k in d:
        if len(d[k]) == 0:
            d.pop(k)
    return d


def multiple_csq_maps(csq_datnums: List[int], datnums_to_map: List[int],
                      sort_func: Optional[Callable] = None,
                      warning_tolerance: Optional[float] = None,
                      overwrite=False) -> True:
    """
    Using `csq_datnums`, will map all `datnums_to_map` based on whichever csq dat has is closest based on `sort_func`
    Args:
        csq_datnums (): All the csq datnums which might be used to do csq mapping (only closest based on sort_func will be used)
        datnums_to_map (): All the data datnums which should be csq mapped
        sort_func (): A function which takes dat as the argument and returns a float or int
        warning_tolerance: The max distance a dat can be from the csq dats based on sort_func without giving a warning
        overwrite (): Whether to overwrite prexisting mapping stuff

    Returns:
        bool: Success
    """
    if sort_func is None:
        sort_func = lambda dat: dat.Logs.fds['ESC']
    csq_dats = get_dats(csq_datnums)
    csq_dict = {sort_func(dat): dat for dat in csq_dats}
    transition_dats = get_dats(datnums_to_map)

    for num in progressbar(csq_datnums):
        setup_csq_dat(num, overwrite=overwrite)

    csq_sort_vals = list(csq_dict.keys())
    for dat in progressbar(transition_dats):
        closest_val = csq_sort_vals[get_data_index(np.array(csq_sort_vals), sort_func(dat))]
        if warning_tolerance is not None:
            if (dist := abs(closest_val-sort_func(dat))) > warning_tolerance:
                logging.warning(f'Dat{dat.datnum}: Closest CSQ dat has distance {dist:.2f} from Dat based on sort_func')
        calculate_csq_map(dat.datnum, csq_dict[closest_val].datnum, overwrite=overwrite)
    return True


def linear_fit_thetas(dats: List[DatHDF], fit_name: str, filter_func: Optional[Callable] = None,
                      show_plots=False) -> FitInfo:
    if filter_func is not None:
        fit_dats = [dat for dat in dats if filter_func(dat)]
    else:
        fit_dats = dats

    thetas = []
    escs = []
    for dat in fit_dats:
        thetas.append(dat.Transition.get_fit(name=fit_name).best_values.theta / 1000)
        escs.append(dat.Logs.fds['ESC'])

    thetas = np.array(U.order_list(thetas, escs))
    escs = np.array(U.order_list(escs))

    line = lm.models.LinearModel()
    fit = calculate_fit(x=escs, data=thetas, params=line.make_params(), func=line.func)

    if show_plots:
        plotter = OneD(dats=dats)
        fig = plotter.figure(xlabel='ESC /mV', ylabel='Theta /mV (real)',
                             title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: '
                                   f'Linear theta fit to Dats{min([dat.datnum for dat in fit_dats])}-'
                                   f'{max([dat.datnum for dat in fit_dats])}')
        fig.add_trace(plotter.trace(data=thetas, x=escs, name='Fit Data', mode='markers'))
        other_dats = [dat for dat in dats if dat not in fit_dats]
        if len(other_dats) > 0:
            other_thetas = []
            other_escs = []
            for dat in other_dats:
                other_thetas.append(dat.Transition.get_fit(name=fit_name).best_values.theta / 1000)
                other_escs.append(dat.Logs.fds['ESC'])
            other_thetas = np.array(U.order_list(other_thetas, other_escs))
            other_escs = np.array(U.order_list(other_escs))
            fig.add_trace(plotter.trace(data=other_thetas, x=other_escs, name='Other Data', mode='markers'))

        all_escs = np.array(sorted([dat.Logs.fds['ESC'] for dat in dats]))
        fig.add_trace(plotter.trace(data=fit.eval_fit(x=all_escs), x=all_escs, name='Fit', mode='lines'))
        fig.show()

    return fit


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


def integrated_entropy_value(dat, fit_name: str) -> float:
    int_info = dat.Entropy.get_integration_info(name=fit_name)
    entropy_signal = dat.SquareEntropy.get_Outputs(name=fit_name).average_entropy_signal
    return float(np.nanmean(int_info.integrate(entropy_signal)[-10:]))