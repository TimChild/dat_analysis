from typing import List, Callable, Optional, Union, Tuple

import lmfit as lm
import scipy
import scipy.io
from scipy.interpolate import RectBivariateSpline
import numpy as np
from deprecation import deprecated
from plotly import graph_objects as go
from progressbar import progressbar
from scipy.interpolate import interp1d
import logging

import UsefulFunctions as U
from Dash.DatPlotting import OneD
from DatObject.DatHDF import DatHDF
from src.DatObject.Attributes.DatAttribute import FitInfo
from src.DatObject.Attributes.SquareEntropy import square_wave_time_array

from src.UsefulFunctions import edit_params
from src.Dash.DatPlotting import OneD
from src.DatObject.Attributes.Transition import i_sense, i_sense_digamma, i_sense_digamma_amplin, \
    get_transition_function
from src.DatObject.DatHDF import DatHDF

from src.DatObject.Make_Dat import get_dats, get_dat, DatHandler
from src.Plotting.Plotly.PlotlyUtil import HoverInfo, additional_data_dict_converter

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


def plot_fit_integrated_comparison(dats: List[DatHDF], x_func: Callable, x_label: str, title_append: Optional[str] = '',
                                   int_info_name: Optional[str] = None, fit_name: str = 'SPS.0045',
                                   plot=True) -> go.Figure():
    if int_info_name is None:
        int_info_name = 'default'
    plotter = OneD(dats=dats)
    fig = plotter.figure(
        title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: Fit and Integrated (\'{int_info_name}\') Entropy{title_append}',
        xlabel=x_label,
        ylabel='Entropy /kB')

    hover_infos = [
        HoverInfo(name='Dat', func=lambda dat: dat.datnum, precision='.d', units=''),
        HoverInfo(name='Temperature', func=lambda dat: dat.Logs.temps.mc * 1000, precision='.1f', units='mK'),
        HoverInfo(name='Bias', func=lambda dat: dat.AWG.max(0) / 10, precision='.1f', units='nA'),
        # HoverInfo(name='Fit Entropy', func=lambda dat: dat.SquareEntropy.get_fit(which_fit='entropy', fit_name=fit_name).best_values.dS,
        #           precision='.2f', units='kB'),
        HoverInfo(name='Integrated Entropy',
                  func=lambda dat: np.nanmean(dat.Entropy.get_integrated_entropy(name=int_info_name,
                                                                                 data=dat.SquareEntropy.get_Outputs(
                                                                                     name=fit_name).average_entropy_signal)[
                                              -10:]),
                  # TODO: Change to using proper output (with setpoints)
                  precision='.2f', units='kB'),
    ]

    funcs, template = additional_data_dict_converter(hover_infos)
    x = [x_func(dat) for dat in dats]
    hover_data = [[func(dat) for func in funcs] for dat in dats]

    entropies = [dat.Entropy.get_fit(name=fit_name).best_values.dS for dat in dats]
    # entropy_errs = [np.nanstd([
    #     f.best_values.dS if f.best_values.dS is not None else np.nan
    #     for f in dat.Entropy.get_row_fits(name=fit_name) for dat in dats
    # ]) / np.sqrt(dat.Data.y_array.shape[0]) for dat in dats]
    entropy_errs = None
    fig.add_trace(plotter.trace(
        data=entropies, data_err=entropy_errs, x=x, name=f'Fit Entropy',
        mode='markers',
        trace_kwargs={'customdata': hover_data, 'hovertemplate': template})
    )
    integrated_entropies = [np.nanmean(
        dat.Entropy.get_integrated_entropy(name=int_info_name,
                                           data=dat.SquareEntropy.get_Outputs(name=fit_name).average_entropy_signal)[
        -10:]) for dat in dats]

    ### For plotting the impurity dot scans (i.e. isolate only the integrated entropy due to the main dot transition)
    # integrated_entropies = []
    # for dat in dats:
    #     out = dat.SquareEntropy.get_Outputs(name=fit_name)
    #     x1, x2 = U.get_data_index(out.x, [-40, 40], is_sorted=True)
    #     integrated = dat.Entropy.get_integrated_entropy(name=int_info_name,
    #                                                     data=out.average_entropy_signal)
    #     integrated_entropies.append(integrated[x2]-integrated[x1])

    fig.add_trace(plotter.trace(
        data=integrated_entropies, x=x, name=f'Integrated',
        mode='markers+lines',
        trace_kwargs={'customdata': hover_data, 'hovertemplate': template})
    )

    if plot:
        fig.show(renderer='browser')
    return fig


def entropy_vs_time_trace(dats: List[DatHDF], trace_name=None, integrated=False, fit_name: str = 'SPS.005',
                          integrated_name: str = 'first'):
    plotter = OneD(dats=dats)
    if integrated is False:
        entropies = [dat.Entropy.get_fit(which='avg', name=fit_name).best_values.dS for dat in dats]
        entropies_err = [np.nanstd(
            [fit.best_values.dS if fit.best_values.dS is not None else np.nan for fit in
             dat.Entropy.get_row_fits(name=fit_name)]
        ) / np.sqrt(len(dats)) for dat in dats]
    else:
        entropies = [np.nanmean(
            dat.Entropy.get_integrated_entropy(
                name=integrated_name,
                data=dat.SquareEntropy.get_Outputs(name=fit_name).average_entropy_signal)[-10:]) for dat in dats]
        entropies_err = [0.0] * len(dats)

    times = [str(dat.Logs.time_completed) for dat in dats]

    trace = plotter.trace(data=entropies, data_err=entropies_err, x=times, text=[dat.datnum for dat in dats],
                          mode='lines', name=trace_name)
    return trace


def entropy_vs_time_fig(title: Optional[str] = None):
    plotter = OneD()
    if title is None:
        title = 'Entropy vs Time'
    fig = plotter.figure(xlabel='Time', ylabel='Entropy /kB', title=title)
    fig.update_xaxes(tickformat="%H:%M\n%a")
    return fig


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


def do_entropy_calc(datnum, save_name: str,
                    setpoint_start: float = 0.005,
                    t_func_name: str = 'i_sense', csq_mapped=False,
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
        csq_mapped (): Whether to use i_sense data mapped back to CSQ gate instead
        theta ():
        gamma ():
        width ():
        overwrite ():

    Returns:

    """
    dat = get_dat(datnum)

    setpoints = [setpoint_start, None]

    setpoint_times = square_wave_time_array(dat.SquareEntropy.square_awg)
    sp_start, sp_fin = [U.get_data_index(setpoint_times, sp) for sp in setpoints]

    s, f = data_rows

    x = dat.Data.get_data('x')
    if csq_mapped:
        data = dat.Data.get_data('csq_mapped')[s:f]
    else:
        data = dat.Transition.get_data('i_sense')[s:f]

    # Run Fits
    t_func, params = _get_transition_fit_func_params(datnum, x=x, data=np.mean(data, axis=0),
                                                     t_func_name=t_func_name,
                                                     theta=theta, gamma=gamma)
    pp = dat.SquareEntropy.get_ProcessParams(name=None,  # Load default and modify from there
                                             setpoint_start=sp_start, setpoint_fin=sp_fin,
                                             transition_fit_func=t_func,
                                             transition_fit_params=params,
                                             save_name=save_name)
    inps = dat.SquareEntropy.get_Inputs(x_array=x, i_sense=data, save_name=save_name)
    out = dat.SquareEntropy.get_Outputs(name=save_name, inputs=inps, process_params=pp, overwrite=overwrite)

    center = float(np.nanmean(out.centers_used))

    ent = calculate_se_entropy_fit(datnum, save_name=save_name, se_output_name=save_name, width=width, center=center,
                                   overwrite=overwrite)

    for t in ['cold', 'hot']:
        calculate_se_transition(datnum, save_name=save_name + f'_{t}', se_output_name=save_name,
                                t_func_name=t_func_name,
                                theta=theta, gamma=gamma,
                                transition_part=t, width=width, center=center, overwrite=overwrite)
    return True


def do_transition_only_calc(datnum, save_name: str,
                            theta=None, gamma=None, width=None, t_func_name='i_sense_digamma',
                            center_func: Optional[str] = None,
                            csq_mapped=False, data_rows: Tuple[Optional[int], Optional[int]] = (None, None),
                            overwrite=False) -> FitInfo:
    dat = get_dat(datnum)

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
        func_name = center_func if center_func is not None else t_func_name
        func, params = _get_transition_fit_func_params(datnum, x=x, data=np.mean(data, axis=0),
                                                       t_func_name=func_name,
                                                       theta=theta, gamma=gamma)

        center_fits = [dat.Transition.get_fit(which='row', row=row, name=f'{name}:{func_name}',
                                              fit_func=func, initial_params=params,
                                              data=d, x=x,
                                              check_exists=False,
                                              overwrite=overwrite) for row, d in zip(rows, data)]

        centers = [fit.best_values.mid for fit in center_fits]
        data_avg, x_avg = U.mean_data(x=x, data=data, centers=centers, method='linear', return_x=True)
        for d, n in zip([data_avg, x_avg], [name, 'x']):
            dat.Data.set_data(data=d, name=f'{n}_avg{data_row_name_append(data_rows)}',
                              data_group_name=data_group_name)

    x = dat.Data.get_data(f'x_avg{data_row_name_append(data_rows)}', data_group_name=data_group_name)
    data = dat.Data.get_data(f'{name}_avg{data_row_name_append(data_rows)}', data_group_name=data_group_name)

    fit = calculate_transition_only_fit(datnum, save_name=save_name, t_func_name=t_func_name, theta=theta,
                                        gamma=gamma, x=x, data=data, width=width,
                                        overwrite=overwrite)
    return fit


def get_default_transition_params(datnum: int, func_name: str,
                                  x: Optional[np.ndarray] = None, data: Optional[np.ndarray] = None) -> lm.Parameters:
    dat = get_dat(datnum)

    x = x if x is not None else dat.Transition.avg_x
    data = data if data is not None else dat.Transition.avg_data

    params = dat.Transition.get_default_params(x=x, data=data)
    if func_name == 'i_sense_digamma':
        params.add('g', 0, min=-50, max=1000, vary=True)
    elif func_name == 'i_sense_digamma_amplin':
        params.add('g', 0, min=-50, max=1000, vary=True)
        params.add('amplin', 0, vary=True)
    return params


def calculate_transition_only_fit(datnum, save_name, t_func_name: str = 'i_sense_digamma', theta=None, gamma=None,
                                  x: Optional[np.ndarray] = None, data: Optional[np.ndarray] = None,
                                  width: Optional[float] = None, center: Optional[float] = None,
                                  overwrite=False) -> FitInfo:
    dat = get_dat(datnum)

    x = x if x is not None else dat.Transition.avg_x
    data = data if data is not None else dat.Transition.avg_data

    x, data = _get_data_in_range(x, data, width, center=center)

    t_func, params = _get_transition_fit_func_params(datnum, x, data, t_func_name, theta, gamma)

    return dat.Transition.get_fit(name=save_name, fit_func=t_func,
                                  data=data, x=x, initial_params=params,
                                  check_exists=False, overwrite=overwrite)


def set_sf_from_transition(entropy_datnums, transition_datnums, fit_name, integration_info_name, dt_from_self=False,
                           fixed_dt=None, fixed_amp=None):
    for enum, tnum in progressbar(zip(entropy_datnums, transition_datnums)):
        edat = get_dat(enum)
        tdat = get_dat(tnum)
        _set_amplitude_from_transition_only(edat, tdat, fit_name, integration_info_name, dt_from_self=dt_from_self,
                                            fixed_dt=fixed_dt, fixed_amp=fixed_amp)


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


def _get_data_in_range(x: np.ndarray, data: np.ndarray, width: Optional[float], center: Optional[float] = None) -> \
        Tuple[np.ndarray, np.ndarray]:
    if center is None:
        center = 0
    if width is not None:
        x, data = np.copy(x), np.copy(data)

        start_ind = np.nanargmin(np.abs(np.add(x, width + center)))
        end_ind = np.nanargmin(np.abs(np.subtract(x, width + center)))

        x[:start_ind] = [np.nan] * start_ind
        x[end_ind:] = [np.nan] * (len(x) - end_ind)

        data[:start_ind] = [np.nan] * start_ind
        data[end_ind:] = [np.nan] * (len(data) - end_ind)
    return x, data


def _get_transition_fit_func_params(datnum, x, data, t_func_name, theta, gamma):
    t_func = get_transition_function(t_func_name)
    params = get_default_transition_params(datnum, t_func_name, x, data)
    if theta:
        params = U.edit_params(params, 'theta', value=theta, vary=False)
    if gamma is not None and 'g' in params:
        params = U.edit_params(params, 'g', gamma, False)
    return t_func, params


def calculate_se_transition(datnum: int, save_name: str, se_output_name: str, t_func_name: str = 'i_sense_digamma',
                            theta=None, gamma=None,
                            transition_part: str = 'cold',
                            width: Optional[float] = None, center: Optional[float] = None,
                            overwrite=False):
    dat = get_dat(datnum)
    data = dat.SquareEntropy.get_transition_part(name=se_output_name, part=transition_part, existing_only=True)
    x = dat.SquareEntropy.get_Outputs(name=se_output_name, check_exists=True).x

    x, data = _get_data_in_range(x, data, width, center=center)

    t_func, params = _get_transition_fit_func_params(datnum, x, data, t_func_name, theta, gamma)

    return dat.SquareEntropy.get_fit(which_fit='transition', transition_part=transition_part, fit_name=save_name,
                                     fit_func=t_func, initial_params=params, data=data, x=x, check_exists=False,
                                     overwrite=overwrite)


def calculate_se_entropy_fit(datnum: int, save_name: str, se_output_name: str,
                             width: Optional[float] = None, center: Optional[float] = None,
                             overwrite=False):
    dat = get_dat(datnum)
    out = dat.SquareEntropy.get_Outputs(name=se_output_name, check_exists=True)
    x = out.x
    data = out.average_entropy_signal

    x, data = _get_data_in_range(x, data, width, center)
    return dat.Entropy.get_fit(name=save_name, x=out.x, data=data, check_exists=False, overwrite=overwrite)


def setup_csq_dat(csq_datnum: int, overwrite=False):
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
        centers = dat.Transition.get_centers()

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


def NRG_fitter() -> Callable:
    NRG = scipy.io.loadmat('NRGResults.mat')
    occ = NRG["Occupation_mat"]
    ens = np.reshape(NRG["Ens"], 401)
    ts = np.reshape(NRG["Ts"], 70)
    ens = np.flip(ens)
    occ = np.flip(occ, 0)
    interp = RectBivariateSpline(ens, np.log10(ts), occ, kx=1, ky=1)

    def interpNRG(x, logt, dx=1, amp=1, center=0, lin=0, const=0):
        ens = np.multiply(np.add(x, center), dx)
        curr = [interp(en, logt)[0][0] for en in ens]
        scaled_current = np.multiply(curr, amp)
        scaled_current += const + np.multiply(lin, x)
        return scaled_current

    return interpNRG


def get_integrated_trace(dats: List[DatHDF], x_func: Callable, x_label: str,
                         trace_name: str,
                         fit_name: str,
                         int_info_name: Optional[str] = None, SE_output_name: Optional[str] = None,
                         ) -> go.Scatter:
    if int_info_name is None:
        int_info_name = 'first'
    if SE_output_name is None:
        SE_output_name = 'SPS.005'

    plotter = OneD(dats=dats)

    hover_infos = [
        HoverInfo(name='Dat', func=lambda dat: dat.datnum, precision='.d', units=''),
        HoverInfo(name=x_label, func=lambda dat: x_func(dat), precision='.1f', units='mV'),
        HoverInfo(name='Bias', func=lambda dat: dat.AWG.max(0) / 10, precision='.1f', units='nA'),
        HoverInfo(name='Fit Entropy',
                  func=lambda dat: dat.Entropy.get_fit(name=fit_name, check_exists=True).best_values.dS,
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
                                               name=SE_output_name, check_exists=True).average_entropy_signal
                                           )[-10:]) for dat in dats]
    trace = plotter.trace(
        data=integrated_entropies, x=x, name=trace_name,
        mode='markers',
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
                            fit_only=False, fit_name: str = 'narrow', transition_only=True,
                            se_output_name: str = 'SPS.005',
                            csq_mapped=False) -> go.Scatter():
    plotter = OneD(dat=dat)
    if transition_only:
        if csq_mapped:
            x = dat.Data.get_data('csq_x_avg')
        else:
            x = dat.Transition.avg_x
    else:
        if csq_mapped:
            raise NotImplementedError
        x = dat.SquareEntropy.avg_x

    if not fit_only:
        if transition_only:
            if csq_mapped:
                data = dat.Data.get_data('csq_mapped_avg')
            else:
                data = dat.Transition.avg_data
        else:
            data = dat.SquareEntropy.get_transition_part(name=se_output_name, part='cold')
    else:
        data = None  # Set below

    if fit_only or subtract_fit:
        if transition_only:
            fit = dat.Transition.get_fit(name=fit_name)
        else:
            fit = dat.SquareEntropy.get_fit(which_fit='transition', fit_name=fit_name)
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
