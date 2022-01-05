from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union, TYPE_CHECKING

import h5py
import lmfit as lm
import numpy as np
from deprecation import deprecated

from .. import useful_functions as U
from ..hdf_file_handler import HDFFileHandler
from .general_fitting import _get_transition_fit_func_params, calculate_se_entropy_fit, \
    calculate_se_transition, calculate_fit
from .square_wave import get_setpoint_indexes_from_times
from .transition import center_from_diff_i_sense

from ..hdf_util import DatDataclassTemplate, with_hdf_write

if TYPE_CHECKING:
    from ..dat_object.dat_hdf import DatHDF


@dataclass
class GammaAnalysisParams(DatDataclassTemplate):
    """All the various things that go into calculating Entropy etc"""
    experiment_name: str
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

    def to_dash_element(self):
        return self.__str__().replace('\t', '&nbsp&nbsp&nbsp').replace('\n', '<br>')
        # lines = self.__str__().replace('\t', '    ').split('\n')
        # dash_lines = [[html.P(l), html.Br()] for l in lines]
        # div = html.Div([c for line in dash_lines for c in line])
        # return div


def save_gamma_analysis_params_to_dat(dat: DatHDF, analysis_params: GammaAnalysisParams,
                                      name: str):
    """Save GammaAnalysisParams to suitable place in DatHDF"""
    @with_hdf_write
    def save_params(d: DatHDF):
        analysis_group = d.hdf.hdf.require_group('Gamma Analysis')
        analysis_params.save_to_hdf(analysis_group, name=name)
    save_params(dat)


def load_gamma_analysis_params(dat: DatHDF, name: str) -> GammaAnalysisParams:
    """Load GammaAnalysisParams from DatHDF"""
    with HDFFileHandler(dat.hdf.hdf_path, 'r') as hdf:
        analysis_group = hdf.get('Gamma Analysis')
        analysis_params = GammaAnalysisParams.from_hdf(analysis_group, name=name)
    return analysis_params


def do_entropy_calc(datnum, save_name: str,
                    setpoint_start: float = 0.005,
                    t_func_name: str = 'i_sense', csq_mapped=False,
                    center_for_avg: Union[bool, float] = True,
                    theta=None, gamma=None, width=None,
                    data_rows: Tuple[Optional[int], Optional[int]] = (None, None),
                    experiment_name: Optional[str] = None,
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
        experiment_name (): which cooldown basically e.g. FebMar21
        overwrite ():

    Returns:

    """
    from ..dat_object.make_dat import get_dat
    dat = get_dat(datnum, exp2hdf=experiment_name)
    print(f'Working on {datnum}')

    if isinstance(center_for_avg, float):
        center_for_avg = True if dat.Logs.fds['ESC'] < center_for_avg else False

    sp_start, sp_fin = get_setpoint_indexes_from_times(dat, start_time=setpoint_start, end_time=None)

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
        ent = calculate_se_entropy_fit(datnum, save_name=save_name, se_output_name=save_name, width=width,
                                       center=center,
                                       experiment_name=experiment_name,
                                       overwrite=overwrite)

        for t in ['cold', 'hot']:
            calculate_se_transition(datnum, save_name=save_name + f'_{t}', se_output_name=save_name,
                                    t_func_name=t_func_name,
                                    theta=theta, gamma=gamma,
                                    experiment_name=experiment_name,
                                    transition_part=t, width=width, center=center, overwrite=overwrite)

    except (TypeError, ValueError):
        print(f'Dat{dat.datnum}: Failed to calculate entropy or transition fit')
        return False
    return True


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


def calculate_new_sf_only(entropy_datnum: int, save_name: str,
                          dt: Optional[float] = None, amp: Optional[float] = None,
                          from_square_transition: bool = False, transition_datnum: Optional[int] = None,
                          fit_name: Optional[str] = None,
                          experiment_name: Optional[str] = None,
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
        experiment_name (): which cooldown basically e.g. FebMar21

    Returns:

    """
    from ..dat_object.make_dat import get_dat
    # Set integration info
    if from_square_transition:  # Only sets dt and amp if not forced
        dat = get_dat(entropy_datnum, exp2hdf=experiment_name)
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
            dat = get_dat(transition_datnum, exp2hdf=experiment_name)
            amp = dat.Transition.get_fit(name=fit_name).best_values.amp

    dat = get_dat(entropy_datnum, exp2hdf=experiment_name)
    dat.Entropy.set_integration_info(dT=dt, amp=amp,
                                     name=save_name, overwrite=True)  # Fast to write


def integrated_entropy_value(dat, fit_name: str) -> float:
    int_info = dat.Entropy.get_integration_info(name=fit_name)
    entropy_signal = dat.SquareEntropy.get_Outputs(name=fit_name).average_entropy_signal
    return float(np.nanmean(int_info.integrate(entropy_signal)[-10:]))


def get_deltaT(dat: DatHDF, fit_name: Optional[str] = None) -> float:
    """
    Returns deltaT of a given dat in mV by comparing hot and cold fit of SquareEntropy data

    Args:
        dat (): Dat to calculate dT from
        fit_name (): Optional name of fits to use for calculating dT

    Returns:
        dT in mV
    """
    cold_fit = dat.SquareEntropy.get_fit(fit_name=fit_name)
    hot_fit = dat.SquareEntropy.get_fit(which_fit='transition', transition_part='hot',
                                        initial_params=cold_fit.params, check_exists=False, output_name=fit_name,
                                        fit_name=fit_name + '_hot')
    if all([fit.best_values.theta is not None for fit in [cold_fit, hot_fit]]):
        dt = hot_fit.best_values.theta - cold_fit.best_values.theta
        return dt
    else:
        raise RuntimeError(f'Failed to calculate dT for dat{dat.dat_id}. cold_fit, hot_fit: \n{cold_fit}\n{hot_fit}')


@deprecated(details='Use get_deltaT instead, this function also includes fallback for getting dT from fixed datnums '
                    'which is not good to use in general')
def _get_deltaT(dat: DatHDF, from_self=False, fit_name: str = None, default_dt=None,
                experiment_name: Optional[str] = None,
                ):
    """
    Returns deltaT of a given dat in mV by comparing hot and cold fit of SquareEntropy data
    Args:
        dat (): Dat to calculate dT for
        from_self ():
        fit_name ():
        default_dt ():
        experiment_name ():

    Returns:

    """
    from ..dat_object.make_dat import get_dats
    if from_self is False:
        ho1 = dat.AWG.max(0)  # 'HO1/10M' gives nA * 10
        t = dat.Logs.temps.mc

        # Datnums to search through (only thing that should be changed)
        # datnums = set(range(1312, 1451 + 1)) - set(range(1312, 1451 + 1, 4))
        datnums = list(range(2143, 2156))

        dats = get_dats(datnums, exp2hdf=experiment_name)

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