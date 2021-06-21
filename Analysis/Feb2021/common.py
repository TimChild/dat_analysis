from typing import List, Callable, Optional, Dict

import numpy as np
from progressbar import progressbar
import lmfit as lm
import logging

from src import UsefulFunctions as U
from src.AnalysisTools.csq_mapping import setup_csq_dat, calculate_csq_map
from src.AnalysisTools.entropy import integrated_data_sub_lin, _get_deltaT
from src.AnalysisTools.general_fitting import FitInfo, calculate_fit
from src.CoreUtil import get_data_index
from src.DatObject.Attributes.SquareEntropy import Output

from src.Dash.DatPlotting import OneD
from src.DatObject.DatHDF import DatHDF

from src.DatObject.Make_Dat import get_dats, get_dat

logger = logging.getLogger(__name__)


def set_sf_from_transition(entropy_datnums, transition_datnums, fit_name, integration_info_name, dt_from_self=False,
                           fixed_dt=None, fixed_amp=None, experiment_name: Optional[str] = None):
    for enum, tnum in progressbar(zip(entropy_datnums, transition_datnums)):
        edat = get_dat(enum, exp2hdf=experiment_name)
        tdat = get_dat(tnum, exp2hdf=experiment_name)
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
        dt = _get_deltaT(ed, from_self=dt_from_self, fit_name=fit_name)
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
                        experiment_name: Optional[str] = None,
                      overwrite=False) -> True:
    """
    Using `csq_datnums`, will map all `datnums_to_map` based on whichever csq dat has is closest based on `sort_func`
    Args:
        csq_datnums (): All the csq datnums which might be used to do csq mapping (only closest based on sort_func will be used)
        datnums_to_map (): All the data datnums which should be csq mapped
        sort_func (): A function which takes dat as the argument and returns a float or int
        warning_tolerance: The max distance a dat can be from the csq dats based on sort_func without giving a warning
        experiment_name (): which cooldown basically e.g. FebMar21
        overwrite (): Whether to overwrite prexisting mapping stuff

    Returns:
        bool: Success
    """
    if sort_func is None:
        sort_func = lambda dat: dat.Logs.fds['ESC']
    csq_dats = get_dats(csq_datnums, exp2hdf=experiment_name)
    csq_dict = {sort_func(dat): dat for dat in csq_dats}
    transition_dats = get_dats(datnums_to_map, exp2hdf=experiment_name)

    for num in progressbar(csq_datnums):
        setup_csq_dat(num, overwrite=overwrite)

    csq_sort_vals = list(csq_dict.keys())
    for dat in progressbar(transition_dats):
        closest_val = csq_sort_vals[get_data_index(np.array(csq_sort_vals), sort_func(dat))]
        if warning_tolerance is not None:
            if (dist := abs(closest_val - sort_func(dat))) > warning_tolerance:
                logging.warning(f'Dat{dat.datnum}: Closest CSQ dat has distance {dist:.2f} from Dat based on sort_func')
        calculate_csq_map(dat.datnum, experiment_name=experiment_name, csq_datnum=csq_dict[closest_val].datnum,
                          overwrite=overwrite,
                          )
    return True


def linear_fit_thetas(dats: List[DatHDF], fit_name: str, filter_func: Optional[Callable] = None,
                      show_plots=False,
                      sweep_gate_divider=100) -> FitInfo:
    if filter_func is not None:
        fit_dats = [dat for dat in dats if filter_func(dat)]
    else:
        fit_dats = dats

    thetas = []
    escs = []
    for dat in fit_dats:
        thetas.append(dat.Transition.get_fit(name=fit_name).best_values.theta / sweep_gate_divider)
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
                other_thetas.append(dat.Transition.get_fit(name=fit_name).best_values.theta / sweep_gate_divider)
                other_escs.append(dat.Logs.fds['ESC'])
            other_thetas = np.array(U.order_list(other_thetas, other_escs))
            other_escs = np.array(U.order_list(other_escs))
            fig.add_trace(plotter.trace(data=other_thetas, x=other_escs, name='Other Data', mode='markers'))

        all_escs = np.array(sorted([dat.Logs.fds['ESC'] for dat in dats]))
        fig.add_trace(plotter.trace(data=fit.eval_fit(x=all_escs), x=all_escs, name='Fit', mode='lines'))
        fig.show()

    return fit


