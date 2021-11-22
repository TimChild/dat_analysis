from typing import Optional, Tuple, List, Callable

import lmfit as lm
import numpy as np
from plotly import graph_objects as go

from dat_analysis.core_util import data_row_name_append
from dat_analysis import useful_functions as U
from dat_analysis.analysis_tools.general_fitting import FitInfo, _get_transition_fit_func_params, calculate_transition_only_fit, \
    calculate_fit
from dat_analysis.dat_object.dat_hdf import DatHDF
from dat_analysis.dat_object.make_dat import get_dat
from dat_analysis.plotting.plotly import OneD


def do_transition_only_calc(datnum, save_name: str,
                            theta=None, gamma=None, width=None, t_func_name='i_sense_digamma',
                            center_func: Optional[str] = None,
                            csq_mapped=False, data_rows: Tuple[Optional[int], Optional[int]] = (None, None),
                            centering_threshold: float = 1000,
                            experiment_name: Optional[str] = None,
                            overwrite=False) -> FitInfo:
    """
    Do calculations on Transition only measurements. Can be run multiprocessed
    Args:
        datnum ():
        save_name (): Name to save fits under in dat.Transition...
        theta (): If set, theta is forced to this value
        gamma (): If set, gamma is forced to this value
        width (): Width of fitting around center of transition
        t_func_name (): Name of fitting function to use for final averaged fit
        center_func (): Name of fitting function to use for centering data
        csq_mapped (): Whether to use CSQ mapped data
        data_rows (): Optionally select only certain rows to fit
        centering_threshold (): If dat.Logs.dacs['ESC'] is below this value, centering will happen
        experiment_name (): which cooldown basically e.g. FebMar21
        overwrite (): Whether to overwrite existing fits

    Returns:
        FitInfo: Returns fit, but fit is also saved in dat.Transition
    """
    dat = get_dat(datnum, exp2hdf=experiment_name)
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
        if dat.Logs.dacs['ESC'] < centering_threshold:
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
            centers = [0] * len(dat.Data.get_data('y'))
        data_avg, x_avg = U.mean_data(x=x, data=data, centers=centers, method='linear', return_x=True)
        for d, n in zip([data_avg, x_avg], [name, 'x']):
            dat.Data.set_data(data=d, name=f'{n}_avg{data_row_name_append(data_rows)}',
                              data_group_name=data_group_name)

    x = dat.Data.get_data(f'x_avg{data_row_name_append(data_rows)}', data_group_name=data_group_name)
    data = dat.Data.get_data(f'{name}_avg{data_row_name_append(data_rows)}', data_group_name=data_group_name)

    try:
        fit = calculate_transition_only_fit(datnum, save_name=save_name, t_func_name=t_func_name, theta=theta,
                                            gamma=gamma, x=x, data=data, width=width,
                                            experiment_name=experiment_name,
                                            overwrite=overwrite)
    except (TypeError, ValueError):
        print(f'Dat{dat.datnum}: Fit Failed. Returning None')
        fit = None
    return fit


def linear_fit_thetas(dats: List[DatHDF], fit_name: str, filter_func: Optional[Callable] = None,
                      show_plots=False,
                      sweep_gate_divider=100,
                      dat_attr_saved_in: str = 'transition',
                      ) -> FitInfo:
    """
    Takes thetas from named fits and plots on graph, then fits a line through any which pass filter_func returning the
    linear FitInfo

    Args:
        dats (): List of dats to include in plot
        fit_name (): Name fit is saved under (also may need to specify which dat_attr it is saved in)
        filter_func (): Function which takes a single dat and returns True or False for whether it should be included
            in linear fit. E.g. lambda dat: True if dat.Logs.dacs['ESC'] < -280 else False
        show_plots (): Whether to show the intermediate plots (i.e. thetas with linear fit)
        sweep_gate_divider (): How much to divide x-axis to get into real mV
        dat_attr_saved_in (): I.e. saved in dat.Transition or dat.NrgOcc

    Returns:

    """
    if filter_func is None:
        filter_func = lambda dat: True

    def _get_theta(dat: DatHDF) -> float:
        """Get theta from a dat"""
        if dat_attr_saved_in == 'transition':
            theta = dat.Transition.get_fit(name=fit_name).best_values.theta
        elif dat_attr_saved_in == 'nrg':
            theta = dat.NrgOcc.get_fit(name=fit_name).best_values.theta
        else:
            raise NotImplementedError
        return theta/sweep_gate_divider

    def _get_x_and_thetas(dats: List[DatHDF]) -> Tuple[np.ndarray, np.ndarray]:
        """Get the x and theta for each dat and return sorted list based on x"""
        x, thetas = [], []
        for dat in dats:
            x.append(dat.Logs.dacs['ESC'])
            thetas.append(_get_theta(dat))
        thetas = np.array(U.order_list(thetas, x))
        x = np.array(U.order_list(x))
        return x, thetas

    def get_data_to_fit() -> Tuple[np.ndarray, np.ndarray]:
        """Get the sorted x and theta values to plot/fit"""
        fit_dats = [dat for dat in dats if filter_func(dat)]
        return _get_x_and_thetas(fit_dats)

    def get_other_data() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        other_dats = [dat for dat in dats if filter_func(dat) is False]
        if other_dats:
            return _get_x_and_thetas(other_dats)
        else:
            return None, None

    def plot_data(fig, x_, thetas_, name: str) -> go.Figure:
        """Add data to figure"""
        fig.add_trace(plotter.trace(data=thetas_, x=x_, name=name, mode='markers'))
        return fig

    def plot_fit(fit: FitInfo, x_) -> go.Figure:
        """Add fit to figure"""
        x_ = np.array((sorted(x_)))
        fig.add_trace(plotter.trace(data=fit.eval_fit(x=x_), x=x_, name='Fit', mode='lines'))
        return fig

    # Data to fit to
    x, thetas = get_data_to_fit()

    # Do linear fit
    line = lm.models.LinearModel()
    fit = calculate_fit(x=x, data=thetas, params=line.make_params(), func=line.func)

    # IF plotting
    if show_plots:
        # Plot fit data
        plotter = OneD(dats=dats)
        fig = plotter.figure(xlabel='ESC /mV', ylabel='Theta /mV (real)',
                             title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: Theta vs ESC')
        fig = plot_data(fig, x, thetas, name='Fit Data')

        # Plot other data
        other_x, other_thetas = get_other_data()
        if other_x is not None:
            fig = plot_data(fig, other_x, other_thetas, name='Other Data')

        # Plot fit line through all
        plot_fit(fit, np.concatenate([x, other_x])).show()
    return fit


def center_from_diff_i_sense(x, data, measure_freq: Optional[float] = None) -> float:
    if measure_freq:
        smoothed = U.decimate(data, measure_freq=measure_freq, numpnts=20)
        x = U.get_matching_x(x, smoothed)
    else:
        smoothed = data
    return x[np.nanargmin(np.diff(smoothed))]