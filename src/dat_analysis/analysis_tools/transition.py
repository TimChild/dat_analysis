from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable, Union, Any, TYPE_CHECKING, Dict
from deprecation import deprecated

import lmfit as lm
import numpy as np
import pandas as pd
import logging
import h5py
from plotly import graph_objects as go
from scipy.signal import savgol_filter
from scipy.special import digamma

from .new_procedures import Process
from ..core_util import get_data_index, mean_data
from .. import useful_functions as U, core_util as CU
from .general_fitting import FitInfo, calculate_fit, get_data_in_range
from ..plotting.plotly import OneD
from ..hdf_util import params_to_HDF, params_from_HDF

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


FIT_NUM_BINS = 1000  # Much faster to bin data down to this size before fitting, and negligible impact on fit result
_NOT_SET = object()


def i_sense(x, mid, theta, amp, lin, const):
    """Weakly coupled charge transition shape including simple linear dependence of sweep gate on charge sensor"""
    arg = (x - mid) / (2 * theta)
    return -amp / 2 * np.tanh(arg) + lin * (x - mid) + const


def i_sense_strong(x, mid, theta, amp, lin, const):
    """Naive strongly coupled charge transition shape including simple linear dependence of sweep gate on charge sensor"""
    arg = (x - mid) / theta
    return (-amp * np.arctan(arg) / np.pi) * 2 + lin * (x - mid) + const


def i_sense_digamma(x, mid, g, theta, amp, lin, const):
    """Strongly coupled charge transition shape for a spinless charge transition (with linear dependence on charge sensor)"""
    def func_no_nans(x_no_nans):
        arg = digamma(0.5 + (x_no_nans - mid + 1j * g) / (2 * np.pi * 1j * theta))  # j is imaginary i
        return amp * (0.5 + np.imag(arg) / np.pi) + lin * (
                x_no_nans - mid) + const - amp / 2  # -amp/2 so const term coincides with i_sense
    return func_no_nan_eval(x, func_no_nans)


def i_sense_digamma_quad(x, mid, g, theta, amp, lin, const, quad):
    """Strongly coupled charge transition shape for a spinless charge transition (with quadratic dependence on charge sensor)"""
    def func_no_nans(x_no_nans):
        arg = digamma(0.5 + (x_no_nans - mid + 1j * g) / (2 * np.pi * 1j * theta))  # j is imaginary i
        return amp * (0.5 + np.imag(arg) / np.pi) + quad * (x_no_nans - mid) ** 2 + lin * (
                x_no_nans - mid) + const - amp / 2  # -amp/2 so const term coincides with i_sense
    return func_no_nan_eval(x, func_no_nans)


def i_sense_digamma_amplin(x, mid, g, theta, amp, lin, const, amplin):
    """Strongly coupled charge transition shape for a spinless charge transition (with linear dependence on charge sensor that varies with occupation of the QD)"""
    def func_no_nans(x_):
        arg = digamma(0.5 + (x_ - mid + 1j * g) / (2 * np.pi * 1j * theta))  # j is imaginary i
        return (amp + amplin * x_) * (0.5 + np.imag(arg) / np.pi) + lin * (
                x_ - mid) + const - (amp + amplin * mid) / 2  # -amp/2 so const term coincides with i_sense
    return func_no_nan_eval(x, func_no_nans)


def func_no_nan_eval(x: Any, func: Callable):
    """Removes nans BEFORE calling function. Necessary for things like scipy.digamma which is EXTREMELY slow with
    np.nans present in input

    Returns similar input (i.e. list if list entered, array if array entered, float if float or int entered)
    """
    if np.sum(np.isnan(np.asanyarray(x))) == 0:
        return func(x)
    t = type(x)
    x = np.array(x, ndmin=1)
    no_nans = np.where(~np.isnan(x))
    arr = np.zeros(x.shape)
    arr[np.where(np.isnan(x))] = np.nan
    arr[no_nans] = func(x[no_nans])
    if t != np.ndarray:  # Return back to original type
        if t == int:
            arr = float(arr)
        else:
            arr = t(arr)
    return arr


@dataclass
class CenteredAveragingProcess(Process):
    def set_inputs(self, x: np.ndarray, datas: Union[np.ndarray, List[np.ndarray]],
                   center_by_fitting: bool = True,
                   fit_start_x: Optional[float] = None,
                   fit_end_x: Optional[float] = None,
                   initial_params: Optional[lm.Parameters] = None,
                   override_centers_for_averaging: Optional[Union[np.ndarray, List[float]]] = None,
                   ):
        """

        Args:
            x ():
            datas (): 2D or list of 2D datas to average (assumes first data is the one to use for fitting to)
            center_by_fitting (): True to use a simple fit to find centres first, False just blind averages (unless
                override_centers_for_averaging is set)
            fit_start_x (): Optionally set a lower x-limit for fitting
            fit_end_x (): Optionally set an upper x-limit for fitting
            initial_params (): Optionally provide some initial paramters for fitting
            override_centers_for_averaging (): Optionally provide a list of center values (will override everything else)

        Returns:

        """
        self.inputs = dict(
            x = x,
            datas = datas,
            center_by_fitting = center_by_fitting,
            fit_start_x = fit_start_x,
            fit_end_x = fit_end_x,
            initial_params = initial_params,
            override_centers_for_averaging = override_centers_for_averaging,
        )

    def _get_centers(self):
        x = self.inputs['x']
        center_by_fitting = self.inputs['center_by_fitting']
        fit_start_x = self.inputs['fit_start_x']
        fit_end_x = self.inputs['fit_end_x']
        initial_params = self.inputs['initial_params']
        override_centers_for_averaging = self.inputs['override_centers_for_averaging']

        datas = self.inputs['datas']
        data = datas[0] if isinstance(datas, list) else datas

        if override_centers_for_averaging is not None:
            centers = override_centers_for_averaging
        elif center_by_fitting is False:
            centers = [0]*data.shape[0]
        else:
            indexes = get_data_index(x, [fit_start_x, fit_end_x])
            s_ = np.s_[indexes[0]:indexes[1]]
            x = x[s_]
            data = data[:, s_]
            if not initial_params:
                initial_params = lm.Parameters()
                initial_params.add_many(
                    # param, value, vary, min, max
                    lm.Parameter('mid', np.mean(x), True, np.min(x), np.max(x)),
                    lm.Parameter('amp', np.nanmax(data) - np.nanmin(data), True, 0, 2),
                    lm.Parameter('const', np.mean(data), True),
                    lm.Parameter('lin', 0, True, 0, 0.01),
                    lm.Parameter('theta', 10, True, 0, 100),
                )
            # fit_processes = []
            # for d in data:
            #     fit_process = TransitionFitProcess()
            #     fit_process.set_inputs(x=x, transition_data=d)
            #
            center_fits = [fit_i_sense1d(x, d, initial_params) for d in data]
            centers = [fit.best_values.get('mid', np.nan) for fit in center_fits]
            centers = [v if v is not np.nan else np.nanmean(centers) for v in centers]  # guess for any bad fits
        return centers

    def process(self):
        x = self.inputs['x']
        datas = self.inputs['datas']
        centers = self._get_centers()

        datas_is_list = isinstance(datas, list)
        if not datas_is_list:
            datas = [datas]

        averaged, new_x, errors = [], [], []
        for data in datas:
            avg, x, errs = mean_data(x, data, centers, return_x=True, return_std=True, nan_policy='omit')
            averaged.append(avg)
            new_x.append(x)
            errors.append(errs)

        new_x = new_x[0]  # all the same anyway
        if not datas_is_list:
            averaged = averaged[0]
            errors = errors[0]

        self.outputs = {
            'x': new_x,  # Worth keeping x-axis even if not modified
            'averaged': averaged,
            'std_errs': errors,
            'centers': centers,
        }
        return self.outputs


@dataclass
class TransitionFitProcess(Process):
    def set_inputs(self, x, transition_data, initial_params=None):
        self.inputs['x'] = x
        self.inputs['data'] = transition_data
        self.inputs['initial_params'] = initial_params

    def process(self):
        x = self.inputs['x']
        data = self.inputs['data']
        params = self.inputs['initial_params']
        ndim = data.ndim  # To know whether to return a single or list of fits in the end

        data = np.atleast_2d(data)  # Might as well always assume 2D data to fit
        fits = transition_fits(x, data, params=params)
        fits = [FitInfo.from_fit(fit) for fit in fits]

        self.outputs = {'fits': fits}
        if ndim == 1:
            return self.outputs['fits'][0]
        else:
            return self.outputs['fits']

    @staticmethod
    def ignore_keys_for_hdf() -> Optional[Union[str, List[str]]]:
        return ['outputs', 'inputs']

    def additional_save_to_hdf(self, dc_group: h5py.Group):
        if self.outputs:
            outputs_group = dc_group.require_group('outputs')
            fits_group = outputs_group.require_group('fits')
            for i, fit in enumerate(self.outputs['fits']):
                fit: FitInfo
                fit.save_to_hdf(fits_group, f'row{i}')

        inputs_group = dc_group.require_group('inputs')
        inputs_group['x'] = self.inputs['x']
        inputs_group['data'] = self.inputs['data']
        if params := self.inputs['initial_params']:
            params_group = inputs_group.require_group('initial_params')
            params_to_HDF(params, params_group)

    @classmethod
    def additional_load_from_hdf(cls, dc_group: h5py.Group) -> Dict[str, Any]:
        additional_load = {}
        output = cls.load_output_only(dc_group)
        if output:
            additional_load = {'outputs': output}

        input_group = dc_group['inputs']
        input_ = dict(
            x=input_group['x'][:],
            data=input_group['data'][:],
        )
        if 'initial_params' in input_group.keys():
            params = params_from_HDF(input_group['initial_params'], initial=True)
            input_['initial_params'] = params
        additional_load['inputs'] = input_
        return additional_load

    @classmethod
    def load_output_only(cls, group: h5py.Group) -> dict:
        outputs = {}
        if 'outputs' in group.keys() and 'fits' in group['outputs'].keys():
            fit_group = group.get('outputs/fits')
            fits = []
            for k in sorted(fit_group.keys()):
                fits.append(FitInfo.from_hdf(fit_group, k))
            outputs['fits'] = fits
        return outputs


def get_param_estimates(x, data: np.array):
    """Return list of estimates of params for each row of data for a charge Transition"""
    if data.ndim == 1:
        return _get_param_estimates_1d(x, data)
    elif data.ndim == 2:
        return [_get_param_estimates_1d(x, z) for z in data]
    else:
        raise NotImplementedError(f"data ndim = {data.ndim}: data shape must be 1D or 2D")


def _get_param_estimates_1d(x, z: np.array) -> lm.Parameters:
    """Returns lm.Parameters for x, z data"""
    assert z.ndim == 1
    z, x = CU.resample_data(z, x, max_num_pnts=500)
    params = lm.Parameters()
    s = pd.Series(z)  # Put into Pandas series so I can work with NaN's more easily
    sx = pd.Series(x, index=s.index)
    z = s[s.first_valid_index():s.last_valid_index() + 1]  # type: pd.Series
    x = sx[s.first_valid_index():s.last_valid_index() + 1]
    if np.count_nonzero(~np.isnan(z)) > 10:  # Prevent trying to work on rows with not enough data
        try:
            smooth_gradient = np.gradient(savgol_filter(x=z, window_length=int(len(z) / 20) * 2 + 1, polyorder=2,
                                                        mode='interp'))  # window has to be odd
        except np.linalg.linalg.LinAlgError:  # Came across this error on 9/9/20 -- Weirdly works second time...
            logger.warning('LinAlgError encountered, retrying')
            smooth_gradient = np.gradient(savgol_filter(x=z, window_length=int(len(z) / 20) * 2 + 1, polyorder=2,
                                                        mode='interp'))  # window has to be odd
        x0i = np.nanargmin(smooth_gradient)  # Index of steepest descent in data
        mid = x.iloc[x0i]  # X value of guessed middle index
        amp = np.nanmax(z) - np.nanmin(z)  # If needed, I should look at max/min near middle only
        lin = (z[z.last_valid_index()] - z[z.first_valid_index()] + amp) / (
                x[z.last_valid_index()] - x[z.first_valid_index()])
        theta = 5
        const = z.mean()
        G = 0
        # add with tuples: (NAME    VALUE   VARY  MIN   MAX     EXPR  BRUTE_STEP)
        params.add_many(('mid', mid, True, None, None, None, None),
                        ('theta', theta, True, 0.01, None, None, None),
                        ('amp', amp, True, 0, None, None, None),
                        ('lin', lin, True, 0, None, None, None),
                        ('const', const, True, None, None, None, None))
    return params


def _append_param_estimate_1d(params: Union[List[lm.Parameters], lm.Parameters],
                              pars_to_add: Optional[Union[List[str], str]] = _NOT_SET) -> None:
    """
    Changes params to include named parameter

    Args:
        params ():
        pars_to_add ():

    Returns:

    """
    if isinstance(params, lm.Parameters):
        params = [params]

    if pars_to_add is _NOT_SET:
        pars_to_add = ['g']

    if pars_to_add:
        for pars in params:
            if 'g' in pars_to_add:
                pars.add('g', 0, vary=True, min=-50, max=1000)
            if 'quad' in pars_to_add:
                pars.add('quad', 0, True, -np.inf, np.inf)
    return None


def fit_i_sense1d(x, z, params: lm.Parameters = None, func: Callable = i_sense, auto_bin=False):
    """Fits charge transition data with function passed
    Other functions could be i_sense_digamma for example"""
    transition_model = lm.Model(func)
    z = pd.Series(z, dtype=np.float32)
    x = pd.Series(x, dtype=np.float32)
    if np.count_nonzero(~np.isnan(z)) > 10:  # Prevent trying to work on rows with not enough data
        z, x = CU.remove_nans(z, x)
        if auto_bin is True and len(z) > FIT_NUM_BINS:
            logger.debug(f'Binning data of len {len(z)} before fitting')
            bin_size = int(np.ceil(len(z) / FIT_NUM_BINS))
            x, z = CU.old_bin_data([x, z], bin_size)
        if params is None:
            params = get_param_estimates(x, z)

        if func in [i_sense_digamma, i_sense_digamma_quad] and 'g' not in params.keys():
            _append_param_estimate_1d(params, ['g'])
        if func == i_sense_digamma_quad and 'quad' not in params.keys():
            _append_param_estimate_1d(params, ['quad'])

        result = transition_model.fit(z, x=x, params=params, nan_policy='omit')
        return result
    else:
        return None


@deprecated(deprecated_in='3.0.0')
def linear_fit_thetas(dats: List[Any], fit_name: str, filter_func: Optional[Callable] = None,
                      show_plots=False,
                      sweep_gate_divider=100,
                      dat_attr_saved_in: str = 'transition',
                      x_gate='ESC',
                      ) -> FitInfo:
    """
    Takes thetas from named fits and plots on graph, then fits a line through any which pass filter_func returning the
    linear FitInfo

    Args:
        dats (): List of dats to include in plot
        fit_name (): Name fit is saved under (also may need to specify which dat_attr it is saved in)
        filter_func (): Function which takes a single dat and returns True or False for whether it should be included
            in linear fit. E.g. lambda dat: True if dat.Logs.dacs[x_gate] < -280 else False
        show_plots (): Whether to show the intermediate plots (i.e. thetas with linear fit)
        sweep_gate_divider (): How much to divide x-axis to get into real mV
        dat_attr_saved_in (): I.e. saved in dat.Transition or dat.NrgOcc
        x_gate (): Which gate varies on the x-axis (i.e. coupling gate)

    Returns:

    """
    if filter_func is None:
        filter_func = lambda dat: True

    def _get_theta(dat: Any) -> float:
        """Get theta from a dat"""
        if dat_attr_saved_in == 'transition':
            theta = dat.Transition.get_fit(name=fit_name).best_values.theta
        elif dat_attr_saved_in == 'nrg':
            theta = dat.NrgOcc.get_fit(name=fit_name).best_values.theta
        else:
            raise NotImplementedError
        return theta/sweep_gate_divider

    def _get_x_and_thetas(dats: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Get the x and theta for each dat and return sorted list based on x"""
        x, thetas = [], []
        for dat in dats:
            x.append(dat.Logs.dacs[x_gate])
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
        fig = plotter.figure(xlabel=f'{x_gate} /mV', ylabel='Theta /mV (real)',
                             title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: Theta vs {x_gate}')
        fig = plot_data(fig, x, thetas, name='Fit Data')

        # Plot other data
        other_x, other_thetas = get_other_data()
        if other_x is not None:
            fig = plot_data(fig, other_x, other_thetas, name='Other Data')

        # Plot fit line through all
        plot_fit(fit, np.concatenate([x, other_x])).show()
    return fit


@deprecated(deprecated_in='3.0.0')
def center_from_diff_i_sense(x, data, measure_freq: Optional[float] = None) -> float:
    if measure_freq:
        smoothed = U.decimate(data, measure_freq=measure_freq, numpnts=20)
        x = U.get_matching_x(x, smoothed)
    else:
        smoothed = data
    return x[np.nanargmin(np.diff(smoothed))]


@deprecated(deprecated_in='3.0.0')
def default_transition_params():
    _pars = lm.Parameters()
    _pars.add_many(
        ('mid', 0, True, None, None, None, None),
        ('theta', 20, True, 0.01, None, None, None),
        ('amp', 1, True, 0, None, None, None),
        ('lin', 0, True, 0, None, None, None),
        ('const', 5, True, None, None, None, None))
    return _pars


# @deprecated(deprecated_in='3.0.0')
def transition_fits(x, z, params: Union[lm.Parameters, List[lm.Parameters]] = None, func=None, auto_bin=False):
    """Returns list of model fits defaulting to simple i_sense fit"""
    if func is None:
        func = i_sense
    assert callable(func)
    assert type(z) == np.ndarray
    if params is None:  # Make list of Nones so None can be passed in each time
        params = [None] * z.shape[0]
    else:
        params = CU.ensure_params_list(params, z)
    if z.ndim == 1:  # For 1D data
        return [fit_i_sense1d(x, z, params[0], func=func, auto_bin=auto_bin)]
    elif z.ndim == 2:  # For 2D data
        fit_result_list = []
        for i in range(z.shape[0]):
            fit_result_list.append(fit_i_sense1d(x, z[i, :], params[i], func=func, auto_bin=auto_bin))
        return fit_result_list


@deprecated(deprecated_in='3.0.0')
def get_transition_function(name: str) -> Callable:
    if name == 'i_sense':
        return i_sense
    elif name == 'i_sense_digamma':
        return i_sense_digamma
    elif name == 'i_sense_digamma_amplin':
        return i_sense_digamma_amplin
    else:
        raise NotImplementedError(f'{name} not found in transition functions (or not in added to this func yet)')


@deprecated(deprecated_in='3.0.0')
def get_default_transition_params(func_name: str,
                                  x: Optional[np.ndarray] = None, data: Optional[np.ndarray] = None) -> lm.Parameters:
    params = get_param_estimates(x=x, data=data)
    if func_name == 'i_sense_digamma':
        params.add('g', 0, min=-50, max=10000, vary=True)
    elif func_name == 'i_sense_digamma_amplin':
        params.add('g', 0, min=-50, max=10000, vary=True)
        params.add('amplin', 0, vary=True)
    return params


@deprecated(deprecated_in='3.0.0')
def calculate_transition_only_fit(datnum, save_name, t_func_name: str = 'i_sense_digamma', theta=None, gamma=None,
                                  x: Optional[np.ndarray] = None, data: Optional[np.ndarray] = None,
                                  width: Optional[float] = None, center: Optional[float] = None,
                                  experiment_name: Optional[str] = None,
                                  overwrite=False) -> FitInfo:
    from ..dat_object.make_dat import get_dat
    dat = get_dat(datnum, exp2hdf=experiment_name)

    x = x if x is not None else dat.Transition.avg_x
    data = data if data is not None else dat.Transition.avg_data

    x, data = get_data_in_range(x, data, width, center=center)

    t_func, params = _get_transition_fit_func_params(x, data, t_func_name, theta, gamma)

    return dat.Transition.get_fit(name=save_name, fit_func=t_func,
                                  data=data, x=x, initial_params=params,
                                  check_exists=False, overwrite=overwrite)


@deprecated(deprecated_in='3.0.0')
def _get_transition_fit_func_params(x, data, t_func_name, theta, gamma):
    """

    Args:
        x ():
        data ():
        t_func_name ():
        theta ():
        gamma ():

    Returns:

    """
    t_func = get_transition_function(t_func_name)
    params = get_default_transition_params(t_func_name, x, data)
    if theta:
        params = U.edit_params(params, 'theta', value=theta, vary=False)
    if gamma is not None and 'g' in params:
        params = U.edit_params(params, 'g', gamma, False)
    return t_func, params


@deprecated(deprecated_in='3.0.0')
def calculate_se_transition(datnum: int, save_name: str, se_output_name: str, t_func_name: str = 'i_sense_digamma',
                            theta=None, gamma=None,
                            transition_part: str = 'cold',
                            width: Optional[float] = None, center: Optional[float] = None,
                            experiment_name: Optional[str] = None,
                            overwrite=False):
    from ..dat_object.make_dat import get_dat
    dat = get_dat(datnum, exp2hdf=experiment_name)
    data = dat.SquareEntropy.get_transition_part(name=se_output_name, part=transition_part, existing_only=True)
    x = dat.SquareEntropy.get_Outputs(name=se_output_name, check_exists=True).x

    x, data = get_data_in_range(x, data, width, center=center)

    t_func, params = _get_transition_fit_func_params(x, data, t_func_name, theta, gamma)

    return dat.SquareEntropy.get_fit(which_fit='transition', transition_part=transition_part, fit_name=save_name,
                                     fit_func=t_func, initial_params=params, data=data, x=x, check_exists=False,
                                     overwrite=overwrite)
