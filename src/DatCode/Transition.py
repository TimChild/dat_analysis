import numpy as np
import types
from typing import List
import src.DatCode.DatAttribute as DA
from scipy.special import digamma
import lmfit as lm
import pandas as pd
from scipy.signal import savgol_filter
import src.CoreUtil as CU


class Transition(DA.DatAttribute):
    __version = '1.0'  # To keep track of whether fitting has changed

    def __init__(self, x_array, transition_data, fit_function=None):
        self.data = np.array(transition_data)
        self.x_array = x_array
        self.version = Transition.__version
        self.full_fits = transition_fits(x_array, self.data, get_param_estimates(x_array, self.data), func=fit_function)
        self.init_params = [fit.init_params for fit in self.full_fits]
        self.params = [fit.params for fit in self.full_fits]

    def recalculate_fits(self, params=None, func=None):
        """Method to recalculate fits using new parameters or new fit_function"""
        if params is None:
            params = self.params
        if func is None:
            func = i_sense
        self.full_fits = transition_fits(self.x_array, self.data, params, func=func)
        self.init_params = [fit.init_params for fit in self.full_fits]
        self.params = [fit.params for fit in self.full_fits]
        self.version = Transition.__version


def i_sense(x, x0, theta, amp, lin, const):
    """ fit to sensor current """
    arg = (x - x0) / (2 * theta)
    return -amp * np.tanh(arg) + lin * (x - x0) + const


def i_sense_strong(x, x0, theta, amp, lin, const):
    arg = (x - x0) / theta
    return (-amp * np.arctan(arg) / np.pi) * 2 + lin * (x - x0) + const


def i_sense_digamma(x, x0, G, theta, amp, lin, const):
    arg = digamma(0.5 + (x0 - x + 1j * G / 2) / (2 * np.pi * 1j * theta))  # j is imaginary i
    return -amp * (0.5 + np.imag(arg) / np.pi) + lin * x + const


def get_param_estimates(x, data: np.array):
    """Return list of estimates of params for each row of data for a charge Transition"""
    if data.ndim == 1:
        return [_get_param_estimates_1d(x, data)]
    elif data.ndim == 2:
        return [_get_param_estimates_1d(x, z) for z in data]
    else:
        raise NotImplementedError("data shape must be 1D or 2D")


def _get_param_estimates_1d(x, z: np.array) -> lm.Parameters:
    """Returns lm.Parameters for x, z data"""
    assert z.ndim == 1
    params = lm.Parameters()
    s = pd.Series(z)  # Put into Pandas series so I can work with NaN's more easily
    z = s[:s.last_valid_index() + 1]  # type: pd.Series
    x = x[:s.last_valid_index() + 1]
    if np.count_nonzero(~np.isnan(z)) > 10:  # Prevent trying to work on rows with not enough data
        smooth_gradient = np.gradient(savgol_filter(x=z, window_length=int(len(z) / 20) * 2 + 1, polyorder=2,
                                                    mode='interp'))  # window has to be odd
        x0i = CU.get_data_index(smooth_gradient, np.nanmin(smooth_gradient))  # Index of steepest descent in data
        x0 = x[x0i]  # X value of guessed middle index
        amp = np.nanmax(z) - np.nanmin(z)  # If needed, I should look at max/min near middle only
        lin = (z[z.last_valid_index()] - z[0] + amp) / (x[-1] - x[0])
        theta = 30
        const = z.mean()
        G = 0
        # add with tuples: (NAME    VALUE   VARY  MIN   MAX     EXPR  BRUTE_STEP)
        params.add_many(('x0', x0, True, None, None, None, None),
                        ('theta', theta, True, 0, None, None, None),
                        ('amp', amp, True, 0, None, None, None),
                        ('lin', lin, True, 0, None, None, None),
                        ('const', const, True, None, None, None, None))
    return params


def _append_digamma_param_estimate_1d(params) -> None:
    """Changes params to include G for digamma fits"""
    params.add('G', 0, vary=True, min=-50, max=1000)
    # params.add('const', value=const + amp / 2, vary=True)
    return None


def i_sense1d(x, z, params: lm.Parameters = None, func: types.FunctionType = i_sense, fixamp=None, fixlin=None,
              fixG=None, fixtheta=None):
    """Fits charge transition data with function passed
    Other functions could be i_sense_digamma for example"""
    transition_model = lm.Model(func)
    z = pd.Series(z, dtype=np.float32)
    if np.count_nonzero(~np.isnan(z)) > 10:  # Prevent trying to work on rows with not enough data
        if params is None:
            params = get_param_estimates(x, z)

        if func == i_sense_digamma and 'G' not in params.keys():
            _append_digamma_param_estimate_1d(params)
        result = transition_model.fit(z, x=x, params=params, nan_policy='propagate')
        return result
    else:
        return None


def transition_fits(x, z, params: List[lm.Parameters] = None, func = None):
    """Returns list of model fits defaulting to simple i_sense fit"""
    if func is None:
        func = i_sense
    assert callable(func)
    if params is None:  # Make list of Nones so None can be passed in each time
        params = [None] * z.shape[1]
    if z.ndim == 1:  # For 1D data
        return [i_sense1d(x, z, params[0], func=func)]
    elif z.ndim == 2:  # For 2D data
        fit_result_list = []
        for i in range(z.shape[1]):
            fit_result_list.append(i_sense1d(x, z[i, :], params[i], func=func))
        return fit_result_list
