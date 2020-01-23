import numpy as np
from typing import Tuple, List
from src.CoreUtil import average_repeats
import src.CoreUtil as CU
from src.CoreUtil import verbose_message
import src.config as cfg
import lmfit as lm
import pandas as pd


class Entropy:
    """
    Optional Dat attribute
        Represents components of the dat which are reserved to measurements of entropy
    """
    __version = '1.0'

    def __init__(self, x_array, entx, enty=None, mids=None, thetas=None):
        """:@param mids: Can pass in real middle values to give a better initial fit param
        :@param thetas: Can pass in theta values to give a better initial fit param"""
        self.x_array = x_array
        self.entx = np.array(entx)
        self.enty = np.array(enty)
        self.version = Entropy.__version

        self.entr = None  # type: np.array
        self.entrav = None  # type: np.array
        self.entangle = None  # type: np.array
        self.calc_r(useangle=True)  # Calculates entr, entrav, and entangle
        if self.entr is not None:
            self.data = self.entr
        else:
            self.data = self.entx
        self.full_fits = entropy_fits(self.x_array, self.data, get_param_estimates(self.x_array, self.data, mids=mids, thetas=thetas))
        self.init_params = [fit.init_params for fit in self.full_fits]
        self.params = [fit.params for fit in self.full_fits]

    def recalculate_fits(self, params=None):
        if params is None:
            params = self.params
        self.full_fits = entropy_fits(self.x_array, self.data, params)
        self.init_params = [fit.init_params for fit in self.full_fits]
        self.params = [fit.params for fit in self.full_fits]
        self.version = Entropy.__version

    def calc_r(self, mid_ids=None, useangle=True):
        # calculate r data using either constant phase determined at largest value or larger signal
        # create averages - Probably this is still the best way to determine phase angle/which is bigger even if it's not repeat data

        if mid_ids is None:
            # region Verbose Entropy calc_r
            if cfg.verbose is True:
                verbose_message(f'Verbose[Entropy][calc_r] - No mid data provided for alignment')
            # endregion
            mid_ids = np.zeros(self.entx.shape[0])
        else:
            mid_ids = mid_ids

        entxav, entxav_err = CU.average_data(self.entx, mid_ids)
        entyav, entyav_err = CU.average_data(self.enty, mid_ids)
        sqr_x = np.square(entxav)
        sqr_y = np.square(entyav)
        sqr_x_orig = np.square(self.entx)
        sqr_y_orig = np.square(self.enty)

        x_max, y_max, which = _get_max_and_sign_of_max(entxav, entyav)  # Gets max of x and y at same location
        # and which was bigger
        angle = np.arctan(y_max / x_max)
        if which == 'x':
            sign = np.sign(entxav)
            sign_orig = np.sign(self.entx)
        elif which == 'y':
            sign = np.sign(entyav)
            sign_orig = np.sign(self.enty)
        else:
            raise ValueError('should have received "x" or "y"')

        if useangle is False:
            self.entrav = np.multiply(np.sqrt(np.add(sqr_x, sqr_y)), sign)
            self.entr = np.multiply(np.sqrt(np.add(sqr_x_orig, sqr_y_orig)), sign_orig)
            self.entangle = None
        elif useangle is True:
            self.entrav = np.array([x * np.cos(angle) + y * np.sin(angle) for x, y in zip(entxav, entyav)])
            self.entr = np.array([x * np.cos(angle) + y * np.sin(angle) for x, y in zip(self.entx, self.enty)])
            self.entangle = angle


def _get_max_and_sign_of_max(x, y) -> Tuple[float, float, np.array]:
    """Returns value of x, y at the max position of the larger of the two and which was larger...
     i.e. x and y value at index=10 if max([x,y]) is x at x[10] and 'x' because x was larger"""

    if max(np.abs(x)) > max(np.abs(y)):
        which = 'x'
        x_max, y_max = _get_values_at_max(x, y)
    else:
        which = 'y'
        y_max, x_max = _get_values_at_max(y, x)
    return x_max, y_max, which


def _get_values_at_max(larger, smaller) -> Tuple[float, float]:
    """Returns values of larger and smaller at position of max in larger"""
    if np.abs(np.nanmax(larger)) > np.abs(np.nanmin(larger)):
        large_max = np.nanmax(larger)
        index = np.nanargmax(larger)
    else:
        large_max = np.nanmin(larger)
        index = np.nanargmin(larger)
    small_max = smaller[index]
    return large_max, small_max


def get_param_estimates(x_array, data, mids, thetas) -> List[lm.Parameters]:
    if data.ndim == 1:
        return [_get_param_estimates_1d(x_array, data, mids, thetas)]
    elif data.ndim == 2:
        return [_get_param_estimates_1d(x_array, z, mid, theta) for z, mid, theta in zip(data, mids, thetas)]


def _get_param_estimates_1d(x, z, mid=None, theta=None) -> lm.Parameters:
    """Returns estimate of params and some reasonable limits. Const forced to zero!!"""
    params = lm.Parameters()
    dT = np.nanmax(z) - np.nanmin(z)
    if mid is None:
        mid = (x[np.nanargmax(z)] + x[np.nanargmin(z)]) / 2  #
    if theta is None:
        theta = abs((x[np.nanargmax(z)] - x[np.nanargmin(z)]) / 2.5)

    params.add_many(('x0', mid, True, None, None, None, None),
                    ('theta', theta, True, 0, 200, None, None),
                    ('const', 0, False, None, None, None, None),
                    ('dS', 0, True, -5, 5, None, None),
                    ('dT', dT, True, -10, 50, None, None))

    return params


def entropy_nik_shape(x, x0, theta, const, dS, dT):
    """fit to entropy curve"""
    arg = ((x - x0) / (2 * theta))
    return -dT * ((x - x0) / (2 * theta) - 0.5 * dS) * (np.cosh(arg)) ** (-2) + const


def entropy_1d(x, z, params: lm.Parameters = None):
    entropy_model = lm.Model(entropy_nik_shape)
    z = pd.Series(z, dtype=np.float32)
    if np.count_nonzero(~np.isnan(z)) > 10:  # Don't try fit with not enough data
        if params is None:
            raise ValueError("entropy_1d requires lm.Parameters with keys 'x0, theta, const, dS, dT'."
                             "\nYou can run _get_param_estimates(x_array, data, mids, thetas) to get them")
        result = entropy_model(z, x=x, params=params, nan_policy='propagate')
        return result
    else:
        return None
    # if sigma is not None:
    #     weights = np.array(1 / sigma, dtype=np.float32)
    # else:
    #     weights = None
    # result = emodel.fit(z, x=x, params=params, nan_policy='propagate', weights=weights)


def entropy_fits(x, z, params: List[lm.Parameters] = None):
    if params is None:
        params = [None]*z.shape[1]
    if z.ndim == 1:  # 1D data
        return [entropy_1d(x, z, params[0])]
    elif z.ndim == 2:  # 2D data
        fit_result_list = []
        for i in range(z.shape[1]):
            fit_result_list.append(entropy_1d(x, z[i, :], params[i]))
        return fit_result_list
