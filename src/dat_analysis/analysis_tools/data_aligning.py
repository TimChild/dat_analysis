import numpy as np
from scipy.interpolate import interp2d, interp1d
import lmfit as lm
from ._shift_tracker_algorithm import imregpoc


class DataAligner:
    def __init__(self, x_match, y_match, data_match):
        """Initialize with data being matched to"""
        self.x_match = x_match
        self.y_match = y_match
        self.data_match = data_match
        self.last_fit = None

    def fit_translation(self, x, y, data, params=None, method='powell'):
        """Find translation fit that minimizes difference"""
        if params is None:
            params = lm.Parameters()
            params.add_many(
                lm.Parameter('x_translate', 0),
                lm.Parameter('y_translate', 0),
            )

        # Ensure reasonable limits on params
        for ax, arrs in zip(['x', 'y'], [(self.x_match, x), (self.y_match, y)]):
            smaller_data_width = min([abs(np.nanmax(_a) - np.nanmin(_a)) for _a in arrs])
            _min = params[f'{ax}_translate'].min
            params[f'{ax}_translate'].min = _min if _min != -np.inf else -smaller_data_width
            _max = params[f'{ax}_translate'].max
            params[f'{ax}_translate'].max = _max if _max != np.inf else smaller_data_width

        fit = lm.minimize(optimize_func, params, method=method,
                          args=[self.x_match, self.y_match, self.data_match, x, y, data])
        self.last_fit = fit
        return fit

    def translate_data_to_match(self, x, y, data, params=None, method='powell'):
        """Translate data to minimize difference"""
        fit = self.fit_translation(x, y, data, params=params, method=method)
        xt, yt, datat = translate_data(x, y, data, self.last_fit.params['x_translate'].value,
                                       self.last_fit.params['y_translate'].value)
        return xt, yt, datat

    def subtract_data(self, x, y, data, translate_first=False, invert=False):
        """Subtract provided data from match data"""
        if translate_first:
            x, y, data = self.translate_data_to_match(x, y, data)
        nx, ny, diff = subtract_data(self.x_match, self.y_match, self.data_match, x, y, data)
        if invert:
            diff *= -1
        return nx, ny, diff


def translate_data(x, y, data, translate_x, translate_y):
    """Translate the data by adding to x or y"""
    return x + translate_x, y + translate_y, data


def subtract_data(x1, y1, data1, x2, y2, data2):
    """Subtract data 2 from data 1 respecting the x and y axes of each

    Returns:
        new_x, new_y, subtracted_data
    """
    if not np.all(x1 == x2):
        nx = np.linspace(max(min(x1), min(x2)), min(max(x1), max(x2)), max(x1.shape[-1], x2.shape[-1]) * 3)
    else:
        nx = x1
    if not np.all(y1 == y2):
        ny = np.linspace(max(min(y1), min(y2)), min(max(y1), max(y2)), max(y1.shape[-1], y2.shape[-1]) * 3)
    else:
        ny = y1
    interper1 = interp2d(x1, y1, data1)
    interper2 = interp2d(x2, y2, data2)
    sub_data = interper1(nx, ny) - interper2(nx, ny)
    return nx, ny, sub_data


def cost_array(data):
    """Turn subtracted data into an array that lmfit.minimize can use for optimization (for leastsq or least_squres)"""
    return data.flatten()


def cost_scalar(data):
    """Turn cost array into a single cost value (for simpler optimization in-case the least squares doesn't work"""
    return np.sum(np.square(data))


def optimize_func(params, x1, y1, data1, x2, y2, data2):
    """Function that follows the required format for lm.minimize
    Expected parameter values are 'x_translate', 'y_translate'
    """
    x_t, y_t = params['x_translate'].value, params['y_translate'].value
    x2, y2, data2 = translate_data(x2, y2, data2, x_t, y_t)
    nx, ny, sub_data = subtract_data(x1, y1, data1, x2, y2, data2)
    return cost_array(sub_data)


class DataAligner1D:
    def __init__(self, x_match, data_match):
        """Initialize with data being matched to"""
        self.x_match = x_match
        self.data_match = data_match
        self.last_fit = None

    def fit_translation(self, x, data, params=None, method='powell', max_shift=None):
        """Find translation fit that minimizes difference"""
        if params is None:
            params = lm.Parameters()
            params.add_many(
                lm.Parameter('x_translate', 0),
            )
        # Ensure reasonable limits on params
        if max_shift is None:
            max_shift = min([abs(np.nanmax(_x) - np.nanmin(_x)) for _x in [x, self.x_match]])
        _min = params['x_translate'].min
        params['x_translate'].min = _min if _min != -np.inf else -max_shift
        _max = params['x_translate'].max
        params['x_translate'].max = _max if _max != np.inf else max_shift

        fit = lm.minimize(optimize_func_1d, params, method=method, args=[self.x_match, self.data_match, x, data])
        self.last_fit = fit
        return fit

    def translate_data_to_match(self, x, data, params=None, method='powell', max_shift=None):
        """Translate data to minimize difference"""
        fit = self.fit_translation(x, data, params=params, method=method, max_shift=max_shift)
        xt, datat = translate_data_1d(x, data, self.last_fit.params['x_translate'].value)
        return xt, datat

    def subtract_data(self, x, data, translate_first=False, invert=False, max_shift=None):
        """Subtract provided data from match data"""
        if translate_first:
            x, data = self.translate_data_to_match(x, data, max_shift=max_shift)
        nx, diff = subtract_data_1d(self.x_match, self.data_match, x, data)
        if invert:
            diff *= -1
        return nx, diff


def translate_data_1d(x, data, translate_x):
    """Translate the data by adding to x"""
    nx, ny, ndata = translate_data(x, 0, data, translate_x, 0)
    return nx, ndata


def subtract_data_1d(x1, data1, x2, data2):
    """Subtract data 2 from data 1 respecting the x axes of each

    Returns:
        new_x, subtracted_data
    """
    if not np.all(x1 == x2):
        nx = np.linspace(max(min(x1), min(x2)), min(max(x1), max(x2)), max(x1.shape[-1], x2.shape[-1]) * 3)
    else:
        nx = x1
    interper1 = interp1d(x1, data1)
    interper2 = interp1d(x2, data2)
    sub_data = interper1(nx) - interper2(nx)
    return nx, sub_data


def optimize_func_1d(params, x1, data1, x2, data2):
    """Function that follows the required format for lm.minimize
    Expected parameter values are 'x_translate'
    """
    x_t = params['x_translate'].value
    x2, data2 = translate_data_1d(x2, data2, x_t)
    nx, sub_data = subtract_data_1d(x1, data1, x2, data2)
    return cost_array(sub_data)



