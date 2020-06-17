import os

import h5py

import src.CoreUtil as CU
import src.DatAttributes.DatAttribute as DA
import numpy as np
from typing import List, NamedTuple
import lmfit as lm
import pandas as pd
from scipy.signal import savgol_filter

import src.DatBuilder.Util
import src.PlottingFunctions as PF
import matplotlib.pyplot as plt
from src import CoreUtil as CU
from src.DatAttributes.Entropy import _get_max_and_sign_of_max


class Li_theta(DA.DatAttribute):
    """Lock-in Theta measurement. Using lock-in on plunger gate to measure differential current through charge sensor"""

    version = '1.0'  # To keep track of whether fitting has changed
    """
    Version Changes:
        1.0 -- First attempt at fitting
        
        """

    def __init__(self, hdf_path, LI_wavenames=('g3x', 'g3y'), wave_multiplier=1):
        """Gets data directly from hdf"""
        self.version = Li_theta.version
        self._hdf_path = hdf_path
        self._wave_names = LI_wavenames
        self._wave_multiplier = wave_multiplier
        self._full_fit = di_sense1d(self.x_array, self.data, params=_get_param_estimates_1d(self.x_array, self.data))

    @property
    def data(self):
        data, phase = calc_r(self.x, self.y)
        return data

    @property
    def phase(self):
        data, phase = calc_r(self.x, self.y)
        return phase

    @property
    def x_array(self):
        return get_data(self._hdf_path, 'x_array')

    @property
    def x(self):
        return get_data(self._hdf_path, self._wave_names[0]) * self._wave_multiplier

    @property
    def y(self):
        if len(self._wave_names) == 2:
            return get_data(self._hdf_path, self._wave_names[1]) * self._wave_multiplier
        else:
            return None

    @property
    def init_params(self):
        return self._full_fit.init_params

    @property
    def params(self):
        return self._full_fit.params

    @property
    def fit_values(self):
        return self.get_fit_values()

    def recalculate_fits(self, params=None):
        """Method to recalculate fits using new parameters"""
        if params is None:
            params = self.params
        self._full_fit = di_sense1d(self.x_array, self.data, params=params)
        self.version = Li_theta.version

    def get_fit_values(self) -> NamedTuple:
        """Takes values from param fits and puts them in NamedTuple"""
        params = self.params
        if params is not None:
            data = {k: params[k].value for k in params.keys()}  # makes dict of all
            return src.DatBuilder.Util.data_to_NamedTuple(data, FitValues)
        else:
            return None

    @staticmethod
    def standard_plot_function():
        return plot_standard_li_theta


class FitValues(NamedTuple):
    mid: List[float]
    theta: List[float]
    amp: List[float]
    const: List[float]


def di_sense(x, mid, theta, amp, const):
    """ differentiated i_sense for lockin-theta measurement """
    arg = (x - mid) / (2 * theta)
    return -amp / 2 * 1 / (2 * theta) * (1 / np.cosh(arg)) ** 2 + const


def _get_param_estimates_1d(x, z: np.array) -> lm.Parameters:
    """Returns lm.Parameters for x, z data"""
    assert z.ndim == 1
    params = lm.Parameters()
    smoothed = savgol_filter(x=z, window_length=int(len(z) / 20) * 2 + 1, polyorder=2, mode='interp')
    mid = x[np.nanargmin(smoothed)]  # Min point is probably middle
    const = np.average([smoothed[0], smoothed[-1]])  # should be a constant
    amp = 1 / 4 * (const - np.nanmin(smoothed))  # 1/4 according to quick plot in desmos
    theta = 1

    # add with tuples: (NAME    VALUE   VARY  MIN   MAX     EXPR  BRUTE_STEP)
    params.add_many(('mid', mid, True, None, None, None, None),
                    ('theta', theta, True, 0, None, None, None),
                    ('amp', amp, True, 0, None, None, None),
                    ('const', const, True, 0, None, None, None),
                    ('const', const, False, None, None, None, None))
    return params


def di_sense1d(x, z, params: lm.Parameters = None):
    """Fits charge transition data with function passed
    Other functions could be i_sense_digamma for example"""
    Li_theta_model = lm.Model(di_sense)
    z = pd.Series(z, dtype=np.float32)
    if params is None:
        params = _get_param_estimates_1d(x, z)
    result = Li_theta_model.fit(z, x=x, params=params, nan_policy='omit')
    return result


def plot_standard_li_theta(dat, axs, plots: List[int] = (1, 11), kwargs_list: List[dict] = None):
    """
    This returns a list of axes which show normal useful LItheta plots
    It requires a dat object to be passed to it so it has access to all other info
    1. 1D data and fit
    11. Add DAC table and other info

    Kwarg hints:
    show_eqn:bool,
    """

    assert len(axs) >= len(plots)
    if kwargs_list is not None:
        assert len(kwargs_list) == len(plots)
        assert type(kwargs_list[0]) == dict
        kwargs_list = [{**k, 'no_datnum': True} if 'no_datnum' not in k.keys() else k for k in kwargs_list]  # Make
        # no_datnum default to True if not passed in.
    else:
        kwargs_list = [{'no_datnum': True}] * len(plots)

    Data = dat.Data

    i = 0

    if 1 in plots:  # 1D data and fit
        ax = axs[i]
        ax.cla()
        data = dat.Li_theta.data
        show_eqn = kwargs_list[i].get('show_eqn', True)
        if show_eqn is True:
            title = '1D Data and Fit: $\\frac{-amp}{2}\\frac{1}{2\\theta}\\frac{1}{\cosh^{2}\left ( \\frac{x-mid}{2\\theta} \\right )}+const$'
        else:
            title = '1D Data and Fit'
        ax = PF.display_1d(Data.x_array, data, ax, x_label=dat.Logs.x_label,
                           y_label='di_sense /nA', dat=dat, title=title, label='Data', **kwargs_list[i])
        fit = dat.Li_theta._full_fit.best_fit
        ax.plot(dat.Li_theta.x_array, fit, label='fit', color='C3')
        PF.ax_text(ax, f'$\\theta$={dat.Li_theta.fit_values.theta:.2f}mV\n', loc=[0.6, 0.3], fontsize=8) # f'amp={dat.Li_theta.fit_values.amp:.3f}nA'
        ax.legend()
        axs[i] = ax
        i += 1  # Ready for next plot to add

    if 11 in plots:  # Add dac table and other info
        ax = axs[i]
        PF.plot_dac_table(ax, dat)
        fig = plt.gcf()
        try:
            fig.suptitle(f'Dat{dat.datnum}')
            PF.add_standard_fig_info(fig)
            PF.add_to_fig_text(fig,
                               f'SRS3bias = {dat.Instruments.srs3.out:.1f}mV, ACbias = {dat.Instruments.srs1.out/50*np.sqrt(2)}/nA, temp = {dat.Logs.temp:.0f}mK')
        except AttributeError:
            print(f'One of the attributes was missing for dat{dat.datnum} so extra fig text was skipped')
        axs[i] = ax
        i += 1


@CU.plan_to_remove  # 9/6
def get_data(hdf_path, wavename) -> np.ndarray:
    """
    Returns array of data from hdf file if given path to hdf file

    @param hdf_path: Path to file
    @type hdf_path: str
    @param wavename: name of wave in hdf file
    @type wavename: str
    @return: array of dat
    @rtype: np.ndarray
    """
    if os.path.isfile(hdf_path):
        hdf = h5py.File(hdf_path, 'r')
        if wavename in hdf.keys():
            array = hdf[wavename][:]
        else:
            print(f'WARNING: wavename [{wavename}] not found in hdf')
            array = None
    else:
        print(f'WARNING: hdf not found at {hdf_path}')
        array = None
    return array


@CU.plan_to_remove  # 9/6
def calc_r(x, y, phase=None):
    """Calculates R from x and y, and returns R, phase"""
    if phase is None:
        x_max, y_max, _ = _get_max_and_sign_of_max(x, y)  # Gets max of x and y at same location
        # and which was bigger
        phase = np.arctan(y_max / x_max)
    r = np.array([x * np.cos(phase) + y * np.sin(phase) for x, y in zip(x, y)])
    return r, phase