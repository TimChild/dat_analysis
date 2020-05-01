import numpy
import numpy as np
from typing import List, NamedTuple

from matplotlib import pyplot

import src.CoreUtil as CU
from src import PlottingFunctions
from src.CoreUtil import verbose_message
import src.Configs.Main_Config as cfg
import lmfit as lm
import pandas as pd
import src.PlottingFunctions as PF
import matplotlib.pyplot as plt


from src.DatCode.Datutil import _get_max_and_sign_of_max
from src.PlottingFunctions import ax_setup


class Entropy:
    """
    Optional Dat attribute
        Represents components of the dat which are reserved to measurements of entropy
    """
    version = '2.6'
    """
    Version updates:
        2.0 -- added integrated entropy
        2.2 -- Somewhat fixed integrated entropy... (3x too big)  
        2.3 -- Added fit average (integrated still not working)
        2.4 -- Omitting NaNs
        2.5 -- Stores _dc_datnum if dcdat passed in to init_int_entropy
        2.6 -- Change default fitting params to allow const to vary
    """

    def __init__(self, x_array, entx, mids, enty=None, thetas=None):
        """:@param mids: Required because it's so integral to integrated entropy
        :@param thetas: Can pass in theta values to give a better initial fit param"""
        self.x_array = x_array
        self.entx = np.array(entx)
        self.enty = np.array(enty)
        self.version = Entropy.version

        # For both fitting and integrated entropy
        self.entxav = None  # type: np.array
        self.entyav = None  # type: np.array
        self.entr = None  # type: np.array
        self.entrav = None  # type: np.array
        self.entangle = None  # type: np.array
        self._mids = mids
        self._calc_r(useangle=True, mid_ids=[CU.get_data_index(self.x_array, mid) for mid in
                                             self._mids])  # Calculates entr, entrav, and entangle

        # For fitted entropy only
        self._full_fits = entropy_fits(self.x_array, self._data,
                                       get_param_estimates(self.x_array, self._data, mids=mids, thetas=thetas))
        self._avg_full_fit = entropy_fits(self.x_array, self._data_average, get_param_estimates(self.x_array, self._data_average, mids=np.average(mids), thetas=np.average(thetas)))[0]

        self.mid = None
        self.theta = None
        self.const = None
        self.dS = None
        self.dT = None
        self._set_average_fit_values()


        # For integrated entropy only
        self._int_dt = None
        self._amp = None
        self._int_x_array = None
        self._int_data = None  # Stores 1D averaged data for integrated entropy
        self.scaling = None
        self.scaling_err = None  # The proportional error. i.e. 1 would mean anything from 0 -> double
        self._integrated_entropy = None
        self._int_entropy_initialized = False
        self._dx = None
        self._dc_datnum = None
        self.int_width = None
        self.integrated_version = None

    @property
    def _data(self):  # Don't need to store second copy of data now
        if hasattr(self, '_altered_data') and self._altered_data is not None:
            return self._altered_data
        elif self.entr is not None:
            return self.entr
        else:
            return self.entx

    @_data.setter
    def _data(self, value):  # for things like subtracting constant
        self._altered_data = value
        self._data_average, self._data_average_err = CU.average_data(self._altered_data, self._mids)
        print('Recalculated _data_average by centering _data with self._mids')

    @property
    def _data_average(self):  # Don't store second copy
        if hasattr(self, '_altered_data_average') and self._altered_data_average is not None:
            return self._altered_data_average
        elif self.entrav is not None:
            return self.entrav
        else:
            return self.entxav

    @_data_average.setter
    def _data_average(self, value):
        self._altered_data_average = value

    @property
    def integrated_entropy(self):  # Don't store second copy
        """1D of averaged data (using width)"""
        if not self.int_entropy_initialized:
            print('need to initialize integrated entropy')
            return None
        return self._integrated_entropy

    @property
    def integrated_entropy_x_array(self):  # Don't store second copy
        """1D x_array around mid_val with width used for integrated entropy"""
        return self._int_x_array

    @property
    def int_entropy_initialized(self):  # For easy outside setting of int_entropy
        return self._int_entropy_initialized

    @property
    def params(self):  # Don't store copy
        return [fit.params for fit in self._full_fits]

    @property
    def avg_params(self):
        return self._avg_full_fit.params

    @property
    def init_params(self):  # Don't store copy
        return [fit.init_params for fit in self._full_fits]

    @property
    def fit_values(self):  # Don't store copy and will update if changed
        return self._get_fit_values()

    @property
    def avg_fit_values(self):
        return self._get_fit_values(avg=True)

    @property
    def avg_x_array(self):  # Most likely necessary when NaNs are omitted
        return self._avg_full_fit.userkws['x']


    @property
    def int_ds(self):
        if self.int_entropy_initialized is False:
            print('Need to initialize first')
            return None
        elif len(self.integrated_entropy_x_array) > 1000:
            return np.average(self.integrated_entropy[-20:])
        else:
            return self.integrated_entropy[-1]

    @property
    def int_entropy_per_line(self):
        """By row data used for average (using width)"""
        if self.int_entropy_initialized is False:
            print('Integrated entropy is not initialized')
            return None
        else:
            int_entropy_per_line = self._get_int_entropy_per_line()
            return int_entropy_per_line

    def _get_fit_values(self, avg=False) -> NamedTuple:
        """Takes values from param fits and puts them in NamedTuple"""
        if avg is False:
            params = self.params
        elif avg is True:
            params = [self.avg_params]
        else:
            params = None
        if self.params is not None:
            data = {k + 's': [param[k].value for param in params] for k in
                    params[0].keys()}  # makes dict of all
            # param values for each key name. e.g. {'mids': [1,2,3], 'thetas':...}
            return CU.data_to_NamedTuple(data, FitValues)
        else:
            return None

    def _set_average_fit_values(self):
        if self.fit_values is not None:
            for i, key in enumerate(self.fit_values._fields):
                avg = np.average(self.fit_values[i])
                exec(f'self.{key[:-1]} = {avg}')  # Keys in fit_values should all end in 's'

    def recalculate_fits(self, params=None):
        if params is None:
            params = self.params
        self._full_fits = entropy_fits(self.x_array, self._data, params)
        self._avg_full_fit = entropy_fits(self.x_array, self._data_average, [params[0]])[0]
        self._set_average_fit_values()
        self.version = Entropy.version

    def _calc_r(self, mid_ids=None, useangle=True):
        # calculate r data using either constant phase determined at largest value or larger signal
        # create averages - Probably this is still the best way to determine phase angle/which is bigger even if it's not repeat data

        if mid_ids is None:
            # region Verbose Entropy calc_r
            if cfg.verbose is True:
                verbose_message(f'Verbose[Entropy][calc_r] - No mid data provided for alignment')
            # endregion
            print('WARNING: Not using mids to center data')
            mid_ids = np.zeros(self.entx.shape[0])
        else:
            mid_ids = mid_ids

        entxav, entxav_err = CU.average_data(self.entx, mid_ids)
        entyav, entyav_err = CU.average_data(self.enty, mid_ids)
        self.entxav = entxav
        self.entyav = entyav
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

    def init_integrated_entropy_average(self, dT_mV: float = None, dT_err: float = None, amplitude: float = None,
                                        amplitude_err: float = None, scaling: float = None, scaling_err: float = None, width: float = None, dcdat=None):
        """
        Initializes integrated entropy attribute of Entropy class if enough info present. scaling/scaling_err is prioritized

        Can also be used to recalculate entropy

        @param amplitude: Amplitude of charge step in nA
        @param dT_mV: dT in units of mV (i.e. theta value) -- This needs to be the REAL dT (i.e. if base temp is 100mK
        and peak current of lock-in results in 130mK then T is really 115+-15mK... 15mK dT)
        @param dT_err: This is uncertainty in dT_mV
        @param scaling: Can just pass a scaling value instead of dT_mV and amplitude
        """
        Entropy.init_integrated_entropy_average._version = "1.1"

        dx = np.abs((self.x_array[-1] - self.x_array[0])/len(self.x_array))

        # Calc scaling if possible
        if scaling is None:
            if None in (dT_mV, amplitude):
                print(
                    'WARNING: Integrated entropy not calculated. Need "scaling" or "dT_mV" and "amplitude" to calculate integrated entropy')
                return None
            scaling = scaling(dT_mV, amplitude, dx)

        # Calculate scaling_err if possible
        if scaling_err is None:
            scaling_err = 1  # temporarily as multiplier so I can add one or both of dT_err, amplitude_err
            if dT_err is not None:
                scaling_err = scaling_err * (1 + dT_err / dT_mV)
            if amplitude_err is not None:
                scaling_err = scaling_err * (1 + amplitude_err / amplitude)
            scaling_err = scaling_err - 1  # back to a fraction of scaling err

        x_array = self.x_array
        data = self._data_average
        assert data.ndim == 1
        if x_array[-1] < x_array[0]:
            x_array = np.flip(x_array)
            data = np.flip(data)  # So that if data was taken backwards, integration still happens from N -> N+1

        # Make x_array and data centered around mid_val if necessary
        mid_val = np.average(self._mids)
        if width is not None:  # only if both not None
            low_index, high_index = CU.data_index_from_width(x_array, mid_val, width)
            x_array = x_array[low_index:high_index]
            data = data[low_index:high_index]

        self._dx = dx
        self._int_dt = dT_mV
        self._amp = amplitude
        self.int_width = width
        self._int_x_array = x_array
        self._int_data = data
        self.scaling = scaling
        self.scaling_err = scaling_err
        self._integrated_entropy = integrate_entropy_1d(self._int_data, self.scaling)
        self.integrated_version = Entropy.init_integrated_entropy_average._version
        self._int_entropy_initialized = True
        if dcdat is not None:
            self._dc_datnum = dcdat.datnum

    def _get_int_entropy_per_line(self):
        if self.int_entropy_initialized is False:
            print('_get_entropy_per_line requires initialized integrated entropy first')
            return None
        data = self._data
        x = self.x_array
        if x[-1] < x[1]:
            data = np.flip(data, axis=1)
            x = np.flip(self.x_array)
        if self.int_width is None:
            return [integrate_entropy_1d(d, self.scaling) for d in data]
        else:
            ids = [CU.data_index_from_width(x, mid, self.int_width) for mid in self._mids]
            data = [data[low_id:high_id] for low_id, high_id, in ids]
            return [integrate_entropy_1d(d, self.scaling) for d in data]

    def plot_integrated_entropy_per_line(self, ax):
        x = self.integrated_entropy_x_array
        data = self.int_entropy_per_line
        _plot_integrated_entropy_per_line(ax, x, data, x_label='/mV', y_label='Entropy/kB', title='Integrated Entropy')

    @staticmethod
    def standard_plot_function():
        return plot_standard_entropy


def _plot_integrated_entropy_per_line(ax, x, data, **kwargs):
    for d in data:
        ax.plot(x, d)
    PF._optional_plotting_args(ax, **kwargs)


def integrate_entropy_1d(data, scaling):
    """Integrates 1D entropy data with scaling factor returns np.array

    @param data: 1D entropy data
    @type data: np.ndarray
    @param dx: spacing of x_array
    @type dx: float
    @param scaling: scaling factor from dT, amplitude etc
    @type scaling: float
    @return: Integrated entropy units of Kb
    @rtype: np.ndarray
    """
    assert data.ndim == 1
    return np.nancumsum(data) * scaling


def integrate_entropy_2d(data: np.ndarray, dx: float, scaling: float) -> List[np.ndarray]:
    """Integrates 2D entropy data with scaling factor returning list of 1D integrated entropies
        @param data: Entropy signal (entx/entr/etc)
        @param dx: spacing of x_array
        @param scaling: scaling factor from dT, amplitude"""
    assert data.ndim == 2
    return [integrate_entropy_1d(d, dx, scaling) for d in data]


def scaling(dt, amplitude, dx):
    return dx / amplitude / dt


class FitValues(NamedTuple):
    mids: List[float]
    thetas: List[float]
    consts: List[float]
    dSs: List[float]
    dTs: List[float]


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

    params.add_many(('mid', mid, True, None, None, None, None),
                    ('theta', theta, True, 0, 200, None, None),
                    ('const', 0, True, None, None, None, None),
                    ('dS', 0, True, -5, 5, None, None),
                    ('dT', dT, True, -10, 50, None, None))

    return params


def entropy_nik_shape(x, mid, theta, const, dS, dT):
    """fit to entropy curve"""
    arg = ((x - mid) / (2 * theta))
    return -dT * ((x - mid) / (2 * theta) - 0.5 * dS) * (np.cosh(arg)) ** (-2) + const


def entropy_1d(x, z, params: lm.Parameters = None):
    entropy_model = lm.Model(entropy_nik_shape)
    z = pd.Series(z, dtype=np.float32)
    if np.count_nonzero(~np.isnan(z)) > 10:  # Don't try fit with not enough data
        if params is None:
            raise ValueError("entropy_1d requires lm.Parameters with keys 'mid, theta, const, dS, dT'."
                             "\nYou can run _get_param_estimates(x_array, data, mids, thetas) to get them")
        result = entropy_model.fit(z, x=x, params=params, nan_policy='omit')
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
        params = [None] * z.shape[0]
    if z.ndim == 1:  # 1D data
        return [entropy_1d(x, z, params[0])]
    elif z.ndim == 2:  # 2D data
        fit_result_list = []
        for i in range(z.shape[0]):
            fit_result_list.append(entropy_1d(x, z[i, :], params[i]))
        return fit_result_list


def plot_standard_entropy(dat, axs, plots: List[int] = (1, 2, 3), kwargs_list: List[dict] = None):
    """This returns a list of axes which show normal useful entropy plots (assuming 2D for now)
    It requires a dat object to be passed to it so it has access to all other info
    1. 2D entr (or entx if no enty)
    2. Centered and averaged 1D entropy
    3. 1D slice of entropy R
    4. Nik_Entropy per repeat
    5. 2D entx
    6. 2D enty
    7. 1D slice of entx
    8. 1D slice of enty
    9. Integrated entropy
    10. Integrated entropy per line
    11. Add DAC table and other info

    Kwarg hints:
    swap_ax:bool, swap_ax_labels:bool, ax_text:bool"""

    Entropy = dat.Entropy
    Data = dat.Data

    assert len(axs) >= len(plots)
    if kwargs_list is not None:
        assert len(kwargs_list) == len(plots)
        assert type(kwargs_list[0]) == dict
        kwargs_list = [{**k, 'no_datnum': True} if 'no_datnum' not in k.keys() else k for k in kwargs_list]  # Make
        # no_datnum default to True if not passed in.
    else:
        kwargs_list = [{'no_datnum': True}] * len(plots)

    i = 0

    if 1 in plots:  # Add 2D entr (or entx)
        ax = axs[i]
        ax.cla()
        if dat.Entropy.entr is not None:
            data = Entropy.entr
            title = 'Entropy R'
        elif dat.Entropy.entx is not None:
            data = Entropy.entx
            title = 'Entropy X'
        else:
            raise AttributeError(f'No entr or entx for dat{dat.datnum}[{dat.datname}]')
        ax = PF.display_2d(Data.x_array, Data.y_array, data, ax, x_label=dat.Logs.x_label,
                           y_label=dat.Logs.y_label, dat=dat, title=title, **kwargs_list[i])

        axs[i] = ax
        i += 1  # Ready for next plot to add

    if 2 in plots:  # Add Centered and Averaged 1D entropy
        ax = axs[i]
        ax.cla()
        if Entropy.entrav is not None:
            data = Entropy.entrav
            title = 'Avg Entropy R'
        elif Entropy.entxav is not None:
            data = Entropy.entxav
            title = 'Avg Entropy X'
        else:
            raise AttributeError(f'No entrav or entxav for dat{dat.datnum}[{dat.datname}]')
        fit = dat.Entropy._avg_full_fit
        ax = PF.display_1d(Data.x_array - np.average(Data.x_array), data, ax=ax, x_label=f'Centered {dat.Logs.x_label}',
                           y_label='1D Avg Entropy Signal', dat=dat, title=title, scatter=True, **kwargs_list[i])
        ax.plot(dat.Entropy.avg_x_array-np.average(Data.x_array), fit.best_fit, color='C3')
        PF.ax_text(ax, f'dS={Entropy.avg_fit_values.dSs[0]:.3f}')
        axs[i] = ax
        i += 1

    if 3 in plots:  # Add 1D entr
        ax = axs[i]
        ax.cla()
        if dat.Entropy.entr is not None:
            data = Entropy.entr[round(dat.Entropy.entr.shape[0] / 2)]
        else:
            raise AttributeError(f'No entyav for dat{dat.datnum}[{dat.datname}]')
        ax = PF.display_1d(Data.x_array, data, ax, x_label=dat.Logs.x_label,
                           y_label='Entropy signal', dat=dat, title='1D entropy R', **kwargs_list[i])
        axs[i] = ax
        i += 1  # Ready for next plot to add

    if 4 in plots:  # Add Nik_Entropy per repeat
        ax = axs[i]
        ax.cla()
        if Entropy.fit_values is not None:
            data = Entropy.fit_values.dSs
            dataerr = [param['dS'].stderr for param in Entropy.params]
        else:
            raise AttributeError(f'No Entropy.fit_values for {dat.datnum}[{dat.datname}]')
        ax = PF.display_1d(Data.y_array, data, errors=dataerr, ax=ax, x_label='Entropy/kB', y_label=dat.Logs.y_label,
                           dat=dat, title='Nik Entropy', swap_ax=True, **kwargs_list[i])
        if kwargs_list[i].get('swap_ax', False) is False:
            ax.axvline(np.log(2), c='k', ls=':')
        else:
            ax.axhline(np.log(2), c='k', ls=':')

        axs[i] = ax
        i += 1

    if 5 in plots:  # Add 2D entx
        ax = axs[i]
        ax.cla()
        if dat.Entropy.entx is not None:
            data = Entropy.entx
        else:
            raise AttributeError(f'No entx for dat{dat.datnum}[{dat.datname}]')
        ax = PF.display_2d(Data.x_array, Data.y_array, data, ax, x_label=dat.Logs.x_label,
                           y_label=dat.Logs.y_label, dat=dat, title='Entropy X', **kwargs_list[i])
        axs[i] = ax
        i += 1  # Ready for next plot to add

    if 6 in plots:  # Add 2D enty
        ax = axs[i]
        ax.cla()
        if dat.Entropy.entx is not None:
            data = Entropy.entx
        else:
            raise AttributeError(f'No entx for dat{dat.datnum}[{dat.datname}]')
        ax = PF.display_2d(Data.x_array, Data.y_array, data, ax, x_label=dat.Logs.x_label,
                           y_label=dat.Logs.y_label, dat=dat, title='Entropy y', **kwargs_list[i])
        axs[i] = ax
        i += 1  # Ready for next plot to add

    if 7 in plots:  # Add 1D entx
        ax = axs[i]
        ax.cla()
        if dat.Entropy.entx is not None:
            data = Entropy.entx[0]
        else:
            raise AttributeError(f'No entxav for dat{dat.datnum}[{dat.datname}]')
        ax = PF.display_1d(Data.x_array, data, ax, x_label=dat.Logs.x_label,
                           y_label='Entropy signal', dat=dat, title='1D entropy X', **kwargs_list[i])
        axs[i] = ax
        i += 1  # Ready for next plot to add

    if 8 in plots:  # Add 1D entx
        ax = axs[i]
        ax.cla()
        if dat.Entropy.enty is not None:
            data = Entropy.enty[0]
        else:
            raise AttributeError(f'No entyav for dat{dat.datnum}[{dat.datname}]')
        ax = PF.display_1d(Data.x_array, data, ax, x_label=dat.Logs.x_label,
                           y_label='Entropy signal', dat=dat, title='1D entropy Y', **kwargs_list[i])
        axs[i] = ax
        i += 1  # Ready for next plot to add

    if 9 in plots:  # Add integrated entropy
        ax = axs[i]
        ax.cla()
        k = kwargs_list[i]
        if 'ax_text' not in k.keys():  # Default to ax_text == True
            k = {**k, 'ax_text': True}
        if 'loc' not in k.keys():
            k = {**k, 'loc': (0.1, 0.5)}

        if dat.Entropy.int_entropy_initialized is True:
            data = dat.Entropy.integrated_entropy
            x = dat.Entropy.integrated_entropy_x_array
            PF.display_1d(x, data, ax, y_label='Entropy /kB', dat=dat, **k)
            if dat.Entropy.int_ds > 0:
                expected = np.log(2)
            else:
                expected = -np.log(2)
            ax.axhline(expected, c='k', ls=':')
            err = dat.Entropy.scaling_err
            ax.fill_between(x, data * (1 - err), data * (1 + err), color='#AAAAAA')
            if k['ax_text'] is True:
                PF.ax_text(ax, f'dS = {dat.Entropy.int_ds:.3f}\n'
                           f'SF = {dat.Entropy.scaling:.4f}\n'
                           f'SFerr = {dat.Entropy.scaling_err*100:.0f}%\n'
                            f'dT = {dat.Entropy._int_dt:.3f}mV\n'
                               f'amp = {dat.Entropy._amp:.2f}nA',
                           loc=(k['loc']))


        else:
            print('Need to initialize integrated entropy first')
        axs[i] = ax
        i += 1

    if 10 in plots:  # Add integrated entropy per line
        ax = axs[i]
        ax.cla()
        k = kwargs_list[i]
        if 'ax_text' not in k.keys():  # Default to ax_text == True
            k = {**k, 'ax_text': True}
        if 'loc' not in k.keys():
            k = {**k, 'loc': (0.1, 0.5)}

        if dat.Entropy.int_entropy_initialized is True:
            x = dat.Entropy.integrated_entropy_x_array
            data = dat.Entropy.int_entropy_per_line
            _plot_integrated_entropy_per_line(ax, x, data, x_label=dat.Logs.x_label, y_label='Entropy/kB',
                                              title='Integrated Entropy')
            if dat.Entropy.int_ds > 0:
                expected = np.log(2)
            else:
                expected = -np.log(2)
            # ax.axhline(expected, c='k', ls=':')
            err = dat.Entropy.scaling_err
            avg_data = dat.Entropy.integrated_entropy
            ax.fill_between(x, avg_data * (1 - err), avg_data * (1 + err), color='#AAAAAA')
            if k['ax_text'] is True:
                PF.ax_text(ax, f'dS = {dat.Entropy.int_ds:.3f}\n'
                           f'SF = {dat.Entropy.scaling:.4f}\n'
                           f'SFerr = {dat.Entropy.scaling_err*100:.0f}%\n'
                            f'dT = {dat.Entropy._int_dt:.3f}mV\n'
                               f'amp = {dat.Entropy._amp:.2f}nA',
                           loc=(k['loc']))

        else:
            print('Need to initialize integrated entropy first')
        axs[i] = ax
        i += 1

    if 11 in plots:
        ax = axs[i]
        PF.plot_dac_table(ax, dat)
        fig = plt.gcf()
        try:
            fig.suptitle(f'Dat{dat.datnum}')
            PF.add_standard_fig_info(fig)
            if dat.Logs.sweeprate is not None:
                sr = f'{dat.Logs.sweeprate:.0f}mV/s'
            else:
                sr = 'N/A'
            PF.add_to_fig_text(fig,
                           f'ACbias = {dat.Instruments.srs1.out / 50 * np.sqrt(2):.1f}nA, sweeprate={sr}, temp = {dat.Logs.temp:.0f}mK')
        except AttributeError:
            print(f'One of the attributes was missing for dat{dat.datnum} so extra fig text was skipped')
        axs[i] = ax
        i+=1

    return axs


def recalculate_entropy_with_offset_subtracted(dat, update=True, save=True, dfname='default'):
    """takes the params for the current fits, changes the const to be allowed to vary, fits again, subtracts that
    offset from each line of data, then fits again. Does NOT recalculate integrated entropy"""
    from src.DFcode.DatDF import update_save
    if dat.Entropy.avg_params['const'].vary is False:
        params = dat.Entropy.params
        for p in params:
            p['const'].vary = True
        dat.Entropy.recalculate_fits(params)
        assert dat.Entropy.avg_params['const'].vary is True

    dat.Entropy._data = np.array(
        [data - c for data, c in zip(dat.Entropy._data, dat.Entropy.fit_values.consts)]).astype(np.float32)
    dat.datname = 'const_subtracted_entropy'
    dat.Entropy.recalculate_fits()
    update_save(dat, update, save, dfname=dfname)


def recalculate_int_entropy_with_offset_subtracted(dat, dc=None, dT_mV=None, make_new=False, update=True,
                                                   save=True, datdf=None):
    """
    Recalculates entropy with offset subtracted, then recalculates integrated entropy with offset subtracted

    @param dc: dcdat object to use for calculating dT_mV for integrated fit. Otherwise can pass in dT_mV value
    @type dc: Dat
    @param make_new: Saves dat with name 'const_subtracted_entropy' otherwise will just overwrite given instance
    @type make_new: bool
    @param update: Whether to update the DF given
    @type update: bool
    @param dfname: Name of dataframe to update changes in
    @type dfname: str
    @return: None
    @rtype: None
    """
    from src.DFcode.DatDF import update_save
    datname = dat.datname
    if datname != 'const_subtracted_entropy' and make_new is False:
        ans = CU.option_input(
            f'datname=[{dat.datname}], do you want to y: create a new copy with entropy subtracted, n: change this '
            f'copy, a: abort?',
            {'y': True, 'n': False, 'a': 'abort'})
        if ans == 'abort':
            return None
        elif ans is True:
            make_new = True
        elif ans is False:
            make_new = False
        else:
            raise NotImplementedError

    recalculate_entropy_with_offset_subtracted(dat, update=False, save=False)  # Creates dat with name changed
    if make_new is True:
        pass
    elif make_new is False:
        dat.datname = datname  # change name back to original before any saving or updating
    else:
        raise NotImplementedError
    if dT_mV is not None:
        dt = dT_mV
        dat.Entropy.init_integrated_entropy_average(dT_mV=dt, dT_err=0,
                                                    amplitude=dat.Transition.avg_fit_values.amps[0],
                                                    amplitude_err=0)
    elif dc is not None:
        dt = dc.DCbias.get_dt_at_current(dat.Instruments.srs1.out / 50 * np.sqrt(2))
        dat.Entropy.init_integrated_entropy_average(dT_mV=dt, dT_err=0,
                                                    amplitude=dat.Transition.avg_fit_values.amps[0],
                                                    amplitude_err=0, dcdat=dc)
    else:
        print('ERROR[E.recalculate_int_entropy_with_offset_corrected]: Must provide either "dT_mV" or "dc" to '
              'calculate integrated entropy.\r Entropy has been recalculated with offset removed, but nothing has been '
              'saved to DF')
        return None

    if datdf is not None:
        update_save(dat, update, save, datdf=datdf)
    elif save is True or update is True:
        print('WARNING[_recalculate_int_entropy_with_offset_subtracted]: No datdf provided to carry out update or save')


def plot_entropy_along_transition(dats, fig=None, axs=None, x_axis='gamma', exclude=None):
    """
    For plotting dats along a transition. I.e. each dat is a repeat measurement somewhere along transition

    @param exclude: datnums to exclude from plot
    @type exclude: List[int]
    @param dats: list of dat objects
    @type dats: src.DatCode.Dat.Dat
    @param fig:
    @type fig: plt.Figure
    @param axs:
    @type axs: List[plt.Axes]
    @return:
    @rtype: plt.Figure, List[plt.Axes]
    """

    if exclude is not None:
        dats = [dat for dat in dats if dat.datnum not in exclude]  # remove excluded dats from plotting

    if axs is None:
        fig, axs = PF.make_axes(3)

    PF.add_standard_fig_info(fig)

    if x_axis.lower() == 'rct':
        xs = [dat.Logs.fdacs[4] for dat in dats]
    elif x_axis.lower() == 'rcss':
        xs = [dat.Logs.fdacs[6] for dat in dats]
    elif x_axis.lower() == 'gamma':
        xs = [dat.Transition.avg_fit_values.gs[0] for dat in dats]
    elif x_axis.lower() == 'mar_sdr':
        xs = [dat.Logs.dacs[13] for dat in dats]
    else:
        print('x_axis has to be one of [rct, gamma, rcss, mar_sdr]')

    ax = axs[0]
    ax_setup(ax, title=f'Nik Entropy vs {x_axis}', x_label=f'{x_axis} /mV', y_label='Entropy /kB', legend=False, fs=10)
    for dat, x in zip(dats, xs):
        y = dat.Entropy.avg_fit_values.dSs[0]
        yerr = np.std(dat.Entropy.fit_values.dSs)
        ax.errorbar(x, y, yerr=yerr, linestyle=None, marker='x')

    ax = axs[1]
    ax_setup(ax, title=f'Integrated Entropy vs {x_axis}', x_label=f'{x_axis} /mV', y_label='Entropy /kB', legend=False,
             fs=10)
    for dat, x in zip(dats, xs):
        y = dat.Entropy.int_ds
        yerr = np.std(dat.Entropy.int_entropy_per_line[-1])
        ax.errorbar(x, y, yerr=yerr, linestyle=None, marker='x')

    ax = axs[2]
    for dat in dats:
        x = dat.Entropy.x_array - dat.Transition.mid
        ax.plot(x, dat.Entropy.integrated_entropy, linewidth=1)
    ax_setup(ax, title=f'Integrated Entropy vs {x_axis}', x_label=dats[0].Logs.x_label, y_label='Entropy /kB', fs=10)

    plt.tight_layout(rect=(0, 0.1, 1, 1))
    PF.add_standard_fig_info(fig)
    return fig, axs