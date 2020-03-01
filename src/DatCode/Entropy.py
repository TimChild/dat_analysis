import numpy as np
from typing import Tuple, List, NamedTuple
import src.CoreUtil as CU
from src.CoreUtil import verbose_message, data_index_from_width
import src.Configs.Main_Config as cfg
import lmfit as lm
import pandas as pd
import src.PlottingFunctions as PF


class Entropy:
    """
    Optional Dat attribute
        Represents components of the dat which are reserved to measurements of entropy
    """
    __version = '2.3'
    """
    Version updates:
        2.0 -- added integrated entropy
        2.2 -- Somewhat fixed integrated entropy... (3x too big)  
        2.3 -- Added fit average (integrated still not working)
    """

    def __init__(self, x_array, entx, mids, enty=None, thetas=None):
        """:@param mids: Required because it's so integral to integrated entropy
        :@param thetas: Can pass in theta values to give a better initial fit param"""
        self.x_array = x_array
        self.entx = np.array(entx)
        self.enty = np.array(enty)
        self.version = Entropy.__version

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
        self._full_fit_average = entropy_fits(self.x_array, self._data_average, get_param_estimates(self.x_array, self._data_average, mids=np.average(mids), thetas=np.average(thetas)))[0]
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
        self.int_width = None
        self.integrated_version = None

    @property
    def _data(self):  # Don't need to store second copy of data now
        if self.entr is not None:
            return self.entr
        else:
            return self.entx

    @property
    def _data_average(self):  # Don't store second copy
        if self.entrav is not None:
            return self.entrav
        else:
            return self.entxav

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
    def init_params(self):  # Don't store copy
        return [fit.init_params for fit in self._full_fits]

    @property
    def fit_values(self):  # Don't store copy and will update if changed
        return self._get_fit_values()

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

    def _get_fit_values(self) -> NamedTuple:
        """Takes values from param fits and puts them in NamedTuple"""
        if self.params is not None:
            data = {k + 's': [param[k].value for param in self.params] for k in
                    self.params[0].keys()}  # makes dict of all
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
        self.version = Entropy.__version

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
                                        amplitude_err: float = None, scaling: float = None, scaling_err: float = None, width: float = None):
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
            scaling = _scaling(dT_mV, amplitude, dx)

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
        self._integrated_entropy = _integrate_entropy_1d(self._int_data, self.scaling)
        self.integrated_version = Entropy.init_integrated_entropy_average._version
        self._int_entropy_initialized = True

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
            return [_integrate_entropy_1d(d, self.scaling) for d in data]
        else:
            ids = [CU.data_index_from_width(x, mid, self.int_width) for mid in self._mids]
            data = [data[low_id:high_id] for low_id, high_id, in ids]
            return [_integrate_entropy_1d(d, self.scaling) for d in data]

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


def _integrate_entropy_1d(data, scaling):
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


def _integrate_entropy_2d(data: np.ndarray, dx: float, scaling: float) -> List[np.ndarray]:
    """Integrates 2D entropy data with scaling factor returning list of 1D integrated entropies
        @param data: Entropy signal (entx/entr/etc)
        @param dx: spacing of x_array
        @param scaling: scaling factor from dT, amplitude"""
    assert data.ndim == 2
    return [_integrate_entropy_1d(d, dx, scaling) for d in data]


def _scaling(dt, amplitude, dx):
    return dx / amplitude / dt


class FitValues(NamedTuple):
    mids: List[float]
    thetas: List[float]
    consts: List[float]
    dSs: List[float]
    dTs: List[float]


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
        large_max = float(np.nanmin(larger))
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

    params.add_many(('mid', mid, True, None, None, None, None),
                    ('theta', theta, True, 0, 200, None, None),
                    ('const', 0, False, None, None, None, None),
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
        result = entropy_model.fit(z, x=x, params=params, nan_policy='propagate')
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
        ax = PF.display_1d(Data.x_array - np.average(Data.x_array), data, ax=ax, x_label=f'Centered {dat.Logs.x_label}',
                           y_label='1D Avg Entropy Signal', dat=dat, title=title, **kwargs_list[i])
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


    return axs
