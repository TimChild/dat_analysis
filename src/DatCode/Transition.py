import numpy as np
import types
from typing import List, NamedTuple
import src.DatCode.DatAttribute as DA
from scipy.special import digamma
import lmfit as lm
import pandas as pd
from scipy.signal import savgol_filter
import src.CoreUtil as CU
import src.PlottingFunctions as PF
import matplotlib.pyplot as plt


def i_sense(x, mid, theta, amp, lin, const):
    """ fit to sensor current """
    arg = (x - mid) / (2 * theta)
    return -amp/2 * np.tanh(arg) + lin * (x - mid) + const


def i_sense_strong(x, mid, theta, amp, lin, const):
    arg = (x - mid) / theta
    return (-amp * np.arctan(arg) / np.pi) * 2 + lin * (x - mid) + const


def i_sense_digamma(x, mid, g, theta, amp, lin, const):
    arg = digamma(0.5 + (x-mid + 1j * g) / (2 * np.pi * 1j * theta))  # j is imaginary i
    return amp * (0.5 + np.imag(arg) / np.pi) + lin * (x-mid) + const - amp/2  # -amp/2 so const term coincides with i_sense


def i_sense_digamma_quad(x, mid, g, theta, amp, lin, const, quad):
    arg = digamma(0.5 + (x-mid + 1j * g) / (2 * np.pi * 1j * theta))  # j is imaginary i
    return amp * (0.5 + np.imag(arg) / np.pi) + quad*(x-mid)**2 + lin * (x-mid) + const - amp/2  # -amp/2 so const term coincides with i_sense


class Transition(DA.DatAttribute):
    version = '4.0'  # To keep track of whether fitting has changed
    """
    Version Changes:
        1.3 -- Added T.g and T.fit_values.gs for digamma_fit
        2.0 -- Added _avg_full_fit and avg_fit_values
        2.1 -- Omitting NaNs
        2.2 -- Change i_sense function to have amp/2 so that it fits with di_gamma function
        2.3 -- Recalculate average values which show up in datdf after refitting data
        2.4 -- self.fit_func defaults to 'i_sense' instead of 'None' now.
        3.0 -- added i_sense_digamma_quad 28/4/20. 
                Also changed i_sense_digamma linear part to be (x-mid) instead of just x. Will affect previous dats
        3.1 -- added self.fit_func_name which should get stored in datdf
        4.0 -- digamma function changed! Change g/2 to g only in order to align with Yigal and others
        """

    def __init__(self, x_array, transition_data, fit_function=None):
        """Defaults to fitting with cosh shape transition"""
        if fit_function is None:
            fit_function = i_sense
        self._data = np.array(transition_data)
        self._avg_data = None  # Initialized in avg_full_fit
        self._x_array = x_array
        self.version = Transition.version
        self._full_fits = transition_fits(x_array, self._data, get_param_estimates(x_array, self._data), func=fit_function)
        self.fit_func = fit_function
        self.fit_func_name = fit_function.__name__
        self._avg_full_fit = self.avg_transition_fits()

        #  Mostly just for convenience when working in console
        self.mid = None  # type: float
        self.theta = None  # type: float
        self.amp = None  # type: float
        self.lin = None  # type: float
        self.const = None  # type: float
        self.g = None  # type: float
        self.set_average_fit_values()

    @property
    def init_params(self):
        return[fit.init_params for fit in self._full_fits]

    @property
    def params(self):
        return [fit.params for fit in self._full_fits]

    @property
    def avg_params(self):
        return self._avg_full_fit.params

    @property
    def fit_values(self):
        return self.get_fit_values()

    @property
    def avg_fit_values(self):
        return self.get_fit_values(avg=True)

    @property
    def avg_x_array(self):
        return self._avg_full_fit.userkws['x']

    def avg_transition_fits(self):
        """Fits to averaged data (based on middle of individual fits and using full_fit[0] params)"""
        self._avg_data, self._avg_data_err = CU.average_data(self._data, [CU.get_data_index(self._x_array, f.best_values['mid']) for f in self._full_fits])
        return transition_fits(self._x_array, self._avg_data, [self._full_fits[0].params], func=self.fit_func)[0]

    def recalculate_fits(self, params=None, func=None):
        """Method to recalculate fits using new parameters or new fit_function"""
        if params is None:
            params = self.params
        if func is None and self.fit_func is not None:
            func = self.fit_func
            print(f'Using self.fit_func as func to recalculate with: [{self.fit_func.__name__}]')
        elif func is None:
            func = i_sense
            print(f'Using standard i_sense as func to recalculate with')
        else:
            pass

        self._full_fits = transition_fits(self._x_array, self._data, params, func=func)
        self._avg_data, _ = CU.average_data(self._data, [CU.get_data_index(self._x_array, f.best_values['mid']) for f in self._full_fits])
        self._avg_full_fit = transition_fits(self._x_array, self._avg_data, [self._full_fits[0].params], func=func)[0]
        self.fit_func = func
        self.fit_func_name = func.__name__
        self.set_average_fit_values()
        self.version = Transition.version

    def set_average_fit_values(self):
        if self.fit_values is not None:
            for i, key in enumerate(self.fit_values._fields):
                if self.fit_values[i] is None:
                    avg = None
                else:
                    avg = np.average(self.fit_values[i])
                exec(f'self.{key[:-1]} = {avg}')  # Keys in fit_values should all end in 's'

    def get_fit_values(self, avg=False) -> NamedTuple:
        """Takes values from param fits and puts them in NamedTuple"""
        if avg is False:
            params = self.params
        elif avg is True:
            params = [self.avg_params]  # Just to make it work with same code below, but totally overkill for avg_values
        else:
            params = None
        if params is not None:
            data = {k+'s': [param[k].value for param in params] for k in params[0].keys()}   # makes dict of all
            # param values for each key name. e.g. {'mids': [1,2,3], 'thetas':...}
            return CU.data_to_NamedTuple(data, FitValues)
        else:
            return None

    def plot_transition1d(self, y_array, yval, ax=None, s=10, transx=0, transy=0, yisindex=0, notext=0):
        if yisindex == 0:
            ylist = y_array
            idy, yval = min(enumerate(ylist), key=lambda x: abs(x[1] - yval))  # Gets the position of the
            # y value closest to yval, and records actual yval
        else:
            idy = yval
            yval = y_array[idy]
        x = self._x_array
        y = self._data[idy]
        ax.scatter(x, y, s=s)
        ax.plot(x, self._full_fits[idy].best_fit, 'r-')
        ax.plot(x, self._full_fits[idy].init_fit, 'b--')
        if notext == 0:
            ax.set_ylabel("i_sense /nA")
            ax.set_xlabel("Plunger /mV")
            return ax
        else:
            return ax, idy


class FitValues(NamedTuple):
    mids: List[float]
    thetas: List[float]
    amps: List[float]
    lins: List[float]
    consts: List[float]
    gs: List[float]





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
        mid = x[x0i]  # X value of guessed middle index
        amp = np.nanmax(z) - np.nanmin(z)  # If needed, I should look at max/min near middle only
        lin = (z[z.last_valid_index()] - z[0] + amp) / (x[-1] - x[0])
        theta = 30
        const = z.mean()
        G = 0
        # add with tuples: (NAME    VALUE   VARY  MIN   MAX     EXPR  BRUTE_STEP)
        params.add_many(('mid', mid, True, None, None, None, None),
                        ('theta', theta, True, 0, None, None, None),
                        ('amp', amp, True, 0, None, None, None),
                        ('lin', lin, True, 0, None, None, None),
                        ('const', const, True, None, None, None, None))
    return params


def _append_param_estimate_1d(params, pars_to_add=None) -> None:
    """
    Changes params to include named parameter

    @param params: full lmfit Parameters
    @type params: lm.Parameters
    @param pars_to_add: list of parameters to add to params
    @type pars_to_add: list[str]
    """
    if pars_to_add is None:
        pars_to_add = ['g']

    if 'g' in pars_to_add:
        params.add('g', 0, vary=True, min=-50, max=1000)
    if 'quad' in pars_to_add:
        params.add('quad', 0, True, -np.inf, np.inf)
    return None


def i_sense1d(x, z, params: lm.Parameters = None, func: types.FunctionType = i_sense):
    """Fits charge transition data with function passed
    Other functions could be i_sense_digamma for example"""
    transition_model = lm.Model(func)
    z = pd.Series(z, dtype=np.float32)
    if np.count_nonzero(~np.isnan(z)) > 10:  # Prevent trying to work on rows with not enough data
        if params is None:
            params = get_param_estimates(x, z)[0]

        if func in [i_sense_digamma, i_sense_digamma_quad] and 'g' not in params.keys():
            _append_param_estimate_1d(params, ['g'])
        if func == i_sense_digamma_quad and 'quad' not in params.keys():
            _append_param_estimate_1d(params, ['quad'])
        result = transition_model.fit(z, x=x, params=params, nan_policy='omit')
        return result
    else:
        return None


def transition_fits(x, z, params: List[lm.Parameters] = None, func = None):
    """Returns list of model fits defaulting to simple i_sense fit"""
    if func is None:
        func = i_sense
    assert callable(func)
    assert type(z) == np.ndarray
    if params is None:  # Make list of Nones so None can be passed in each time
        params = [None] * z.shape[0]
    if z.ndim == 1:  # For 1D data
        return [i_sense1d(x, z, params[0], func=func)]
    elif z.ndim == 2:  # For 2D data
        fit_result_list = []
        for i in range(z.shape[0]):
            fit_result_list.append(i_sense1d(x, z[i, :], params[i], func=func))
        return fit_result_list



def plot_standard_transition(dat, axs, plots: List[int] = (1, 2, 3), kwargs_list: List[dict] = None):
    """This returns a list of axes which show normal useful transition plots (assuming 2D for now)
    It requires a dat object to be passed to it so it has access to all other info
    1. 2D i_sense
    2. Centered and averaged i_sense
    3. 1D slice of i_sense with fit
    4. amplitude_per_line
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

    if 1 in plots:  # Add 2D i_sense
        ax = axs[i]
        ax.cla()
        data = dat.Transition._data
        title = '2D i_sense'
        ax = PF.display_2d(Data.x_array, Data.y_array, data, ax, x_label=dat.Logs.x_label,
                           y_label=dat.Logs.y_label, dat=dat, title=title, **kwargs_list[i])

        axs[i] = ax
        i += 1  # Ready for next plot to add

    if 2 in plots:  # Add averaged i_sense
        ax = axs[i]
        ax.cla()
        data = dat.Transition._avg_data
        title = 'Averaged Data'
        fit = dat.Transition._avg_full_fit
        ax = PF.display_1d(dat.Transition.x_array, data, ax, dat=dat, title=title, x_label=dat.Logs.x_label, y_label='Current/nA')
        ax.plot(dat.Transition.avg_x_array, fit.best_fit)
        axs[i] = ax
        i+=1

    if 3 in plots:  # 1D slice of i_sense with fit
        ax = axs[i]
        ax.cla()
        data = dat.Transition._data[0]
        title = '1D slice of Data'
        fit = dat.Transition._full_fits[0]
        ax = PF.display_1d(dat.Transition.x_array, data, ax, dat=dat, title=title,
                           x_label=dat.Logs.x_label, y_label='Current/nA')
        ax.plot(dat.Transition.avg_x_array, fit.best_fit)
        axs[i] = ax
        i += 1

    if 4 in plots:  # Amplitude per line
        ax = axs[i]
        ax.cla()
        data = dat.Transition.fit_values.amps
        title = 'Amplitude per row'
        ax = PF.display_1d(dat.Data.y_array, data, ax, dat=dat, title=title, x_label=dat.Logs.y_array, y_label='Amplitude /nA')
        axs[i] = ax
        i += 1

    if 11 in plots:  # Add dac table and other info
        ax = axs[i]
        PF.plot_dac_table(ax, dat)
        fig = plt.gcf()
        try:
            fig.suptitle(f'Dat{dat.datnum}')
            PF.add_standard_fig_info(fig)
            PF.add_to_fig_text(fig,
                               f'fit func = {dat.Transition.fit_func.__name__}, ACbias = {dat.Instruments.srs1.out / 50 * np.sqrt(2):.1f}nA, sweeprate={dat.Logs.sweeprate:.0f}mV/s, temp = {dat.Logs.temp:.0f}mK')
        except AttributeError:
            print(f'One of the attributes was missing for dat{dat.datnum} so extra fig text was skipped')
        axs[i] = ax
        i += 1