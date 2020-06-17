import numpy
import numpy as np
import types
from typing import List, NamedTuple
import src.DatCode.DatAttribute as DA
import lmfit as lm
import pandas as pd
import src.CoreUtil as CU
import src.Configs.Main_Config as cfg
import src.PlottingFunctions as PF
import matplotlib.pyplot as plt


# TODO: Finish this one
class NewDCBias(DA.DatAttribute):
    version = '1.0'
    group_name = 'DCbias'

    def __init__(self, hdf):
        super().__init__(hdf)

    def get_from_HDF(self):
        dg = self.group['Data']

    def update_HDF(self):
        super().update_HDF()

    def _set_default_group_attrs(self):
        pass







class DCbias(object):
    version = '1.3'  # To keep track of whether fitting has changed
    """
    Version updates:
        1.1 -- added option to fit over given width (in +- nA)
        1.2 -- drop theta values which are zero before fitting
        1.3 -- get_dt_at_current returns the true amplitude, not peak to peak now. 
    """

    def __init__(self, x_array, y_array, i_sense, transition_fit_values):
        """Tries to fit quadratic to whole data set. Can later call to provide range to fit over"""
        assert transition_fit_values is not None

        self._data = i_sense
        self._x_array = x_array
        self._y_array = _y_to_current(y_array)
        self.version = DCbias.version
        self.thetas = transition_fit_values.thetas

        self._width = None
        self._full_fit = self._fit_dc_bias(self._dcbias_init_params(), width=None)

    @property
    def init_params(self):
        return self._full_fit.init_params

    @property
    def params(self):
        return self._full_fit.params

    @property
    def full_fit(self):
        return self._full_fit

    @property
    def fit_values(self):
        return self._get_fit_values()

    @property
    def x_array_for_fit(self):
        if hasattr(self, '_x_array_for_fit') is False:  # This should keep it working for older versions
            self._x_array_for_fit = self._y_array
        return self._x_array_for_fit

    @x_array_for_fit.setter
    def x_array_for_fit(self, x):
        self._x_array_for_fit = x

    @property
    def min_theta(self):
        return _get_quad_min(self._full_fit)

    def recalculate_fit(self, width=None):
        self._full_fit = self._fit_dc_bias(self.params, width=width)
        self.version = DCbias.version

    def _get_fit_values(self):
        a = self._full_fit.best_values['a']
        b = self._full_fit.best_values['b']
        c = self._full_fit.best_values['c']
        return {'a': a, 'b': b, 'c': c}

    def _fit_dc_bias(self, params, width=None) -> lm.model.ModelResult:
        y = np.array(self._y_array, dtype=np.float32)
        thetas = np.array(self.thetas, dtype=np.float32)
        if width is not None:  # Then only fit over given width
            self._width = width
            neg_index, pos_index = CU.get_data_index(y, -width), CU.get_data_index(y, width)
            y = y[neg_index:pos_index]
            thetas = thetas[neg_index:pos_index]
        y, thetas = zip(*((y, t) for y, t in zip(y, thetas) if not np.isclose(t, 0, atol=0.1)))  # Drop thetas
        # that didn't fit
        self.x_array_for_fit = y
        quad = lm.models.QuadraticModel()
        fit = quad.fit(thetas, x=y, params=params, nan_policy='propagate')
        return fit

    def get_current_for_target_heat(self, multiplier=1.3) -> float:
        fit = self._full_fit
        a = fit.best_values['a']
        b = fit.best_values['b']
        c = fit.best_values['c']
        miny = _get_quad_min(fit)
        target_y = miny * multiplier
        pos_value = (-b + np.sqrt(b ** 2 - 4 * a * (c - target_y)) / (2 * a))  # Solving a^2 + bx + c = target_y
        neg_value = (-b - np.sqrt(b ** 2 - 4 * a * (c - target_y)) / (2 * a))
        return np.average([np.abs(pos_value), np.abs(neg_value)])  # average of two values since going to be using AC current.

    def get_dt_at_current(self, current: float) -> float:
        fit = self._full_fit
        pos_value = fit.eval(fit.params, x=current)
        neg_value = fit.eval(fit.params, x=-current)
        value = np.average([np.abs(pos_value), np.abs(neg_value)]) - _get_quad_min(fit)
        return float(value/2)  # Over 2 because dT is an amplitude of oscillation, not peak to peak

    def _dcbias_init_params(self) -> lm.Parameters:
        thetas = self.thetas
        params = lm.Parameters()
        # add with tuples: (NAME    VALUE   VARY  MIN   MAX     EXPR  BRUTE_STEP)
        params.add_many(('a', 0, True, None, None, None, None),
                        ('b', 0, True, None, None, None, None),
                        ('c', np.nanmin(thetas), True, None, None, None, None))
        return params

    @staticmethod
    def standard_plot_function():
        return plot_standard_dcbias

    @staticmethod
    def plot_self(dc, dat=None):
        """
        Plot standard DCbias plots, optionally add markers for where dT was calculated for given dat

        @param dc: dcbias dat
        @type dc: Dat
        @param dat: entropy dat
        @type dat: Dat
        @return: fig, axs
        @rtype: Tuple[plt.Figure, list[plt.Axes]]
        """
        fig, axs = PF.make_axes(4)
        plot_standard_dcbias(dc, axs, plots=[1, 2, 3, 4])

        if dat is not None:  # Replace axs[1] with version that has dT on it too.
            plot_dc_with_dt_points(dc, dat, ax=axs[1], add_fig_title=False)

        return fig, axs


def _get_quad_min(fit: lm.model.ModelResult) -> float:
    a = fit.best_values['a']
    b = fit.best_values['b']
    c = fit.best_values['c']
    miny = fit.model.eval(fit.params, x=(-b / (2 * a)))
    return float(miny)


def _y_to_current(y_array):
    """Takes x_array in mV and returns x_array in nA"""
    # TODO: Better if this doesn't look directly at cfg file and instead uses something stored in dat
    DC_HQPC_current_bias_resistance = cfg.DC_current_bias_resistance
    print(f'Using DC_current_bias_resistance of {DC_HQPC_current_bias_resistance}ohms')
    return (y_array/1e3) / DC_HQPC_current_bias_resistance * 1e9  # /1e3 is to V, then *1e9 is to nA


def plot_standard_dcbias(dat, axs, plots: List[int] = (1, 2, 3), kwargs_list: List[dict] = None):
    """This returns a list of axes which show normal useful DCbias plots (assuming 2D type for now)
    It requires a dat object to be passed to it so it has access to all other info
    1.  2D data
    2.  Theta vs Current
    3.  Fit Values (in a table)
    4.  Adds dac values (and other fig info)


    Kwarg hints:
    swap_ax, swap_ax_labels"""

    plot_standard_dcbias.plot_names = {
        1: '2D data',
        2: 'Theta vs Current',
        3: 'Fit values',
        4: 'Dac values'  # Also adds figure info
    }

    dc = dat.DCbias
    assert isinstance(dc, DCbias)


    assert len(axs) >= len(plots)
    kwargs_list = PF.set_kwarg_defaults(kwargs_list, len(plots))  # Sets things like no datnum for all plots

    i = 0
    if 1 in plots:  # Add 2D CS data
        ax = axs[i]
        ax.cla()
        data = dc._data
        title = 'DCbias Data'
        ax = PF.display_2d(dc._x_array, dc._y_array, data, ax, x_label=dat.Logs.x_label,
                           y_label='Current /nA', dat=dat, title=title, **kwargs_list[i])
        axs[i] = ax
        i += 1  # Ready for next plot to add

    if 2 in plots:  # add Theta vs Current
        ax = axs[i]
        ax.cla()
        data = dc.thetas
        title = 'Theta vs Bias/nA'
        ax = PF.display_1d(dc._y_array, data, ax=ax, x_label='Current/ nA', y_label='Theta /mV', dat=dat, label='data', scatter=True, title = title, **kwargs_list[i])
        ax.plot(dc.x_array_for_fit, dc.full_fit.best_fit, color='xkcd:dark red', label='best fit')
        ax.legend()
        axs[i] = ax
        i += 1  # Ready for next plot to add

    if 3 in plots:  # add Fit values as table
        ax = axs[i]
        ax.cla()
        ax.axis('off')
        ax.axis('tight')
        title = 'Fit info'

        rownames = ['a', 'b', 'c']
        series = pd.Series([dc.fit_values['a'], dc.fit_values['b'], dc.fit_values['c']])

        series = series.apply(lambda x: round(x, 3 - int(np.floor(np.log10(abs(x))))))

        table = ax.table(cellText=[[val] for val in series.values], rowLabels=rownames, colLabels=['Fit eqn: ax^2 + bx + c'], loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        PF._optional_plotting_args(ax, title=title)
        axs[i] = ax
        i += 1

    if 4 in plots:
        ax = axs[i]
        ax.cla()
        fig = plt.gcf()
        fig.suptitle(f'Dat{dat.datnum}')
        PF.add_standard_fig_info(fig)
        temp = dat.Logs.temps['mc'] * 1000
        PF.add_to_fig_text(fig, f'Temp = {temp:.1f}mK')
        if dat.Logs.sweeprate is not None:
            PF.add_to_fig_text(fig, f'Sweeprate = {dat.Logs.sweeprate:.1f}mV/s')
        PF.plot_dac_table(ax, dat)
        axs[i] = ax
        i += 1
    return axs


def plot_dc_with_dt_points(dc, dat, ax=None, add_fig_title=True, **kwargs):
    """
    Single plot of theta vs DCbias with fit shown as well as points where dT would be calculated for a given dat

    @param ax: axes to plot on
    @type ax: Union[plt.Axes, None]
    @param dc: DCbias dat
    @param dat: Entropy dat to get i_heat from
    """
    if ax is None:
        fig, ax = PF.make_axes(1)
        ax = ax[0]
    else:
        fig = ax.figure

    # region Plot 2 from standard DCbias plots which puts data and fit on ax
    ax.cla()

    data = dc.DCbias.thetas
    title = 'Theta vs Bias/nA'
    ax = PF.display_1d(dc.DCbias._y_array, data, ax=ax, x_label='Current/ nA', y_label='Theta /mV', dat=dat, label='data',
                       scatter=True, title=title, **kwargs)
    ax.plot(dc.DCbias.x_array_for_fit, dc.DCbias.full_fit.best_fit, color='xkcd:dark red', label='best fit')
    # endregion

    i_heat_ac = dat.Instruments.srs1.out / 50 * np.sqrt(2)
    fit = dc.DCbias.full_fit
    y_min, _ = ax.get_ylim()
    x_min, x_max = ax.get_xlim()
    fit_min = np.nanmin(fit.eval(x=np.linspace(-10, 10, 10000)))
    dt = dc.DCbias.get_dt_at_current(i_heat_ac)
    ax.margins(x=0, y=0)
    for i_heat in [i_heat_ac, -i_heat_ac]:
        y_val = fit.eval(x=i_heat)
        ax.plot([i_heat, i_heat], [y_min, y_val], color='k', linestyle=':')  # Vertical lines to fit
        ax.plot([x_min, i_heat], [y_val, y_val], color='k', linestyle='--')  # Horizontal lines to fit
        ax.plot([x_min, x_min + (x_max - x_min) / 10], [fit_min + dt * 2, fit_min + dt * 2],
                color='C3')  # Lines near y_axis showing dT
        ax.plot([x_min, x_min + (x_max - x_min) / 10], [fit_min, fit_min],
                color='C3')  # Lines near y_axis showing dT
    ax.plot([], [], color='C3', label=f'2*dT = {dt * 2:.2f}')
    ax.legend()

    if add_fig_title is True:
        fig.suptitle(f'DC[{dc.datnum}], Dat[{dat.datnum}]')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, ax