import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import src.Plotting.Mpl.PlotUtil
import src.UsefulFunctions
from src import CoreUtil as CU
from src.Plotting.Mpl.PlotUtil import xy_to_meshgrid, addcolorlegend, make_axes, ax_setup, get_colors, bin_for_plotting


def power_spectrum(data, meas_freq, normalization=1, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1)
        ax: plt.Axes
    freq, power = src.UsefulFunctions.power_spectrum(data, meas_freq, normalization)
    display_1d(freq[1:], power[1:], ax, **kwargs)  # First data point is super tiny
    ax.set_yscale("log", nonposy='clip')
    ax_setup(ax, 'Power Spectrum', 'Frequency /Hz', 'Power')
    return ax


def deviation_from_fit(x, data, best_fit, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1)
    data, x = CU.remove_nans(data, x)  # Mostly shouldn't need to do anything
    # kwargs['label'] = kwargs.get('label', 'Deviation from fit')
    display_1d(x, data-best_fit, ax, **kwargs)
    ax_setup(ax, title='Deviation from fit')
    return ax


def display_2d(x: np.array, y: np.array, data: np.array, ax: plt.Axes,
               norm=None, colorscale: bool = False, x_label: str = None, y_label: str = None, auto_bin=True, **kwargs):
    """Displays 2D data with axis x, y
    @param data: 2D numpy array
    @param norm: Normalisation for the colorscale if provided
    @param colorscale: Bool for show colorscale or not
    Function should only draw on values from kwargs, option args are just there for type hints but should immediately be added to kwargs
    """
    kwargs['colorscale'] = colorscale

    if auto_bin is True:
        x, data = bin_for_plotting(x, data)

    xx, yy = xy_to_meshgrid(x, y)
    ax.pcolormesh(xx, yy, data, norm=norm)

    if x_label is not None and kwargs.get('x_label', None) is None:  # kwargs gets precedence
        kwargs['x_label'] = x_label
    if y_label is not None and kwargs.get('y_label', None) is None:
        kwargs['y_label'] = y_label
    # kwarg options specific to 2D
    if kwargs.get('colorscale', False) is True:
        addcolorlegend(ax)
    del kwargs['colorscale']

    _optional_plotting_args(ax, **kwargs)
    return ax


def display_1d(x: np.array, data: np.array, ax: plt.Axes = None, x_label: str = None, y_label: str = None, errors: np.array = None, auto_bin=True, **kwargs):
    """Displays 1D data with axis x
    Function should only draw on values from kwargs, option args are just there for type hints but should immediately
     be added to kwargs
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if auto_bin is True:
        x, data = bin_for_plotting(x, data)

    if x_label is not None and kwargs.get('x_label', None) is None:
        kwargs['x_label'] = x_label
    if y_label is not None and kwargs.get('y_label', None) is None:
        kwargs['y_label'] = y_label

    label = kwargs.get('label', None)

    if errors is not None:
        errors = [np.nan if e in [None] else e for e in errors]  # fix errors if None values given...

    if kwargs.get('swap_ax', False) is False:
        x1 = x
        y1 = data
        error_arg = {'yerr': errors}
    else:  # Swap axis for plotting
        x1 = data
        y1 = x
        error_arg = {'xerr': errors}

    scatter = kwargs.get('scatter', False)
    cmap = kwargs.get('cmap', None)
    marker = kwargs.get('marker', '../..')
    linewidth = kwargs.get('linewidth', 2)
    color = kwargs.get('color', None)
    if scatter is True:
        ax.scatter(x1, y1, label=label, cmap=cmap, marker=marker, color=color)
    elif errors is None:
        ax.plot(x1, y1, label=label, marker=marker, linewidth=linewidth, color=color)
    else:
        ax.errorbar(x1, y1, label=label, linewidth=linewidth, **error_arg, color=color)

    # kwarg options specific to 1D
    src.Plotting.Mpl.PlotUtil.del_kwarg(['swap_ax', 'scatter', 'label', 'linewidth', 'marker', 'color'], kwargs)

    _optional_plotting_args(ax, **kwargs)
    return ax


def df_table(df: pd.DataFrame, title=None, sig_fig=3):
    """
    Makes a new figure which is sized based on table dimensions and optionally plots a title at the top

    @param df: dataframe to display
    @type df: pd.DataFrame
    @param title: optional title to display at top
    @type title: str
    @return: fig, ax of table
    @rtype: tuple[plt.Figure, plt.Axes]
    """
    width = 1.3*(len(df.columns))
    height = 0.75+0.35*(len(df.index))
    fig, ax = plt.subplots(1, figsize=(width, height))
    ax.axis('tight')
    ax.axis('off')
    if title is not None:
        ax.set_title(title)
    table = ax.table(cellText=(CU.sig_fig(df, sig_fig)).values, colLabels=df.columns, loc='center', bbox=(0, 0, 1, 1))
    for cell in table.get_celld().values():
        cell.PAD = 0.03
    table.auto_set_font_size(True)
    table.auto_set_column_width(range(df.shape[1]+1))
    fig.tight_layout()
    return fig, ax


def waterfall_plot(x, data, ax=None, y_spacing=1, y_add=None, x_spacing=0, x_add=None, every_nth=1, plot_args=None, ptype='plot',
                   label=False, index=None, color=None, cmap_name='viridis', auto_bin=True):
    """
    Plot 2D data as a waterfall plot

    Args:
        x (np.ndarray): x_array for data
        data (np.ndarray): 2D data
        ax (plt.Axes): Axes to plot on
        y_spacing (float): Fractional amount to space plots (1 is good starting point)
        y_add (float): Absolute amount to space plots
        x_spacing (float): Fractional amount to space plots (0 is good starting point)
        x_add (float): Absolute amount to space plots
        every_nth (int): Every nth line plotted from Data
        plot_args (dict): Args to pass to plotting function
        ptype (str): Plot type, ('plot', 'scatter', 'error')
        label (bool): Whether to add row num labels to data
        index (List[Union[str, int]]): List of labels to use for legend
        color (str): Force color of waterfall lines
        cmap_name (str): cmap to use for waterfall
        auto_bin (bool): Whether to bin data before plotting

    Returns:
        Tuple[float, float]: True spacing in y and x
    """

    if plot_args is None:
        plot_args = {}

    def get_plot_fn(ptype):
        """Returns plotting function"""
        if ptype == 'plot':
            return ax.plot
        elif ptype == 'scatter':
            return ax.scatter
        elif ptype == 'error':
            return ax.errorbar
        else:
            print('ERROR[waterfall_plot]: Need to specify ptype (plot_type, e.g. plot, scatter, error)')

    def get_2d_of_color(c, vals):
        if ptype == 'scatter':
            return np.repeat(np.atleast_2d(c), vals.shape[0], axis=0)
        else:
            return c

    if {'color', 'c'} & set(plot_args.keys()):  # Needed for how I deal with colors in plotting
        print(f'WARNING[waterfall_plot]: Provide color as keyword arg, not plot_arg')
        if 'color' in plot_args.keys():
            del plot_args['color']
        if 'c' in plot_args.keys():
            del plot_args['c']

    def get_colors_list(color, num):
        """Return list of colors, one for each row"""
        if color is None:
            return get_colors(num, cmap_name=cmap_name)
        elif type(color) == str:
            return [color] * int(num)
        elif type(color) == np.ndarray:
            return color
        else:
            print(f'WARNING[waterfall_plot]: Color must be a str or np.ndarray with len(y)')
            return get_colors(num, cmap_name=cmap_name)

    assert data.ndim == 2
    if auto_bin is True:
        x, data = bin_for_plotting(x, data)

    if ax is None:
        fig, ax = make_axes(1)
        ax = ax[0]
    else:
        fig = ax.figure

    y_num = int(np.floor(data.shape[0] / every_nth))
    if y_add is None:
        y_scale = np.nanmax(data) - np.nanmin(data)
        y_add = y_scale/y_num*y_spacing
    else:
        pass

    if x_add is None:
        x_scale = np.abs(x[-1]-x[0])
        x_add = x_scale/y_num*x_spacing

    plot_fn = get_plot_fn(ptype)
    cs = get_colors_list(color, y_num)
    if index is None or len(index) != y_num:
        index = range(y_num)

    if x.ndim == data.ndim:
        xs = x
    elif x.ndim == data.ndim-1:
        xs = np.array([x] * data.shape[0])
    else:
        raise ValueError(f'ERROR[PF.waterfall_plot]: Wrong shape of x, y passed. Got ([{x.shape}], [{data.shape}] '
                         f'(after possible binning)')

    for i, (label_text, x, row, c) in enumerate(zip(index, xs, data[::every_nth], cs)):
        plot_args['c'] = get_2d_of_color(c, row)
        plot_fn(x+x_add*i, row + y_add * i, **plot_args)
        if label is True:
            ax.plot([], [], label=f'{label_text}', c=c)
    return y_add, x_add


def _optional_plotting_args(ax, **kwargs):
    """Handles adding standard optional kwargs to ax"""

    keys = kwargs.keys()
    if 'swap_ax_labels' in keys:
        x_label = kwargs.get('y_label', None)
        y_label = kwargs.get('x_label', None)
        kwargs['x_label'] = x_label
        kwargs['y_label'] = y_label
        del kwargs['swap_ax_labels']
    if 'x_label' in keys and kwargs['x_label']:
        ax.set_xlabel(kwargs['x_label'])
        del kwargs['x_label']
    if 'y_label' in keys and kwargs['y_label']:
        ax.set_ylabel(kwargs['y_label'])
        del kwargs['y_label']
    if 'axtext' in keys and kwargs['axtext']:
        axtext = kwargs['axtext']
        ax.text(0.1, 0.8, f'{axtext}', fontsize=12, transform=ax.transAxes)
        del kwargs['axtext']
    if 'add_datnum' in keys and kwargs['add_datnum']:
        datnum = kwargs['add_datnum']
        ax.text(0.01, 0.94, f'Dat{datnum}', bbox=dict(boxstyle='round', facecolor='wheat'), color='k', fontsize=8, transform=ax.transAxes)
        del kwargs['add_datnum']
    if 'title' in keys and kwargs['title']:
        ax.set_title(kwargs['title'])
        del kwargs['title']


    unusued_args = [key for key in kwargs.keys() if kwargs[key] is not None]
    if len(unusued_args) > 0:
        print(f'Unused plotting arguments are: {unusued_args}')
    return ax


def dac_table(ax: plt.Axes, dat, fontsize=6):
    """Shows all non-zero DAC values in a table"""
    ax.axis('tight')
    ax.axis('off')
    df = pd.DataFrame(list(range(len(dat.Logs.dacs))), columns=['#'])
    df = df.join(pd.DataFrame.from_dict(dat.Logs.dacs, orient='index', columns=['DAC/mV']))
    df = df.join(pd.DataFrame.from_dict(dat.Logs.dacnames, orient='index', columns=['DACname']))
    df = df[df['DAC/mV'] != 0]
    df = df.fillna('')

    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='upper center',
                     colWidths=[0.2, 0.3, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)

    if hasattr(dat.Logs, 'fdacs') and dat.Logs.fdacs is not None:
        df = pd.DataFrame(list(range(len(dat.Logs.dacs))), columns=['#'])
        df = df.join(pd.DataFrame.from_dict(dat.Logs.fdacs, orient='index', columns=['FDAC/mV']))
        df = df.join(pd.DataFrame.from_dict(dat.Logs.fdacnames, orient='index', columns=['FDACname']))
        df = df.fillna('')
        df = df[(df['FDAC/mV'] != '') & (df['FDAC/mV'] != 0)]
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='lower center',
                         colWidths=[0.2, 0.3, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)

    return ax