import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import inspect
import re
from typing import List, Tuple, Union
import src.Configs.Main_Config as cfg
import src.CoreUtil as CU
import src.DatCode.Dat as Dat
import datetime

import pandas as pd


def xy_to_meshgrid(x, y):
    """ returns a meshgrid that makes sense for pcolorgrid
        given z data that should be centered at (x,y) pairs """
    nx = len(x)
    ny = len(y)

    dx = (x[-1] - x[0]) / float(nx - 1)
    dy = (y[-1] - y[0]) / float(ny - 1)

    # shift x and y back by half a step
    x = x - dx / 2.0
    y = y - dy / 2.0

    xn = x[-1] + dx
    yn = y[-1] + dy

    return np.meshgrid(np.append(x, xn), np.append(y, yn))


def get_ax(ax=None) -> plt.Axes:
    """Either return ax passed, or get current ax if None passed. Can add further functionality here later"""
    if ax is None:
        ax = plt.gca()
    return ax


def addcolorlegend(ax) -> None:
    """Adds colorscale to ax"""
    for pcm in ax.get_children():
        if type(pcm) == mpl.collections.QuadMesh:
            break
    plt.colorbar(pcm, ax=ax, fraction=0.1, pad=-0.05)


def display_2d(x: np.array, y: np.array, data: np.array, ax: plt.Axes,
               norm=None, colorscale: bool = False, x_label: str = None, y_label: str = None, dat=None, **kwargs):
    """Displays 2D data with axis x, y
    @param data: 2D numpy array
    @param norm: Normalisation for the colorscale if provided
    @param colorscale: Bool for show colorscale or not
    Function should only draw on values from kwargs, option args are just there for type hints but should immediately be added to kwargs
    """
    kwargs['colorscale'] = colorscale

    xx, yy = xy_to_meshgrid(x, y)
    ax.pcolormesh(xx, yy, data, norm=norm)

    if dat is not None:
        if x_label is None:
            x_label = dat.Logs.x_label
        if y_label is None:
            y_label = dat.Logs.y_label
        if kwargs.get('no_datnum', False) is False:
            kwargs['add_datnum'] = dat.datnum
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


def display_1d(x: np.array, data: np.array, ax: plt.Axes = None, x_label: str = None, y_label: str = None,
               dat: Dat = None, errors: np.array = None, **kwargs):
    """Displays 2D data with axis x, y
    Function should only draw on values from kwargs, option args are just there for type hints but should immediately
     be added to kwargs
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if dat is not None:
        if x_label is None:
            x_label = dat.Logs.x_label
        if y_label is None:
            y_label = dat.Logs.y_label
        if kwargs.get('no_datnum', False) is False:
            kwargs['add_datnum'] = dat.datnum
    if x_label is not None and kwargs.get('x_label', None) is None:
        kwargs['x_label'] = x_label
    if y_label is not None and kwargs.get('y_label', None) is None:
        kwargs['y_label'] = y_label

    if kwargs.get('label', None) is None:
        label = None
    else:
        label = kwargs['label']

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
    marker = kwargs.get('marker', '.')
    linewidth = kwargs.get('linewidth', 2)
    if scatter is True:
        ax.scatter(x1, y1, label=label, cmap=cmap, marker=marker)
    elif errors is None:
        ax.plot(x1, y1, label=label, marker=marker, linewidth=linewidth)
    else:
        ax.errorbar(x1, y1, label=label, linewidth=linewidth, **error_arg)

    # kwarg options specific to 1D
    try:  # Don't need it anymore
        del kwargs['swap_ax']
    except KeyError:
        pass

    _optional_plotting_args(ax, **kwargs)
    return ax


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


_fig_text_position = (0.5, 0.02)


def set_figtext(fig: plt.Figure, text: str):
    """Replaces current figtext with new text"""
    fig = plt.figure(fig.number)  # Just to set as current figure to add text to
    for i, t in enumerate(fig.texts):  # Remove any fig text that has been added previously
        if t.get_position() == _fig_text_position:
            t.remove()
    plt.figtext(_fig_text_position[0], _fig_text_position[1], text, horizontalalignment='center', wrap=True)
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])  # rect=(left, bottom, right, top)


def add_standard_fig_info(fig: plt.Figure):
    """Add file info etc to figure"""
    text = []
    stack = inspect.stack()
    filename = ''
    for f in stack:
        filename = f.filename
        if re.search('/',
                     filename):  # Seems to be that only the file the code is initially run from has forward slashes in the filename...
            break
    if not re.search('PyDatAnalysis', filename):
        print('No filename found')
        return fig
    _, short_name = filename.split('PyDatAnalysis', 1)
    text = [short_name]

    dmy = '%Y-%b-%d'  # Year:month:day
    text.append(f'{datetime.datetime.now().strftime(dmy)}')
    for t in text:
        add_to_fig_text(fig, t)
    return fig


def add_to_fig_text(fig: plt.Figure, text: str):
    """Adds text to figtext at the front"""
    existing_text = ''
    for t in fig.texts:  # Grab any fig_info_text that is already on figure
        if t.get_position() == _fig_text_position:
            existing_text = f' ,{t._text}'
            break
    text = text + existing_text
    set_figtext(fig, text)


def reuse_plots(num: int = 1, loc: Union[int, tuple] = 0) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Will reuse the last selected plot if it has the right number of axes and will bring it to front at loc (which
    can be 0, 1, 2 for screens, or a tuple for a location"""
    mpluse('qt')
    global figexists
    try:
        if figexists == True:
            fig = plt.gcf()
            if len(fig.axes) == num:
                ax = fig.axes
            for a in ax:
                a.cla()
        else:
            fig, ax = make_axes(num)
    except NameError:
        fig, ax = make_axes(num)
        figexists = True
    # fig_to_front(fig, loc)
    return fig, ax


def make_axes(num: int = 1, plt_kwargs={}) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Makes required number of axes in grid"""
    if num == 1:
        fig, ax = plt.subplots(1, 1, figsize=(3.3, 3.3), **plt_kwargs)  # 5, 5
        ax = [ax]
    elif 1 < num <= 2:
        fig, ax = plt.subplots(2, 1, figsize=(3.3, 6), **plt_kwargs)  # 5, 10
        ax = ax.flatten()
    elif 2 < num <= 4:
        fig, ax = plt.subplots(2, 2, figsize=(6.6, 6.6), **plt_kwargs)  # 9, 9 or 11.5, 9
        ax = ax.flatten()
    elif 4 < num <= 6:
        fig, ax = plt.subplots(2, 3, figsize=(9, 6.6), **plt_kwargs)
        ax = ax.flatten()
    elif 6 < num <= 9:
        fig, ax = plt.subplots(3, 3, figsize=(10, 10), **plt_kwargs)
        ax = ax.flatten()
    elif 9 < num <= 12:
        fig, ax = plt.subplots(3, 4, figsize=(12, 10), **plt_kwargs)
        ax = ax.flatten()
    elif 12 < num <= 16:
        fig, ax = plt.subplots(4, 4, figsize=(12, 12), **plt_kwargs)
        ax = ax.flatten()
    else:
        raise OverflowError(f'Can\'t build more than 16 axes in one go: User asked for {num}')
    fig: plt.Figure
    ax: list[plt.Axes]
    return fig, ax


def mpluse(backend: str = 'qt') -> None:
    if backend == 'qt':
        mpl.use('qt5agg')
    elif backend == 'bi':
        mpl.use('module://backend_interagg')
    else:
        print('Please use \'qt\' for qt5agg, or \'bi\' for built in sciview')
    return None


def standard_dat_plot(dat, mpl_backend: str = 'qt', raw_data_names: List[str] = None, fit_attrs: dict = {},
                      dfname='default', **kwargs):
    """Make a nice standard layout of figs/info for dats
    Designed to be used as a method for Dat objects as they can pass in many kwarg params
    @param fit_attrs: e.g. {'transition':['mids', 'amps', 'lins'], 'entropy': ['dSs']} to show data of transition
    and entropy fits plus plots for the named fit params. Otherwise should be empty list e.g. {'pinch':[]}
    """
    # mpluse(mpl_backend)

    if raw_data_names is None:
        dat_waves = dat.Data.get_names()
        raw_data_names = list(
            set(dat_waves) & set(cfg.common_raw_wavenames))  # Names that correspond to raw data and are in dat

    for v in fit_attrs.values():  # names for additional graphs must be in a list
        assert type(v) == list

    num_fit_attrs = len(fit_attrs.keys())
    num_fit_attr_values = sum([len(v) for v in fit_attrs.values()])
    num_raw_data = len(raw_data_names)

    num_graphs = num_fit_attrs + num_fit_attr_values + num_raw_data
    if num_graphs == 0:
        print(f'No graphs to plot, maybe you want to add some of these to raw_data_names: {cfg.common_raw_wavenames}')
        return None
    num_rows = int(np.ceil(num_graphs / 2))
    widths = [1, 1, 2]
    heights = [1] * num_rows + [1]  # last ratio is for other data

    fig = plt.figure(constrained_layout=True, figsize=(7, 2 * (num_rows + 1)))
    gs = fig.add_gridspec(nrows=num_rows + 1, ncols=3, width_ratios=widths,
                          height_ratios=heights)  # Extra row for metadata

    if dat.Logs.dim not in [1, 2]:
        raise ValueError(f'dat.Logs.dim should be 1 or 2, not {dat.Logs.dim}')

    axs = []
    axs = [fig.add_subplot(gs[int(np.floor(i / 2)), i % 2]) for i in range(num_graphs)]

    # Add data that fit values used
    for ax, attr_name in zip(axs[:num_fit_attrs], fit_attrs.keys()):
        fit_attr = getattr(dat, attr_name, None)
        assert fit_attr is not None
        data = fit_attr._data  # all fit_attrs should store 'raw' data as ._data
        if data is not None:
            if dat.Logs.dim == 1:
                display_1d(dat.Data.x_array, data, ax=ax, dat=dat, no_datnum=True)
            elif dat.Logs.dim == 2:
                display_2d(dat.Data.x_array, dat.Data.y_array, data, ax=ax, colorscale=True, dat=dat, no_datnum=True)

    #  Now add fit values if asked for
    for attr_name in fit_attrs.keys():  # TODO: Make axs work for more than one fit attr
        fit_attr = getattr(dat, attr_name, None)
        if getattr(fit_attr, 'fit_values', None) is not None:
            fit_values = fit_attr.fit_values
            for ax, fit_values_name in zip(axs[num_fit_attrs:num_fit_attr_values + num_fit_attrs],
                                           fit_attrs[attr_name]):
                values = fit_values._asdict().get(fit_values_name, None)
                if values is None:
                    print(
                        f'No fit values found for {attr_name}.{fit_values_name}. Keys available are {fit_values._fields}')
                else:
                    display_1d(dat.Data.y_array, values, ax, dat.Logs.y_label, fit_values_name, swap_ax=True,
                               swap_ax_labels=True, no_datnum=True, scatter=True)

    # Add raw data
    for ax, raw_name in zip(axs[-num_raw_data:], raw_data_names):
        data = getattr(dat.Data, raw_name, None)
        if data is not None:
            if dat.Logs.dim == 1:
                display_1d(dat.Data.x_array, data, ax)
            elif dat.Logs.dim == 2:
                display_2d(dat.Data.x_array, dat.Data.y_array, data, ax, dat=dat, colorscale=True, no_datnum=True)

    # Add Fit Values
    for i, attr_name in enumerate(fit_attrs.keys()):
        fit_attr = getattr(dat, attr_name, None)
        if fit_attr is not None:
            ax = fig.add_subplot(gs[i, 2])
            ax.axis('off')
            ax.axis('tight')
            fit_values = {k: np.average(v) for k, v in fit_attr.fit_values._asdict().items() if v is not None}
            data = [[round(v, 3)] for v in fit_values.values()]
            names = [k for k in fit_values.keys()]
            table = ax.table(cellText=data, rowLabels=names, colLabels=['Avg Fit value'], loc='center',
                             colWidths=[0.5, 0.5])
            table.auto_set_font_size(False)
            table.set_fontsize(8)

    try:
        from src.DFcode.DatDF import DatDF
        # Add Other data
        fig.suptitle(f'Dat{dat.datnum}')
        datdf = DatDF(dfname=dfname)
        ax = fig.add_subplot(gs[-1, :1])
        ax.axis('off')
        ax.axis('tight')
        columns = [('Logs', 'time_elapsed'), ('Logs', 'fdacfreq')]
        s1 = datdf.df.loc[(dat.datnum, dat.datname), columns]
        colnames = ['Time /s', 'Rate /Hz']
        s2 = pd.Series([round(dat.Logs.temps['mc'] * 1000, 0)])
        colnames.append('Temp/mK')
        series = s1.append(s2)

        table = ax.table(cellText=[series.values], colLabels=colnames, loc='center', colWidths=[0.4, 0.4, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
    except KeyError:  # If not saved in datdf yet, then will throw errors
        print(f'dat{dat.datnum} is not saved in datdf[{dfname}] yet so no df data added')
        pass

    # Add Dac/FDac info
    ax = fig.add_subplot(gs[-1, 2])
    ax = plot_dac_table(ax, dat)
    return fig, axs


def plot_dac_table(ax: plt.Axes, dat, fontsize=6):
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

    if hasattr(dat.Logs, 'fdacs'):
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


def set_kwarg_defaults(kwargs_list: List[dict], length: int):
    """Sets default kwargs for things dat attribute plots"""

    if kwargs_list is not None:
        assert len(kwargs_list) == length
        assert type(kwargs_list[0]) == dict
        kwargs_list = [{**k, 'no_datnum': True} if 'no_datnum' not in k.keys() else k for k in kwargs_list]  # Make
        # no_datnum default to True if not passed in.
    else:
        kwargs_list = [{'no_datnum': True}] * length
    return kwargs_list


def ax_text(ax, text, **kwargs):
    """adds text"""
    if 'fontsize' not in kwargs.keys():  # Default to ax_text == True
        kwargs = {**kwargs, 'fontsize': 10}
    if 'loc' not in kwargs.keys():
        kwargs = {**kwargs, 'loc': (0.1, 0.7)}
    ax.text(*kwargs['loc'], f'{text}', transform=ax.transAxes)


def ax_setup(ax, title=None, x_label=None, y_label=None, legend=None, fs=10):
        """
        A quicker way to make axes look good... Will overwrite where it can, and will try to avoid cluttering upon repeated
        calls

        @param fs: fontsize
        @type fs: int
        @param ax:  axes to modify
        @type ax: plt.Axes
        @param title: Ax title
        @type title: str
        @param x_label:
        @type x_label: str
        @param y_label:
        @type y_label: str
        @param legend:
        @type legend: bool
        @return: None -- Only edits axes passed
        @rtype: None
        """

        ax.set_title(title, fontsize=fs * 1.2)
        ax.set_xlabel(x_label, fontsize=fs)
        ax.set_ylabel(y_label, fontsize=fs)

        if legend is True:
            ax.legend(fontsize=fs)
