import datetime
import inspect
import re
from typing import Union, Tuple, List, Optional
import logging
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from src import core_util as CU

logger = logging.getLogger(__name__)

PF_binning = True
PF_num_points_per_row = 1000


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


def addcolorlegend(ax) -> None:
    """Adds colorscale to ax"""
    for pcm in ax.get_children():
        if type(pcm) == mpl.collections.QuadMesh:
            break
    plt.colorbar(pcm, ax=ax, fraction=0.1, pad=-0.05)


_fig_text_position = (0.5, 0.02)


def set_figtext(fig: plt.Figure, text: str):
    """Replaces current figtext with new text"""
    fig = plt.figure(fig.number)  # Just to set as current figure to add text to
    for i, t in enumerate(fig.texts):  # Remove any fig text that has been added previously
        if t.get_position() == _fig_text_position:
            t.remove()
    plt.figtext(_fig_text_position[0], _fig_text_position[1], text, horizontalalignment='center', wrap=True)
    fig.tight_layout(rect=[0, 0.1, 1, 0.98])  # rect=(left, bottom, right, top)


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
    if text not in existing_text:
        text = text + existing_text
    else:
        text = existing_text
    set_figtext(fig, text)


def reuse_plots(num: int = 1, loc: Union[int, tuple] = 0) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Will reuse the last selected plot if it has the right number of axes and will bring it to front at loc (which
    can be 0, 1, 2 for screens, or a tuple for a location"""
    # mpluse('qt')
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


def make_axes(num: int = 1, single_fig_size=None, plt_kwargs: dict = None) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Makes required number of axes in grid where each axes is ~3.3x3.3 in by default

    @param num: How many axes to make
    @type num: int
    @param single_fig_size: Tuple of how big to make each axes
    @type single_fig_size: tuple
    @param plt_kwargs: Any additional kwargs to pass to plt.subplots
    @type plt_kwargs: dict
    @return: fig, List of axes subplots
    @rtype: Tuple[plt.Figure, List[plt.Axes]]
    """

    if plt_kwargs is None:
        plt_kwargs = {}
    if single_fig_size is None:
        single_fig_size = (3.3, 3.3)

    assert type(single_fig_size) == tuple
    ax: np.ndarray[plt.Axes]
    if num == 1:
        fig, ax = plt.subplots(1, 1, figsize=(single_fig_size[0], single_fig_size[1]), **plt_kwargs)  # 5, 5
        ax = np.ndarray([ax])
    elif 1 < num <= 2:
        fig, ax = plt.subplots(2, 1, figsize=(single_fig_size[0], 2 * single_fig_size[1]), **plt_kwargs)  # 5, 10
        ax = ax.flatten()
    elif 2 < num <= 4:
        fig, ax = plt.subplots(2, 2, figsize=(2 * single_fig_size[0], 2 * single_fig_size[1]),
                               **plt_kwargs)  # 9, 9 or 11.5, 9
        ax = ax.flatten()
    elif 4 < num <= 6:
        fig, ax = plt.subplots(2, 3, figsize=(3 * single_fig_size[0], 2 * single_fig_size[1]), **plt_kwargs)
        ax = ax.flatten()
    elif 6 < num <= 9:
        fig, ax = plt.subplots(3, 3, figsize=(3 * single_fig_size[0], 3 * single_fig_size[1]), **plt_kwargs)
        ax = ax.flatten()
    elif 9 < num <= 12:
        fig, ax = plt.subplots(3, 4, figsize=(4 * single_fig_size[0], 3 * single_fig_size[1]), **plt_kwargs)
        ax = ax.flatten()
    elif 12 < num <= 16:
        fig, ax = plt.subplots(4, 4, figsize=(4 * single_fig_size[0], 4 * single_fig_size[1]), **plt_kwargs)
        ax = ax.flatten()
    else:
        raise OverflowError(f'Can\'t build more than 16 axes in one go: User asked for {num}')
    fig: plt.Figure
    ax: List[plt.Axes]
    return fig, ax


def require_axs(num, axs, clear=False):
    """
    Checks if number of axs passed is >= num. If so, returns original axes, otherwise creates new.
    Args:
        num (int): Desired min number of axs
        axs (Union(np.ndarray[plt.Axes], plt.Axes, None)): The possibly existing axs to check if suitable
        clear (bool): Whether to clear axes passed

    Returns:
        np.ndarray[plt.Axes]: Either original fig, axes or new if original weren't sufficient (or closed)
    """
    # If None then just make new and return, no other thought necessary
    if axs is None:
        _, axs = make_axes(num)
    else:
        # Otherwise make sure in array form for checks
        axs = np.atleast_1d(axs)
        if axs.size < num or not plt.fignum_exists(axs[0].figure.number):
            _, axs = make_axes(num)
        elif clear is True:
            for ax in axs:
                ax.cla()
        else:
            raise NotImplementedError
    return axs


def mpluse(backend: str = 'qt') -> None:
    if backend == 'qt':
        mpl.use('qt5agg')
    elif backend == 'bi':
        mpl.use('module://backend_interagg')
    else:
        print('Please use \'qt\' for qt5agg, or \'bi\' for built in sciview')
    return None


def ax_text(ax, text, **kwargs):
    """adds text"""
    if 'fontsize' not in kwargs.keys():  # Default to ax_text == True
        kwargs = {**kwargs, 'fontsize': 10}
    if 'loc' in kwargs.keys():
        loc = kwargs['loc']
        del kwargs['loc']
    else:
        loc = (0.1, 0.7)
    ax.text(*loc, f'{text}', transform=ax.transAxes, **kwargs)


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

    if title is not None:
        ax.set_title(title, fontsize=fs * 1.2)
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=fs)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=fs)

    if legend is True:
        ax.legend(fontsize=fs)
    elif legend is False:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    for axis in [ax.xaxis, ax.yaxis]:
        for tick in axis.get_major_ticks():
            tick.label.set_fontsize(fs * 0.8)


def get_colors(num, cmap_name='viridis') -> list:
    """
    Returns list of colors evenly spaced with length num

    @param num: How many colors to return
    @type num: int
    @param cmap_name: name of cmap to use
    @type cmap_name: str
    @return: list of colors
    @rtype: list
    """

    cmap = plt.get_cmap(cmap_name)
    return cmap(np.linspace(0, 1, num))


def close_all_figures():
    """
    Quick command to close all figures
    @return: None
    @rtype: None
    """
    for num in plt.get_fignums():
        plt.close(plt.figure(num))


def remove_last_plot():
    ax = plt.gca()
    ax.lines[-1].remove()


def remove_last_scatter():
    ax = plt.gca()
    ax.collections[-1].remove()


def add_legend_label(label, ax: Optional[plt.Axes] = None, color: Optional[str] = None, size: Optional[float] = None,
                     marker: Optional[str] = None, linestyle: Optional[str] = None):
    """
    For adding labels to scatter plots where the scatter points are very small. Will default to using the same color as
    whatever was last added to whatever axes was last used, but those can be specified otherwise

    Args:
        label (): Label for legend
        ax (): plt.Axes to add to
        color ():
        size (): size of marker to use
        marker (): marker style
        linestyle (): If set, will add a line as well as marker to legend

    Returns:

    """
    if ax is None:
        ax = plt.gca()
    if color is None:
        color = ax.collections[-1].get_facecolor()
    if size is None:
        size = 10
    if linestyle is not None:
        ax.plot([], [], markersize=size, c=color, label=label, linestyle=linestyle, marker=marker)
    else:
        ax.scatter([], [], s=size, c=color, label=label, marker=marker)


def bin_for_plotting(x, data, num=None):
    """
    Returns reduced sized dataset so that there are no more than 'num' datapoints per row
    @param x: x_data
    @type x: np.ndarray
    @param data: plotting data, either 1D or 2D
    @type data: np.ndarray
    @return: reduced x, data tuple
    @rtype: tuple[np.ndarray]
    """
    if PF_binning is True:
        if num is None:
            num = PF_num_points_per_row
        bin_size = np.ceil(len(x) / num)
        if bin_size > 1:
            logger.info(f'PF.bin_for_plotting: auto_binning with bin_size [{bin_size}] applied')
        return CU.bin_data([x, data], bin_size)
    else:
        return [x, data]


def get_gridspec(fig: plt.Figure, num: int, return_list=True):
    if num > 12:
        logger.warning('[add_gridspec]: got request for more than 12 plots, only returning 12')
        num = 12
    if num != 0:
        shape_dict = {1: (1, 1), 2: (2, 1), 3: (2, 2), 5: (3, 2), 7: (3, 3), 10: (4, 3)}
        arr = np.array(list(shape_dict.keys()))  # array of index
        key = arr[arr <= num].max()  # largest element lower than num (i.e. previous size which fits required num
        shape = shape_dict[key]
        gs = fig.add_gridspec(*shape)
        if return_list is True:
            gs = [gs[i, j] for i in range(shape[0]) for j in range(shape[1])]
    else:
        if return_list is True:
            gs = []
        else:
            gs = None
    return gs


def toggle_draggable_legend(fig=None, axs=None):
    all_axs = []
    if fig is not None:
        all_axs += fig.axes
    if axs is not None:
        axs = CU.ensure_list(axs)
        all_axs += axs
    ax: plt.Axes
    for ax in all_axs:
        leg = ax.get_legend()
        if leg:
            leg.set_draggable(not leg.get_draggable())


def adjust_lightness(color, amount=-0.1):
    """
        Lightens the given color by multiplying (1-luminosity) by the given amount.
        Input can be matplotlib color string, hex string, or RGB tuple.
        https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib

        Examples:
        >> adjust_lightness('g', 0.3)
        >> adjust_lightness('#F034A3', 0.6)
        >> adjust_lightness((.3,.55,.1), 0.5)
        """
    amount = amount + 1  # So that 0 does nothing, -ve darkens, +ve lightens
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def remove_line(ax: plt.Axes, label: str, verbose=True):
    """Removes labelled line from plt.Axes"""
    found = False
    for line in ax.lines:
        if line.get_label().lower() == label.lower():
            line.remove()
            found = True
    if found is False and verbose:
        logger.info(f'"{label}" not found in ax.lines')


def edit_title(ax, text, prepend=False, append=False):
    """
    Adds text to beginning or end of existing title on axes

    Args:
        ax (plt.Axes): Axes with title to edit
        text (str): Text to add to beginning or end of title
        prepend (bool): Prepend text (can't do both at same time)
        append (bool): Append text (can't do both at same time)

    Returns:
        plt.Axes: ax adjusted
    """
    assert prepend ^ append  # Assert 1 and only 1 is True
    t = ax.title.get_text()
    if prepend:
        ax.title.set_text(f'{text}{t}')
    elif append:
        ax.title.set_text(f'{t}{text}')
    else:
        raise NotImplementedError
    return ax


def del_kwarg(name, kwargs):
    """
    Deletes name(s) from kwargs if present in kwargs
    @param kwargs: kwargs to try deleting args from
    @type kwargs: dict
    @param name: name or names of kwargs to delete
    @type name: Union[str, List[str]]
    @return: None
    @rtype: None
    """

    def del_1_kwarg(n, ks):
        try:
            del ks[n]
        except KeyError:
            pass

    names = np.atleast_1d(name)
    for name in names:
        del_1_kwarg(name, kwargs)


def set_default_rcParams():
    mpl.rcParams.update(
        {
            'figure.figsize': [3.375, 2.5],
            'axes.labelsize': 8,
            'font.size': 8,
            'legend.fontsize': 8,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'text.usetex': False,
        }
    )
