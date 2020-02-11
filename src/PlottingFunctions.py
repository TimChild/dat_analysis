import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import inspect
import re
from typing import List, Tuple
import src.Configs.Main_Config as cfg
import src.CoreUtil as CU
import src.DatCode.Dat as Dat
import datetime

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
    plt.colorbar(pcm, ax=ax)


def display_2d(x: np.array, y: np.array, data: np.array, ax: plt.Axes,
               norm=None, colorscale: bool = False, xlabel: str = None, ylabel: str = None, **kwargs):
    """Displays 2D data with axis x, y
    @param data: 2D numpy array
    @param norm: Normalisation for the colorscale if provided
    @param colorscale: Bool for show colorscale or not
    Function should only draw on values from kwargs, option args are just there for type hints but should immediately be added to kwargs
    """
    kwargs = dict(kwargs, **{'norm': norm, 'colorscale': colorscale, 'x_label': xlabel,
                             'y_label': ylabel})  # TODO: better way of adding all optional params to kwargs?

    xx, yy = xy_to_meshgrid(x, y)
    ax.pcolormesh(xx, yy, data, norm=norm)

    # kwarg options
    if 'colorscale' in kwargs.keys() and kwargs['colorscale'] is True:
        addcolorlegend(ax)
    _optional_plotting_args(ax, **kwargs)


def display_1d(x: np.array, data: np.array, ax: plt.Axes = None, x_label: str = None, y_label: str = None, dat: Dat = None, **kwargs):
    """Displays 2D data with axis x, y
    Function should only draw on values from kwargs, option args are just there for type hints but should immediately
     be added to kwargs
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if dat is not None:
        x_label = dat.Logs.x_label
        y_label = dat.Logs.y_label
    if x_label is not None and kwargs.get('x_label', None) is None:
        kwargs['x_label'] = x_label
    if y_label is not None and kwargs.get('y_label', None) is None:
        kwargs['y_label'] = y_label

    ax.plot(x, data)

    # kwarg options
    _optional_plotting_args(ax, **kwargs)
    return ax


def _optional_plotting_args(ax, **kwargs):
    """Handles adding standard optional kwargs to ax"""
    if 'x_label' in kwargs.keys() and kwargs['x_label']:
        ax.set_xlabel(kwargs['x_label'])
        del kwargs['x_label']
    if 'y_label' in kwargs.keys() and kwargs['y_label']:
        ax.set_ylabel(kwargs['y_label'])
        del kwargs['y_label']
    if 'axtext' in kwargs.keys() and kwargs['axtext']:
        axtext = kwargs['axtext']
        ax.text(0.1, 0.8, f'{axtext}', fontsize=12, transform=ax.transAxes)
        del kwargs['axtext']
    unusued_args = [key for key in kwargs.keys()]
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
    for f in stack:
        filename = f.filename
        if re.search('/', filename):  # Seems to be that only the file the code is initially run from has forward slashes in the filename...
            break
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


def make_axes(num: int = 1) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Makes required number of axes in grid"""
    if num == 1:
        fig, ax = plt.subplots(1, 1, figsize=(3.3, 3.3))  # 5, 5
        ax = np.array(ax)
    elif 1 < num <= 2:
        fig, ax = plt.subplots(2, 1, figsize=(3.3, 6))  # 5, 10
        ax = ax.flatten()
    elif 2 < num <= 4:
        fig, ax = plt.subplots(2, 2, figsize=(6.6, 6.6))  # 9, 9 or 11.5, 9
        ax = ax.flatten()
    elif 4 < num <= 6:
        fig, ax = plt.subplots(2, 3, figsize=(9, 6.6))
        ax = ax.flatten()
    elif 6 < num <= 9:
        fig, ax = plt.subplots(3, 3, figsize=(10, 10))
        ax = ax.flatten()
    elif 9 < num <= 12:
        fig, ax = plt.subplots(3, 4, figsize=(12, 10))
        ax = ax.flatten()
    elif 12 < num <= 16:
        fig, ax = plt.subplots(4, 4, figsize=(12, 12))
        ax = ax.flatten()
    else:
        raise OverflowError("Can't build more than 16 axes in one go")
    fig: plt.Figure
    ax: list[plt.Axes]
    return fig, ax