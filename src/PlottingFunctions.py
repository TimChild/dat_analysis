import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import src.config as cfg
import inspect
import re

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
    kwargs = dict(kwargs, **{'norm': norm, 'colorscale': colorscale, 'xlabel': xlabel,
                             'ylabel': ylabel})  # TODO: better way of adding all optional params to kwargs?

    xx, yy = xy_to_meshgrid(x, y)
    ax.pcolormesh(xx, yy, data, norm=norm)

    # kwarg options
    if 'colorscale' in kwargs.keys() and kwargs['colorscale'] is True:
        addcolorlegend(ax)
    _optional_plotting_args(ax, **kwargs)


def display_1d(x: np.array, data: np.array, ax: plt.Axes, xlabel: str = None, ylabel: str = None, **kwargs):
    """Displays 2D data with axis x, y
    Function should only draw on values from kwargs, option args are just there for type hints but should immediately
     be added to kwargs
    """
    kwargs = dict(kwargs,
                  **{'xlabel': xlabel, 'ylabel': ylabel})  # TODO: better way of adding all optional params to kwargs?

    ax.plot(x, data)

    # kwarg options
    _optional_plotting_args(ax, **kwargs)
    return ax


def _optional_plotting_args(ax, **kwargs):
    """Handles adding standard optional kwargs to ax"""
    if 'xlabel' in kwargs.keys() and kwargs['xlabel']:
        ax.set_xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs.keys() and kwargs['ylabel']:
        ax.set_ylabel(kwargs['ylabel'])
    if 'axtext' in kwargs.keys() and kwargs['axtext']:
        axtext = kwargs['axtext']
        ax.text(0.1, 0.8, f'{axtext}', fontsize=12, transform=ax.transAxes)




_fig_text_position = (0.5, 0.02)

def set_figtext(fig: plt.Figure, text: str):
    """Replaces current figtext with new text"""
    fig = plt.figure(fig.number)  # Just to set as current figure to add text to
    for i, text in enumerate(fig.texts):  # Remove any fig text that has been added previously
        if text.get_position() == _fig_text_position:
            text.remove()
    plt.figtext(_fig_text_position[0], _fig_text_position[1], text, horizontalalignment='center', wrap=True)
    plt.tight_layout(rect=[0, 0.07, 1, 1])


def add_standard_fig_info(fig: plt.Figure):
    """Add file info etc to figure"""
    stack = inspect.stack()
    for f in stack:
        filename = f.filename
        if re.search('/', filename):  # Seems to be that only the file the code is initially run from has forward slashes in the filename...
            break
    _, short_name = filename.split('PyDatAnalysis', 1)
    text = short_name
    add_to_fig_text(fig, text)
    return stack


def add_to_fig_text(fig: plt.Figure, text: str):
    """Adds text to figtext at the front"""
    existing_text = ''
    for t in fig.texts:  # Grab any fig_info_text that is already on figure
        if t.get_position() == _fig_text_position:
            existing_text = f' ,{t._text}'
            break
    text = text + existing_text
    set_figtext(fig, text)