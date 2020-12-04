from __future__ import annotations
import plotly.graph_objs as go
import plotly.express as px
from src import CoreUtil as CU
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.DatObject.DatHDF import DatHDF

allowed_datnums = range(4000, 10000)
max_points_per_row = 1000


def plot_avg(dat: DatHDF, type='transition'):
    y_label = "Current /nA"
    if type == 'transition':
        x = dat.Transition.x
        y = dat.Transition.avg_data
    elif type == 'entropy':
        if hasattr(dat, 'SquareEntropy'):
            x = dat.SquareEntropy.Processed.outputs.x
            y = dat.SquareEntropy.Processed.outputs.entropy_signal
        else:
            x = dat.Entropy.x
            z = dat.Entropy.avg_data
    else:
        raise ValueError(f'{type} is not valid, should be "transition" or "entropy"')

    x_label = dat.Logs.x_label
    x, y = CU.bin_data([x, y], bin_size=np.ceil(y.shape[-1] / max_points_per_row))

    fig = go.Figure()
    fig.add_trace(go.Scatter(mode='markers', x=x, y=y))
    fig.update_layout(xaxis_title=x_label,
                      yaxis_title=y_label,
                      title=f'Dat{dat.datnum}')
    return fig


def avg_ct(dat: DatHDF):
    return plot_avg(dat, type='transition')


def avg_entropy(dat: DatHDF):
    return plot_avg(dat, type='entropy')


def plot_2d(dat: DatHDF, type='transition'):
    z_label = "Current /nA"
    if type == 'transition':
        x = dat.Transition.x
        z = dat.Transition.data
    elif type == 'entropy':
        if hasattr(dat, 'SquareEntropy'):
            x = dat.SquareEntropy.Processed.outputs.x
            z = dat.SquareEntropy.entropy_data
        else:
            x = dat.Entropy.x
            z = dat.Entropy.data
    else:
        raise ValueError(f'{type} is not valid, should be "transition" or "entropy"')

    y = list(range(z.shape[0]))
    x_label = dat.Logs.x_label
    y_label = dat.Logs.y_label

    x, z = CU.bin_data([x, z], bin_size=np.ceil(z.shape[-1] / max_points_per_row))
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=x, y=y, z=z))
    fig.update_layout(xaxis_title=x_label,
                      yaxis_title=y_label,
                      coloraxis=dict(colorbar=dict(title=z_label)),
                      title=f'Dat{dat.datnum}')
    return fig


def all_ct(dat: DatHDF):
    return plot_2d(dat, type='transition')


def all_entropy(dat: DatHDF):
    return plot_2d(dat, type='entropy')


def plot_fit_values(dat: DatHDF, type='transition'):
    if type == 'transition':
        from src.DatObject.Attributes import Transition as T
        x = dat.Transition.x
        z = dat.Transition.data
        title_prefix = 'Transition'
    elif type == 'entropy':
        from src.DatObject.Attributes import Entropy as E
        if hasattr(dat, 'SquareEntropy'):
            title_prefix = 'Square Entropy'
            x = dat.SquareEntropy.Processed.outputs.x
            z = dat.SquareEntropy.entropy_data
        else:
            title_prefix = 'Entropy'
            x = dat.Entropy.x
            z = dat.Entropy.data
    else:
        raise ValueError(f'{type} is not valid, should be "transition" or "entropy"')

    y = np.array(range(z.shape[0]))

    # TODO: Should not be doing any fitting here!! Should be working off of already calculated fits
    x, z = CU.bin_data([x, z], bin_size=np.ceil(z.shape[-1] / max_points_per_row))
    if type == 'transition':
        pars = T.get_param_estimates(x, z)
        fits = T.transition_fits(x, z, func=T.i_sense, params=pars, auto_bin=True)
    elif type == 'entropy':
        pars = E.get_param_estimates(x, z)
        fits = E.entropy_fits(x, z, params=pars, auto_bin=True)
    else:
        raise NotImplementedError

    fig = go.Figure()
    for k in fits[0].best_values.keys():
        fvals = [f.best_values.get(k, None) for f in fits]
        fig.add_trace(go.Scatter(mode='markers', x=fvals, y=y, name=f'{k} values'))

    fig.update_layout(xaxis_title='Fit values', yaxis_title='Y array', title=f'{title_prefix} fit values for Dat{dat.datnum}')
    return fig


def transition_values(dat: DatHDF):
    return plot_fit_values(dat, type='transition')


def entropy_values(dat: DatHDF):
    return plot_fit_values(dat, type='entropy')

