from src.Scripts.StandardImports import *
from src.Scripts.Jun30_dot_tuning import _plot_dat_array
from src.Plotting.Plotly.PlotlyUtil import PlotlyViewer
import plotly.graph_objects as go
from scipy.interpolate import interp2d, RectBivariateSpline

from dataclasses import dataclass

@dataclass
class DotTuningData:
    dats: List[DatHDF]
    full_x: np.ndarray = None
    full_y: np.ndarray = None
    datas: np.ndarray = None
    diff_datas: np.ndarray = None


def interpolate_2d(x, y, z, xnew, ynew, **kwargs):
    """
    Interpolates 2D data returning NaNs where NaNs in original data.
    Taken from https://stackoverflow.com/questions/51474792/2d-interpolation-with-nan-values-in-python
    Scipy.interp2d by itself doesn't handle NaNs (and isn't documented!!)
    Args:
        x (np.ndarray):
        y (np.ndarray):
        z (np.ndarray):
        xnew (np.ndarray):
        ynew (np.ndarray):
        **kwargs (dict):

    Returns:
        np.ndarray: interpolated data
    """
    nan_map = np.zeros_like(z)
    nan_map[np.isnan(z)] = 1

    filled_z = z.copy()
    filled_z[np.isnan(z)] = 0

    f = interp2d(x, y, filled_z, **kwargs)
    f_nan = interp2d(x, y, nan_map, **kwargs)

    z_new = f(xnew, ynew)
    nan_new = f_nan(xnew, ynew)
    z_new[nan_new > 0.1] = np.nan
    return z_new


def remove_first_last_non_nan(data):
    data = np.asanyarray(data)
    assert data.ndim in [1, 2]
    if data.ndim == 2:
        row = data[0]
    else:
        row = data
    first_index = row.shape[-1] - np.argmax(np.cumsum(~np.isnan(np.flip(row))))-1
    last_index = np.argmax(np.cumsum(~np.isnan(row)))
    if data.ndim == 2:
        data[:, first_index] = np.nan
        data[:, last_index] = np.nan
        return data
    else:
        data[first_index] = np.nan
        data[last_index] = np.nan
        return data

def _get_dot_tuning_data(data):
    """
    Interpolates data onto the same x, y grid for easier plotting, also calculates differentiated data

    Args:
        data (Union(List[DatHDF], DotTuningData)): Either a list of dats, or a DotTuningData instance

    Returns:
        DotTuningData: DTD object which includes calculated info
    """
    if not isinstance(data, DotTuningData):
        dtd = DotTuningData(data)
    else:
        dtd = data

    dats = dtd.dats
    full_x = np.linspace(np.nanmin([np.nanmin(dat.Data.x_array) for dat in dats]),
                         np.nanmax([np.nanmax(dat.Data.x_array) for dat in dats]), 1000)
    full_y = np.linspace(np.nanmin([np.nanmin(dat.Data.y_array) for dat in dats]),
                         np.nanmax([np.nanmax(dat.Data.y_array) for dat in dats]),
                         int(np.nanmax([dat.Data.y_array.shape[-1] for dat in dats])))
    datas = dict()
    for dat in dats:
        isense = dat.Data.Exp_cscurrent_2d
        isense = CU.decimate(isense, dat.Logs.Fastdac.measure_freq, 30, return_freq=False)
        x = np.linspace(dat.Data.x_array[0], dat.Data.x_array[-1], isense.shape[-1]).astype(np.float32)
        datas[dat.datnum] = interpolate_2d(x, dat.Data.y_array, isense, full_x, full_y)

    diff_data = dict()
    for k in datas:
        # diff_data[k] = remove_first_last_non_nan(np.diff(datas[k], prepend=np.NaN, append=np.NaN))
        diff_data[k] = remove_first_last_non_nan(np.diff(datas[k], append=np.NaN, axis=1))

    dtd.datas = list(datas.values())
    dtd.diff_datas = list(diff_data.values())
    dtd.full_x = full_x
    dtd.full_y = full_y
    return dtd


def _plot_dot_tuning(dtd, differentiated = True):
    """
    Plots DotTuningData
    Args:
        dtd (DotTuningData): DotTuningData calculated with _get_dot_tuning_data()

    Returns:
        go.Figure: plotly Figure instance
    """

    if differentiated:
        data = dtd.diff_datas
    else:
        data = dtd.datas

    fig = go.Figure()
    for dat, d in zip(dtd.dats, data):
        fig.add_trace(
            go.Heatmap(
                visible=False,
                x=dtd.full_x,
                y=dtd.full_y,
                z=d,
            ))
        fig.data[0].visible = True

    steps = []
    for i, dat in enumerate(dtd.dats):
        fds = dat.Logs.fds
        if 'RP/0.16' in fds.keys():
            rp_key = 'RP/0.16'
        elif 'RP*2' in fds.keys():
            rp_key = 'RP*2'
        else:
            raise KeyError("Couldn't find RP key... Come add another option here!")
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": f'Dat{dat.datnum}: RCSS={fds["RCSS"]:.1f}mV, {rp_key}={fds[rp_key]:.1f}mV'}],  # layout attribute
            label=f'{dat.datnum}'
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Dat: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(sliders=sliders)
    fig.update_layout(xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text='RCT /mV')), yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text='RCB /mV')))
    return fig


if __name__ == '__main__':
    # dats = get_dats(range(49, 68+1))
    # dats = get_dats(range(75, 80+1))
    dats = get_dats(range(81, 88))
    dtd = _get_dot_tuning_data(dats)
    fig = _plot_dot_tuning(dtd, differentiated=True)
    PlotlyViewer(fig)

    # _plot_dat_array(dats, rows=5, cols=4, fixed_scale=False)
