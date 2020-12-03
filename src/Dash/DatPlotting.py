"""
This is where all general dat plotting functions should live... To use in other pages, import the more general plotting
function from here, and make a little wrapper plotting function which calls with the relevant arguments
"""
from __future__ import annotations
from src.UsefulFunctions import bin_data_new, get_matching_x
from src.CoreUtil import get_nested_attr_default

import plotly.graph_objects as go
import numpy as np
import logging
import abc
from typing import Optional, Union, List, Tuple, Dict, Any
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.DatObject.DatHDF import DatHDF

logger = logging.getLogger(__name__)


class DatPlotter(abc.ABC):
    """Generally useful functions for all Dat Plotters"""

    MAX_POINTS = 1000  # Maximum number of points to plot in x or y
    RESAMPLE_METHOD = 'bin'  # Whether to resample down to 1000 points by binning or just down sampling (i.e every nth)

    @abc.abstractmethod
    def plot(self, trace_kwargs: Optional[dict], fig_kwargs: Optional[dict]) -> go.Figure:
        """Override to make something which returns a competed plotly go.Figure

        Note: Should also include: trace_kwargs: Optional[dict] = None, fig_kwargs: Optional[dict] = None
        """
        pass

    @abc.abstractmethod
    def trace(self, trace_kwargs: Optional[dict]) -> go.Trace:
        """Override to make something which returns the completed trace only

        Note: Should also include: trace_kwargs: Optional[dict] = None
        """
        pass

    def __init__(self, dat: Optional[DatHDF] = None, dats: Optional[List[DatHDF]] = None):
        """Initialize with a dat or dats to provide some ability to get defaults"""
        if dat:
            self.dat = dat
        elif dats:
            self.dat = dats[0]
        else:
            self.dat = dat
            logger.warning(f'No Dat supplied, no values will be supplied by default')
        self.dats = dats

    def save_to_dat(self, fig, name: Optional[str] = None, sub_group_name: Optional[str] = None, overwrite: bool = False):
        """Saves to the Figures attribute of the dat"""
        self.dat.Figures.save_fig(fig, name=name, sub_group_name=sub_group_name, overwrite=overwrite)

    def _resample_data(self, data : np.ndarray,
                       x: Optional[np.ndarray] = None,
                       y: Optional[np.ndarray] = None,
                       z: Optional[np.ndarray] = None):
        """
        Resamples given data using self.MAX_POINTS and self.RESAMPLE_METHOD.
        Will always return data, then optionally ,x, y, z incrementally (i.e. can do only x or only x, y but cannot do
        e.g. x, z)
        Args:
            data (): Data to resample down to < self.MAX_POINTS in each dimension
            x (): Optional x array to resample the same amount as data
            y (): Optional y ...
            z (): Optional z ...

        Returns:
            (Any): Matching combination of what was passed in (e.g. data, x, y ... or data only, or data, x, y, z)
        """
        def chunk_size(orig, desired):
            """chunk_size can be for binning or downsampling"""
            s = round(orig/desired)
            if orig > desired and s == 1:
                s = 2  # At least make sure it is sampled back below desired
            elif s == 0:
                s = 1  # Make sure don't set zero size
            return s

        ndim = data.ndim
        data = np.array(data, ndmin=3)
        shape = data.shape
        if any([s > self.MAX_POINTS for s in shape]):
            chunk_sizes = [chunk_size(s, self.MAX_POINTS) for s in reversed(shape)]  # (shape is z, y, x otherwise)
            if self.RESAMPLE_METHOD == 'bin':
                data = bin_data_new(data, *chunk_sizes)
                x, y, z = [bin_data_new(arr, cs) if arr is not None else arr for arr, cs in zip([x, y, z], chunk_sizes)]
            elif self.RESAMPLE_METHOD == 'downsample':
                data = data[::chunk_sizes[-1], ::chunk_sizes[-2], ::chunk_sizes[-3]]
                x, y, z = [arr[::cs] if arr is not None else None for arr, cs in zip([x, y, z], chunk_sizes)]
            else:
                raise ValueError(f'{self.RESAMPLE_METHOD} is not a valid option')

        if ndim == 1:
            data = data[0, 0]
            if x is not None:
                return data, x
            return data

        elif ndim == 2:
            data = data[0]
            if x is not None:
                if y is not None:
                    return data, x, y
                return data, x
            return data

        elif ndim == 3:
            if x is not None:
                if y is not None:
                    if z is not None:
                        return data, x, y, z
                    return data, x, y
                return data, x
            return data
        raise ValueError(f'Most likely something wrong with {data}')

    # Get values from dat if value passed in is None
    # General version first, then specific ones which will be used more frequently
    def _get_any(self, any_name: str, any_value: Optional[Any] = None):
        """Can use this to get any value from dat by passing a '.' separated string path to the attr
        Note: will default to None if not found instead of raising error
        'any_value' will be returned if it is not None.
        """
        if any_value is None and self.dat:
            return get_nested_attr_default(self.dat, any_name, None)
        return any_value

    def _get_x(self, x):
        if x is None and self.dat:
            return self.dat.Data.x
        return x

    def _get_y(self, y):
        if y is None and self.dat:
            return self.dat.Data.y
        return y

    def _get_xlabel(self, xlabel):
        if xlabel is None and self.dat:
            return self.dat.Logs.xlabel
        return xlabel

    def _get_ylabel(self, ylabel):
        if ylabel is None and self.dat:
            return self.dat.Logs.ylabel
        return ylabel


class OneD(DatPlotter):
    """
    For 1D plotting
    """

    def _get_mode(self, mode):
        if mode is None:
            mode = 'markers'
        return mode

    def _get_ylabel(self, ylabel):
        if ylabel is None:
            ylabel = 'Arbitrary'
        return ylabel

    def plot(self, data: np.ndarray, x: Optional[np.ndarray] = None,
             xlabel: Optional[str] = None, ylabel: Optional[str] = None,
             trace_name: Optional[str] = None,
             title: Optional[str] = None,
             mode: Optional[str] = None,
             trace_kwargs: Optional[dict] = None, fig_kwargs: Optional[dict] = None):
        if fig_kwargs is None:
            fig_kwargs = {}

        xlabel = self._get_xlabel(xlabel)
        ylabel = self._get_ylabel(ylabel)

        fig = go.Figure(self.trace(data=data, x=x, mode=mode, name=trace_name, trace_kwargs=trace_kwargs), **fig_kwargs)
        fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel, title=title)
        self._plot_autosave(fig, name=title)
        return fig

    def _plot_autosave(self, fig: go.Figure, name: Optional[str] = None):
        self.save_to_dat(fig, name=name)

    def trace(self, data: np.ndarray, x: Optional[np.ndarray] = None,
              mode: Optional[str] = None,
              name: Optional[str] = None,
              trace_kwargs: Optional[dict] = None):
        if trace_kwargs is None:
            trace_kwargs = {}
        x = self._get_x(x)
        mode = self._get_mode(mode)

        data, x = self._resample_data(data, x)  # Makes sure not plotting more than self.MAX_POINTS in any dim

        trace = go.Scatter(x=x, y=data, mode=mode, name=name,**trace_kwargs)
        return trace


class TwoD(DatPlotter):
    """
    For 2D plotting
    """

    def plot(self, data: np.ndarray, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
             xlabel: Optional[str] = None, ylabel: Optional[str] = None,
             title: Optional[str] = None,
             plot_type: Optional[str] = None,
             trace_kwargs: Optional[dict] = None, fig_kwargs: Optional[dict] = None):
        if fig_kwargs is None:
            fig_kwargs = {}
        if plot_type is None:
            plot_type = 'heatmap'
        xlabel = self._get_xlabel(xlabel)
        ylabel = self._get_ylabel(ylabel)

        fig = go.Figure(self.trace(data=data, x=x, y=y, trace_type=plot_type, trace_kwargs=trace_kwargs), **fig_kwargs)
        fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel, title=title)
        self._plot_autosave(fig, name=title)
        return fig

    def _plot_autosave(self, fig: go.Figure, name: Optional[str] = None):
        self.save_to_dat(fig, name=name)

    def trace(self, data: np.ndarray, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
              trace_type: Optional[str] = None,
              trace_kwargs: Optional[dict] = None):
        if trace_type is None:
            trace_type = 'heatmap'
        if trace_kwargs is None:
            trace_kwargs = {}
        x = self._get_x(x)
        y = self._get_y(y)

        data, x = self._resample_data(data, x)  # Makes sure not plotting more than self.MAX_POINTS in any dim

        if trace_type == 'heatmap':
            trace = go.Heatmap(x=x, y=y, z=data, **trace_kwargs)
        elif trace_type == 'waterfall':
            trace = [go.Scatter3d(mode='lines', x=x, y=[yval]*len(x), z=row, name=f'{yval:.3g}', **trace_kwargs) for row, yval in zip(data, y)]
        else:
            raise ValueError(f'{trace_type} is not a recognized trace type for TwoD.trace')
        return trace


class ThreeD(DatPlotter):
    """
    For 3D plotting
    """

    def plot(self, trace_kwargs: Optional[dict] = None, fig_kwargs: Optional[dict] = None) -> go.Figure:
        pass

    def trace(self, trace_kwargs: Optional[dict] = None) -> go.Trace:
        # data, x = self._resample_data(data, x)  # Makes sure not plotting more than self.MAX_POINTS in any dim
        pass
