"""
This is where all general dat plotting functions should live... To use in other pages, import the more general plotting
function from here, and make a little wrapper plotting function which calls with the relevant arguments
"""
from src.UsefulFunctions import bin_data_new, get_matching_x

import plotly.graph_objects as go
import numpy as np
import logging
import abc
from typing import Optional, Union, List, Tuple, Dict
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

    def _resample_data(self, data, x=None, y=None, z=None):
        def chunk_size(orig, desired):
            """chunk_size can be for binning or downsampling"""
            s = round(orig/desired)
            if orig > desired and s == 1:
                s = 2  # At least make sure it is sampled back below desired
            elif s == 0:
                s = 1  # Make sure don't set zero size
            return s

        def resample(arr, chunk_size, method):
            if method == 'bin':
                return bin_data_new(arr, chunk_size)
            elif method == 'downsample':
                return arr[round(chunk_size/2)::chunk_size]
            else:
                raise ValueError(f'{method} is not a valid option')

        if data.ndim == 1:
            shape = data.shape[-1]
            if shape > self.MAX_POINTS:
                cs = chunk_size(shape, self.MAX_POINTS)
                data = resample(data, cs, self.RESAMPLE_METHOD)
                if x is not None:
                    x = resample(x, cs, self.RESAMPLE_METHOD)
                    return data, x
                return data
            else:
                if x is not None:
                    return data, x
                return data

        elif data.ndim == 2:
            shape = data.shape
            if any([v > self.MAX_POINTS for v in shape]):
                css = [chunk_size(s, self.MAX_POINTS) for s in reversed(shape)]
                data = resample(data, )

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
             mode: Optional[str] = None,
             trace_kwargs: Optional[dict] = None, fig_kwargs: Optional[dict] = None):

        xlabel = self._get_xlabel(xlabel)
        ylabel = self._get_ylabel(ylabel)

        fig = go.Figure(self.trace(data=data, x=x, mode=mode, **trace_kwargs), **fig_kwargs)
        fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)
        return fig

    def trace(self, data: np.ndarray, x: Optional[np.ndarray] = None,
              mode: Optional[str] = None,
              trace_kwargs: Optional[dict] = None):
        x = self._get_x(x)
        mode = self._get_mode(mode)

        trace = go.Scatter(x=x, y=data, mode=mode, **trace_kwargs)
        return trace


class TwoD(DatPlotter):
    """
    For 2D plotting
    """

    def plot(self, data: np.ndarray, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
             xlabel: Optional[str] = None, ylabel: Optional[str] = None,
             trace_kwargs: Optional[dict] = None, fig_kwargs: Optional[dict] = None):
        xlabel = self._get_xlabel(xlabel)
        ylabel = self._get_ylabel(ylabel)

        fig = go.Figure(self.trace(data=data, x=x, y=y, trace_kwargs=trace_kwargs), **fig_kwargs)
        fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)
        return fig

    def trace(self, data: np.ndarray, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
              trace_kwargs: Optional[dict] = None):
        x = self._get_x(x)
        y = self._get_y(y)
        trace = go.Heatmap(x=x, y=y, z=data, **trace_kwargs)
        return trace


class ThreeD(DatPlotter):
    """
    For 3D plotting
    """

    def plot(self, trace_kwargs: Optional[dict] = None, fig_kwargs: Optional[dict] = None) -> go.Figure:
        pass

    def trace(self, trace_kwargs: Optional[dict] = None) -> go.Trace:
        pass
