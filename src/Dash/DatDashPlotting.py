"""
This is where all general dat plotting functions should live... To use in other pages, import the more general plotting
function from here, and make a little wrapper plotting function which calls with the relevant arguments
"""
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

    @abc.abstractmethod
    def plot(self, *args, **kwargs) -> go.Figure:
        """Override to make something which returns a competed plotly go.Figure"""
        pass

    @abc.abstractmethod
    def trace(self, *args, **kwargs) -> go.Trace:
        """Overridie to make something which returns the completed trace only"""
        pass

    def __init__(self, dat: Optional[DatHDF], dats: Optional[List[DatHDF]]):
        """Initialize with a dat or dats to provide some ability to get defaults"""
        if dat:
            self.dat = dat
        elif dats:
            self.dat = dats[0]
        else:
            self.dat = dat
            logger.warning(f'No Dat supplied, no values will be supplied by default')
        self.dats = dats

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
             mode: Optional[str] = None):
        xlabel = self._get_xlabel(xlabel)
        ylabel = self._get_ylabel(ylabel)

        fig = go.Figure(self.trace(data=data, x=x, mode=mode))
        fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)
        return fig

    def trace(self, data: np.ndarray, x: Optional[np.ndarray] = None,
              mode: Optional[str] = None):
        x = self._get_x(x)
        mode = self._get_mode(mode)
        trace = go.Scatter(x=x, y=data, mode=mode)
        return trace


class TwoD(DatPlotter):
    """
    For 2D plotting
    """

    def plot(self, data: np.ndarray, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
             xlabel: Optional[str] = None, ylabel: Optional[str] = None,
             ):
        xlabel = self._get_xlabel(xlabel)
        ylabel = self._get_ylabel(ylabel)

        fig = go.Figure(self.trace(data=data, x=x, y=y))
        fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)
        return fig

    def trace(self, data: np.ndarray, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
              ):
        x = self._get_x(x)
        y = self._get_y(y)
        trace = go.Heatmap(x=x, y=y, z=data)
        return trace


class ThreeD(DatPlotter):
    """
    For 3D plotting
    """
    pass