import plotly.graph_objects as go
import plotly
import dash
import dash_core_components as dcc
import dash_html_components as html
import src.CoreUtil as CU
import numpy as np
from src.HDF_Util import with_hdf_write, with_hdf_read
import src.HDF_Util as HDU

from typing import TYPE_CHECKING, Union, List, Dict, Optional

if TYPE_CHECKING:
    from src.DatObject.DatHDF import DatHDF
    from src.DatObject.Attributes.DatAttribute import FittingAttribute, DatAttribute


# Is this a good idea? Should I just make a class which takes fig as an argument instead?
class MyFigure(go.Figure):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Plot:
    def __init__(self, dat: Optional[DatHDF] = None):
        """
        Option to initialize a plotting class with some information which can be drawn from if data isn't
        passed into each plotting function directly
        Args:
            dat (DatHDF): Optional instance of a dat which should be used to draw extra information from
        """
        self.dat = dat

    # Would it be good to make some sort of class that mostly handles this stuff
    # A lot of it is going to look the same for every plot... But then maybe there aren't too many plots
    # to worry about...
    def plot_1d(self, data: np.ndarray,
                x: Optional[np.ndarray] = None,
                title: str = None,
                x_label: str = None,
                y_label: str = None,
                label: str = None,
                fig: go.Figure = None,
                mode: str = None,
                plot_kwargs=None,
                ) -> go.Figure:
        if fig is None:
            fig = go.Figure()
        if mode is None:
            mode = 'markers+lines'
        if self.dat:
            if title is None:
                title = f'Dat{self.dat.datnum}'
            if x_label is None:
                x_label = self.dat.Logs.x_label
            if y_label is None:
                y_label = self.dat.Logs.y_label
            if label is None:
                label = f'Dat{self.dat.datnum}'
        if plot_kwargs is None:
            plot_kwargs = dict()

        trace = go.Scatter(mode=mode, x=x, y=data, name=label, **plot_kwargs)
        fig.add_trace(trace)
        fig.update_layout(title=title, xaxis_label=x_label, yaxis_label=y_label)
        return fig


class Figures(DatAttribute):
    version = '1.0.0'
    group_name = 'Figures'
    description = 'A place to store all figures which are plotted, partly so that it is easy to see a history of what' \
                  'has been plotted, but also so that figures can be shared between kernels and with Dash apps (i.e. ' \
                  'they can save data to the HDF here, and it can be read anywhere else)'

    def __init__(self, dat: DatHDF):
        super().__init__(dat)
        self.Plot = Plot(self.dat)  # For easy access to plotting which will automatically fill blanks from dat

    @property
    def existing_fig_names(self):
        """Something which returns all the figure names (and group name?) of saved figures"""
        raise NotImplementedError

    def initialize_minimum(self):
        self.initialized = True

    def save_fig(self, fig, name: Optional[str] = None, group_name: Optional[str] = None):
        if not name:
            name = self._get_fig_name(fig)
        self._save_fig(fig, name, group_name)

    @with_hdf_write
    def _save_fig(self, fig: go.Figure, name: str, group_name: Optional[str] = None):
        if group_name:
            group = self.hdf.group.require_group(group_name)
        else:
            group = self.hdf.group
        fig_group = group.require_group(name)
        HDU.save_dict_to_hdf_group(fig_group, fig.to_dict())

    def get_fig(self, name: str, group_name: Optional[str] = None):
        self._get_fig(name, group_name)

    @with_hdf_read
    def _get_fig(self, name: str, group_name: Optional[str] = None) -> go.Figure:
        if group_name:
            group = self.hdf.group.get(group_name)
        else:
            group = self.hdf.group
        fig_group = group.get(name)
        fig = go.Figure(HDU.load_dict_from_hdf_group(fig_group))
        return fig


