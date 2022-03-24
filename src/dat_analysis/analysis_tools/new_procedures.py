from __future__ import annotations
import abc
from typing import Any, List
from functools import lru_cache
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType
import h5py
import numpy as np

import dash_bootstrap_components as dbc
from dash import html, dcc


from dat_analysis.plotting.plotly import OneD, TwoD
from dat_analysis.useful_functions import get_matching_x
from dat_analysis.hdf_util import set_attr, get_attr, DatDataclassTemplate


@dataclass
class PlottableData:
    """
    Data in a plottable format (i.e. a rectangular grid of datapoints). Intended to be used by DataPlotter which will
    take PlottableData and give labels etc

    A subclass could be made to handle non-rectangular data
    """
    data: np.ndarray
    x: np.ndarray = None
    y: np.ndarray = None
    z: np.ndarray = None
    axes: np.ndarray = None

    def __post_init__(self):
        ndim = self.data.ndim
        # Set default axes (just numbered if not provided)
        if self.x is None and ndim >= 1:
            self.x = np.arange(self.data.shape[-1])
        if self.y is None and ndim >= 2:
            self.y = np.arange(self.data.shape[-2])
        if self.z is None and ndim >= 3:
            self.z = np.arange(self.data.shape[-3])
        if self.axes is None and ndim >= 4:
            self.axes = np.array([np.arange(s) for s in self.data.shape])

        # Make sure axes have correct dimensions for plotting data
        for ax, dim in zip([self.x, self.y, self.z], [-1, -2, -3]):
            if ax is not None:
                assert ax.shape[-1] == self.data.shape[dim]
        assert np.all([a.shape for a in self.axes] == self.data.shape)


@dataclass
class DataPlotter:
    """
    Collection of functions which take PlottableData and plot it various ways
    e.g. 1D, 2D, heatmap, waterfall, single row of 2d
    """
    data: PlottableData
    xlabel: str = ''
    ylabel: str = ''
    data_label: str = ''
    title: str = ''

    xspacing: float = 0
    yspacing: float = 0

    def plot_1d(self, s_: np.s_ = None) -> go.Figure:
        p = OneD(dat=None)
        fig = p.figure(self.xlabel, self.data_label, self.title)
        fig.add_trace(self.trace_1d(s_=s_))
        return fig

    def trace_1d(self, s_: np.s_ = None, axis: np.ndarray = None) -> go.Scatter:
        p = OneD(dat=None)
        data = self.data.data
        if axis is None:
            axis = self.data.x
        if slice is not None:
            data = data[s_]

        axis = get_matching_x(axis, data)
        if data.ndim > 1:
            raise ValueError(f'Data has shape {data.shape} after slicing with {s_}, cannot be plot 1D')
        return p.trace(x=axis, data=data)

    def plot_heatmap(self, s_: np.s_ = None) -> go.Figure:
        p = TwoD(dat=None)
        fig = p.figure(self.xlabel, self.ylabel, self.title)
        fig.add_trace(self.trace_heatmap(s_=s_))
        return fig
        pass

    def trace_heatmap(self, s_: np.s_ = None, axis_x: np.ndarray = None, axis_y: np.ndarray = None) -> go.Heatmap:
        p = TwoD(dat=None)
        data = self.data.data
        if axis_x is None:
            axis_x = self.data.x
        if axis_y is None:
            axis_y = self.data.y

        if slice is not None:
            data = data[s_]

        axis_x = get_matching_x(axis_x, shape_to_match=data.shape[-1])
        axis_y = get_matching_x(axis_y, shape_to_match=data.shape[-2])
        if data.ndim > 1:
            raise ValueError(f'Data has shape {data.shape} after slicing with {s_}, cannot be plot 1D')
        return p.trace(x=axis_x, y=axis_y, data=data, trace_type='heatmap')

    def plot_waterfall(self) -> go.Figure:
        raise NotImplementedError

    def plot_row(self) -> go.Figure:
        raise NotImplementedError


class Process(abc.ABC):
    """
    Standardize what should be written for any analysis process, can recursively call sub processes as well.

    Intention is that any step of analysis has some ways to view the inputs and outputs of that step of analysis

    Should be some mixture of human friendly input and data passed in.

    Also provide some options for saving progress to HDF file in given group location.

    E.g. splitting data into square wave parts

    mixture of human friendly input, as well as data passed in, no loading from file
    """
    default_name = 'SubProcess'  # Name of this Process (e.g. group name in HDF)

    def __init__(self):
        self.child_processes: List[Process] = []  # if calling other processes, append to this list (e.g. in input_data)

    @abc.abstractmethod
    def input_data(self, data) -> Any:
        """
        Pass minimal data in to get job done and call any sub processes here adding them to children
        e.g.
        self.inputs = dict(...)
        sub_process = OtherProcess(...)
        self.child_processes.append(sub_process)
        part_processed = sub_process.output_data(...)

        self.data_to_process = PlottableData(..., part_processed.x, ...)

        return self.data_to_process

        Returns:

        """
        pass

    @abc.abstractmethod
    def output_data(self) -> PlottableData:  # Note: can make more different outputs as well
        """
        Do something with self.data_to_process
        Anything intensive should be put in a cached self._process or similar

        Returns:

        """
        raise NotImplementedError

    @lru_cache
    def _process(self):
        """Do the actual processing in cached function"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_input_plotter(self) -> DataPlotter:
        """
        Initialize the DataPlotter with reasonable title and labels etc
        i.e.
        return DataPlotter(self.data_to_process(...), self.inputs, ...)
        Returns:

        """
        pass

    @abc.abstractmethod
    def get_output_plotter(self) -> DataPlotter:
        """
        Same as self.input_plotter, but for the output data
        i.e.
        return DataPlotter(self.output_data(...), ...)
        Returns:

        """
        pass

    @abc.abstractmethod
    def save_progress(self, group: h5py.Group, **kwargs):
        """
        Save necessary info to given hdf group/file with minimal additional input (in a single group)
        """
        self._save_progress_default(group)
        # Any data to save for this Process specifically (i.e. inputs/outputs)

    @classmethod
    @abc.abstractmethod
    def load_progress(cls, group: h5py.Group) -> Process:
        """
        Given the location of a process h5py group, load data back into a Process object

        Args:
            group: location of Process to load

        Returns:

        """
        inst = cls()
        inst.child_processes = cls._load_progress_default(group)
        # Then load any more input/output data
        return inst

    def _save_progress_default(self, group: h5py.Group):
        """
        Generally it will be necessary to also save all sub_processes
        Args:
            group ():

        Returns:

        """
        sub_process_locations = []
        for child in self.child_processes:
            child_group = group.require_group(f'{group.name}/{child.default_name}')
            child.save_progress(child_group)
            sub_process_locations.append(child_group.name)

        # Save location of any Sub processes required for loading
        self._write_to_group(group, 'subprocess_locations', sub_process_locations)

    @staticmethod
    def _write_to_group(group: h5py.Group, name: str, data, dat_dataclass: DatDataclassTemplate = None):
        """
        Save some part of data to file

        Returns:

        """
        set_attr(group=group, name=name, value=data, dataclass=dat_dataclass)
        return True

    @staticmethod
    def _read_from_group(group: h5py.Group, name: str,
                         default=None, check_exists=True,
                         dat_dataclass: DatDataclassTemplate = None) -> Any:
        """
        Load info back from h5py Group

        Args:
            group ():
            name ():

        Returns:

        """
        value = get_attr(group=group, name=name, default=default,
                         check_exists=check_exists, dataclass=dat_dataclass)
        return value

    @classmethod
    def _load_progress_default(cls, group: h5py.Group) -> List[Process]:
        """Equivalent of self._save_progress_default for loading that data"""
        subprocess_locations = cls._read_from_group(group, 'subprocess_locations')
        sub_processes = []
        if subprocess_locations:
            for location in subprocess_locations:
                sub_process = Process.load_progress(location)
                sub_processes.append(sub_process)
        return sub_processes


class ProcessViewer(abc.ABC):
    """
    Take a process and combine input/output views into a nice overall summary... Something which can be easily plotted
    in a dash page. or output to a plotly multigraph fig or mpl, or saved to pdf
    """
    def __init__(self, process: Process, **options):
        self.process = process

    @abc.abstractmethod
    def dash_full(self, **more_options):  # -> Component
        """
        Combine multiple figures/text/etc to make an output component
        (can assume 3/4 page wide, any height)
        Args:
            **more_options ():

        Returns:

        """
        input_plotter = self.process.get_input_plotter()
        output_plotter = self.process.get_output_plotter()

        input_fig = ProcessComponentOutputGraph(fig=input_plotter.plot_1d())
        output_fig = ProcessComponentOutputGraph(fig=output_plotter.plot_1d())

        return html.Div([
            input_fig.layout(),
            output_fig.layout()
        ])

        pass

    @abc.abstractmethod
    def mpl_full(self, **more_options):  # -> plt.Figure
        raise NotImplementedError

    @abc.abstractmethod
    def pdf_full(self, **more_options):  # -> pdf
        raise NotImplementedError


class ProcessInterface(abc.ABC):
    """
    Things necessary to put a process into a dash page or report with easy human friendly input. I.e. for building a
    new dash page

    human friendly input, and id of file to load from (or data passed in)
    """
    def __init__(self):
        self.store_id = ''
        self.sub_store_ids = []

    @abc.abstractmethod
    def required_input_components(self) -> List[ProcessComponentInput]:
        """
        Give the list of components that need to be placed in order for Process to be carried out
        Should all update stores or possibly even the dat file
        Returns:

        """
        return []

    @abc.abstractmethod
    def all_outputs(self) -> List[ProcessComponentOutput]:
        """Lit of components that display the process in detail"""
        return []

    @abc.abstractmethod
    def main_outputs(self) -> List[ProcessComponentOutput]:
        """List of components that display the main features of the process"""
        return []


class ProcessComponent(abc.ABC):
    """
    A dash component with callbacks etc to interface with user
    """
    @abc.abstractmethod
    def run_callbacks(self, **kwargs):
        pass

    @abc.abstractmethod
    def layout(self):
        pass


class ProcessComponentInput(ProcessComponent, abc.ABC):
    """
    Component mostly for user input
    """
    def __init__(self, title, num_buttons, etc):
        super().__init__()


class ProcessComponentOutput(ProcessComponent, abc.ABC):
    """
    Component mostly for output to graph/table/file etc
    """


class ProcessComponentOutputGraph(ProcessComponentOutput):
    """
    Component for displaying a figure (with additional options)
    """
    def __init__(self, fig: go.Figure):
        super().__init__()
        self.fig = fig

    def layout(self):
        return dcc.Graph(figure=self.fig)

    def run_callbacks(self, **kwargs):
        pass


#####################################################################################################




if __name__ == '__main__':
    import dash
    from dash import html
    import dash_bootstrap_components as dbc

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = html.Div([
        dbc.NavbarSimple('Testing'),
        dbc.Container([
            dbc.Row([
                # Sidebar
                dbc.Col([
                    html.H2(f'Sidebar'),

                ],
                    width=3,
                    class_name='border',
                ),
                # Main area
                dbc.Col([
                    html.H2(f'Display Area'),

                ],
                    width=9,
                    class_name='border',
                )
            ])
        ],
            fluid=True
        )

    ])

    app.run_server(debug=True, threaded=True, port=8050)
