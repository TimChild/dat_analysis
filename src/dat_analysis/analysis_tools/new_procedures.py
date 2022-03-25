from __future__ import annotations
import abc
from typing import Any, List, Union, Optional, Dict, TYPE_CHECKING
from functools import lru_cache
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType
import h5py
import numpy as np

import dash_bootstrap_components as dbc
from dash import html, dcc

from dat_analysis.plotting.plotly import OneD, TwoD
from dat_analysis.useful_functions import get_matching_x, get_data_index
from dat_analysis.hdf_util import set_attr, get_attr, DatDataclassTemplate
from dat_analysis.dat_object.attributes.square_entropy import get_transition_part

if TYPE_CHECKING:
    from dash.development.base_component import Component


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
        if self.axes is None:
            self.axes = np.array([np.arange(s) for s in self.data.shape])

        # Make sure axes have correct dimensions for plotting data
        for ax, dim in zip([self.x, self.y, self.z], [-1, -2, -3]):
            if ax is not None:
                if ax.shape[-1] != self.data.shape[dim]:
                    print(ax.shape, self.data.shape, dim)
        # assert np.all([a.shape for a in self.axes] == self.data.shape)


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
        if s_ is not None:
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

    mixture of human friendly input, as well as data passed in, no loading input data from file (only save or restore
    whole process to or from and open HDF Group)
    """
    default_name = 'SubProcess'  # Name of this Process (e.g. group name in HDF)

    def __init__(self):
        self.child_processes: List[Process] = []  # if calling other processes, append to this list (e.g. in input_data)
        self._data_input: Dict[str, Union[np.ndarray, Any]] = {}  # Store data as provided
        self._data_preprocessed: Dict[
            str, Union[np.ndarray, Any]] = {}  # Store data from sub processes or basic pre-processing
        self._data_output: Dict[str, Union[np.ndarray, Any]] = {}  # Store data produced

    @abc.abstractmethod
    def input_data(self, *args, **kwargs):
        """
        Pass minimal data in to get job done

        self._data_input = {'data': ..., 'y': ...}

        Returns:

        """
        pass

    @abc.abstractmethod
    def preprocess(self):
        """
        Do basic preprocessing or call any sub processes here adding them to children

        Note: This need not be saved to HDF, should be quick to run again
        (child_process should be quick to load if previously run)

        sub_process = OtherProcess(...)
        self.child_processes.append(sub_process)

        sub_process.input_data(...)
        part_processed = sub_process.output_data(...)

        self._data_preprocessed = {..., 'data': part_processed.data, ...}

        e.g.
        Returns:
        """

    @abc.abstractmethod
    def output_data(self) -> PlottableData:  # Note: can make more different outputs as well
        """
        Do something with self._data_input and self._data_preprocessed

        Returns:

        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_input_plotter(self) -> DataPlotter:
        """
        Initialize the DataPlotter with reasonable title and labels etc
        i.e.
        return DataPlotter(self._data_preprocessed.x, self._data_input.data, ...)
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

        i.e. save inputs and outputs. Preprocessed data should not need to be saved.
        (If processing is quick, and data large, can also not save outputs and just recalculate them in load_progress)
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

#####################################################################################################


# Now to create this process for separating square wave i_sense data into separate parts
class SeparateSquareProcess(Process):
    def input_data(self, i_sense_2d: np.ndarray, x: np.ndarray,
                   measure_frequency: float,
                   samples_per_setpoint: int,
                   setpoint_average_delay: float,

                   y: Optional[np.ndarray] = None,
                   ):
        self._data_input = {
            'i_sense': i_sense_2d,
            'x': x,
            'measure_freq': measure_frequency,
            'samples_per_setpoint': samples_per_setpoint,
            'setpoint_average_delay': setpoint_average_delay,
            'y': y,
        }

    def preprocess(self):
        i_sense = np.atleast_2d(self._data_input['i_sense'])
        y = self._data_input['y']
        y = y if y is not None else np.arange(i_sense.shape[-2]),

        data_by_setpoint = i_sense.reshape((i_sense.shape[0], -1, 4, self._data_input['samples_per_setpoint']))

        delay_index = round(self._data_input['setpoint_average_delay'] * self._data_input['measure_freq'])
        assert delay_index < self._data_input['samples_per_setpoint']

        setpoint_duration = self._data_input['samples_per_setpoint'] / self._data_input['measure_freq']

        self._data_preprocessed = {
            'y': y,
            'data_by_setpoint': data_by_setpoint,
            'delay_index': delay_index,
            'setpoint_duration': setpoint_duration,
        }

    def output_data(self,
                    ) -> dict:
        separated = np.mean(
            self._data_preprocessed['data_by_setpoint'][:, :, :, self._data_preprocessed['delay_index']:], axis=-1)

        x = self._data_input['x']
        x = np.linspace(x[0], x[-1], separated.shape[-1])
        y = self._data_preprocessed['y']
        self._data_output = {
            'x': x,
            'separated': separated,
            'y': y,
        }
        return self._data_output

    def get_input_plotter(self,
                          xlabel: str = 'Sweepgate /mV', data_label: str = 'Current /nA',
                          title: str = 'Data Averaged to Single Square Wave',
                          start_x: Optional[float] = None, end_x: Optional[float] = None,  # To only average between
                          start_y: Optional[float] = None, end_y: Optional[float] = None,  # To only average between
                          ) -> DataPlotter:
        by_setpoint = self._data_preprocessed['data_by_setpoint']
        x = self._data_input['x']
        y = self._data_preprocessed['y']

        if start_y or end_y:
            indexes = get_data_index(y, [start_y, end_y])
            s_ = np.s_[indexes[0], indexes[1]]
            by_setpoint = by_setpoint[s_]  # slice of rows, all_dac steps, 4 parts, all datapoints

        if start_x or end_x:
            indexes = get_data_index(x, [start_x, end_x])
            s_ = np.s_[indexes[0], indexes[1]]
            by_setpoint = by_setpoint[:, s_]  # All rows, slice of dac steps, 4 parts, all datapoints

        averaged = np.nanmean(by_setpoint, axis=0)  # Average rows together
        averaged = np.moveaxis(averaged, 1, 0)  # 4 parts, num steps, samples
        averaged = np.nanmean(averaged, axis=-1)  # 4 parts, num steps
        averaged = averaged.flatten()  # single 1D array with all 4 setpoints sequential

        duration = self._data_preprocessed['setpoint_duration']
        time_x = np.linspace(0, 4 * duration, averaged.shape[-1])

        data = PlottableData(
            data=averaged,
            x=time_x,
        )

        plotter = DataPlotter(
            data=data,
            xlabel=xlabel,
            data_label=data_label,
            title=title,
        )
        return plotter

    def get_output_plotter(self,
                           xlabel: str = 'Sweepgate /mV', data_label: str = 'Current* /nA',
                           ylabel: str = 'Repeats',
                           part: Union[str, int] = 'cold',  # e.g. hot, cold, vp, vm, or 0, 1, 2, 3
                           title: str = 'Separated into Square Wave Parts',
                           xspacing: float = 0,
                           yspacing: float = 0.3,
                           ) -> DataPlotter:
        separated = self._data_output['separated']  # rows, 4 parts, dac steps
        separated = np.moveaxis(separated, 2, 1)
        print(separated.shape)

        data_part = get_transition_part(separated, part)  # TODO: I think separated shape is wrong order

        data = PlottableData(
            data=data_part,
            x=self._data_output['x'],
            y=self._data_output['y'],
        )
        plotter = DataPlotter(
            data=data,
            xlabel=xlabel,
            ylabel=ylabel,
            data_label=data_label,
            title=title,
            xspacing=xspacing,
            yspacing=yspacing,
        )
        return plotter

    def save_progress(self, group: h5py.Group, **kwargs):
        self._write_to_group(group, 'data_input', self._data_input)
        if self._data_output:
            set_attr(group, 'data_output', self._data_output)

    @classmethod
    def load_progress(cls, group: h5py.Group) -> Process:
        inst = cls()
        inst._data_input = cls._read_from_group(group, 'data_input')
        inst.preprocess()
        inst._data_output = cls._read_from_group(group, 'data_output', default=dict(), check_exists=False)
        return inst


if __name__ == '__main__':
    import dash

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
