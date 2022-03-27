from __future__ import annotations
import abc
from typing import Any, List, Union, Optional, Dict, TYPE_CHECKING, TypeVar, Type
from functools import lru_cache
from dataclasses import dataclass, field
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType
import h5py
import numpy as np

import dash_bootstrap_components as dbc
from dash import html, dcc

from dat_analysis.plotting.plotly import OneD, TwoD
from dat_analysis.useful_functions import get_matching_x, get_data_index
from dat_analysis.hdf_util import set_attr, get_attr, DatDataclassTemplate, NotFoundInHdfError
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


T = TypeVar('T', bound='Process')  # Required in order to make subclasses return their own subclass


@dataclass
class Process(DatDataclassTemplate, abc.ABC):
    """
    Standardize what should be written for any analysis process, can recursively call sub processes as well.

    Intention is that any step of analysis has some ways to view the inputs and outputs of that step of analysis

    Should be some mixture of human friendly input and data passed in.

    Also provide some options for saving progress to HDF file in given group location.

    E.g. splitting data into square wave parts

    mixture of human friendly input, as well as data passed in, no loading input data from file (only save or restore
    whole process to or from and open HDF Group)
    """
    # TODO: Not sure if this needs to be here
    data_input: Dict[str, Union[np.ndarray, Any]] = field(init=False)  # Store data as provided
    data_output: Dict[str, Union[np.ndarray, Any]] = field(init=False)  # Store data produced
    # child_processes: List[Process] = field(default_factory=list)  # if calling other processes, append to this list (e.g. in input_data)

    def __post_init__(self):
        # TODO: Or here (or maybe just need to use field(...) here?) Or need to consider https://stackoverflow.com/questions/57601705/annotations-doesnt-return-fields-from-a-parent-dataclass
        self.data_input: Dict[str, Union[np.ndarray, Any]] = {}  # Store data as provided
        self.data_output: Dict[str, Union[np.ndarray, Any]] = {}  # Store data produced
        # self.child_processes: List[Process] = []  # if calling other processes, append to this list (e.g. in input_data)

    @abc.abstractmethod
    def input_data(self, *args, **kwargs):
        """
        Require minimal data in to get job done and save into self.data_input

        self.data_input = {'data': ..., 'y': ...}

        Returns:

        """
        pass

    @abc.abstractmethod
    def process(self):
        """
        Do the process with self.data_input and fill self.data_output

        Returns:

        """
        pass

    def get_input_plotter(self) -> DataPlotter:
        """
        Initialize the DataPlotter with reasonable title and labels etc
        i.e.
        return DataPlotter(self.data_input.x, self.data_input.data, ...)
        Returns:

        """
        raise NotImplementedError

    def get_output_plotter(self) -> DataPlotter:
        """
        Same as self.input_plotter, but for the output data
        i.e.
        return DataPlotter(self.output_data(...), ...)
        Returns:

        """
        raise NotImplementedError

    @property
    def processed(self) -> bool:
        """Easy way to check if processing has been done or not"""
        return True if self.data_output else False

    def save_progress(self, parent_group: h5py.Group, name: str = None):
        self.save_to_hdf(parent_group=parent_group, name=name)

    @classmethod
    def load_progress(cls: Type[T], parent_group: h5py.Group, name: Optional[str] = None) -> T:
        return cls.from_hdf(parent_group=parent_group, name=name)

    @classmethod
    def load_output_only(cls, parent_group: h5py.Group, name: Optional[str] = None) -> dict:
        if name is None:
            name = cls._default_name()
        name = name.replace('/', '-')  # Because I make this substitution on saving, also make it for loading.
        dc_group = parent_group.get(name)

        if dc_group is None:
            raise NotFoundInHdfError(f'No {name} group in {parent_group.name}')

        output = get_attr(dc_group, 'data_output', check_exists=True)
        return output

    def ignore_keys_for_hdf(self) -> Optional[Union[str, List[str]]]:
        return ['child_processes']

    def additional_save_to_hdf(self, dc_group: h5py.Group):
        # TODO: Figure out how to deal with sub_processes later
        # sub_process_locations = []
        # for child in self.child_processes:
        #     child_group = dc_group.require_group(f'{dc_group.name}/{child._default_name()}')
        #     child.save_to_hdf(child_group, name=child._default_name())
        #     sub_process_locations.append(child_group.name)
        #
        # # Save location of any Sub processes required for loading
        # self._write_to_group(dc_group, 'subprocess_locations', sub_process_locations)
        pass

    @staticmethod
    def additional_load_from_hdf(dc_group: h5py.Group) -> Dict[str, Any]:
        # TODO: Figure out how to load subclasses later (maybe needs to be done explicitly to know the subclass types)
        pass

#####################################################################################################


# Now to create this process for separating square wave i_sense data into separate parts
class SeparateSquareProcess(Process):
    def input_data(self, i_sense_2d: np.ndarray, x: np.ndarray,
                   measure_frequency: float,
                   samples_per_setpoint: int,
                   setpoint_average_delay: float,

                   y: Optional[np.ndarray] = None,
                   ):
        self.data_input = {
            'i_sense': i_sense_2d,
            'x': x,
            'measure_freq': measure_frequency,
            'samples_per_setpoint': samples_per_setpoint,
            'setpoint_average_delay': setpoint_average_delay,
            'y': y,
        }

    def _preprocess(self):
        i_sense = np.atleast_2d(self.data_input['i_sense'])
        y = self.data_input['y']
        y = y if y is not None else np.arange(i_sense.shape[-2]),

        data_by_setpoint = i_sense.reshape((i_sense.shape[0], -1, 4, self.data_input['samples_per_setpoint']))

        delay_index = round(self.data_input['setpoint_average_delay'] * self.data_input['measure_freq'])
        assert delay_index < self.data_input['samples_per_setpoint']

        setpoint_duration = self.data_input['samples_per_setpoint'] / self.data_input['measure_freq']

        self._data_preprocessed = {
            'y': y,
            'data_by_setpoint': data_by_setpoint,
            'delay_index': delay_index,
            'setpoint_duration': setpoint_duration,
        }

    def process(self,
                ) -> dict:
        self._preprocess()
        separated = np.mean(
            self._data_preprocessed['data_by_setpoint'][:, :, :, self._data_preprocessed['delay_index']:], axis=-1)

        x = self.data_input['x']
        x = np.linspace(x[0], x[-1], separated.shape[-1])
        y = self._data_preprocessed['y']
        self.data_output = {
            'x': x,
            'separated': separated,
            'y': y,
        }
        return self.data_output

    def get_input_plotter(self,
                          xlabel: str = 'Sweepgate /mV', data_label: str = 'Current /nA',
                          title: str = 'Data Averaged to Single Square Wave',
                          start_x: Optional[float] = None, end_x: Optional[float] = None,  # To only average between
                          start_y: Optional[float] = None, end_y: Optional[float] = None,  # To only average between
                          ) -> DataPlotter:
        self._preprocess()
        by_setpoint = self._data_preprocessed['data_by_setpoint']
        x = self.data_input['x']
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
        separated = self.data_output['separated']  # rows, 4 parts, dac steps
        separated = np.moveaxis(separated, 2, 1)
        print(separated.shape)

        data_part = get_transition_part(separated, part)  # TODO: I think separated shape is wrong order

        data = PlottableData(
            data=data_part,
            x=self.data_output['x'],
            y=self.data_output['y'],
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
