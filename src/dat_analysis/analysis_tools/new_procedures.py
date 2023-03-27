import abc
from typing import Any, List, Union, Optional, Dict, TYPE_CHECKING, TypeVar, Type
from dataclasses import dataclass, field
import plotly.graph_objects as go
import h5py
import numpy as np
from deprecation import deprecated

from dat_analysis.plotting.plotly import OneD, TwoD
from dat_analysis.useful_functions import get_matching_x, get_data_index
from dat_analysis.hdf_util import set_attr, get_attr, HDFStoreableDataclass, NotFoundInHdfError

if TYPE_CHECKING:
    pass


@deprecated(deprecated_in='3.2.0', details='Included as part of new Data class in useful_functions')
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
    axes: List[np.ndarray] = None
    data_err: Optional[np.ndarray] = None

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
            self.axes = [np.arange(s) for s in self.data.shape]


@deprecated(deprecated_in='3.2.0', details='Included as part of new Data class in useful_functions')
@dataclass
class DataPlotter:
    """
    Collection of functions which take PlottableData and plot it various ways (adding labels etc)
    e.g. 1D, 2D, heatmap, waterfall, single row of 2d
    """
    data: Optional[PlottableData]  #
    xlabel: str = ''
    ylabel: str = ''
    data_label: str = ''
    title: str = ''

    xspacing: float = 0
    yspacing: float = 0

    def fig_1d(self, fig_kwargs=None) -> go.Figure:
        """
        Usef
        Args:
            s_ ():
            avg ():

        Returns:

        """
        if fig_kwargs is None:
            fig_kwargs = {}
        p = OneD(dat=None)
        fig = p.figure(self.xlabel, self.data_label, self.title, fig_kwargs=fig_kwargs)
        return fig

    def trace_1d(self, s_: np.s_ = None,
                 axis: np.ndarray = None,
                 avg: bool = False,
                 trace_kwargs=None) -> go.Scatter:
        """

        Args:
            s_ ():
            axis ():
            avg (): Whether to average 2D data before plotting
            trace_kwargs ():

        Returns:

        """
        if trace_kwargs is None:
            trace_kwargs = {}
        p = OneD(dat=None)  # Temporarily piggybacking off this
        data = self.data.data
        if avg and data.ndim > 1:
            data = np.nanmean(data, axis=0)

        if axis is None:
            axis = self.data.x
        if s_ is not None:
            data = data[s_]

        if data.ndim > 1:
            raise ValueError(f'Data has shape {data.shape} after slicing with {s_}, cannot be plot 1D')

        axis = get_matching_x(axis, data)
        return p.trace(x=axis, data=data, data_err=self.data.data_err, trace_kwargs=trace_kwargs)

    def fig_heatmap(self, fig_kwargs=None) -> go.Figure:
        if fig_kwargs is None:
            fig_kwargs = {}
        p = TwoD(dat=None)  # Temporarily piggybacking off this
        fig = p.figure(self.xlabel, self.ylabel, self.title, fig_kwargs=fig_kwargs)
        return fig

    def trace_heatmap(self, s_: np.s_ = None, axis_x: np.ndarray = None, axis_y: np.ndarray = None) -> go.Heatmap:
        p = TwoD(dat=None)  # Temporarily piggybacking off this
        data = self.data.data
        if axis_x is None:
            axis_x = self.data.x
        if axis_y is None:
            axis_y = self.data.y

        if s_ is not None:
            data = data[s_]

        axis_x = get_matching_x(axis_x, shape_to_match=data.shape[-1])
        axis_y = get_matching_x(axis_y, shape_to_match=data.shape[-2])
        if data.ndim > 2:
            raise ValueError(f'Data has shape {data.shape} after slicing with {s_}, cannot be plot 2D')
        return p.trace(x=axis_x, y=axis_y, data=data, trace_type='heatmap')

    def plot_waterfall(self) -> go.Figure:
        raise NotImplementedError


T = TypeVar('T', bound='Process')  # Required in order to make subclasses return their own subclass


@deprecated(deprecated_in='3.2.0', details='Moving away from the use of this (never implemented it enough). Might be a good idea to reintroduce again later but needs to be easier to use')
@dataclass
class Process(HDFStoreableDataclass, abc.ABC):
    """
    Standardize what should be written for any analysis process, can recursively call sub processes as well.

    Intention is that any step of analysis has some ways to view the inputs and outputs of that step of analysis

    Should be some mixture of human friendly input and data passed in.

    Also provide some options for saving progress to HDF file in given group location.

    E.g. splitting data into square wave parts
    """
    inputs: Dict[str, Union[np.ndarray, Any]] = field(default_factory=dict)  # Store data as provided
    outputs: Dict[str, Union[np.ndarray, Any]] = field(default_factory=dict)  # Store data produced

    @abc.abstractmethod
    def set_inputs(self, *args, **kwargs):
        """
        Require minimal data in to get job done and save into self.input

        self.inputs = {'data': ..., 'y': ...}

        Returns:

        """
        pass

    @abc.abstractmethod
    def process(self):
        """
        Do the process with self.inputs and fill self.outputs

        Returns:

        """
        pass

    @property
    def processed(self) -> bool:
        """Easy way to check if processing has been done or not"""
        return True if self.outputs else False

    def save_progress(self, parent_group: h5py.Group, name: str = None):
        save_group = self.save_to_hdf(parent_group=parent_group, name=name)
        return save_group

    @classmethod
    def load_progress(cls: Type[T], group: h5py.Group) -> T:
        """Load state of Process from HDF file
        Note: Process may not have been processed yet (i.e. no self.outputs)

        To override loading behaviour override the HDFStoreableDataclass methods [ignore_keys_for_hdf,
        additional_save_to_hdf, additional_load_from_hdf]
        Note: also override cls.load_output_only if necessary
        """
        return cls.from_hdf(parent_group=group.parent, name=group.name.split('/')[-1])

    @classmethod
    def _load_progress(cls, group: h5py.Group):
        """Override this to change behaviour of cls.load_progress() -- This is to avoid having to deal with TypeVar"""

    @classmethod
    def load_output_only(cls, group: h5py.Group) -> dict:
        output = get_attr(group, 'outputs', check_exists=True)
        return output


@deprecated(deprecated_in='3.2.0', details='Moving away from the use of this (never implemented it enough). Might be a good idea to reintroduce again later but needs to be easier to use')
@dataclass
class TemplateProcess(Process):
    def set_inputs(self, x: np.ndarray, data: np.ndarray,
                   other_variable: float,
                   ):
        self.inputs = dict(
            x=x,
            data=data,
            other_variable=other_variable,
        )

    def process(self):
        x = self.inputs['x']
        data = self.inputs['data']
        var = self.inputs['other_variable']
        new_data = data*var
        self.outputs = {
            'x': x,  # Worth keeping x-axis even if not modified
            'new_data': new_data,
        }
        return self.outputs

    def get_input_plotter(self,
                          xlabel: str = 'Sweepgate /mV', data_label: str = 'Current /nA',
                          title: str = 'Standard Title for Plotting Inputs',
                          ) -> DataPlotter:
        x = self.inputs['x']
        data = self.inputs['data']
        var = self.inputs['other_variable']

        data = PlottableData(
            data=data,
            x=x,
        )

        plotter = DataPlotter(
            data=data,
            xlabel=xlabel,
            data_label=data_label,
            title=title,
        )
        return plotter

    def get_output_plotter(self,
                           y: Optional[np.ndarray] = None,
                           xlabel: str = 'Sweepgate /mV', data_label: str = 'Current* /nA',
                           title: str = 'Standard Title for Plotting Outputs',
                           ) -> DataPlotter:
        x = self.outputs['x']
        data = self.outputs['data']

        data = PlottableData(
            data=data,
            x=x,
        )

        plotter = DataPlotter(
            data=data,
            xlabel=xlabel,
            data_label=data_label,
            title=title,
        )
        return plotter


if __name__ == '__main__':
    pass
