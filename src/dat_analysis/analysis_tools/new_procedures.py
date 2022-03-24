from __future__ import annotations
from typing import Any, List
from functools import lru_cache
from dataclasses import dataclass
import plotly.graph_objects as go
import numpy as np


@dataclass
class PlottableData:
    x: np.ndarray
    # y
    # data


@dataclass
class DataPlotter:
    """
    Collection of functions which take PlottableData and plot it various ways
    e.g. 1D, 2D, heatmap, waterfall, single row of 2d
    """
# labels
# title
#
# optionals:
# xspacing
# yspacing (if waterfalled etc)

    def plot_1d(self) -> go.Figure:
        pass

    def plot_2d(self) -> go.Figure:
        pass

    def plot_waterfall(self) -> go.Figure:
        pass

    def plot_row(self) -> go.Figure:
        pass


class Procedure:
    """
    E.g. splitting data into square wave parts

    mixture of human friendly input, as well as data passed in, no loading from file
    """

    def __init__(self, data=None):
        self.child_procedures: List[Procedure] = []  # if calling other procedures, append to this list (e.g. in input_data)

        if data:
            self.data_to_process = self.input_data(data)

    @property
    def default_name(self):
        """Default name of specific procedure"""
        return 'SubProcess'
    
    def input_data(self, data) -> Any:
        """
        Pass minimal data in to get job done and call any sub procedures here adding them to children
        e.g.
        self.inputs = dict(...)
        sub_process = OtherProcedure(...)
        self.child_procedures.append(sub_process)
        part_processed = sub_process.output_data(...)

        self.data_to_process = PlottableData(..., part_processed.x, ...)

        return self.data_to_process

        Returns:

        """
        pass

    @lru_cache
    def _process(self):
        """Do the actual processing in cached function"""
        pass
    
    def output_data1(self) -> PlottableData:  # Rename as please...
        """
        Do something with self.data_to_process
        (possibly cache processing part)
        Returns:

        """
        pass

    def input_plotter(self) -> DataPlotter:
        """
        Initialize the DataPlotter with reasonable title and labels etc
        i.e.
        return DataPlotter(self.data_to_process(...), self.inputs, ...)
        Returns:

        """
        pass

    def output_plotter(self) -> DataPlotter:
        """
        Same as self.input_plotter, but for the output data
        i.e.
        return DataPlotter(self.output_data(...), ...)
        Returns:

        """
        pass

    def save_progress(self, location, **kwargs):
        """
        Save necessary info to given dat/hdf group/file with minimal additional input (in a single group)
        """
        sub_process_locations = []
        for child in self.child_procedures:
            child_location = f'{location}/{child.default_name}'
            child.save_progress(child_location)
            sub_process_locations.append(child_location)
        self._write_to_file(sub_process_locations)

    def _write_to_file(self, subprocess_locations, **kwargs):
        """
        Save all necessary info to load again to file (only can rely on sub processes doing the same)

        Returns:

        """

    @classmethod
    def load_progress(cls, location) -> Procedure:
        """

        Args:
            **kwargs ():

        Returns:

        """
        subprocess_locations = []  # Get this from file
        for location in subprocess_locations:
            sub_process = Procedure.load_progress(location)
            # self.child_procedures.append(sub_process)  Need to think about how I want init to work etc for this



    def _load_from_file(self):
        """
        Get all data from hdf again (for this process only)
        Returns:

        """




class ProcessViewer:
    """
    Take a process and combine input/output views into a nice overall summary... Something which can be easily plotted
    in a dash page. or output to a plotly muligraph fig or mpl, or saved to pdf
    """
    def __init__(self, process, **options):
        self.process = process

    def dash_full(self, **more_options):  # -> Component
        pass

    def mpl_full(self, **more_options):  # -> plt.Figure
        pass

    def pdf_full(self, **more_options):  # -> pdf
        pass


class ProcedureInterface:
    """
    Things necessary to put a procedure into a dash page or report with easy human friendly input. I.e. for building a
    new dash page

    human friendly input, and id of file to load from (or data passed in)
    """
    def __init__(self):
        self.store_id = ''  # To stor
        self.sub_store_ids = []

    @property
    def required_input_components(self) -> List[ProcedureComponentInput]:
        """
        Give the list of components that need to be placed in order for Procedure to be carried out
        Should all update stores or possibly even the dat file
        Returns:

        """
        return []

    @property
    def all_outputs(self) -> List[ProcedureComponentOutput]:
        """Lit of components that display the procedure in detail"""
        return []

    @property
    def main_outputs(self) -> List[ProcedureComponentOutput]:
        return []


class ProcedureComponent:
    """
    A dash component with callbacks etc to interface with user
    """
    def run_callbacks(self, **kwargs):
        pass

    def layout(self):
        pass


class ProcedureComponentInput(ProcedureComponent):
    """
    Component mostly for user input
    """
    def __init__(self, title, num_buttons, etc):
        super().__init__()

    def run_callbacks(self, **kwargs):
        super().run_callbacks(**kwargs)


class ProcedureComponentOutput(ProcedureComponent):
    """
    Component mostly for output to graph/table/file etc
    """


