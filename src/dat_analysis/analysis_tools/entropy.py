from dataclasses import dataclass
from typing import Optional, Tuple, Union, TYPE_CHECKING
import numpy as np

from .new_procedures import Process, DataPlotter, PlottableData
from .square_wave import get_transition_parts
from ..characters import DELTA


if TYPE_CHECKING:
    pass


@dataclass
class EntropySignalProcess(Process):
    """
    Taking data which has been separated into the 4 parts of square heating wave and making entropy signal (2D)
    by averaging together cold and hot parts, then subtracting to get entropy signal
    """

    def set_inputs(self, x: np.ndarray, separated_data: np.ndarray,
                   ):
        self.inputs = dict(
            x=x,
            separated_data=separated_data,
        )

    def process(self):
        x = self.inputs['x']
        data = self.inputs['separated_data']
        data = np.atleast_3d(data)  # rows, steps, 4 heating setpoints
        cold = np.nanmean(np.take(data, get_transition_parts('cold'), axis=2), axis=2)
        hot = np.nanmean(np.take(data, get_transition_parts('hot'), axis=2), axis=2)
        entropy = cold-hot
        self.outputs = {
            'x': x,  # Worth keeping x-axis even if not modified
            'entropy': entropy
        }
        return self.outputs

    def get_input_plotter(self) -> DataPlotter:
        return DataPlotter(data=None, xlabel='Sweepgate /mV', ylabel='Repeats', data_label='Current /nA')

    def get_output_plotter(self,
                           y: Optional[np.ndarray] = None,
                           xlabel: str = 'Sweepgate /mV', data_label: str = f'{DELTA} Current /nA',
                           title: str = 'Entropy Signal',
                           ) -> DataPlotter:
        x = self.outputs['x']
        data = self.outputs['entropy']

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
