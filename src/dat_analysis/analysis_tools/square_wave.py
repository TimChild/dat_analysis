from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Any
from deprecation import deprecated

import numpy as np

from .new_procedures import Process, DataPlotter, PlottableData
from ..core_util import get_data_index

from .. import useful_functions as U


@deprecated(deprecated_in='3.0.0', details='uses old dat object')
def get_setpoint_indexes_from_times(dat: Any,
                                    start_time: Optional[float] = None,
                                    end_time: Optional[float] = None) -> Tuple[Union[int, None], Union[int, None]]:
    """
    Gets the indexes of setpoint_start/fin from a start and end time in seconds
    Args:
        dat (): Dat for which this is being applied
        start_time (): Time after setpoint change in seconds to start averaging setpoint
        end_time (): Time after setpoint change in seconds to finish averaging setpoint
    Returns:
        start, end indexes
    """
    setpoints = [start_time, end_time]
    setpoint_times = square_wave_time_array(dat.SquareEntropy.square_awg)
    sp_start, sp_fin = [U.get_data_index(setpoint_times, sp) for sp in setpoints]
    return sp_start, sp_fin


@deprecated(deprecated_in='3.2.0', details="Un-deprecate if used in new subclass of Data, but otherwise remove")
def get_transition_part(data: np.ndarray, part: Union[str, int]) -> np.ndarray:
    """
    Returns the specified part of I_sense data (i.e. for square wave heating analysis)
    Args:
        data (): I_sense data where axis [-2] has shape 4 (i.e. split into the separate parts of square wave)
        part (): Which part out of 'cold', 'hot', 0, 1, 2, 3 to return

    Returns:

    """
    assert data.shape[-2] == 4  # If not 4, then it isn't square wave transition data

    parts = get_transition_parts(part=part)

    data = np.take(data, parts, axis=-2)
    data = np.mean(data, axis=-2)
    return data


def get_transition_parts(part: Union[str, int]) -> Union[tuple, int]:
    if isinstance(part, str):
        part = part.lower()
        if part == 'cold':
            parts = (0, 2)
        elif part == 'hot':
            parts = (1, 3)
        elif part == 'vp':
            parts = (1,)
        elif part == 'vm':
            parts = (3,)
        else:
            raise ValueError(f'{part} not recognized. Should be in ["hot", "cold", "vp", "vm"]')
    elif isinstance(part, int):
        parts = (part,)
    else:
        raise ValueError(f'{part} not recognized. Should be in ["hot", "cold", "vp", "vm"]')
    return parts


@deprecated(deprecated_in='3.0.0', details='was used for old dat. OK to undeprecate if it becomes useful again')
def square_wave_time_array(awg: dict) -> np.ndarray:
    """Returns time array of single square wave (i.e. time in s for each sample in a full square wave cycle)"""
    num_pts = awg['wave_len']
    duration = num_pts / awg['measure_freq']
    x = np.linspace(0, duration, num_pts)  # In seconds
    return x


@deprecated(deprecated_in='3.2.0', details="Moving away from use of Process class")
@dataclass
class SeparateSquareProcess(Process):
    def set_inputs(self, x: np.ndarray, i_sense: np.ndarray,
                   measure_freq: float,
                   samples_per_setpoint: int,

                   setpoint_average_delay: Optional[float] = 0,
                   ):
        self.inputs = dict(
            x=x,
            i_sense=i_sense,
            measure_freq=measure_freq,
            samples_per_setpoint=samples_per_setpoint,
            setpoint_average_delay=setpoint_average_delay,
        )

    def _preprocess(self):
        i_sense = np.atleast_2d(self.inputs['i_sense'])

        data_by_setpoint = i_sense.reshape((i_sense.shape[0], -1, 4, self.inputs['samples_per_setpoint']))

        delay_index = round(self.inputs['setpoint_average_delay'] * self.inputs['measure_freq'])
        if delay_index > self.inputs['samples_per_setpoint']:
            setpoint_duration = self.inputs['samples_per_setpoint'] / self.inputs['measure_freq']
            raise ValueError(f'setpoint_average_delay ({self.inputs["setpoint_average_delay"]} s) is longer than '
                             f'setpoint duration ({setpoint_duration:.5f} s)')

        setpoint_duration = self.inputs['samples_per_setpoint'] / self.inputs['measure_freq']

        self._preprocessed = {
            'data_by_setpoint': data_by_setpoint,
            'delay_index': delay_index,
            'setpoint_duration': setpoint_duration,
        }

    def process(self):
        self._preprocess()
        separated = np.mean(
            self._preprocessed['data_by_setpoint'][:, :, :, self._preprocessed['delay_index']:], axis=-1)

        x = self.inputs['x']
        x = np.linspace(x[0], x[-1], separated.shape[-2])
        self.outputs = {
            'x': x,
            'separated': separated,
        }
        return self.outputs

    def get_input_plotter(self,
                          xlabel: str = 'Sweepgate /mV', data_label: str = 'Current /nA',
                          title: str = 'Data Averaged to Single Square Wave',
                          start_x: Optional[float] = None, end_x: Optional[float] = None,  # To only average between
                          y: Optional[np.ndarray] = None,
                          start_y: Optional[float] = None, end_y: Optional[float] = None,  # To only average between
                          ) -> DataPlotter:
        self._preprocess()
        by_setpoint = self._preprocessed['data_by_setpoint']
        x = self.inputs['x']

        if start_y or end_y:
            if y is None:
                raise ValueError(f'Need to pass in y_array to use start_y and/or end_y')
            indexes = get_data_index(y, [start_y, end_y])
            s_ = np.s_[indexes[0], indexes[1]]
            by_setpoint = by_setpoint[s_]  # slice of rows, all_dac steps, 4 parts, all datapoints

        if start_x or end_x:
            indexes = get_data_index(x, [start_x, end_x])
            s_ = np.s_[indexes[0], indexes[1]]
            by_setpoint = by_setpoint[:, s_]  # All rows, slice of dac steps, 4 parts, all datapoints

        averaged = np.nanmean(by_setpoint, axis=0)  # Average rows together
        averaged = np.moveaxis(averaged, 1, 0)  # 4 parts, num steps, samples
        averaged = np.nanmean(averaged, axis=-2)  # 4 parts, samples  (average steps together)
        averaged = averaged.flatten()  # single 1D array with all 4 setpoints sequential

        duration = self._preprocessed['setpoint_duration']
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
                           y: Optional[np.ndarray] = None,
                           xlabel: str = 'Sweepgate /mV', data_label: str = 'Current* /nA',
                           ylabel: str = 'Repeats',
                           part: Union[str, int] = 'cold',  # e.g. hot, cold, vp, vm, or 0, 1, 2, 3
                           title: str = 'Separated into Square Wave Parts',
                           x_spacing: float = 0,
                           y_spacing: float = 0.3,
                           ) -> DataPlotter:
        if not self.processed:
            self.process()
        separated = self.outputs['separated']  # rows, dac steps, 4 parts
        y = y if y is not None else np.arange(separated.shape[0])

        data_part = self.data_part_out(part)

        data = PlottableData(
            data=data_part,
            x=self.outputs['x'],
            y=y,
        )
        plotter = DataPlotter(
            data=data,
            xlabel=xlabel,
            ylabel=ylabel,
            data_label=data_label,
            title=title,
            xspacing=x_spacing,
            yspacing=y_spacing,
        )
        return plotter

    def data_part_out(self,
                      part: Union[str, int] = 'cold',  # e.g. hot, cold, vp, vm, or 0, 1, 2, 3
                      ) -> np.ndarray:
        if not self.processed:
            self.process()
        separated = self.outputs['separated']
        part = get_transition_parts(part)  # Convert to Tuple (e.g. (1,3) for 'hot')
        data_part = np.nanmean(np.take(separated, part, axis=2), axis=2)
        return data_part
