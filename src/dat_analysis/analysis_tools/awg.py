from __future__ import annotations
from dataclasses import dataclass, field
import copy
import numpy as np
from typing import TYPE_CHECKING, Optional

from ..plotting.plotly.util import figures_to_subplots, default_fig
from ..core_util import get_data_index
from .data import Data, PlottingInfo

if TYPE_CHECKING:
    from ..dat.dat_hdf import DatHDF


@dataclass(frozen=True)
class AWGInfo:
    awg_used: int
    setpoint_samples: tuple[int]
    setpoint_values: np.ndarray = field(
        compare=False
    )  # compare doesn't work for arrays
    num_setpoints: int
    wave_len: int
    num_cycles: int
    num_steps: int

    aw_dac_channels: tuple[int]
    aw_dac_names: tuple[str]
    aw_waves: tuple[int]
    num_waves: int
    sampling_freq: float
    num_adcs: int
    measure_freq: float

    @classmethod
    def from_dat(cls, dat: DatHDF) -> AWGInfo:
        fd_info = dat.Logs.get_fastdac(1)
        aw_waves = [dat.Data.get_data(f"fdAW_{i}", None) for i in range(9)]
        aw_waves = [arr for arr in aw_waves if arr is not None]
        awg_dict = fd_info.AWG

        aw_dac_channels = tuple([int(v) for v in awg_dict["AW_Dacs"].split(",")])
        dac_names = list(dat.Logs.dacs.keys())
        aw_dac_names = [dac_names[num] for num in aw_dac_channels]
        setpoint_values = np.stack([arr[0] for arr in aw_waves])
        num_setpoint = aw_waves[0].shape[1]
        setpoint_samples = tuple(aw_waves[0][1].astype(int))
        d = {
            "awg_used": awg_dict.get("AWG_used", None),
            "setpoint_samples": setpoint_samples,
            "setpoint_values": setpoint_values,
            "num_setpoints": num_setpoint,
            "wave_len": awg_dict["waveLen"],
            "num_cycles": awg_dict["numCycles"],
            "num_steps": awg_dict["numSteps"],
            "aw_dac_channels": aw_dac_channels,
            "aw_dac_names": aw_dac_names,
            "aw_waves": tuple([int(v) for v in awg_dict["AW_Waves"].split(",")]),
            "num_waves": awg_dict["numWaves"],
            "sampling_freq": awg_dict["samplingFreq"],
            "num_adcs": awg_dict["numADCs"],
            "measure_freq": awg_dict["measureFreq"],
        }
        return cls(**d)


@dataclass
class AWGData(Data):
    awg_info: AWGInfo = None

    # For internal use
    _last_settling_time: float = 0.0

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        if not isinstance(self.awg_info, AWGInfo):
            raise ValueError(
                f"EntropyData requires AWGInfo, provide `awg_info` when initializing EntropyData"
            )

    @classmethod
    def from_Data(cls, data: Data, awg_info: AWGInfo):
        plot_info = data.plot_info
        plot_info.title = f"{plot_info.title} as AWGData"
        return cls(
            data=data.data,
            x=data.x,
            y=data.y,
            xerr=data.xerr,
            yerr=data.yerr,
            plot_info=plot_info,
            awg_info=awg_info,
        )

    def slice_values(
        self,
        x_range: tuple[Optional[int], Optional[int]] = None,
        y_range: tuple[Optional[int], Optional[int]] = None,
    ):
        """Select data based on x and y axes"""

        def replace_nones_in_indexes(arr: np.ndarray, indexes: tuple):
            """Replace None with either 0 or last index value"""
            new_indexes = list(indexes)
            if indexes[0] is None:
                indexes[0] = 0
            if indexes[1] is None:
                indexes[1] = arr.shape[-1]
            return indexes

        x_slice = ...
        if x_range is not None:
            x_indexes = get_data_index(self.x, x_range)
            x_indexes = replace_nones_in_indexes(self.x, x_indexes)
            if None not in x_indexes:
                x_indexes = min(x_indexes), max(x_indexes)
            #### Specific to AWG Data  ####

            # To always end up with exact multiples of full AWG
            x_indexes = [
                ind - ind % self.awg_info.wave_len if ind else None for ind in x_indexes
            ]
            x_slice = slice(x_indexes[0], x_indexes[1])  # Note: do NOT +1 for AWGData
            ###################################
        if self.data.ndim == 1:
            return self[x_slice]
        elif self.data.ndim == 2:
            y_slice = ...
            if y_range is not None:
                y_indexes = get_data_index(self.y, y_range)
                y_indexes = replace_nones_in_indexes(self.y, y_indexes)
                if None not in y_indexes:
                    y_indexes = min(y_indexes), max(y_indexes)
                y_slice = slice(y_indexes[0], y_indexes[1] + 1)
            return self[y_slice, x_slice]
        else:
            raise NotImplementedError

    def get_awg_averaged(
        self,
        x_range: Optional[tuple[int, int]] = None,
        y_range: Optional[tuple[int, int]] = None,
    ) -> Data:
        """
        Average data to a single AWG wave

        E.g. Good for seeing the settling time after AWG steps
        """
        if x_range:
            data = self.slice_values(x_range=x_range, y_range=y_range)
        else:
            data = self.copy()

        new_d = np.reshape(data.data, (-1, self.awg_info.wave_len)).mean(axis=0)
        new_x = np.linspace(
            0, self.awg_info.wave_len / self.awg_info.measure_freq, new_d.shape[-1]
        )

        title = f"Averaged AWG"
        if x_range is not None:
            title += f" x_range=({x_range})"
        if y_range is not None:
            title += f" y_range=({y_range})"
        return Data(
            x=new_x,
            data=new_d,
            plot_info=PlottingInfo(x_label="Time /s", y_label="", title=title),
        )

    def separate_and_average_setpoints(self, settling_time: float = 0.0) -> list[Data]:
        """Separate AWG into the individual parts, excluding settling time"""
        self._last_settling_time = settling_time
        delay_samples = round(settling_time * self.awg_info.measure_freq)
        if np.any(delay_samples > np.array(self.awg_info.setpoint_samples)):
            setpoint_durations = (
                np.array(self.awg_info.setpoint_samples) / self.awg_info.measure_freq
            ).round(4)
            raise ValueError(
                f"settling_time ({settling_time} s) is longer than "
                f"a setpoint duration ({str(setpoint_durations)} s)"
            )

        # New x has shape of number of full DAC steps (note: cannot use self.awg_info.num_setpoints in case data
        # already sliced)
        new_x = np.linspace(
            self.x[0], self.x[-1], int(len(self.x) / self.awg_info.wave_len)
        )

        setpoint_lengths = self.awg_info.setpoint_samples
        d = np.atleast_2d(self.data)
        by_full_awg = d.reshape((d.shape[0], -1, sum(setpoint_lengths)))

        start_index = 0
        new_datas = []
        for i, length in enumerate(setpoint_lengths):
            # Select only the part that is for this setpoint (excluding delay), then average values at setpoint
            section = by_full_awg[
                :, :, start_index + delay_samples : start_index + length
            ]
            std = section.std(axis=-1)
            section = section.mean(axis=-1)
            start_index += length

            # Put that into a new Data object (modifying necessary things)
            self_dict = copy.deepcopy(self.__dict__)
            self_dict.pop("awg_info")
            self_dict.pop("_last_settling_time")
            new_data = Data(**self_dict)
            new_data.plot_info.title = f"AWG Data Setpoint {i}"
            new_data.x = new_x
            new_data.yerr = std
            new_data.data = section
            new_datas.append(new_data)
        return new_datas

    def plot_setpoints_2d(self, settling_time=None, resample=True):
        if settling_time is None:
            settling_time = self._last_settling_time
        setpoint_datas = self.separate_and_average_setpoints(
            settling_time=settling_time
        )
        figs = [data.plot(resample=resample) for data in setpoint_datas]
        # return figs
        title = f"AWG Data separated into averaged setpoints"
        if settling_time:
            title += f" with time delay = {settling_time}"
        fig = figures_to_subplots(figs, title=title, shared_data=True)
        return fig

    def plot_setpoints_averaged(self, settling_time=None, resample=False):
        if settling_time is None:
            settling_time = self._last_settling_time
        setpoint_datas = self.separate_and_average_setpoints(
            settling_time=settling_time
        )
        fig = default_fig()
        for i, data in enumerate(setpoint_datas):
            fig.add_traces(
                data.mean().get_traces(
                    name=f"Setpoint {i}", max_num_pnts=1000 if resample else 1000000
                )[0]
            )
        fig.update_traces(error_y=None)
        title = f"AWG Data separated into averaged setpoints then averaged to 1D"
        if settling_time:
            title += f"<br>With time delay = {settling_time}"
        fig.update_layout(title=title, xaxis_title=self.plot_info.x_label)
        return fig
