"""
2022-06: Making a simpler interface for DatHDF files
Generally will be useful to have an easy way to navigate through Logs

This should not require specific things to exist though (i.e. should not be a problem to initialize without
temperature data, and maybe raise a reasonable error if asked for temperature data?)

Not sure yet how I want to handle the possibility of different logs existing...
-- Hard coding has the advantage of typing, but will be misleading as to what Logs actually exist
    -- But I could just make a method which tries to get all the different logs excepting any missing
-- Dynamically adding attributes for Logs would be nice in Jupyter/console, but bad for static typing...
"""
from __future__ import annotations
from ..hdf_file_handler import HDFFileHandler
from dataclasses import dataclass
import logging
import json
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from ..hdf_util import get_attr, HDFStoreableDataclass, NotFoundInHdfError

logger = logging.getLogger(__name__)


class Logs:
    def __init__(self, hdf_path: str, path_to_logs_group: str):
        self._hdf_path = hdf_path
        self._group_path = path_to_logs_group
        self._dacs = None
        self._fastdac_sweepgates = None

    @property
    def hdf_read(self) -> HDFFileHandler:
        return HDFFileHandler(self._hdf_path, "r")  # with self.hdf_read as f: ...

    @property
    def hdf_write(self) -> HDFFileHandler:
        return HDFFileHandler(self._hdf_path, "r+")  # with self.hdf_write as f: ...

    @property
    def logs_keys(self) -> list[str]:
        all_keys = self._get_all_keys()
        keys = list(set(all_keys) - set())
        return keys

    @property
    def sweeplogs_string(self) -> str:
        """Sweeplogs as saved in experiment file"""
        with self.hdf_read as f:
            sweeplogs_string = f["Logs"].attrs["sweep_logs_string"]
            if isinstance(sweeplogs_string, bytes):
                sweeplogs_string = sweeplogs_string.decode("utf-8")
        return sweeplogs_string

    @property
    def sweeplogs(self) -> dict:
        """Sweeplogs loaded into a dictionary"""
        sl_string = self.sweeplogs_string
        return json.loads(sl_string)

    @property
    def comments(self) -> list[str]:
        with self.hdf_read as f:
            f = f.get(self._group_path)
            comments = f["General"].attrs.get("comments", "")
        comments = [c.strip() for c in comments.split(",")]
        return comments

    @property
    def dacs(self) -> dict:
        """All DACs connected (i.e. babydacs, fastdacs, ... combined)"""
        if self._dacs is None:
            keys = self.logs_keys
            dacs = {}
            for i in range(1, 10):  # Up to 10 FastDACs
                if f"FastDAC{i}" in keys:
                    fd = self.get_fastdac(i)
                    dacs.update(fd.dacs)

            for i in range(1, 10):  # Up to 10 BabyDACs
                pass  # TODO: add BabyDacs

            self._dacs = dacs
        return self._dacs

    @property
    def fastdac_sweepgates(self) -> SweepGates:
        """The gates swept during a FastDAC Scan"""

        # TODO: move the creation of SweepGates to initialization of Dat (and check it actually is HDFStoreable)
        def str_to_float(str_val):
            try:
                val = float(str_val)
            except ValueError:
                val = np.nan
            return val

        if self._fastdac_sweepgates is None:
            with self.hdf_read as f:
                scan_vars = json.loads(f["Logs"].attrs["scan_vars_string"].decode())
                axis_gates = []
                for axis in ["x", "y"]:
                    starts = scan_vars.get(f"start{axis}s")
                    fins = scan_vars.get(f"fin{axis}s")
                    starts, fins = [
                        tuple([str_to_float(v) for v in vals])
                        for vals in [starts.split(","), fins.split(",")]
                    ]
                    channels = scan_vars.get(f"channels{axis}")
                    channels = tuple(
                        [int(v) if v != "null" else None for v in channels.split(",")]
                    )
                    numpts = scan_vars.get(f"numpts{axis}")
                    channel_names = scan_vars.get(f"{axis}_label")
                    if channel_names.endswith(" (mV)"):
                        channel_names = channel_names[:-5]
                    else:
                        channel_names = ",".join([f"DAC{num}" for num in channels])
                    channel_names = tuple(
                        [name.strip() for name in channel_names.split(",")]
                    )
                    axis_gates.append(
                        AxisGates(channels, channel_names, starts, fins, numpts)
                    )
                self._fastdac_sweepgates = SweepGates(*axis_gates)
        return self._fastdac_sweepgates

    @property
    def temperatures(self) -> Temperatures:
        temps = self._get_temps()
        return temps

    @property
    def magnets(self) -> Magnets:
        mags = self._get_mags()
        return mags

    @property
    def x_label(self) -> str:
        with self.hdf_read as f:
            f = f.get(self._group_path)
            label = f["General"].attrs.get("x_label", None)
        return label

    @property
    def y_label(self) -> str:
        with self.hdf_read as f:
            f = f.get(self._group_path)
            label = f["General"].attrs.get("y_label", None)
        return label

    @property
    def measure_freq(self) -> float:
        with self.hdf_read as f:
            f = f.get(self._group_path)
            freq = f["General"].attrs.get("measure_freq", None)
        return freq

    @property
    def time_completed(self) -> str:
        with self.hdf_read as f:
            f = f.get(self._group_path)
            time = f["General"].attrs.get("time_completed", None)
        return time

    def get_fastdac(self, num=1) -> FastDAC:
        """Load entry for FastDAC logs (i.e. dict of Dac channels where keys are labels or channel num)"""
        fd_logs = None
        with self.hdf_read as f:
            f = f.get(self._group_path)
            try:
                fd_logs = FastDAC.from_hdf(f, f"FastDAC{num}")
            except NotFoundInHdfError:
                logger.warning(f"FastDAC{num} not found in DatHDF")
            except Exception as e:
                logger.warning(f"need to set exceptions I should catch here")
                raise e
        return fd_logs

    def _get_temps(self):
        temps = None
        if "Temperatures" in self.logs_keys:
            with self.hdf_read as f:
                f = f.get(self._group_path)
                temps = Temperatures.from_hdf(f, "Temperatures")
        return temps

    def _get_mags(self):
        mags = Magnets()
        for axis in ["x", "y", "z"]:
            if f"Magnet {axis}" in self.logs_keys:
                with self.hdf_read as f:
                    f = f.get(self._group_path)
                    mag = Magnet.from_hdf(f, f"Magnet {axis}")
                setattr(mags, axis, mag)
        return mags

    def _get_all_keys(self):
        """Get all keys that are groups or attrs of top group"""
        keys = []
        with self.hdf_read as f:
            f = f.get(self._group_path)
            keys.extend(f.attrs.keys())
            keys.extend(f.keys())
        return keys

    @property
    def scan_vars(self):
        """Get the dictionary of Scan Vars from HDF File"""
        with self.hdf_read as f:
            scan_vars = json.loads(f["Logs"].attrs["scan_vars_string"])
        return scan_vars

    @property
    def get_sweepgates(self):
        scan_vars = self.scan_vars
        axis_gates = {}
        for axis in ["x", "y"]:
            starts = scan_vars.get(f"start{axis}s")
            fins = scan_vars.get(f"fin{axis}s")
            if starts == "null" or fins == "null":
                continue
            starts, fins = [
                tuple([float(v) for v in vals])
                for vals in [starts.split(","), fins.split(",")]
            ]
            channels = scan_vars.get(f"channels{axis}")
            channels = tuple([int(v) for v in channels.split(",")])
            numpts = scan_vars.get(f"numpts{axis}")
            channel_names = scan_vars.get(f"{axis}_label")
            if channel_names.endswith(" (mV)"):
                channel_names = channel_names[:-5]
            else:
                channel_names = ",".join([f"DAC{num}" for num in channels])
            channel_names = tuple([name.strip() for name in channel_names.split(",")])
            axis_gates[axis] = AxisGates(channels, channel_names, starts, fins, numpts)
        return SweepGates(**axis_gates)

@dataclass
class FastDAC(HDFStoreableDataclass):
    dac_vals: dict
    dac_names: dict
    adcs: dict
    sample_freq: float
    measure_freq: float
    AWG: dict
    visa_address: str

    @property
    def dacs(self):
        """Dict of {dacname: dacval}"""
        return {
            name: val
            for name, val in zip(self.dac_names.values(), self.dac_vals.values())
        }


@dataclass
class Temperatures(HDFStoreableDataclass):
    fiftyk: float
    fourk: float
    magnet: float
    still: float
    mc: float


@dataclass
class Magnet(HDFStoreableDataclass):
    axis: str
    field: float
    rate: float


@dataclass
class Magnets:
    x: Magnet = None
    y: Magnet = None
    z: Magnet = None

@dataclass(frozen=True)
class AxisGates:
    dacs: tuple[int]
    channels: tuple[str]
    starts: tuple[float]
    fins: tuple[float]
    numpts: int

    def to_df(self) -> pd.DataFrame:
        df = pd.DataFrame(
            [self.starts, self.fins, self.dacs],
            columns=self.channels,
            index=["starts", "fins", "dac"],
        )
        return df

    def values_at(self, value: float, channel: str = None) -> dict[str, float]:
        """Return dict of DAC values at given value of channel
        Args:
            value: DAC value of channel to evaluate other DAC values at
            channel: Which channel the value corresponds to (by default the first channel)
        """
        channel = channel if channel else self.channels[0]
        index = self.channels.index(channel)
        start = self.starts[index]
        fin = self.fins[index]

        proportion = (value - start) / (fin - start)
        return self.calculate_axis_gate_vals_from_sweep_proportion(proportion)

    def calculate_axis_gate_vals_from_sweep_proportion(self,
                                                       proportion
                                                       ) -> dict[str, float]:
        """
        Return dict of DAC values at proportion along sweep

        Args:
            proportion: Proportion of sweep to return values for (e.g. 0.0 is start of sweep, 1.0 is end of sweep)
        """
        return {
            k: s + proportion * (f - s)
            for k, s, f in zip(self.channels, self.starts, self.fins)
        }


@dataclass(frozen=True)
class SweepGates:
    x: AxisGates = None
    y: AxisGates = None

    def plot(self, axis: str = 'x', numpts=200):
        fig = go.Figure()
        df = getattr(self, axis).to_df()
        main_gate = df.T.iloc[0]
        main_x = np.linspace(main_gate.starts, main_gate.fins, numpts)
        for name, row in df.T.iterrows():
            y = np.linspace(row.starts, row.fins, numpts)
            fig.add_trace(go.Scatter(x=main_x, y=y, name=name))
        fig.update_layout(
            title=f"{axis} axis sweepgates",
            xaxis_title=f"Main sweep gate ({main_gate.name}) /mV",
            yaxis_title="Other Gate Values /mV",
            hovermode="x unified",
        )
        return fig

    def convert_to_real(self):
        """If gates have voltage dividers and are named e.g. "P*200", this will return the real applied voltages and gate names"""
        return convert_sweepgates_to_real(self)

def _dividers_from_gate_names(gate_names) -> np.ndarray:
    dividers = [
        float(re.search("\*(\d+)", gate_name).groups()[0]) for gate_name in gate_names
    ]
    return np.array(dividers)


def _gate_names_excluding_dividers(gate_names) -> list[str]:
    gates = [re.search("(.*)\*", gate_name).groups()[0] for gate_name in gate_names]
    return gates


def convert_sweepgates_to_real(sweepgates: SweepGates) -> SweepGates:
    """If gates have voltage dividers and are named e.g. "P*200", this will return the real applied voltages and gate names"""
    real_axis_gates = {}
    for axis in ["x", "y"]:
        axis_gates = getattr(sweepgates, axis)
        if axis_gates is not None:
            dividers = _dividers_from_gate_names(axis_gates.channels)
            real_starts = axis_gates.starts / dividers
            real_fins = axis_gates.fins / dividers
            real_channels = _gate_names_excluding_dividers(axis_gates.channels)
            real_axis_gates[axis] = AxisGates(
                dacs=axis_gates.dacs,
                channels=real_channels,
                starts=real_starts,
                fins=real_fins,
                numpts=axis_gates.numpts,
            )
    return SweepGates(**real_axis_gates)
