"""
2022-06: Trying to make a simpler Dat interface.

The aim of the DatHDF class is to provide an easy interface to the HDF files that are made here (i.e. not the experiment
files directly which differ too much from experiment to experiment)
"""
from __future__ import annotations
import os.path
import tempfile
import importlib.machinery
import re
from typing import Callable, Optional, Union
import logging
from dataclasses import dataclass, field
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import h5py

from .build_dat_hdf import check_hdf_meets_requirements, default_exp_to_hdf

from .data_attr import Data
from .logs_attr import Logs

from ..hdf_file_handler import HDF, GlobalLock
from ..hdf_util import NotFoundInHdfError
from .dat_util import get_local_config
from ..core_util import get_full_path, slugify

logger = logging.getLogger(__name__)


class DatHDF(HDF):
    def __init__(self, hdf_path: str):
        super().__init__(hdf_path)
        passed_checks, message = check_hdf_meets_requirements(hdf_path)
        if not passed_checks:
            raise Exception(
                f"DatHDF at {hdf_path} does not meet requirements:\n{message}"
            )

        self.Logs = Logs(hdf_path, "/Logs")
        self.Data = Data(hdf_path, "/Data")
        # self.Analysis = ... maybe add some shortcuts to standard analysis stuff? Fitting etc

        self._datnum = None

    def __repr__(self):
        """Give a more useful summary of DatHDF e.g. when left at end of jupyter cell"""
        return f"Dat{self.datnum} - {self.Logs.time_completed}"

    @property
    def datnum(self):
        if not self._datnum:
            with self.hdf_read as f:
                self._datnum = f.attrs.get("datnum", -1)
        return self._datnum

    @property
    def standard_fig(self: DatHDF):
        """
        Shortcut to getting a Figure with x_label, y_label, and title with standard layout
        Returns
            go.Figure: With default layout and xaxis_title, yaxis_title, title etc
        """
        from ..plotting.plotly.util import (
            default_fig,
        )  # Avoiding circular imports, and this is just a shortcut anyway

        fig = default_fig()
        fig.update_layout(
            xaxis_title=self.Logs.x_label,
            yaxis_title=self.Logs.y_label,
            title=f"Dat{self.datnum}: ",
        )
        return fig

    def get_Data(self, key) -> Data:
        """
        Get more complete Data object directly from Dat

        I.e. including x (and y) axes, x_label, y_label, title (with datnum)
        """
        from ..analysis_tools.data import (
            Data,
            PlottingInfo,
        )  # Avoiding circular imports. This is only intended as a

        # shortcut anyway
        keys = self.Data._get_all_data_keys()
        if key in keys:
            data = self.Data._load_data(key)
        else:
            raise NotFoundInHdfError(
                f"{key} not found. Existing keys are {self.Data.data_keys}"
            )
        data = Data(
            data=data,
            x=self.Data.x,
            y=self.Data.y,
            plot_info=PlottingInfo(
                x_label=self.Logs.x_label,
                y_label=self.Logs.y_label,
                title=f"Dat{self.datnum}: {key}",
            ),
        )
        return data

    def plot(self, fig: go.Figure, overwrite=False) -> go.Figure:
        """Adds extra info to figure from Dat file and also saves the figure to the HDF"""
        # TODO: Decide where to put "Saved in..." annotation based on Figure height (for larger height, smaller number works better)
        if not np.any([annotation.y == -0.15 for annotation in fig.layout.annotations]):
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=1.0,
                y=-0.15,
                text=f"Saved in Dat{self.datnum}",
                showarrow=False,
            )
        self.save_fig(fig, overwrite=overwrite)
        return fig

    def save_fig(
        self, fig: go.Figure, filename: str = None, overwrite=False
    ) -> FigInfo:
        """Save Figure to HDF
        Args:
            filename: optionally provide the name to store under (defaults to fig title)
        """
        assert isinstance(fig, go.Figure)
        fig_info = FigInfo.from_fig(fig, filename=filename)

        if not overwrite:
            # Avoid entering write mode if not necessary
            existing = self._load_fig_info(filename=fig_info.filename, load_fig=False)
            if existing == fig_info:
                logging.info(
                    f"Fig ({fig_info.filename}) already saved in Dat{self.datnum}, to overwrite, set `overwrite` = True"
                )
                return fig_info
            elif existing:
                logging.info(
                    f"Ovewriting Fig ({fig_info.filename}) in Dat{self.datnum}. Existing fig had same title but was different"
                )
        # Write fig to HDF
        with self.hdf_write as f:
            fig_group = f.require_group("Figs")
            fig_info.to_group(parent_group=fig_group, overwrite=overwrite)
        logging.info(f"Fig ({fig_info.filename}) saved in Dat{self.datnum}")
        return fig_info

    def load_fig(self, filename: str = None) -> Optional[go.Figure]:
        """Load the named fig from HDF, or if None, the last saved fig"""
        fig_info = self._load_fig_info(filename=filename)
        if fig_info:
            return fig_info.fig
        return None

    def saved_figs(self, load_figs=True) -> Optional[FigInfos]:
        """Load saved Figs from HDF
        Args:
            load_figs: If False will load the attrs only (fast), otherwise will also load the full figure
        """
        fig_infos = None
        with self.hdf_read as f:
            if "Figs" in f.keys():
                fig_infos = FigInfos.from_group(f["Figs"], load_figs=load_figs)
        return fig_infos

    def _load_fig_from_hdf(self, filename, load_fig=True) -> FigInfo:
        with self.hdf_read as f:
            if "Figs" in f.keys():
                fig_group = f["Figs"]
                if filename in fig_group.keys():
                    fig_info = FigInfo.from_group(
                        fig_group[filename], load_fig=load_fig
                    )
                    return fig_info
        return None

    def _load_fig_info(self, filename: str = None, load_fig=True) -> Optional[FigInfo]:
        """Load the named fig from HDF, or if None, the last saved fig"""
        # Get Info on Saved Figs
        saved = self.saved_figs(load_figs=False)

        # If nothing saved, return None
        if not saved or not saved.infos:
            return None

        # If no filename specified, get latest saved fig
        filename = filename if filename else saved.latest_fig.filename

        # If present, load it
        if filename in saved.filenames:
            fig_info = self._load_fig_from_hdf(filename, load_fig=load_fig)
            return fig_info
        return None


def get_dat(
    datnum: int,
    # host_name = None, user_name = None, experiment_name = None,
    host_name,
    user_name,
    experiment_name,
    raw=False,
    overwrite=False,
    override_save_path=None,
    **loading_kwargs,
) -> DatHDF:
    """
    Function to help with loading DatHDF object.

    Note: Not all arguments should be provided from (datnum, host_name, user_name, experiment_name)
        Either:
            - datnum only: Uses the 'current_experiment_path' in config.toml to decide where to look
            - datnum, host_name, user_name, experiment_name: Look for dat at 'path_to_measurement_data/host_name/user_name/experiment_name/dat{datnum}.h5'
        To load from direct data path, use 'get_dat_from_exp_filepath' instead.
        To load from already existing DatHDF.h5 without checking for existing experiment data etc, initialize DatHDF directly with DatHDF(hdf_path=filepath, mode='r')

    Args:
        datnum (): Optionally provide datnum
        host_name (): Optionally provide host_name of computer measurement was taken from (e.g. qdev-xld)
        user_name (): Optionally provide the user_name the data was stored under (i.e. matching how it appears in the server file structure)
        experiment_name (): Optionally provide the folder (or path to folder using '/' to join folders) of where to find data under host_name/user_name/... (e.g. 'tests/test1_date/')
        raw (): Bool to specify whether to load datXX_RAW.h5 or datXX.h5 (defaults to False)
        overwrite (): Whether to overwrite a possibly existing DatHDF (defaults to False)
        override_save_path (): Optionally override the path where the DatHDF will be looked for/stored
        **loading_kwargs (): Any other args that are needed in order to load the dat.h5

    Returns:
        DatHDF: A python object for easy interaction with a standardized HDF file.
    """
    config = get_local_config()
    # host_name = host_name if host_name else config['loading']['default_host_name']
    # user_name = user_name if user_name else config['loading']['default_user_name']
    # experiment_name = experiment_name if experiment_name else config['loading']['default_experiment_name']

    # Get path to directory containing datXX.h5 files
    exp_path = os.path.join(host_name, user_name, experiment_name)

    # Get path to specific datXX.h5 file and check it exists
    measurement_data_path = config["loading"]["path_to_measurement_data"]
    if raw is True:
        filepath = os.path.join(measurement_data_path, exp_path, f"dat{datnum}_RAW.h5")
    else:
        filepath = os.path.join(measurement_data_path, exp_path, f"dat{datnum}.h5")
    filepath = get_full_path(filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath}")

    return get_dat_from_exp_filepath(
        filepath,
        overwrite=overwrite,
        override_save_path=override_save_path,
        **loading_kwargs,
    )


def get_dat_from_exp_filepath(
    experiment_data_path: str,
    overwrite: bool = False,
    override_save_path: Optional[str] = None,
    override_exp_to_hdf: Optional[Callable] = None,
    **loading_kwargs,
) -> DatHDF:
    """
    Get a DatHDF for given experiment data path... Uses experiment data path to decide where to save DatHDF if
    override_save_path not provided

    Args:
        experiment_data_path (): Path to the hdf file as saved at experiment time (i.e. likely some not quite standard format)
        overwrite (): Whether to overwrite an existing DatHDF file
        override_save_path (): Optionally provide a string path of where to save DatHDF (or look for existing DatHDF)
        override_exp_to_hdf (): Optionally provide a function which will be used to turn the experiment hdf file into the standardized DatHDF file.
            Function should take arguments (experiment_data_path, save_path, **kwargs) and need not return anything.
        **loading_kwargs (): Any other args that are needed in order to load the dat.h5

    Returns:

    """

    experiment_data_path = get_full_path(experiment_data_path)
    if override_save_path:
        override_save_path = get_full_path(override_save_path)

    # Figure out path to DatHDF (existing or not)
    if override_save_path is None:
        save_path = save_path_from_exp_path(experiment_data_path)
    elif isinstance(override_save_path, str):
        if os.path.isdir(override_save_path):
            raise IsADirectoryError(
                f"To override_save_path, must specify a full path to a file not a directory. Got ({override_save_path})"
            )
        save_path = override_save_path
    else:
        raise ValueError(
            f"If providing 'override_save_path' it should be a path string. Got ({override_save_path}) instead"
        )

    # If already existing, return or delete if overwriting
    if os.path.exists(save_path) and os.path.isfile(save_path) and not overwrite:
        return DatHDF(hdf_path=save_path)

    # If not already returned, then create new standard DatHDF file from non-standard datXX.h5 file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    lock = GlobalLock(
        save_path + ".init.lock"
    )  # Can't just use '.lock' as that is used by FileQueue
    # lock = GlobalLock(save_path+'.lock')
    with (
        lock
    ):  # Only one thread/process should be doing this for any specific save path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            if overwrite:
                os.remove(
                    save_path
                )  # Possible thaht this was just created by another thread, but if trying to overwrite, still better to remove again
            else:
                return DatHDF(
                    hdf_path=save_path
                )  # Must have been created whilst this thread was waiting
        if override_exp_to_hdf is not None:  # Use the specified function to convert
            override_exp_to_hdf(experiment_data_path, save_path, **loading_kwargs)
        elif (
            get_local_config()
            and get_local_config()["loading"]["path_to_python_load_file"]
        ):  # Use the file specified in config to convert
            # module = importlib.import_module(config['loading']['path_to_python_load_file'])
            config = get_local_config()
            module = importlib.machinery.SourceFileLoader(
                "python_load_file", config["loading"]["path_to_python_load_file"]
            ).load_module()
            fn = module.create_standard_hdf
            fn(experiment_data_path, save_path, **loading_kwargs)
        else:  # Do a basic default convert
            default_exp_to_hdf(experiment_data_path, save_path)

    # Return a DatHDF object from the standard DatHDF.h5 file
    return DatHDF(hdf_path=save_path)


def save_path_from_exp_path(experiment_data_path: str) -> str:
    config = get_local_config()
    match = re.search(r"measurement[-_]data[\\:/]*(.+)", experiment_data_path)
    after_measurement_data = (
        match.groups()[0]
        if match
        else re.match(
            r"[\\:/]*(.+)", os.path.splitdrive(experiment_data_path)[1]
        ).groups()[0]
    )  # TODO: better way to handle this? This could make some crazy file locations...
    save_path = get_full_path(
        os.path.join(
            config["loading"]["path_to_save_directory"],
            os.path.normpath(after_measurement_data),
        )
    )
    return save_path


@dataclass(frozen=True)
class FigInfo:
    """Info about a Figure for saving/loading from HDF"""

    filename: str
    time_saved: pd.Timestamp = field(compare=False)
    title: str
    trace_type: str
    datashape: Union[tuple[int], None]
    hdf_path: str = field(default=None, compare=False)
    fig: Optional[go.Figure] = field(default=None, compare=False)

    @classmethod
    def from_fig(cls, fig: go.Figure, filename: str = None) -> FigInfo:
        info = {}
        if filename is None:
            if fig.layout.title.text:
                filename = fig.layout.title.text
            else:
                raise ValueError(
                    f"If the fig has no title, a filename must be passed, got neither."
                )

        # Make filename filename safe
        filename = slugify(filename, allow_unicode=True)
        info["filename"] = filename
        info["time_saved"] = pd.Timestamp.now()
        info["title"] = fig.layout.title.text if fig.layout.title.text else ""
        info["trace_type"] = str(type(fig.data[0])) if fig.data else ""
        if fig.data:
            trace = fig.data[0]
            data = getattr(trace, "z", None)
            if data is None:
                data = getattr(trace, "y", None)
            if data is not None:
                info["datashape"] = tuple(data.shape)
        info["fig"] = fig
        return cls(**info)

    @classmethod
    def from_group(cls, group: h5py.Group, load_fig=True) -> FigInfo:
        """Read from group to build instance of FigInfo
        Note: Read only!
        """
        if group.attrs.get("dataclass", None) != "FigInfo":
            raise NotFoundInHdfError(f"{group.name} is not a saved FigInfo")

        info = {
            k: group.attrs.get(k, None) for k in ["filename", "title", "trace_type"]
        }
        info["time_saved"] = pd.Timestamp(group.attrs["time_saved"])
        info["datashape"] = tuple(group.attrs.get("datashape", tuple()))
        if load_fig:
            info["fig"] = pio.from_json(group.attrs["fig"])
        info["hdf_path"] = group.file.filename
        return cls(**info)

    def to_group(self, parent_group: h5py.Group, overwrite=False):
        """Write to group everything that is needed to load again
        Note: Write allowed. Try not to write big things if not necessary (I think HDFs do not reclaim reused space)
        """
        if not self.fig:
            raise ValueError(
                f"This FigInfo does not contain a go.Figure (`self.fig = None`). Not saving"
            )
        if overwrite:
            if self.filename in parent_group.keys():
                logging.info(
                    f"Overwriting {self.filename} in {parent_group.file.filename}"
                )
                del parent_group[self.filename]
        group = parent_group.require_group(self.filename)
        if group.attrs.get("dataclass", None) != "FigInfo":
            group.attrs["dataclass"] = "FigInfo"
        for k in ["filename", "title", "trace_type"]:
            if group.attrs.get(k, None) != getattr(self, k) or overwrite:
                group.attrs[k] = getattr(self, k)

        if pd.Timestamp(group.attrs.get("time_saved", None)) != self.time_saved:
            group.attrs["time_saved"] = str(self.time_saved)

        if tuple(group.attrs.get("datashape", tuple())) != self.datashape:
            group.attrs["datashape"] = self.datashape

        fig_json = self.fig.to_json()
        if group.attrs.get("fig", None) != fig_json:
            group.attrs["fig"] = "" if self.fig is None else fig_json
        return self


@dataclass
class FigInfos:
    """Collection of FigInfo objects for looking at all saved Figs in HDF"""

    infos: tuple[FigInfo]
    latest_fig: FigInfo

    @property
    def filenames(self) -> list[str]:
        return [info.filename for info in self.infos]

    @classmethod
    def from_group(cls, fig_group: h5py.Group, load_figs=True) -> FigInfos:
        """
        Get all FigInfos saved in HDF group (optionally there info only if load_figs is False (fast))
        Note: Read only!
        """
        fig_infos = []
        for k in fig_group.keys():
            single_fig_group = fig_group[k]
            if single_fig_group.attrs.get("dataclass", None) == "FigInfo":
                fig_info = FigInfo.from_group(single_fig_group, load_fig=load_figs)
                fig_infos.append(fig_info)
        if not fig_infos:
            return None

        # Order newest first
        fig_infos = tuple(reversed(sorted(fig_infos, key=lambda info: info.time_saved)))
        latest_fig = fig_infos[0]
        inst = cls(fig_infos, latest_fig)
        return inst
