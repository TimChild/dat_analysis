from __future__ import annotations

import DatObject.Attributes.Logs
from src.DatObject.Attributes.Logs import replace_in_json
from src import CoreUtil as CU
import os
import h5py
import abc
import subprocess
from subprocess import PIPE
import logging
from typing import TYPE_CHECKING, Optional, Tuple
from dataclasses import dataclass
import pathlib
from src.DataStandardize.ExpConfig import ExpConfigBase

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class Directories:
    """For keeping directories together in Config.Directories"""
    hdfdir: Optional[str] = None  # DatHDFs (My HDFs)
    ddir: Optional[str] = None  # Experiment data


def get_expected_sub_dir_paths(base_path: str) -> Tuple[str, str]:
    """
    Helper method to get the usual directories given a base_path. Takes care of looking at shortcuts etc

    Args:
        base_path (str):

    Returns:
        Tuple[str, str, str, str]: The standard paths that Directories needs to be fully initialized
    """
    hdfdir = os.path.join(base_path, 'Dat_HDFs')
    ddir = os.path.join(base_path, 'Experiment_Data')

    # Replace paths with shortcuts with real paths
    paths = []
    for path in [hdfdir, ddir]:
        try:
            paths.append(CU.get_full_path(path))
        except FileNotFoundError:
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            paths.append(os.path.abspath(path))
    hdfdir, ddir, = [path for path in paths]
    return hdfdir, ddir


class SysConfigBase(abc.ABC):
    """
    Base SysConfig class to outline what info needs to be in system specific config

    This is for things like where to find data, where to save data, how to synchronize data etc
    """
    def __init__(self, datnum: Optional[int] = None):
        self.datnum = datnum

    @property
    @abc.abstractmethod
    def main_folder_path(self) -> str:
        """ Override to return a string of the path to the main folder where all experiments are saved"""
        return r'D:\OneDrive\UBC LAB\My work\Fridge_Measurements_and_Devices\Fridge Measurements with PyDatAnalysis'

    @property
    @abc.abstractmethod
    def dir_name(self) -> str:
        """Name to use inside of main_folder_path"""
        return ''

    @property
    def Directories(self) -> Directories:
        return self.get_directories()

    def get_directories(self):
        """Something that sets self.Directories with the relevant paths"""
        hdfdir, ddir = get_expected_sub_dir_paths(
            os.path.join(self.main_folder_path, self.dir_name))
        return Directories(hdfdir, ddir)

    @abc.abstractmethod
    def synchronize_data_batch_file(self) -> str:
        """Path to a batch file which will synchronize data from experiment PC to local data folder"""
        #  e.g.  path = r'D:\OneDrive\UBC LAB\Machines\Remote Connections\WinSCP Scripts\Jun20.bat'
        path = ''
        return path


class Exp2HDF(abc.ABC):
    """Base class for standard functions going from Experiment to my standard of data for Builders
    Then Builders are responsible for making my Dats.

    This will also interact with both ExpConfig and SysConfig (and can pass datnums for future proofing)
    """

    def __init__(self, datnum, datname='base'):
        """ Basic info to go from exp data to Dat

        Args:
            datnum (int): Datnum
        """
        self.datnum = datnum
        self.datname = datname

    @property
    @abc.abstractmethod
    def ExpConfig(self) -> ExpConfigBase:
        """Override to return a ExpConfig for the Experiment"""
        return ExpConfigBase(self.datnum)

    @property
    @abc.abstractmethod
    def SysConfig(self) -> SysConfigBase:
        """Override to return a SysConfig for the Experiment"""
        return SysConfigBase(self.datnum)

    def synchronize_data(self):
        """Run to update local data folder from remote"""
        path = self._get_update_batch_path()
        if path is not None:
            subprocess.call(path)

    def get_dat_types_from_comments(self) -> set:  # TODO: Is this actually used?
        sweep_logs = self.get_sweeplogs()
        comments = sweep_logs.get('comment', None)
        possible_dat_types = self.ExpConfig.get_possible_dat_types()
        return self._get_dat_type_from_comments(possible_dat_types, comments)

    @staticmethod
    def _get_dat_type_from_comments(possible_dat_types, comments):
        """Something which should return a list of the dat types which are in comments (Doesn't need dependencies
        they will be created automatically, although it doesn't hurt to have them)"""
        raise NotImplemented  # TODO: Implement this

    @abc.abstractmethod
    def get_sweeplogs(self) -> dict:
        """If this fails you need to override to make it work"""
        path = self.get_exp_dat_path()
        with h5py.File(path, 'r') as dat_hdf:
            sweeplogs = dat_hdf['metadata'].attrs['sweep_logs']
            sweeplogs = replace_in_json(sweeplogs, self.ExpConfig.get_sweeplogs_json_subs())
        return sweeplogs

    def get_hdfdir(self):
        return self.SysConfig.Directories.hdfdir

    def get_ddir(self):
        return self.SysConfig.Directories.ddir

    def get_datHDF_path(self):
        dat_id = CU.get_dat_id(self.datnum, self.datname)
        return os.path.join(self.get_hdfdir(), dat_id + '.h5')

    def get_exp_dat_path(self):
        return os.path.join(self.get_ddir(), f'dat{self.datnum}.h5')

    def get_name(self, datname):
        name = CU.get_dat_id(self.datnum, datname)
        return name

    def _get_update_batch_path(self):
        """Returns path to update_batch.bat file to update local data from remote"""
        path = self.SysConfig.synchronize_data_batch_file()
        if path is None:
            logger.warning(f'No path found to batch file for synchronizing remote data')
        else:
            return path

    def _check_data_exists(self, suppress_output=False):
        """Checks whether the Exp dat file exists. If not and update_batch is not None, will run update_batch to
        synchronize data"""
        hdfpath = os.path.join(self.get_ddir(), f'dat{self.datnum:d}.h5')
        update_batch = self._get_update_batch_path()
        if os.path.isfile(hdfpath):
            return True
        elif update_batch is not None:
            if os.path.isfile(update_batch):
                stdout = PIPE if suppress_output else None
                comp_process = subprocess.run(update_batch, shell=True, stdout=stdout)
                if os.path.isfile(hdfpath):
                    return True
                else:
                    if suppress_output is False:
                        logger.warning(
                            f'Tried updating local data folder, but still can\'t find Exp data for dat{self.datnum}')
                    return False
            else:
                raise FileNotFoundError(f'Path to update_batch.bat in config in but not found:\r {update_batch}')
        else:
            return False


# TODO: Make this class -- It should basically be above, but with the ability to manually enter any necessary info
# TODO: to make at least a temporary dat.. Also should load most things from a config file (json or something).
class File2HDF(Exp2HDF):
    pass
