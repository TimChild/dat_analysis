from __future__ import annotations
from src import CoreUtil as CU
import src.DataStandardize.Standardize_Util as Util
import os
import h5py
import abc
import subprocess
from subprocess import PIPE
import logging
from typing import TYPE_CHECKING, Optional, Tuple
from dataclasses import dataclass

if TYPE_CHECKING:
    from src.DFcode.SetupDF import SetupDF

logger = logging.getLogger(__name__)


@dataclass
class Directories(object):
    """For keeping directories together in Config.Directories"""
    hdfdir: Optional[str] = None  # DatHDFs (My HDFs)
    ddir: Optional[str] = None  # Experiment data
    dfsetupdir: Optional[str] = None  # SetupDF
    dfbackupdir: Optional[str] = None  # Where SetupDF is backed up to

    def set_dirs(self, hdfdir, ddir, dfsetupdir, dfbackupdir):
        """Should point to the real folders (i.e. after any substitutions for shortcuts etc)"""
        self.hdfdir = hdfdir
        self.ddir = ddir
        self.dfsetupdir = dfsetupdir
        self.dfbackupdir = dfbackupdir


class SysConfigBase(abc.ABC):
    """
    Base SysConfig class to outline what info needs to be in system specific config

    This is for things like where to find data, where to save data, how to synchronize data etc
    """
    def __init__(self, datnum: Optional[int] = None):
        self.Directories = Directories()
        self.set_directories(datnum=datnum)

    @property
    @abc.abstractmethod
    def main_folder_path(self) -> str:
        """ Override to return a string of the path to the main folder where all experiments are saved"""
        pass

    @property
    @abc.abstractmethod
    def dir_name(self) -> str:
        """Name to use inside of main_folder_path"""
        return ''

    @abc.abstractmethod
    def set_directories(self, datnum: Optional[int] = None):
        """Something that sets self.Directories with the relevant paths"""
        hdfdir, ddir, dfsetupdir, dfbackupdir = self.get_expected_sub_dir_paths(
            os.path.join(self.main_folder_path, self.dir_name))
        self.Directories.set_dirs(hdfdir, ddir, dfsetupdir, dfbackupdir)
        # datnum is here for potential future use. Can't decide whether I should use datnum for each method, or init
        # whole class with datnum and use that. At the moment I don't really see an advantage either wayV
        pass

    @staticmethod
    def get_expected_sub_dir_paths(base_path: str) -> Tuple[str, str, str, str]:
        """
        Helper method to get the usual directories given a base_path. Takes care of looking at shortcuts etc

        Args:
            base_path (str):

        Returns:
            Tuple[str, str, str, str]: The standard paths that Directories needs to be fully initialized
        """
        hdfdir = os.path.join(base_path, 'Dat_HDFs')
        ddir = os.path.join(base_path, 'Experiment_Data')
        dfsetupdir = os.path.join(base_path, 'DataFrames/setup/')
        dfbackupdir = os.path.join(base_path, 'DataFramesBackups')

        # Replace paths with shortcuts with real paths
        hdfdir = CU.get_full_path(hdfdir, None)
        ddir = CU.get_full_path(ddir, None)
        dfsetupdir = CU.get_full_path(dfsetupdir, None)
        dfbackupdir = CU.get_full_path(dfbackupdir, None)
        return hdfdir, ddir, dfsetupdir, dfbackupdir

    @abc.abstractmethod
    def synchronize_data_batch_file(self, datnum: Optional[int] = None) -> str:
        """Path to a batch file which will synchronize data from experiment PC to local data folder"""
        #  e.g.  path = r'D:\OneDrive\UBC LAB\Machines\Remote Connections\WinSCP Scripts\Jun20.bat'
        path = ''
        return path


class ExpConfigBase(abc.ABC):
    """
    Base Config class to outline what info needs to be in any exp/system specific config
    """

    def __init__(self, datnum=None):
        # datnum is here for potential future use. Can't decide whether I should use datnum for each method, or init
        # whole class with datnum and use that. At the moment I don't really see an advantage either wayV
        pass

    @abc.abstractmethod
    def get_sweeplogs_json_subs(self,  datnum: Optional[int] = None):
        """Something that returns a list of re match/repl strings to fix sweeplogs JSON for a given datnum
        [(match, repl), (match, repl),..]"""
        return [('FastDAC 1', 'FastDAC')]

    @abc.abstractmethod
    def get_dattypes_list(self, datnum: Optional[int] = None) -> set:
        """Something that returns a list of dattypes that exist in experiment"""
        return {'none', 'entropy', 'transition', 'square entropy'}

    @abc.abstractmethod
    def get_exp_names_dict(self, datnum: Optional[int] = None) -> dict:
        """Override to return a dictionary of experiment wavenames for each standard name
        standard names are: i_sense, entx, enty, x_array, y_array"""
        d = dict(x_array=['x_array'], y_array=['y_array'],
                 i_sense=['cscurrent', 'cscurrent_2d'])
        return d


class Exp2HDF(abc.ABC):
    """Base class for standard functions going from Experiment to my standard of data for Builders
    Then Builders are responsible for making my Dats.

    This will also interact with both ExpConfig and SysConfig (and can pass datnums for future proofing)
    """

    def __init__(self, datnum):
        """ Basic info to go from exp data to Dat

        Args:
            datnum (int): Datnum
        """
        self.datnum = datnum
        self._dat_types = None

    @property
    @abc.abstractmethod
    def setupdf(self) -> SetupDF:
        """override to return a SetupDF for the Experiment"""
        return SetupDF()

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

    def _get_update_batch_path(self):
        """Returns path to update_batch.bat file to update local data from remote"""
        path: str = self.SysConfig.synchronize_data_batch_file(datnum=self.datnum)
        if path is None:
            logger.warning(f'No path found to batch file for synchronizing remote data')
        else:
            return path

    def synchronize_data(self):
        """Run to update local data folder from remote"""
        path = self._get_update_batch_path()
        if path is not None:
            subprocess.call(path)

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

    @property
    def dat_types(self) -> set:
        return self._get_dat_types()

    def _get_dat_types(self):
        if self._dat_types is None:  # Only load dattypes the first time, then store
            sweep_logs = self.get_sweeplogs()
            comments = sweep_logs.get('comment', None)
            dat_types_list = self.ExpConfig.get_dattypes_list()
            self._dat_types = Util.get_dattypes(None, comments, dat_types_list)
        return self._dat_types

    @dat_types.setter
    def dat_types(self, dattypes):
        """For forcing the dattypes to be something other than what is returned by get_dattypes"""
        self._set_dat_types(dattypes)

    def _set_dat_types(self, dattypes):
        if dattypes is not None:
            self._dat_types = dattypes

    @abc.abstractmethod
    def get_sweeplogs(self) -> dict:
        """If this fails you need to override to make it work"""
        path = self.get_exp_dat_path()
        with h5py.File(path, 'r') as dat_hdf:
            sweeplogs = dat_hdf['metadata'].attrs['sweep_logs']
            sweeplogs = Util.replace_in_json(sweeplogs, self.ExpConfig.get_sweeplogs_json_subs(self.datnum))
            sweeplogs = Util.clean_basic_sweeplogs(sweeplogs)  # Simple changes which apply to many exps
        return sweeplogs

    def get_hdfdir(self):
        return self.SysConfig.Directories.hdfdir

    def get_ddir(self):
        return self.SysConfig.Directories.ddir

    def get_datHDF_path(self, name='base'):
        dat_id = CU.get_dat_id(self.datnum, name)
        return os.path.join(self.get_hdfdir(), dat_id + '.h5')

    def get_exp_dat_path(self):
        return os.path.join(self.get_ddir(), f'Dat{self.datnum}.h5')

    def get_data_setup_dict(self):
        exp_names_dict = self.ExpConfig.get_exp_names_dict()
        sweep_logs = self.get_sweeplogs()
        dattypes = self.dat_types
        setup_dict = Util.get_data_setup_dict(self.datnum, dattypes, self.setupdf, exp_names_dict, sweep_logs)
        return setup_dict

    def get_name(self, datname):
        name = CU.get_dat_id(self.datnum, datname)
        return name


# TODO: Make this class -- It should basically be above, but with the ability to manually enter any necessary info
# TODO: to make at least a temporary dat.
class File2HDF(Exp2HDF):
    pass
