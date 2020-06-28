from __future__ import annotations
from src import CoreUtil as CU
import src.DataStandardize.Standardize_Util as Util
import os
import h5py
import abc
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.DFcode.SetupDF import SetupDF

from src.Main_Config import main_data_path


class Directories(object):
    """For keeping directories together in Config.Directories"""
    def __init__(self):
        self.hdfdir = None  # DatHDFs saves
        self.ddir = None  # Experiment data
        self.dfsetupdir = None  # SetupDF
        self.dfbackupdir = None  # Where SetupDF is backed up to

    def set_dirs(self, hdfdir, ddir, dfsetupdir, dfbackupdir):
        """Should point to the real folders (i.e. after any substitutions for shortcuts etc)"""
        self.hdfdir = hdfdir
        self.ddir = ddir
        self.dfsetupdir = dfsetupdir
        self.dfbackupdir = dfbackupdir


class ConfigBase(abc.ABC):
    """
    Base Config class to outline what info needs to be in any exp specific config
    """
    def __init__(self):
        self.Directories = Directories()
        self.main_folder_path = main_data_path
        self.set_directories()

    @property
    @abc.abstractmethod
    def dir_name(self):
        """Required attribute of subclass, doesn't need to be a whole property!"""
        return

    @staticmethod
    def get_expected_sub_dir_paths(base_path):
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
    def set_directories(self):
        """Something that sets self.Directories"""
        pass

    @abc.abstractmethod
    def get_sweeplogs_json_subs(self, datnum):
        """Something that returns a list of re match/repl strings to fix sweeplogs JSON for a given datnum
        [(match, repl), (match, repl),..]"""
        pass

    @abc.abstractmethod
    def get_dattypes_list(self):
        """Something that returns a list of dattypes that exist in experiment"""
        pass

    @abc.abstractmethod
    def get_exp_names_dict(self):
        """Override to return a dictionary of experiment wavenames for each standard name
        standard names are: i_sense, entx, enty, x_array, y_array"""


class ExperimentSpecificInterface(abc.ABC):
    """Base class for standard functions going from Experiment to my standard of data for Builders
    Then Builders are responsible for making my Dats"""

    def __init__(self, datnum):
        """ Basic info to go from exp data to Dat

        Args:
            datnum (int): Datnum
        """
        self.datnum = datnum
        self.setupdf = None
        self.Config: ConfigBase = None
        self.set_setupdf()
        self.set_Config()
        self._dattypes = None

    @abc.abstractmethod
    def set_setupdf(self) -> SetupDF:
        """override to return a SetupDF for the Experiment"""
        pass

    @abc.abstractmethod
    def set_Config(self) -> ConfigBase:
        """Override to return a config for the Experiment"""
        pass

    def set_dattypes(self, dattypes):
        """May want to override to prevent just overwriting existing dattypes"""
        if dattypes is not None:
            self._dattypes = dattypes

    def get_dattypes(self) -> set:
        if self._dattypes is None:
            sweep_logs = self.get_sweeplogs()
            comments = sweep_logs.get('comment', None)
            dat_types_list = self.Config.get_dattypes_list()
            self._dattypes = Util.get_dattypes(None, comments, dat_types_list)
        return self._dattypes

    def get_exp_dat_hdf(self):
        ddir = self.Config.Directories.ddir
        path = os.path.join(ddir, f'dat{self.datnum:d}.h5')
        dat_hdf = h5py.File(path, 'r')
        return dat_hdf

    @abc.abstractmethod
    def get_sweeplogs(self) -> dict:
        """If this fails you need to override to make it work"""
        dat_hdf = self.get_exp_dat_hdf()
        sweeplogs = dat_hdf['metadata'].attrs['sweep_logs']
        sweeplogs = Util.replace_in_json(sweeplogs, self.Config.get_sweeplogs_json_subs(self.datnum))
        return sweeplogs

    def get_hdfdir(self):
        return self.Config.Directories.hdfdir

    def get_ddir(self):
        return self.Config.Directories.ddir

    def get_HDF_path(self, name='base'):
        dat_id = CU.get_dat_id(self.datnum, name)
        return os.path.join(self.get_ddir(), dat_id+'.h5')

    def get_data_setup_dict(self):
        exp_names_dict = self.Config.get_exp_names_dict()
        sweep_logs = self.get_sweeplogs()
        dattypes = self.get_dattypes()
        setup_dict = Util.get_data_setup_dict(self.datnum, dattypes, self.setupdf, exp_names_dict, sweep_logs)
        return setup_dict


