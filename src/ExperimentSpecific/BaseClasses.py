import src.Configs.ConfigBase
from src.DFcode.SetupDF import SetupDF
from src.Configs.ConfigBase import ConfigBase
from src.DatBuilder import Util

import os
import h5py
import abc


class ExperimentSpecificInterface(abc.ABC):
    """Base class for standard functions going from Experiment to Dat"""

    def __init__(self, datnum):
        """ Basic info to go from exp data to Dat

        Args:
            datnum (int): Datnum
        """
        self.datnum = datnum
        self.setupdf = None
        self.Config = None
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
        sweeplogs = Util.replace_in_json(sweeplogs, self.Config.get_json_subs())
        return sweeplogs

    def get_hdfdir(self):
        return self.Config.Directories.hdfdir

    def get_ddir(self):
        return self.Config.Directories.ddir

    def get_HDF_path(self, name='base'):
        dat_id = Util.get_dat_id(self.datnum, name)
        return os.path.join(self.get_ddir(), dat_id+'.h5')

    def get_data_setup_dict(self):
        exp_names_dict = self.Config.get_exp_names_dict()
        sweep_logs = self.get_sweeplogs()
        dattypes = self.get_dattypes()
        setup_dict = Util.get_data_setup_dict(self.datnum, dattypes, self.setupdf, exp_names_dict, sweep_logs)
        return setup_dict
