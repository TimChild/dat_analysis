import abc
import os
import src.CoreUtil as CU

main_data_path = 'D:\\OneDrive\\UBC LAB\\My work\\Fridge_Measurements_and_Devices\\Fridge Measurements with PyDatAnalysis'


class Directories(object):
    """For keeping directories together"""
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
        """Something that returns a list of re match/repl strings to fix sweeplogs JSON for a given datnum"""
        pass

    @abc.abstractmethod
    def get_dattypes_list(self):
        """Something that returns a list of dattypes that exist in experiment"""
        pass

    def get_json_subs(self):
        """Something that returns a list of tuples of json_subs that need to be made to make jsons valid
        [(match, repl), (match, repl),..]"""
        return []

    @abc.abstractmethod
    def get_exp_names_dict(self):
        """Override to return a dictionary of experiment wavenames for each standard name
        standard names are: i_sense, entx, enty, x_array, y_array"""