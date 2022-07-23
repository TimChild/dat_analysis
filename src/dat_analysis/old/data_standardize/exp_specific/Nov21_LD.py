from __future__ import annotations
from ..base_classes import Exp2HDF, SysConfigBase, Directories
from ..exp_config import ExpConfigBase

from .Nov21 import Nov21ExpConfig, Nov21SysConfig, Fixes


Nov21ExpConfig_LD = Nov21ExpConfig


class Nov21SysConfig_LD(Nov21SysConfig):
    def synchronize_data_batch_file(self) -> str:
        raise ValueError("Shouldn't need to sychronize LD PC")

    @property
    def main_folder_path(self) -> str:
        return r"C:\Users\folklab\Documents\local-measurement-data\Tim\2021Oct_OneCK\PythonAnalysis"

    @property
    def main_archive_path(self) -> str:
        raise ValueError("No Archive on LD PC")

    def get_directories(self):
        hdfdir = r"C:\Users\folklab\Documents\local-measurement-data\Tim\2021Oct_OneCK\PythonAnalysis\DatHDFs"
        ddir = r"C:\Users\folklab\Documents\local-measurement-data\Tim\2021Oct_OneCK"
        return Directories(hdfdir, ddir)


class Nov21Exp2HDF_LD(Exp2HDF):
    unique_exp2hdf_name = 'nov21ld'

    def __init__(self, datnum: int, datname: str = 'base'):
        super().__init__(datnum, datname)

    @property
    def ExpConfig(self) -> ExpConfigBase:
        return Nov21ExpConfig_LD(self.datnum)

    @property
    def SysConfig(self) -> SysConfigBase:
        return Nov21SysConfig_LD(self.datnum)

    def get_hdfdir(self):
        return self.SysConfig.Directories.hdfdir

    def _get_update_batch_path(self):
        return self.SysConfig.synchronize_data_batch_file()
