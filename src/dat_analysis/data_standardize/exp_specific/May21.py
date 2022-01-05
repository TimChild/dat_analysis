from __future__ import annotations
import numpy as np
from ..base_classes import Exp2HDF, SysConfigBase
from ..exp_config import ExpConfigBase, DataInfo
import logging
from typing import TYPE_CHECKING, Dict


if TYPE_CHECKING:
    from ...dat_object.dat_hdf import DatHDF
logger = logging.getLogger(__name__)


class May21ExpConfig(ExpConfigBase):
    dir_name = 'May21'

    def __init__(self, datnum=None):
        super().__init__(datnum)

    def get_sweeplogs_json_subs(self, datnum=None):
        return {'FastDAC 1': 'FastDAC'}

    def get_sweeplog_modifications(self) -> dict:
        switch = {'Lakeshore.Temperature': 'Temperatures'}
        remove = ['Lakeshore']  # Nothing else in 'Lakeshore' after 'Temperatures' are switched out
        # add = {}
        # return {'switch': switch, 'remove': remove, 'add': add}
        mods = {'switch': switch, 'remove': remove, 'add': {}}
        add = {}
        mods['add'] = add
        return mods

    def get_default_data_info(self) -> Dict[str, DataInfo]:
        info = super().get_default_data_info()
        return info


class May21SysConfig(SysConfigBase):
    @property
    def dir_name(self) -> str:
        return 'May21'

    def synchronize_data_batch_file(self) -> str:
        return r'D:\OneDrive\UBC LAB\Machines\Remote Connections\WinSCP scripts\May21.bat'

    @property
    def main_folder_path(self) -> str:
        return super().main_folder_path


class May21Exp2HDF(Exp2HDF):

    def __init__(self, datnum: int, datname: str = 'base'):
        super().__init__(datnum, datname)

    @property
    def ExpConfig(self) -> ExpConfigBase:
        return May21ExpConfig(self.datnum)

    @property
    def SysConfig(self) -> SysConfigBase:
        return May21SysConfig(self.datnum)

    def get_hdfdir(self):
        return self.SysConfig.Directories.hdfdir

    def _get_update_batch_path(self):
        return self.SysConfig.synchronize_data_batch_file()


class Fixes(object):
    """Just a place to collect together functions for fixing HDFs/Dats/sweeplogs/whatever"""
    pass

