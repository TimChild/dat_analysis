from __future__ import annotations
import numpy as np
from dat_analysis.data_standardize.base_classes import Exp2HDF, SysConfigBase
from dat_analysis.data_standardize.exp_config import ExpConfigBase, DataInfo
from dat_analysis.hdf_file_handler import HDFFileHandler
import logging
from typing import TYPE_CHECKING, Dict


if TYPE_CHECKING:
    from dat_analysis.dat_object.dat_hdf import DatHDF
logger = logging.getLogger(__name__)


class Nov21ExpConfig(ExpConfigBase):
    dir_name = 'Nov21_OneCK'

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


class Nov21SysConfig(SysConfigBase):
    @property
    def dir_name(self) -> str:
        return 'Nov21_OneCK'

    def synchronize_data_batch_file(self) -> str:
        """Path to a file that can be run to synchronize data"""
        raise FileNotFoundError(f'dat{self.datnum} not downloaded, and dont want to download')
        # return r'D:\NextCloud\Documents\Machines\Remote Connections\WinSCP Scripts\Nov21.bat'

    @property
    def main_folder_path(self) -> str:
        """Can override this with a root directory"""
        return super().main_folder_path


class Nov21Exp2HDF(Exp2HDF):

    def __init__(self, datnum: int, datname: str = 'base'):
        super().__init__(datnum, datname)

    @property
    def ExpConfig(self) -> ExpConfigBase:
        return Nov21ExpConfig(self.datnum)

    @property
    def SysConfig(self) -> SysConfigBase:
        return Nov21SysConfig(self.datnum)

    def get_hdfdir(self):
        return self.SysConfig.Directories.hdfdir

    def _get_update_batch_path(self):
        return self.SysConfig.synchronize_data_batch_file()


class Fixes(object):
    """Just a place to collect together functions for fixing HDFs/Dats/sweeplogs/whatever"""
    @staticmethod
    def fix_axis_arrays(dat: DatHDF):
        if not dat.Data.get_group_attr('arrays_fixed', False):
            with HDFFileHandler(dat.hdf.hdf_path) as f:
                infos = {}
                for array in ['x_array', 'y_array']:
                    if array in f['Experiment Copy']:
                        infos[array] = f['Experiment Copy'][array].attrs['IGORWaveScaling']
            for key in infos:
                scaling = infos[key]
                start = scaling[1, 1]
                step = scaling[1, 0]
                num = len(dat.Data.get_data(key[0]))
                new = np.linspace(start, start + (num - 1) * step, num)
                dat.Data.set_data(new, key[0])
            dat.Data.set_group_attr('arrays_fixed', True)
        else:
            logger.info(f'Already fixed arrays for dat{dat.datnum}')

