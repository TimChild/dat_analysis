from __future__ import annotations
import numpy as np
from src.DataStandardize.BaseClasses import Exp2HDF, SysConfigBase
from src.DataStandardize.ExpConfig import ExpConfigBase, DataInfo
import logging
from typing import TYPE_CHECKING, Dict



if TYPE_CHECKING:
    from src.DatObject.DatHDF import DatHDF
logger = logging.getLogger(__name__)


class Feb21ExpConfig(ExpConfigBase):
    dir_name = 'Feb21'

    def __init__(self, datnum=None):
        super().__init__(datnum)

    def get_sweeplogs_json_subs(self, datnum=None):
        return {'FastDAC 1': 'FastDAC'}

    def get_sweeplog_modifications(self) -> dict:
        # switch = {'Lakeshore.Temperature': 'Temperatures',
        #           'Temperatures. 50K Plate K': 'Temperatures.50K Plate K'}
        # remove = ['Lakeshore']  # Nothing else in 'Lakeshore' after 'Temperatures' are switched out
        # add = {}
        # return {'switch': switch, 'remove': remove, 'add': add}
        mods = {'switch': {}, 'remove': [], 'add': {}}
        add = {}
        if self.datnum <= 627:
            for key in ['50K Plate K', '4K Plate K', 'Still K', 'Magnet K', 'MC K']:
                add[f'Temperatures.{key}'] = None
        mods['add'] = add
        return mods

    def get_default_data_info(self) -> Dict[str, DataInfo]:
        info = super().get_default_data_info()
        if 646 <= self.datnum <= 664:  # Current amp was at 1e9, Igor thought it was at 1e8
            info['cscurrent_2d'] = DataInfo('i_sense', multiply=0.1)
            info['cscurrent'] = DataInfo('i_sense', multiply=0.1)
        return info



class Feb21SysConfig(SysConfigBase):
    @property
    def dir_name(self) -> str:
        return 'Feb21'

    def synchronize_data_batch_file(self) -> str:
        return r'D:\OneDrive\UBC LAB\Machines\Remote Connections\WinSCP Scripts\Feb21.bat'

    @property
    def main_folder_path(self) -> str:
        return super().main_folder_path


class Feb21Exp2HDF(Exp2HDF):
    @property
    def ExpConfig(self) -> ExpConfigBase:
        return Feb21ExpConfig(self.datnum)

    @property
    def SysConfig(self) -> SysConfigBase:
        return Feb21SysConfig(self.datnum)

    def get_hdfdir(self):
        return self.SysConfig.Directories.hdfdir


class Fixes(object):
    """Just a place to collect together functions for fixing HDFs/Dats/sweeplogs/whatever"""
    pass
    # @staticmethod
    # def _add_magy(dat):  # TODO: Change this fairly soon, it's a bit hacky
    #     if not hasattr(dat.Other, 'magy'):
    #         dat.Other.magy = _get_mag_field(dat)
    #         dat.Other.update_HDF()
    #         dat.old_hdf.flush()
    #
    # @staticmethod
    # def fix_magy(dat: DatHDF):
    #     import src.HDF_Util as HDU
    #     if not hasattr(dat.Logs, 'magy') and 'LS625 Magnet Supply' in dat.Logs.full_sweeplogs.keys():
    #         mag = _get_mag_field(dat)
    #         group = dat.Logs.group
    #         mags_group = group.require_group(f'mags')  # Make sure there is an srss group
    #         HDU.set_attr(mags_group, mag.name, mag)  # Save in srss group
    #         dat.old_hdf.flush()
    #         dat.Logs.get_from_HDF()

    # @staticmethod
    # def add_part_of(dat: DatHDF):
    #     if not hasattr(dat.Logs, 'part_of'):
    #         from DatObject.Attributes.Logs import get_part
    #         part_of = get_part(dat.Logs.comments)
    #         dat.Logs.group.attrs['part_of'] = part_of
    #         dat.old_hdf.flush()
    #         dat.Logs.get_from_HDF()
    #
    # @staticmethod
    # def setpoint_averaging_fix(dat: DatHDF):
    #     from src.CoreUtil import get_nested_attr_default
    #     if get_nested_attr_default(dat, 'SquareEntropy.Processed.process_params', None):
    #         pp = dat.SquareEntropy.Processed.process_params
    #         if pp.setpoint_start is None:
    #             pp.setpoint_start = int(np.round(1.2e-3*dat.AWG.measure_freq))
    #             logger.info(f'Recalculating Square entropy for Dat{dat.datnum}')
    #             dat.SquareEntropy.Processed.calculate()
    #             dat.SquareEntropy.update_HDF()
#
#
# from src.DatObject.Attributes.Logs import Magnet
#
#
# def _get_mag_field(dat: DatHDF) -> Magnet:
#     sl = dat.Logs.full_sweeplogs
#     field = sl['LS625 Magnet Supply']['field mT']
#     rate = sl['LS625 Magnet Supply']['rate mT/min']
#     variable_name = sl['LS625 Magnet Supply']['variable name']
#     mag = Magnet(variable_name, field, rate)
#     return mag
#
#
# def get_lct_name(dat: DatHDF):
#     """
#     Returns the name which is being used for LCT (based on which divider was in there
#
#     Args:
#         dat (DatHDF):  Dat to look for LCT name in
#
#     Returns:
#         str: LCT name
#     """
#     fds = dat.Logs.fds
#     if 'LCT' in fds:
#         return 'LCT'
#     elif 'LCT/0.16' in fds:
#         return 'LCT/0.16'
#     elif 'LCT/0.196' in fds:
#         return 'LCT/0.196'
#     else:
#         raise NotImplementedError(f'No recognised LCT name found in dat.Logs.fds')
#
#
# def get_real_lct(dat: DatHDF):
#     """
#     Returns the real value of LCT from the dat (i.e. accounting for divider)
#
#     Args:
#         dat (DatHDF): Dat to get real LCT value from
#
#     Returns:
#         float: Real LCT value in mV
#     """
#     key = get_lct_name(dat)
#     val = dat.Logs.fds.get(key)
#     if key == 'LCT':
#         return val
#     elif key == 'LCT/0.16':
#         return val * 0.163
#     elif key == 'LCT/0.196':
#         return val * 0.196
#     else:
#         raise NotImplementedError(f'No recognised LCT name found in dat.Logs.fds')
#
if __name__ == '__main__':
    datnums = range(646, 664+1)
    from src.DatObject.Make_Dat import get_dat
    # for num in datnums:
    #     dat = get_dat(num, exp2hdf=Feb21Exp2HDF, overwrite=True)
