from __future__ import annotations
import numpy as np
from src.data_standardize.base_classes import Exp2HDF, SysConfigBase
from src.data_standardize.exp_config import ExpConfigBase
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.dat_object.dat_hdf import DatHDF, get_nested_attr_default
logger = logging.getLogger(__name__)


class SepExpConfig(ExpConfigBase):
    dir_name = 'Sep20'

    def __init__(self, datnum=None):
        super().__init__(datnum)

    def get_sweeplogs_json_subs(self, datnum=None):
        return {'FastDAC 1': 'FastDAC'}

    def get_sweeplog_modifications(self) -> dict:
        switch = {'Lakeshore.Temperature': 'Temperatures',
                  'Temperatures. 50K Plate K': 'Temperatures.50K Plate K'}
        remove = ['Lakeshore']  # Nothing else in 'Lakeshore' after 'Temperatures' are switched out
        add = {}
        return {'switch': switch, 'remove': remove, 'add': add}


class SepSysConfig(SysConfigBase):

    @property
    def dir_name(self) -> str:
        return 'Sep20'

    def synchronize_data_batch_file(self) -> str:
        return r'D:\OneDrive\UBC LAB\Machines\Remote Connections\WinSCP scripts\Sep20.bat'

    @property
    def main_folder_path(self) -> str:
        return super().main_folder_path


class SepExp2HDF(Exp2HDF):
    @property
    def ExpConfig(self) -> ExpConfigBase:
        return SepExpConfig(self.datnum)

    @property
    def SysConfig(self) -> SysConfigBase:
        return SepSysConfig(self.datnum)

    def get_hdfdir(self):
        if self.datnum < 6000:  # TODO: Owen, I'm using this to store old data elsewhere, how should we make this work between us better?
            return r'Z:\10UBC\Measurement_Data\2020Sep\Dat_HDFs'
        else:
            return self.SysConfig.Directories.hdfdir


class Fixes(object):
    """Just a place to collect together functions for fixing HDFs/Dats/sweeplogs/whatever"""

    @staticmethod
    def _add_magy(dat):  # TODO: Cange this fairly soon, it's a bit hacky
        if not hasattr(dat.Other, 'magy'):
            dat.Other.magy = _get_mag_field(dat)
            dat.Other.update_HDF()
            dat.old_hdf.flush()

    @staticmethod
    def fix_magy(dat: DatHDF):
        import src.hdf_util as HDU
        if not hasattr(dat.Logs, 'magy') and 'LS625 Magnet Supply' in dat.Logs.full_sweeplogs.keys():
            mag = _get_mag_field(dat)
            group = dat.Logs.group
            mags_group = group.require_group(f'mags')  # Make sure there is an srss group
            HDU.set_attr(mags_group, mag.name, mag)  # Save in srss group
            dat.old_hdf.flush()
            dat.Logs.get_from_HDF()

    @staticmethod
    def add_part_of(dat: DatHDF):
        if not hasattr(dat.Logs, 'part_of'):
            from DatObject.Attributes.Logs import get_part
            part_of = get_part(dat.Logs.comments)
            dat.Logs.group.attrs['part_of'] = part_of
            dat.old_hdf.flush()
            dat.Logs.get_from_HDF()

    @staticmethod
    def setpoint_averaging_fix(dat: DatHDF):
        if get_nested_attr_default(dat, 'SquareEntropy.Processed.process_params', None):
            pp = dat.SquareEntropy.Processed.process_params
            if pp.setpoint_start is None:
                pp.setpoint_start = int(np.round(1.2e-3*dat.AWG.measure_freq))
                logger.info(f'Recalculating Square entropy for Dat{dat.datnum}')
                dat.SquareEntropy.Processed.calculate()
                dat.SquareEntropy.update_HDF()


from src.dat_object.attributes.Logs import Magnet


def _get_mag_field(dat: DatHDF) -> Magnet:
    sl = dat.Logs.full_sweeplogs
    field = sl['LS625 Magnet Supply']['field mT']
    rate = sl['LS625 Magnet Supply']['rate mT/min']
    variable_name = sl['LS625 Magnet Supply']['variable name']
    mag = Magnet(variable_name, field, rate)
    return mag


def get_lct_name(dat: DatHDF):
    """
    Returns the name which is being used for LCT (based on which divider was in there

    Args:
        dat (DatHDF):  Dat to look for LCT name in

    Returns:
        str: LCT name
    """
    fds = dat.Logs.fds
    if 'LCT' in fds:
        return 'LCT'
    elif 'LCT/0.16' in fds:
        return 'LCT/0.16'
    elif 'LCT/0.196' in fds:
        return 'LCT/0.196'
    else:
        raise NotImplementedError(f'No recognised LCT name found in dat.Logs.fds')


def get_real_lct(dat: DatHDF):
    """
    Returns the real value of LCT from the dat (i.e. accounting for divider)

    Args:
        dat (DatHDF): Dat to get real LCT value from

    Returns:
        float: Real LCT value in mV
    """
    key = get_lct_name(dat)
    val = dat.Logs.fds.get(key)
    if key == 'LCT':
        return val
    elif key == 'LCT/0.16':
        return val * 0.163
    elif key == 'LCT/0.196':
        return val * 0.196
    else:
        raise NotImplementedError(f'No recognised LCT name found in dat.Logs.fds')


