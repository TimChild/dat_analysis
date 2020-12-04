from __future__ import annotations
from typing import Optional
from dictor import dictor
from src.DataStandardize import Standardize_Util as Util
import numpy as np
from src.DataStandardize.BaseClasses import Exp2HDF, SysConfigBase, Directories
from src.DataStandardize.ExpConfig import ExpConfigBase
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.DatObject.DatHDF import DatHDF
logger = logging.getLogger(__name__)


class SepExpConfig(ExpConfigBase):

    dir_name = 'Sep20'

    def __init__(self, datnum=None):
        super().__init__(datnum)

    def get_sweeplogs_json_subs(self, datnum=None):
        return {'FastDAC 1': 'FastDAC'}

    def get_sweeplog_modifications(self) -> dict:
        switch = {'Lakeshore.Temperature': 'Temperatures'}
        remove = ['Lakeshore']  # Nothing else in 'Lakeshore' after 'Temperatures' are switched out
        add = {}
        return {'switch': switch, 'remove': remove, 'add': add}


    # def get_dattypes_list(self, datnum=None):
    #     return ['none', 'entropy', 'transition', 'dcbias', 'square entropy']
    #
    # def get_exp_names_dict(self, datnum=None):
    #     d = dict(x_array=['x_array'], y_array=['y_array'],
    #              i_sense=['cscurrent', 'cscurrent_2d'],
    #              entx=['entropy_x_2d', 'entropy_x'],
    #              enty=['entropy_y_2d', 'entropy_y'])
    #     return d

    # def synchronize_data_batch_file(self):
    #     if platform == "darwin":
    #         path = "/Users/owensheekey/Nextcloud/Shared/measurement-data/Owen"
    #     elif platform == "win32":
    #         path = r'D:\OneDrive\UBC LAB\Machines\Remote Connections\WinSCP Scripts\Sep20.bat'
    #     else:
    #         raise ValueError("System unsupported -- Add to config")
    #     return path


class SepSysConfig(SysConfigBase):

    @property
    def dir_name(self) -> str:
        return 'Sep20'

    def synchronize_data_batch_file(self) -> str:
        return r'D:\OneDrive\UBC LAB\Machines\Remote Connections\WinSCP Scripts\Sep20.bat'

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

    # def set_setupdf(self) -> SetupDF:
    #     self.setupdf = SetupDF(config=SepExpConfig())
    #     return self.setupdf  # Just to stop type hints
    #
    # def set_ExpConfig(self) -> ExpConfigBase:
    #     self.Config = SepExpConfig()
    #     return self.Config  # Just to stop type hints



    def get_sweeplogs(self) -> dict:
        sweep_logs = super().get_sweeplogs()
        if 'Lakeshore' in sweep_logs:
            sweep_logs['Temperatures'] = sweep_logs['Lakeshore']['Temperature']
            del sweep_logs['Lakeshore']

        # if self.datnum < 218:
        #     if 'FastDAC' in sweep_logs.keys():
        #         fd = sweep_logs.get('FastDAC')
        #         fd['SamplingFreq'] = float(fd['SamplingFreq'])
        #         fd['MeasureFreq'] = float(fd['MeasureFreq'])
        return sweep_logs

    def _get_dat_types(self) -> set:
        if self._dat_types is None:  # Only load dattypes the first time, then store
            sweep_logs = self.get_sweeplogs()
            comments = sweep_logs.get('comment', None)
            if comments and 'square_entropy' in [val.strip() for val in comments.split(',')]:
                comments += ", square entropy"

            dat_types_list = self.ExpConfig.get_dattypes_list()
            self._dat_types = Util.get_dattypes(None, comments, dat_types_list)
        # Maybe I forgot to add AWG to comments but there are AWG logs in FastDAC sweeplogs
        if 'AWG' not in self._dat_types:
            sweeplogs = self.get_sweeplogs()
            awg_logs = dictor(sweeplogs, 'FastDAC.AWG', None)
            if awg_logs is not None:
                self._dat_types.add('AWG')
        return self._dat_types

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
        import src.HDF_Util as HDU
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
        from src.CoreUtil import get_nested_attr_default
        if get_nested_attr_default(dat, 'SquareEntropy.Processed.process_params', None):
            pp = dat.SquareEntropy.Processed.process_params
            if pp.setpoint_start is None:
                pp.setpoint_start = int(np.round(1.2e-3*dat.AWG.measure_freq))
                logger.info(f'Recalculating Square entropy for Dat{dat.datnum}')
                dat.SquareEntropy.Processed.calculate()
                dat.SquareEntropy.update_HDF()

    # @staticmethod
    # def fix_magy(dat):
    #     raise NotImplementedError
    #     pass #  TODO: do the better fix which I started writing in InitLogs etc.. need to figure out how to parse multiple mags

    # @staticmethod
        # def log_temps(dat):
        #     import src.DatObject.DatBuilder as DB
        #     if dat.Logs.temps is None:
        #         print(f'Fixing logs in dat{dat.datnum}')
        #         esi = SepESI(dat.datnum)
        #         sweep_logs = esi.get_sweeplogs()
        #         DB.InitLogs.set_temps(dat.Logs.group, sweep_logs['Temperatures'])
        #         dat.hdf.flush()
        #         dat.Logs.get_from_HDF()
        #
        # @staticmethod
        # def add_full_sweeplogs(dat):
        #     import src.HDF_Util as HDU
        #     if dat.Logs.full_sweeplogs is None:
        #         print(f'Fixing logs in dat{dat.datnum}')
        #         esi = SepESI(dat.datnum)
        #         sweep_logs = esi.get_sweeplogs()
        #         HDU.set_attr(dat.Logs.group, 'Full sweeplogs', sweep_logs)
        #         dat.hdf.flush()
        #         dat.Logs.get_from_HDF()


from src.DatObject.Attributes.Logs import Magnet


def _get_mag_field(dat:DatHDF) -> Magnet:
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


