"""
2022/01/05 -- Started updating, but it's too out of date.
"""
import os
from dictor import dictor
from ...dat_object.dat_hdf import DatHDF
from sys import platform

from ..base_classes import Exp2HDF
from ..exp_config import ExpConfigBase
from dataclasses import dataclass


class AugExpConfig(ExpConfigBase):
    dir_name = 'Aug20'

    def __init__(self):
        super().__init__()

    def set_directories(self):
        hdfdir, ddir, dfsetupdir, dfbackupdir = self.get_expected_sub_dir_paths(
            os.path.join(self.main_folder_path, self.dir_name))
        self.Directories.set_dirs(hdfdir, ddir, dfsetupdir, dfbackupdir)

    def get_sweeplogs_json_subs(self, datnum):
        return [('FastDAC 1', 'FastDAC')]

    def get_dattypes_list(self):
        return ['none', 'entropy', 'transition', 'dcbias', 'square entropy']

    def get_exp_names_dict(self):
        d = dict(x_array=['x_array'], y_array=['y_array'],
                 i_sense=['cscurrent', 'cscurrent_2d'],
                 entx=['entropy_x_2d', 'entropy_x'],
                 enty=['entropy_y_2d', 'entropy_y'])
        return d

    def synchronize_data_batch_file(self):
        if platform == "darwin":
            path = "/Users/owensheekey/Nextcloud/Shared/measurement-data/Owen"
        elif platform == "win32":
            path = r'D:\OneDrive\UBC LAB\Machines\Remote Connections\WinSCP Scripts\Aug20.bat'
        else:
            raise ValueError("System unsupported -- Add to config")
        return path

class AugESI(Exp2HDF):
    def set_ExpConfig(self) -> ExpConfigBase:
        self.Config = AugExpConfig()
        return self.Config  # Just to stop type hints

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

    def get_dattypes(self) -> set:
        dattypes = super().get_dattypes()

        # Maybe I forgot to add AWG to comments but there are AWG logs in FastDAC sweeplogs
        if 'AWG' not in dattypes:
            sweeplogs = self.get_sweeplogs()
            awg_logs = dictor(sweeplogs, 'FastDAC.AWG', None)
            if awg_logs is not None:
                dattypes.add('AWG')

        return dattypes


class Fixes(object):
    """Just a place to collect together functions for fixing HDFs/Dats/sweeplogs/whatever"""

    @staticmethod
    def _add_magy(dat):  # TODO: Cange this fairly soon, it's a bit hacky
        if not hasattr(dat.Other, 'magy'):
            dat.Other.magy = _get_magy_field(dat)
            dat.Other.update_HDF()
            dat.old_hdf.flush()

    @staticmethod
    def fix_magy(dat):
        raise NotImplementedError
        pass #  TODO: do the better fix which I started writing in InitLogs etc.. need to figure out how to parse multiple mags

    # @staticmethod
        # def log_temps(dat):
        #     import dat_analysis.dat_object.DatBuilder as DB
        #     if dat.Logs.temps is None:
        #         print(f'Fixing logs in dat{dat.datnum}')
        #         esi = AugESI(dat.datnum)
        #         sweep_logs = esi.get_sweeplogs()
        #         DB.InitLogs.set_temps(dat.Logs.group, sweep_logs['Temperatures'])
        #         dat.hdf.flush()
        #         dat.Logs.get_from_HDF()
        #
        # @staticmethod
        # def add_full_sweeplogs(dat):
        #     import dat_analysis.HDF_Util as HDU
        #     if dat.Logs.full_sweeplogs is None:
        #         print(f'Fixing logs in dat{dat.datnum}')
        #         esi = AugESI(dat.datnum)
        #         sweep_logs = esi.get_sweeplogs()
        #         HDU.set_attr(dat.Logs.group, 'Full sweeplogs', sweep_logs)
        #         dat.hdf.flush()
        #         dat.Logs.get_from_HDF()


from ...dat_object.attributes.logs import Magnet


def _get_magy_field(dat:DatHDF) -> Magnet:
    sl = dat.Logs.full_sweeplogs
    field = sl['LS625 Magnet Supply']['field mT']
    rate = sl['LS625 Magnet Supply']['rate mT/min']
    mag = Magnet('magy', field, rate)
    return mag


