import os
from dictor import dictor
from src.DFcode.SetupDF import SetupDF
from src.DatObject.DatHDF import DatHDF
import numpy as np
from src.DataStandardize.BaseClasses import ConfigBase, ExperimentSpecificInterface
import logging
logger = logging.getLogger(__name__)


class SepConfig(ConfigBase):
    dir_name = 'Sep20'

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
        path = r'D:\OneDrive\UBC LAB\Machines\Remote Connections\WinSCP Scripts\Sep20.bat'
        return path


class SepESI(ExperimentSpecificInterface):
    def set_setupdf(self) -> SetupDF:
        self.setupdf = SetupDF(config=SepConfig())
        return self.setupdf  # Just to stop type hints

    def set_Config(self) -> ConfigBase:
        self.Config = SepConfig()
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

    def get_hdfdir(self):
        if self.datnum < 3000:  # TODO: Owen, I'm using this to store old data elsewhere, how should we make this work between us better?
            return r'Z:\10UBC\Measurement_Data\2020Sep\Dat_HDFs'
        else:
            return self.Config.Directories.hdfdir


class Fixes(object):
    """Just a place to collect together functions for fixing HDFs/Dats/sweeplogs/whatever"""

    @staticmethod
    def _add_magy(dat):  # TODO: Cange this fairly soon, it's a bit hacky
        if not hasattr(dat.Other, 'magy'):
            dat.Other.magy = _get_mag_field(dat)
            dat.Other.update_HDF()
            dat.hdf.flush()

    @staticmethod
    def fix_magy(dat: DatHDF):
        import src.HDF_Util as HDU
        if not hasattr(dat.Logs, 'magy') and 'LS625 Magnet Supply' in dat.Logs.full_sweeplogs.keys():
            mag = _get_mag_field(dat)
            group = dat.Logs.group
            mags_group = group.require_group(f'mags')  # Make sure there is an srss group
            HDU.set_attr(mags_group, mag.name, mag)  # Save in srss group
            dat.hdf.flush()
            dat.Logs.get_from_HDF()

    @staticmethod
    def add_part_of(dat: DatHDF):
        if not hasattr(dat.Logs, 'part_of'):
            from src.DatObject.DatBuilder import get_part
            part_of = get_part(dat.Logs.comments)
            dat.Logs.group.attrs['part_of'] = part_of
            dat.hdf.flush()
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


from src.DatObject.Attributes.Logs import MAGs


def _get_mag_field(dat:DatHDF) -> MAGs:
    sl = dat.Logs.full_sweeplogs
    field = sl['LS625 Magnet Supply']['field mT']
    rate = sl['LS625 Magnet Supply']['rate mT/min']
    variable_name = sl['LS625 Magnet Supply']['variable name']
    mag = MAGs(variable_name, field, rate)
    return mag


