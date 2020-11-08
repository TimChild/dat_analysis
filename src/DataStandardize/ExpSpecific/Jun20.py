import os
from dictor import dictor
from src.DFcode.SetupDF import SetupDF
from src.DataStandardize.BaseClasses import ExpConfigBase, Exp2HDF


class JunExpConfig(ExpConfigBase):
    dir_name = 'Jun20'

    def __init__(self):
        super().__init__()

    def set_directories(self):
        hdfdir, ddir, dfsetupdir, dfbackupdir = self.get_expected_sub_dir_paths(
            os.path.join(self.main_folder_path, self.dir_name))
        self.Directories.set_dirs(hdfdir, ddir, dfsetupdir, dfbackupdir)

    def get_sweeplogs_json_subs(self, datnum):
        return [('FastDAC 1', 'FastDAC')]

    def get_dattypes_list(self):
        return ['none', 'entropy', 'transition', 'dcbias']

    def get_exp_names_dict(self):
        d = dict(x_array=['x_array'], y_array=['y_array'],
                 i_sense=['i_sense', 'cscurrent', 'cscurrent_2d', 'current_dc_2d', 'current_dc'],
                 entx=['entx', 'entropy_x_2d', 'entropy_x'],
                 enty=['enty', 'entropy_y_2d', 'entropy_y'])
        return d

    def synchronize_data_batch_file(self):
        path = r'D:\OneDrive\UBC LAB\Machines\Remote Connections\WinSCP Scripts\Jun20.bat'
        return path


class JunESI(Exp2HDF):
    def set_setupdf(self) -> SetupDF:
        self.setupdf = SetupDF(config=JunExpConfig())
        return self.setupdf  # Just to stop type hints

    def set_ExpConfig(self) -> ExpConfigBase:
        self.Config = JunExpConfig()
        return self.Config  # Just to stop type hints

    def get_sweeplogs(self) -> dict:
        sweep_logs = super().get_sweeplogs()
        if self.datnum < 218:
            if 'FastDAC' in sweep_logs.keys():
                fd = sweep_logs.get('FastDAC')
                fd['SamplingFreq'] = float(fd['SamplingFreq'])
                fd['MeasureFreq'] = float(fd['MeasureFreq'])
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
    def log_temps(dat):
        import src.DatObject.DatBuilder as DB
        if dat.Logs.temps is None:
            print(f'Fixing logs in dat{dat.datnum}')
            esi = JunESI(dat.datnum)
            sweep_logs = esi.get_sweeplogs()
            DB.InitLogs.set_temps(dat.Logs.group, sweep_logs['Temperatures'])
            dat.hdf.flush()
            dat.Logs.get_from_HDF()

    @staticmethod
    def add_full_sweeplogs(dat):
        import src.HDF_Util as HDU
        if dat.Logs.full_sweeplogs is None:
            print(f'Fixing logs in dat{dat.datnum}')
            esi = JunESI(dat.datnum)
            sweep_logs = esi.get_sweeplogs()
            HDU.set_attr(dat.Logs.group, 'Full sweeplogs', sweep_logs)
            dat.hdf.flush()
            dat.Logs.get_from_HDF()