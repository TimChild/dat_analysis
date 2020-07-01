import os
from dictor import dictor
from src.DFcode.SetupDF import SetupDF

from src.DataStandardize.BaseClasses import ConfigBase, ExperimentSpecificInterface


class JunConfig(ConfigBase):
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


dir_name = 'Jun20'  # Name of folder inside main data folder specified by src.config.main_data_path


class JunESI(ExperimentSpecificInterface):
    def set_setupdf(self) -> SetupDF:
        self.setupdf = SetupDF(config=JunConfig())
        return self.setupdf  # Just to stop type hints

    def set_Config(self) -> ConfigBase:
        self.Config = JunConfig()
        return self.Config  # Just to stop type hints

    def get_sweeplogs(self) -> dict:
        return super().get_sweeplogs()

    def get_dattypes(self) -> set:
        dattypes = super().get_dattypes()

        # Maybe I forgot to add AWG to comments but there are AWG logs in FastDAC sweeplogs
        if 'AWG' not in dattypes:
            sweeplogs = self.get_sweeplogs()
            awg_logs = dictor(sweeplogs, 'FastDAC.AWG', None)
            if awg_logs is not None:
                dattypes.add('AWG')

        return dattypes

    # import os
    #
    # path_replace = None  # ('work\\Fridge Measurements with PyDatAnalysis', 'work\\Fridge_Measurements_and_Devices\\Fridge Measurements with PyDatAnalysis')
    #
    # instruments = {'srs': 'srs830', 'dmm': 'hp34401a', 'dac': 'babydac', 'fastdac': 'fastdac', 'magnet': 'ls625',
    #                'fridge': 'ls370'}
    # instrument_num = {'srs': 3, 'dmm': 1, 'dac': 16, 'fastdac': 8, 'magnet': 3}
    #
    # dat_types_list = ['none', 'i_sense', 'entropy', 'transition', 'pinch', 'dot tuning', 'dcbias']
    #
    # ### Path to all Data (e.g. dats, dataframes, pickles etc). Hopefully this will allow moving out of project without
    # #  losing access to everything
    # abspath = os.path.abspath('../..').split('PyDatAnalysis')[0]
    #
    # wavenames = ['x_array', 'y_array', 'i_sense', 'entx', 'enty']
    # raw_wavenames = [f'ADC{i:d}' for i in range(4)] + [f'ADC{i:d}_2d' for i in range(4)] + ['g1x', 'g1y', 'g2x', 'g2y']
    #
    # x_array_keys = ['x_array']
    # y_array_keys = ['y_array']
    # i_sense_keys = ['i_sense', 'cscurrent', 'cscurrent_2d']
    # entx_keys = ['entx', 'entropy_x_2d', 'entropy_x']
    # enty_keys = ['enty', 'entropy_y_2d', 'entropy_y']
    # # For old code below
    # entropy_x_keys = ['entx', 'entropy_x_2d', 'entropy_x']
    # entropy_y_keys = ['enty', 'entropy_y_2d', 'entropy_y']
    # # li_theta_x_keys = ['g3x']
    # # li_theta_y_keys = ['g3y']
    #
    # # json_subs = [('"comment": "{"gpib_address":4, "units":"VOLT", "range":.1.000000E.0., "resolution":...000000E-0.}"',
    # #                 '"comment": "replaced to fix json"'),
    # #              (":\+", ':'),
    # #              ('\r', '')]
    # json_subs = []
    #
    # DC_HQPC_current_bias_resistance = 10e6  # Resistor inline with DC bias for HQPC
    #
    # # num_adc_record = len([key for key in dat.Data.data_keys if re.search('.*RAW.*', key)])
