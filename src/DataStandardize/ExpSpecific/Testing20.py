import os

from dictor import dictor

import src.DataStandardize
from src.DFcode.SetupDF import SetupDF

from src.DataStandardize.BaseClasses import ConfigBase, ExperimentSpecificInterface as ESI
from src.DataStandardize.ExpSpecific.Jan20 import convert_babydac_json, get_num_adc_from_hdf, convert_fastdac_json


class TestingConfig(ConfigBase):
    """As of Jun20 this is based on Jan20 Config as I have lots of data there to test on"""
    # TODO: Change this and tests to Jun20 based once I have new data
    dir_name = 'TestingDir'

    def __init__(self):
        super().__init__()

    def set_directories(self):
        hdfdir, ddir, dfsetupdir, dfbackupdir = self.get_expected_sub_dir_paths(
            os.path.join(self.main_folder_path, self.dir_name))  # self.dir_name set in subclass
        # To fix shortcuts in paths
        self.Directories.set_dirs(hdfdir, ddir, dfsetupdir, dfbackupdir)

    def get_sweeplogs_json_subs(self, datnum):
        return [('"comment": "{"gpib_address":4, "units":"VOLT", "range":.1.000000E.0., "resolution":...000000E-0.}"',
                 '"comment": "replaced to fix json"'), (":\+", ':'), ('\r', '')]

    def get_dattypes_list(self):
        return ['none', 'i_sense', 'entropy', 'transition']  # , 'pinch', 'dot tuning', 'dcbias']

    def get_exp_names_dict(self):
        d = dict(x_array=['x_array'], y_array=['y_array'],
                 i_sense=['i_sense', 'cscurrent', 'cscurrent_2d'],
                 entx=['entropy_x_2d', 'entropy_x'],
                 enty=['entropy_y_2d', 'entropy_y'])  # took out entx, enty because not in SetupDF
        return d


class TestingESI(ESI):
    def set_setupdf(self) -> SetupDF:
        self.setupdf = SetupDF(config=src.DataStandardize.ExpSpecific.Testing20.TestingConfig())
        return self.setupdf  # Just to stop type hints

    def set_Config(self) -> ConfigBase:
        self.Config = src.DataStandardize.ExpSpecific.Testing20.TestingConfig()
        return self.Config  # Just to stop type hints

    def get_sweeplogs(self) -> dict:
        sweep_logs = super().get_sweeplogs()

        # Then need to change the format of BabyDAC and FastDAC
        bdacs_json = dictor(sweep_logs, 'BabyDAC', None)
        if bdacs_json is not None:
            sweep_logs['BabyDAC'] = convert_babydac_json(bdacs_json)

        fdacs_json = dictor(sweep_logs, 'FastDAC', None)
        if fdacs_json is not None:
            hdf = self.get_exp_dat_hdf()
            num_adc = get_num_adc_from_hdf(hdf)
            hdf.close()
            sweep_logs['FastDAC'] = convert_fastdac_json(fdacs_json, num_adc)
        return sweep_logs