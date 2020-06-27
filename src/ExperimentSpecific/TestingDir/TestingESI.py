from src.Configs.ConfigBase import ConfigBase
from src.DFcode.SetupDF import SetupDF
from src.ExperimentSpecific.BaseClasses import ExperimentSpecificInterface as ESI
from src.ExperimentSpecific.TestingDir import TestingConfig
from src.ExperimentSpecific.Jan20.Jan20ESI import convert_babydac_json, convert_fastdac_json, get_num_adc_from_hdf
from dictor import dictor


class TestingESI(ESI):
    def set_setupdf(self) -> SetupDF:
        self.setupdf = SetupDF(config=TestingConfig.TestingConfig())
        return self.setupdf  # Just to stop type hints

    def set_Config(self) -> ConfigBase:
        self.Config = TestingConfig.TestingConfig()
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

