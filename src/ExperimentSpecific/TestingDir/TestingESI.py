from src.Configs.ConfigBase import ConfigBase
from src.DFcode.SetupDF import SetupDF
from src.ExperimentSpecific.BaseClasses import ExperimentSpecificInterface as ESI
from src.ExperimentSpecific.TestingDir import TestingConfig
from src.DatBuilder import Util




class TestingESI(ESI):
    def set_setupdf(self) -> SetupDF:
        self.setupdf = SetupDF(config=TestingConfig.TestingConfig())
        return self.setupdf  # Just to stop type hints

    def set_Config(self) -> ConfigBase:
        self.Config = TestingConfig.TestingConfig()
        return self.Config  # Just to stop type hints

    def get_sweeplogs(self) -> dict:
        return super().get_sweeplogs()

