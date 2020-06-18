from src.Configs.ConfigBase import ConfigBase
from src.DFcode.SetupDF import SetupDF
from src.ExperimentSpecific.BaseClasses import ExperimentSpecificInterface
from src.ExperimentSpecific.Jun20 import Jun20Config
from src.DatBuilder import Util


class JunESI(ExperimentSpecificInterface):
    def set_setupdf(self) -> SetupDF:
        self.setupdf = SetupDF()
        return self.setupdf  # Just to stop type hints

    def set_Config(self) -> ConfigBase:
        self.Config = Jun20Config.JunConfig()
        return self.Config  # Just to stop type hints

    def get_sweeplogs(self) -> dict:
        return super().get_sweeplogs()

