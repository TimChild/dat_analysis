from src.Configs.ConfigBase import ConfigBase
from src.DFcode.SetupDF import SetupDF
from src.ExperimentSpecific.BaseClasses import ExperimentSpecificInterface
from src.ExperimentSpecific.Jun20 import Jun20Config
from dictor import dictor


class JunESI(ExperimentSpecificInterface):
    def set_setupdf(self) -> SetupDF:
        self.setupdf = SetupDF()
        return self.setupdf  # Just to stop type hints

    def set_Config(self) -> ConfigBase:
        self.Config = Jun20Config.JunConfig()
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

