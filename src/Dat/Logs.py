from typing import Dict, List, NamedTuple


class Logs(object):
    def __init__(self, infodict: Dict):
        self.srs1 = None
        self.srs2 = None
        self.srs3 = None
        self.srs4 = None
        self.magx = None
        self.magy = None
        self.magz = None

        self.sweeplogs = infodict['sweeplogs']  # type: dict  # Full JSON formatted sweeplogs
        self.sc_config = infodict['sc_config']  # type: dict  # Full JSON formatted sc_config
        self.time_elapsed = self.sweeplogs['time_elapsed']
        self.temps = infodict['temperatures'] # Stores temperatures in tuple e.g. self.temps.mc

    def instr_vals(self, name: str, data: List[NamedTuple]):
        if data is not None:
            for ntuple in data:  # data should be a List of namedtuples for instrument, First field should be ID (e.g. 1 or x)
                evalstr = f'self.{name}{ntuple[0]} = {ntuple}'
                exec(evalstr)
        return None