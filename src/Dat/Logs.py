from typing import Dict, List, NamedTuple

from src.Dat.DatAttribute import DatAttribute

'''
Required Dat attribute
    Represents all basic logging functionality from SRSs, magnets, temperature probes, and anything else of interest
'''
class Logs(DatAttribute):
    def __init__(self, infodict: Dict):
        logs = infodict['Logs']
        self.srs1 = None
        self.srs2 = None
        self.srs3 = None
        self.srs4 = None
        self.magx = None
        self.magy = None
        self.magz = None

        self.dacs = logs['dacs']
        self.dacnames = logs['dacnames']

        self.x_array = logs['xarray']  # type:np.ndarray
        self.y_array = logs['yarray']  # type:np.ndarray
        self.x_label = logs['axis_labels']['x']
        self.y_label = logs['axis_labels']['y']
        self.dim = logs['dim']  # type: int  # Number of dimensions to data

        self.time_elapsed = logs['time_elapsed']
        self.time_completed = logs['time_completed']
        self.temps = logs['temperatures']  # Stores temperatures in tuple e.g. self.temps.mc

        # self.set_instr_vals('mag', infodict['mags'])
        # self.set_instr_vals('srs', infodict['srss'])

    # TODO: I think this fucntion is not fully complete, not sure
    def set_instr_vals(self, name: str, data: List[NamedTuple]):
        if data is not None:
            # data should be a List of namedtuples for instrument, First field should be ID (e.g. 1 or x)
            for ntuple in data:
                evalstr = f'self.{name}{ntuple[0]} = {ntuple}'
                exec(evalstr)
        return None


