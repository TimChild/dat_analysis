from typing import Dict, List, NamedTuple

from src.DatCode.DatAttribute import DatAttribute

'''
Required Dat attribute
    Represents all basic logging functionality from SRSs, magnets, temperature probes, and anything else of interest
'''


class Logs(DatAttribute):
    """Holds most standard data from sweeplogs. Including x_array and y_array"""
    def __init__(self, infodict: Dict):
        logs = infodict['Logs']
        if 'srss' in logs.keys():
            if 'srs1' in logs['srss'].keys():
                self.srs1 = logs['srss']['srs1']
            if 'srs2' in logs['srss'].keys():
                self.srs2 = logs['srss']['srs2']
            if 'srs3' in logs['srss'].keys():
                self.srs3 = logs['srss']['srs3']
            if 'srs4' in logs['srss'].keys():
                self.srs4 = logs['srss']['srs4']

        # FIXME: DO magnets... Json needs working with first
        self.magx = None
        self.magy = None
        self.magz = None

        self.dacs = logs['dacs']
        self.dacnames = logs['dacnames']

        self.x_array = logs['x_array']  # type:np.ndarray
        self.y_array = logs['y_array']  # type:np.ndarray
        self.x_label = logs['axis_labels']['x']
        self.y_label = logs['axis_labels']['y']
        self.dim = logs['dim']  # type: int  # Number of dimensions to data

        self.time_elapsed = logs['time_elapsed']
        self.time_completed = logs['time_completed']
        self.temps = logs['temperatures']  # Stores temperatures in tuple e.g. self.temps.mc
