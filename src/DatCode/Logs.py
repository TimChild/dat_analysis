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
        self._logs = logs
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

        self.fdacs = logs['fdacs']
        self.fdacnames = logs['fdacnames']
        self.fdacfreq = logs['fdacfreq']

        self.x_array = logs['x_array']  # type:np.ndarray
        self.y_array = logs['y_array']  # type:np.ndarray
        self.x_label = logs['axis_labels']['x']
        self.y_label = logs['axis_labels']['y']
        self.dim = logs['dim']  # type: int  # Number of dimensions to data

        self.time_elapsed = logs['time_elapsed']
        self.time_completed = logs['time_completed']
        self.temps = logs['temperatures']  # Stores temperatures in dict e.g. self.temps['mc']
        self.mc_temp = self.temps['mc']*1000

        self.comments = logs['comments']
        self.sweeprate = None
        self.calc_sweeprate()

    @property
    def temp(self):
        return self.temps['mc']*1000

    def calc_sweeprate(self):
        self.sweeprate = _get_sweeprate(self.x_array, self.y_array, self.fdacfreq, self.time_elapsed)


def _get_sweeprate(x, y, freq, time):
    width = abs(x[-1] - x[1])
    numpnts = len(x)
    numadc = _num_adc(x, y, freq, time)  # TODO: Improve this... Trying to guess this based on total numpnts and time taken for now
    if numadc is not None and freq is not None:
        return width * freq / (numpnts * numadc)
    else:
        return None


def _num_adc(x, y, freq, duration):
    """Temporary fn which guesses num adc based on duration of scan"""
    if y is None:  # if 1D array
        numy = 1
    else:
        numy = len(y)
    if freq is None or duration is None:
        return None
    numpnts = len(x)*numy
    num_adc = round(freq*(duration-numy*0.5)/numpnts)  # assuming 0.5s to ramp/delay per line
    if 1 <= num_adc <= 4:
        return num_adc
    else:
        return None