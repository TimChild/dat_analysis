from typing import Dict, List, NamedTuple
import re
import h5py
from src.DatCode.DatAttribute import DatAttribute
import src.CoreUtil as CU
'''
Required Dat attribute
    Represents all basic logging functionality from SRSs, magnets, temperature probes, and anything else of interest
'''


class Logs(DatAttribute):
    """Holds most standard data from sweeplogs. Including x_array and y_array"""
    version = '1.1'
    """
    Version History
        1.0 -- means hdfpath added manually
        1.1 -- Added version and changed sweeprate to sweep_rate temporarily
    """
    def __init__(self, infodict: Dict):
        self.version = Logs.version
        logs = infodict['Logs']
        self._logs = logs
        self._hdf_path = infodict.get('hdfpath', None)
        if 'srss' in logs.keys():
            if 'srs1' in logs['srss'].keys():
                self.srs1 = logs['srss']['srs1']
            if 'srs2' in logs['srss'].keys():
                self.srs2 = logs['srss']['srs2']
            if 'srs3' in logs['srss'].keys():
                self.srs3 = logs['srss']['srs3']
            if 'srs4' in logs['srss'].keys():
                self.srs4 = logs['srss']['srs4']

        if 'mags' in logs.keys():
            if 'magx' in logs['mags'].keys():
                self.magx = logs['mags']['magx']
            if 'magy' in logs['mags'].keys():
                self.magy = logs['mags']['magy']
            if 'magz' in logs['mags'].keys():
                self.magz = logs['mags']['magz']

        self.dacs = logs.get('dacs', None)
        self.dacnames = logs.get('dacnames', None)

        self.fdacs = logs.get('fdacs', None)
        self.fdacnames = logs.get('fdacnames', None)
        self.fdacfreq = logs.get('fdacfreq', None)

        self.x_array = logs.get('x_array', None)  # type:np.ndarray
        self.y_array = logs.get('y_array', None)  # type:np.ndarray
        if 'axis_labels' in logs:
            axis_labels = logs['axis_labels']
            self.x_label = axis_labels.get('x', None)
            self.y_label = axis_labels.get('y', None)
        self.dim = logs.get('dim', None)  # type: int  # Number of dimensions to data

        self.time_elapsed = logs.get('time_elapsed', None)
        self.time_completed = logs.get('time_completed', None)
        self.temps = logs.get('temperatures', None)  # Stores temperatures in dict e.g. self.temps['mc']

        self.mc_temp = self.temps['mc']*1000 if self.temps else None

        self.comments = logs.get('comments', None)


    @property
    def temp(self):
        return self.temps['mc']*1000 if self.temps else None

    @property
    def sweeprate(self):  # TODO: make it so select 'property' values are loaded into datdf.. For now quick fix is to set self.sweep_rate
        self.sweep_rate = _get_sweeprate(self.x_array, self.y_array, self.fdacfreq, self.time_elapsed, hdf_path = CU.get_full_path(self._hdf_path))
        return self.sweep_rate

    def set_hdf_path(self, hdf_path):
        self._hdf_path = hdf_path
        self.version = '1.0'

    def add_mags(self, mag_dict):  # Mostly a temporary function for adding magnet data after initialization
        if 'magx' in mag_dict.keys():
            self.magx = mag_dict['magx']
        if 'magy' in mag_dict.keys():
            self.magy = mag_dict['magy']
        if 'magz' in mag_dict.keys():
            self.magz = mag_dict['magz']


def _get_sweeprate(x, y, freq, time, hdf_path=None):
    width = abs(x[-1] - x[1])
    numpnts = len(x)
    numadc = _num_adc(x, y, freq, time, hdf_path)  # TODO: Improve this... Trying to guess this based on total numpnts and time taken for now
    if numadc is not None and freq is not None:
        return width * freq / (numpnts * numadc)
    else:
        return None


def _num_adc(x, y, freq, duration, hdf_path=None):
    """Temporary fn which guesses num adc based on duration of scan"""
    if hdf_path is not None:  # Look at how many waves are saved with ADC...
        data_keys = _get_data_keys(hdf_path)
        i=0
        for name in data_keys:  # Look for all datawaves saved as ADC... relies on saving raw data!!
            if re.search('ADC', name):
                i+=1
        if i!=0:
            return i
    else:  # Else move on to trying to estimate from numpnts etc
        pass

    if y is None:  # if 1D array
        numy = 1
    else:
        numy = len(y)
    if freq is None or duration is None:
        return None
    numpnts = len(x)*numy
    x_dist = abs(x[-1]-x[1])
    num_adc = round(freq*(duration-numy*0.5-numy*x_dist/1000)/numpnts)  # assuming 0.5s to ramp/delay per line
    if 1 <= num_adc <= 4:
        return num_adc
    else:
        return None

def _get_data_keys(hdf_path):
    hdf = h5py.File(hdf_path, 'r')
    keylist = hdf.keys()
    data_keys = []
    for key in keylist:
        if isinstance(hdf[key], h5py.Dataset):  # Make sure it's a dataset not metadata
            data_keys.append(key)
    return data_keys