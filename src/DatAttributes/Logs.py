from typing import Dict, NamedTuple
from collections import namedtuple
import re
import h5py

import src.DatBuilder.Util
from src.DatBuilder import Exp_to_standard as E2S
from src.Configs import Main_Config as cfg
from src.DatAttributes import DatAttribute as DA
from src.DatAttributes.DatAttribute import DatAttribute
import src.CoreUtil as CU
import logging
import ast
from dictor import dictor

logger = logging.getLogger(__name__)

'''
Required Dat attribute
    Represents all basic logging functionality from SRSs, magnets, temperature probes, and anything else of interest
'''

EXPECTED_TOP_ATTRS = ['version', 'comments', 'filenum', 'x_label', 'y_label', 'current_config', 'time_completed', 'time_elapsed']


class NewLogs(DatAttribute):
    version = '3.0'
    group_name = 'Logs'

    def __init__(self, hdf):
        super().__init__(hdf)
        self.dacs = None
        self.dacnames = None

        self.fdacs = None
        self.fdacnames = None
        self.fdacfreq = None

        self.comments = None
        self.filenum = None
        self.x_label = None
        self.y_label = None
        self.current_config = None
        self.time_completed = None
        self.time_elapsed = None

        self.dim = None
        self.temps = None
        self.mc_temp = None
        self.get_from_HDF()

    def _set_default_group_attrs(self):
        super()._set_default_group_attrs()

    def get_from_HDF(self):
        group = self.group
        # Get top level attrs
        for k, v in group.attrs.items():
            if k in EXPECTED_TOP_ATTRS:
                setattr(self, k, v)
            else:
                logger.info(f'Attr [{k}] in Logs group attrs unexpectedly')

        # Get instr attrs
        fdac_group = group.get('FastDACs', None)
        if fdac_group:
            fdac_json = load_simple_dict_from_hdf(fdac_group)
            self._set_fdacs(fdac_json)

        bdac_group = group.get('BabyDACs', None)
        if bdac_group:
            bdac_json = load_simple_dict_from_hdf(bdac_group)
            self._set_bdacs(bdac_json)

        srss_group = group.get('srss', None)
        if srss_group:
            for key in srss_group.keys():
                if key[:3] == 'srs':
                    srs_group = srss_group[key]
                    setattr(self, key, load_group_to_namedtuple(srss_group))

    def _set_bdacs(self, bdac_json):
        """Set values from BabyDAC json"""
        """dac dict should be stored in format:
                            visa_address: ...
                    """  # TODO: Fill this in
        dacs = {k: v for k, v in bdac_json.items() if k[:3] == 'DAC'}
        vals = None  # Need to look at how this comes back again
        nums = None
        names = None
        self.dacs = dict(zip(nums, vals))
        self.dacnames = dict(zip(nums, names))
        raise NotImplementedError

    def _set_fdacs(self, fdac_json):
        """Set values from FastDAC json"""  # TODO: Make work for more than one fastdac
        """fdac dict should be stored in format:
                                visa_address: ...
                                SamplingFreq:
                                DAC#{<name>}: <val>
                                ADC#: <val>

                                ADCs not currently required
                                """
        fdacs = {k: v for k, v in fdac_json.items() if k[:3] == 'DAC'}
        nums = [int(re.search('\d+', k)[0]) for k in fdacs.keys()]
        names = [re.search('(?<={).*(?=})', k)[0] for k in fdacs.keys()]
        self.fdacs = dict(zip(nums, fdacs.values()))
        self.fdacnames = dict(zip(nums, names))
        self.fdacfreq = dictor(fdac_json, 'SamplingFreq', None)


def _init_logs_set_babydac(group, babydac_json):
    """Puts info into Dat HDF"""
    """dac dict should be stored in format:
                    visa_address: ...
            """  # TODO: Fill this in
    if babydac_json is not None:
        dacs_group = group.create_group('BabyDACs')
        bdac_dict = dictor(babydac_json, 'BabyDAC')
        save_simple_dict_to_hdf(dacs_group, bdac_dict)
    else:
        logger.info(f'No "BabyDAC" found in json')


def _init_logs_set_fastdac(group, fdac_json):
    """Puts info into Dat HDF"""  # TODO: Make work for more than one fastdac
    """fdac dict should be stored in format:
                            visa_address: ...
                            SamplingFreq:
                            DAC#{<name>}: <val>
                            ADC#: <val>

                            ADCs not currently required
                            """
    if fdac_json is not None:
        fdac_group = group.create_group('FastDACs')
        save_simple_dict_to_hdf(fdac_group, fdac_json)
    else:
        logger.info(f'No "FastDAC" found in json')


def _init_logs_set_srss(group, json):
    """Sets SRS values in Dat HDF from either full sweeplogs or minimally json which contains SRS_{#} keys"""
    for i in range(1, cfg.current_config.instrument_num['srs'] + 1 + 1):
        if f'SRS_{i}' in json.keys():
            srs_dict = dictor(json, f'SRS_{i}', checknone=True)
            srs_data = E2S.srs_from_json(srs_dict, i)  # Converts to my standard
            srs_id, srs_tuple = DA.get_key_ntuple('srs', i)  # Gets named tuple to store
            ntuple = src.DatBuilder.Util.data_to_NamedTuple(srs_data, srs_tuple)  # Puts data into named tuple

            srs_group = group.require_group(f'srss/{srs_id}')  # Make group in HDF
            save_namedtuple_to_group(ntuple, srs_group)
        else:
            logger.info(f'No "SRS_{i}" found in json')


def _init_logs_set_simple_attrs(group, json):
    """Sets top level attrs in Dat HDF from sweeplogs"""
    group.attrs['comments'] = dictor(json, 'comment', '')
    group.attrs['filenum'] = dictor(json, 'filenum', 0)
    group.attrs['x_label'] = dictor(json, 'axis_labels.x', 'None')
    group.attrs['y_label'] = dictor(json, 'axis_labels.y', 'None')
    group.attrs['current_config'] = dictor(json, 'current_config', None)
    group.attrs['time_completed'] = dictor(json, 'time_completed', None)
    group.attrs['time_elapsed'] = dictor(json, 'time_elapsed', None)


def save_simple_dict_to_hdf(group: h5py.Group, simple_dict: dict):
    """
    Saves simple dict where depth is only 1 and values can be stored in h5py attrs
    @param group:
    @type group:
    @param simple_dict:
    @type simple_dict:
    @return:
    @rtype:
    """
    for k, v in simple_dict.items():
        group.attrs[k] = v


def load_simple_dict_from_hdf(fdac_group: h5py.Group):
    """Inverse of save_simple_dict_to_hdf returning to same form"""
    d = {}
    for k, v in fdac_group.attrs.items():
        d[k] = v
    return d


def save_namedtuple_to_group(ntuple: NamedTuple, group: h5py.Group):
    """Saves named tuple inside group given"""
    for key in ntuple:
        group.attrs[key] = ntuple[key]  # Store as attrs of group in HDF


def load_group_to_namedtuple(group: h5py.Group):
    """Returns namedtuple with name of group and key: values of group attrs
    e.g. srs1 group which has gpib: 1... will be returned as an srs1 namedtuple with .gpib etc
    """
    name = group.name.split('/')[-1]
    d = {key: val for key, val in group.attrs.items()}
    for k, v in d.items():
        try:
            temp = ast.literal_eval(v)
            if isinstance(temp, dict):
                d[k] = temp
        except ValueError as e:
            pass

    ntuple = namedtuple(name, d.keys())
    return ntuple

############################# OLD LOGS BELOW #########################


class Logs(object):
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


