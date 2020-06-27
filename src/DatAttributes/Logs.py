from typing import Dict, NamedTuple
import re
import h5py
from src.DatBuilder import Exp_to_standard as E2S, Util
from src.DatAttributes.DatAttribute import DatAttribute
import logging
from dictor import dictor
import src.HDF.Util as HDU

logger = logging.getLogger(__name__)

'''
Required Dat attribute
    Represents all basic logging functionality from SRSs, magnets, temperature probes, and anything else of interest
'''

EXPECTED_TOP_ATTRS = ['version', 'comments', 'filenum', 'x_label', 'y_label', 'current_config', 'time_completed',
                      'time_elapsed']


class NewLogs(DatAttribute):
    version = '1.0'
    group_name = 'Logs'

    def __init__(self, hdf):
        super().__init__(hdf)
        self.Babydac = None
        self.Fastdac = None
        self.AWG = None

        self.comments = None
        self.filenum = None
        self.x_label = None
        self.y_label = None

        self.time_completed = None
        self.time_elapsed = None

        self.dim = None
        self.temps = None
        self.get_from_HDF()

    def update_HDF(self):
        logger.warning('Calling update_HDF on Logs attribute has no effect')
        pass

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
        fdac_json = HDU.get_attr(group, 'FastDACs', None)
        if fdac_json:
            self._set_fdacs(fdac_json)

        bdac_json = HDU.get_attr(group, 'BabyDACs', None)
        if bdac_json:
            self._set_bdacs(bdac_json)

        awg_tuple = HDU.get_attr(group, 'AWG', None)
        if awg_tuple:
            self.AWG = awg_tuple

        srss_group = group.get('srss', None)
        if srss_group:
            for key in srss_group.keys():
                if isinstance(srss_group[key], h5py.Group) and srss_group[key].attrs.get('description',
                                                                                         None) == 'NamedTuple':
                    setattr(self, key, HDU.get_attr(srss_group, key))

    def _set_bdacs(self, bdac_json):
        """Set values from BabyDAC json"""
        """dac dict should be stored in format:
                            visa_address: ...
                    """  # TODO: Fill this in
        dacs = {k: v for k, v in bdac_json.items() if k[:3] == 'DAC'}
        nums = [int(re.search('\d+', k)[0]) for k in dacs.keys()]
        names = [re.search('(?<={).*(?=})', k)[0] for k in dacs.keys()]
        dacs = dict(zip(nums, dacs.values()))
        dacnames = dict(zip(nums, names))
        self.Babydac = BABYDACtuple(dacs=dacs, dacnames=dacnames)


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

        dacs = dict(zip(nums, fdacs.values()))
        dacnames = dict(zip(nums, names))
        sample_freq = dictor(fdac_json, 'SamplingFreq', None)
        measure_freq = dictor(fdac_json, 'MeasureFreq', None)
        visa_address = dictor(fdac_json, 'visa_address', None)
        self.Fastdac = FASTDACtuple(dacs=dacs, dacnames=dacnames, sample_freq=sample_freq, measure_freq=measure_freq,
                                    visa_address=visa_address)



class InitLogs(object):
    """Class to contain all functions required for setting up Logs in HDF (so that Logs DA can get_from_hdf())"""
    BABYDAC_KEYS = ['com_port',
                    'DAC#{<name>}']  # TODO: not currently using DAC#{}, should figure out a way to check this
    FASTDAC_KEYS = ['SamplingFreq', 'MeasureFreq', 'visa_address', 'AWG', 'numADCs', 'DAC#{<name>}']
    AWG_KEYS = ['AW_Waves', 'AW_Dacs', 'waveLen', 'numADCs', 'samplingFreq', 'measureFreq', 'numWaves', 'numCycles',
                'numSteps']

    @staticmethod
    def check_key(k, expected_keys):
        if k in expected_keys:
            return
        elif k[0:3] == 'DAC':
            return
        else:
            logger.warning(f'Unexpected key in logs: k = {k}, Expected = {expected_keys}')
            return

    @staticmethod
    def set_babydac(group, babydac_json):
        """Puts info into Dat HDF"""
        """dac dict should be stored in format:
                        com_port: ...
                        DAC#{<name>}: <val>
                """
        if babydac_json is not None:
            for k, v in babydac_json.items():
                InitLogs.check_key(k, InitLogs.BABYDAC_KEYS)
            HDU.set_attr(group, 'BabyDACs', babydac_json)
        else:
            logger.info(f'No "BabyDAC" found in json')

    @staticmethod
    def set_fastdac(group, fdac_json):
        """Puts info into Dat HDF"""  # TODO: Make work for more than one fastdac
        """fdac dict should be stored in format:
                                visa_address: ...
                                SamplingFreq:
                                DAC#{<name>}: <val>
                                ADC#: <val>
    
                                ADCs not currently required
                                """
        if fdac_json is not None:
            for k, v in fdac_json.items():
                InitLogs.check_key(k, InitLogs.FASTDAC_KEYS)
            HDU.set_attr(group, 'FastDACs', fdac_json)
        else:
            logger.info(f'No "FastDAC" found in json')

    @staticmethod
    def set_awg(group, fdac_json):
        """Put info into Dat HDF"""
        if awg_logs := dictor(fdac_json, 'AWG', None) is not None:
            # Check keys make sense
            for k, v in awg_logs.items():
                InitLogs.check_key(k, InitLogs.AWG_KEYS)

            # Get dict of data how I like
            awg_data = E2S.awg_from_json(awg_logs)

            # Store in NamedTuple
            ntuple = Util.data_to_NamedTuple(awg_data, AWGtuple)
            HDU.set_attr(group, 'AWG', ntuple)
        else:
            logger.info(f'No "AWG" found in "FastDAC" part of json or not "FastDAC" json')

    @staticmethod
    def set_srss(group, json):
        """Sets SRS values in Dat HDF from either full sweeplogs or minimally json which contains SRS_{#} keys"""
        srs_ids = [key[4] for key in json.keys() if key[:3] == 'SRS']

        for num in srs_ids:
            if f'SRS_{num}' in json.keys():
                srs_data = E2S.srs_from_json(json, num)  # Converts to my standard
                ntuple = Util.data_to_NamedTuple(srs_data, SRStuple)  # Puts data into named tuple
                srs_group = group.require_group(f'srss')  # Make sure there is an srss group
                HDU.set_attr(srs_group, f'srs{num}', ntuple)  # Save in srss group
            else:
                logger.error(f'No "SRS_{num}" found in json')  # Should not get to here

    @staticmethod
    def set_simple_attrs(group, json):
        """Sets top level attrs in Dat HDF from sweeplogs"""
        group.attrs['comments'] = dictor(json, 'comment', '')
        group.attrs['filenum'] = dictor(json, 'filenum', 0)
        group.attrs['x_label'] = dictor(json, 'axis_labels.x', 'None')
        group.attrs['y_label'] = dictor(json, 'axis_labels.y', 'None')
        group.attrs['current_config'] = dictor(json, 'current_config', None)
        group.attrs['time_completed'] = dictor(json, 'time_completed', None)
        group.attrs['time_elapsed'] = dictor(json, 'time_elapsed', None)


class SRStuple(NamedTuple):
    gpib: int
    out: int
    tc: float
    freq: float
    phase: float
    sens: float
    harm: int
    CH1readout: int


class MAGtuple(NamedTuple):
    field: float
    rate: float


class TEMPtuple(NamedTuple):
    mc: float
    still: float
    mag: float
    fourk: float
    fiftyk: float


class AWGtuple(NamedTuple):
    outputs: dict  # The AW_waves with corresponding dacs outputting them. i.e. {0: [1,2], 1: [3]} for dacs 1,2
    # outputting AW 0
    wave_len: int  # in samples
    num_adcs: int  # how many ADCs being recorded
    samplingFreq: float
    measureFreq: float
    num_cycles: int  # how many repetitions of wave per dac step
    num_steps: int  # how many DAC steps


class FASTDACtuple(NamedTuple):
    dacs: dict
    dacnames: dict
    sample_freq: float
    measure_freq: float
    visa_address: str


class BABYDACtuple(NamedTuple):
    dacs: dict
    dacnames: dict


############################# OLD LOGS BELOW #########################

#
# class Logs(object):
#     """Holds most standard data from sweeplogs. Including x_array and y_array"""
#     version = '1.1'
#     """
#     Version History
#         1.0 -- means hdfpath added manually
#         1.1 -- Added version and changed sweeprate to sweep_rate temporarily
#     """
#     def __init__(self, infodict: Dict):
#         self.version = Logs.version
#         logs = infodict['Logs']
#         self._logs = logs
#         self._hdf_path = infodict.get('hdfpath', None)
#         if 'srss' in logs.keys():
#             if 'srs1' in logs['srss'].keys():
#                 self.srs1 = logs['srss']['srs1']
#             if 'srs2' in logs['srss'].keys():
#                 self.srs2 = logs['srss']['srs2']
#             if 'srs3' in logs['srss'].keys():
#                 self.srs3 = logs['srss']['srs3']
#             if 'srs4' in logs['srss'].keys():
#                 self.srs4 = logs['srss']['srs4']
#
#         if 'mags' in logs.keys():
#             if 'magx' in logs['mags'].keys():
#                 self.magx = logs['mags']['magx']
#             if 'magy' in logs['mags'].keys():
#                 self.magy = logs['mags']['magy']
#             if 'magz' in logs['mags'].keys():
#                 self.magz = logs['mags']['magz']
#
#         self.dacs = logs.get('dacs', None)
#         self.dacnames = logs.get('dacnames', None)
#
#         self.fdacs = logs.get('fdacs', None)
#         self.fdacnames = logs.get('fdacnames', None)
#         self.fdacfreq = logs.get('fdacfreq', None)
#
#         self.x_array = logs.get('x_array', None)  # type:np.ndarray
#         self.y_array = logs.get('y_array', None)  # type:np.ndarray
#         if 'axis_labels' in logs:
#             axis_labels = logs['axis_labels']
#             self.x_label = axis_labels.get('x', None)
#             self.y_label = axis_labels.get('y', None)
#         self.dim = logs.get('dim', None)  # type: int  # Number of dimensions to data
#
#         self.time_elapsed = logs.get('time_elapsed', None)
#         self.time_completed = logs.get('time_completed', None)
#         self.temps = logs.get('temperatures', None)  # Stores temperatures in dict e.g. self.temps['mc']
#
#         self.mc_temp = self.temps['mc']*1000 if self.temps else None
#
#         self.comments = logs.get('comments', None)
#
#
#     @property
#     def temp(self):
#         return self.temps['mc']*1000 if self.temps else None
#
#     @property
#     def sweeprate(self):  # TODO: make it so select 'property' values are loaded into datdf.. For now quick fix is to set self.sweep_rate
#         self.sweep_rate = _get_sweeprate(self.x_array, self.y_array, self.fdacfreq, self.time_elapsed, hdf_path = CU.get_full_path(self._hdf_path))
#         return self.sweep_rate
#
#     def set_hdf_path(self, hdf_path):
#         self._hdf_path = hdf_path
#         self.version = '1.0'
#
#     def add_mags(self, mag_dict):  # Mostly a temporary function for adding magnet data after initialization
#         if 'magx' in mag_dict.keys():
#             self.magx = mag_dict['magx']
#         if 'magy' in mag_dict.keys():
#             self.magy = mag_dict['magy']
#         if 'magz' in mag_dict.keys():
#             self.magz = mag_dict['magz']
#
#
# def _get_sweeprate(x, y, freq, time, hdf_path=None):
#     width = abs(x[-1] - x[1])
#     numpnts = len(x)
#     numadc = _num_adc(x, y, freq, time, hdf_path)  # TODO: Improve this... Trying to guess this based on total numpnts and time taken for now
#     if numadc is not None and freq is not None:
#         return width * freq / (numpnts * numadc)
#     else:
#         return None
#
#
# def _num_adc(x, y, freq, duration, hdf_path=None):
#     """Temporary fn which guesses num adc based on duration of scan"""
#     if hdf_path is not None:  # Look at how many waves are saved with ADC...
#         data_keys = _get_data_keys(hdf_path)
#         i=0
#         for name in data_keys:  # Look for all datawaves saved as ADC... relies on saving raw data!!
#             if re.search('ADC', name):
#                 i+=1
#         if i!=0:
#             return i
#     else:  # Else move on to trying to estimate from numpnts etc
#         pass
#
#     if y is None:  # if 1D array
#         numy = 1
#     else:
#         numy = len(y)
#     if freq is None or duration is None:
#         return None
#     numpnts = len(x)*numy
#     x_dist = abs(x[-1]-x[1])
#     num_adc = round(freq*(duration-numy*0.5-numy*x_dist/1000)/numpnts)  # assuming 0.5s to ramp/delay per line
#     if 1 <= num_adc <= 4:
#         return num_adc
#     else:
#         return None
#
#
# def _get_data_keys(hdf_path):
#     hdf = h5py.File(hdf_path, 'r')
#     keylist = hdf.keys()
#     data_keys = []
#     for key in keylist:
#         if isinstance(hdf[key], h5py.Dataset):  # Make sure it's a dataset not metadata
#             data_keys.append(key)
#     return data_keys
#
