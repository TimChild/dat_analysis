from __future__ import annotations

import json
from typing import NamedTuple, Tuple, List, TYPE_CHECKING
import re
import h5py
import datetime
from dataclasses import dataclass
import logging
from dictor import dictor

from src.data_standardize.standardize_util import logger
from src.dat_object.Attributes.DatAttribute import DatAttribute
import src.hdf_util as HDU
from src.hdf_util import with_hdf_read, with_hdf_write, NotFoundInHdfError, DatDataclassTemplate
import src.core_util as CU
from src.core_util import my_partial, data_to_NamedTuple

if TYPE_CHECKING:
    from src.data_standardize.exp_config import ExpConfigGroupDatAttribute
    from src.dat_object.dat_hdf import DatHDF

logger = logging.getLogger(__name__)


'''
Required Dat attribute
    Represents all basic logging functionality from SRSs, magnets, temperature probes, and anything else of interest
'''

EXPECTED_TOP_ATTRS = ['version', 'comments', 'filenum', 'x_label', 'y_label', 'current_config', 'time_completed',
                      'time_elapsed', 'part_of']


@dataclass
class SRSs(DatDataclassTemplate):
    srs1: SRStuple = None
    srs2: SRStuple = None
    srs3: SRStuple = None
    srs4: SRStuple = None


@dataclass
class MAGs(DatDataclassTemplate):
    magx: Magnet = None
    magy: Magnet = None
    magz: Magnet = None


@dataclass
class FastDac(DatDataclassTemplate):
    measure_freq: float
    sampling_freq: float
    visa_address: str
    dacs: dict


# Just to shorten making properties below
pp = DatAttribute.property_prop
mp = my_partial


class Logs(DatAttribute):
    version = '2.0.0'
    group_name = 'Logs'
    description = 'Information stored in experiment sweeplogs plus some other information drawn from other recorded ' \
                  'such as Fastdac measure speed etc'

    def __init__(self, dat: DatHDF):
        super().__init__(dat)
        self._srss = None
        self._mags = None

    sweeplogs: dict = property(mp(pp, 'sweeplogs'),)
    bds: dict = property(mp(pp, 'BabyDACs'),)
    Fastdac: FastDac = property(mp(pp, 'FastDACs', dataclass=FastDac),)
    fds: dict = property(lambda self: self.Fastdac.dacs)  # shorter way to get to just the dac values
    dacs: dict = property(lambda self: {**self.bds, **self.fds})  # All DACs in a single dict, Fastdacs overwrite BabyDacs with same name
    awg: dict = property(mp(pp, 'AWG'),)
    temps: dict = property(mp(pp, 'Temperatures'),)
    mags: dict = property(mp(pp, 'Mags'),)
    xlabel: str = property(mp(pp, 'xlabel'))
    ylabel: str = property(mp(pp, 'ylabel'))
    comments: str = property(mp(pp, 'comments'))
    sweeprate: float = property(mp(pp, 'sweeprate'))
    measure_freq: float = property(mp(pp, 'measure_freq'))
    sampling_freq: float = property(mp(pp, 'sampling_freq'))
    time_completed: datetime.datetime = property(mp(pp, 'time_completed'))
    time_elapsed: float = property(mp(pp, 'time_elapsed'))

    @property
    def srss(self):
        if not self._srss:
            self._srss = self._get_srss()
        return self._srss

    @with_hdf_read
    def _get_srss(self) -> SRSs:
        group = self.hdf.group
        srss_group = group.get('srss', None)
        if srss_group:
            srss = {}
            for key in srss_group.keys():
                if isinstance(srss_group[key], h5py.Group) and srss_group[key].attrs.get('description',
                                                                                         None) == 'NamedTuple':
                    srss[key] = HDU.get_attr(srss_group, key)
            return SRSs(**srss)
        else:
            raise KeyError(f'"srss" not found in Logs group')

    @property
    def mags(self):
        if not self._mags:
            self._mags = self._get_mags()
        return self._mags

    @with_hdf_read
    def _get_mags(self):
        group = self.hdf.group
        mags_group = group.get('Magnets', None)
        if mags_group:
            mags = {}
            for key in mags_group.keys():
                if isinstance(mags_group[key], h5py.Group) and mags_group[key].attrs.get('description', None) == 'NamedTuple':
                    mags[key] = HDU.get_attr(mags_group, key)
            return MAGs(**mags)
        else:
            raise NotFoundInHdfError(f'Magnets not found in Logs group. Available keys are {group.keys()}')

    def initialize_minimum(self):
        """Initialize data into HDF"""
        self._init_sweeplogs()  # Do this first so that other inits can use it
        self._init_srss()
        self._init_babydac()
        self._init_fastdac()
        self._init_awg()
        self._init_temps()
        self._init_mags()
        self._init_other()
        self.initialized = True

    def _init_sweeplogs(self):
        """Save fixed sweeplogs in Logs group"""
        sweeplogs = self._get_sweeplogs_from_exp()
        self.set_group_attr('sweeplogs', sweeplogs)

    def _get_sweeplogs_from_exp(self) -> dict:
        """Get fixed sweeplogs through ExpConfig attribute (which can fix things from experiment first)"""
        Exp_config: ExpConfigGroupDatAttribute = self.dat.ExpConfig
        return Exp_config.get_sweeplogs()

    @with_hdf_write
    def _init_srss(self):
        group = self.hdf.group
        sweeplogs = self.sweeplogs
        InitLogs.set_srss(group, sweeplogs)

    def _init_babydac(self):
        sweeplogs = self.sweeplogs
        babydac_dict = sweeplogs.get('BabyDAC', None)
        if babydac_dict:
            self._set_babydac(babydac_dict)

    def _init_fastdac(self):
        sweeplogs = self.sweeplogs
        fastdac_dict = sweeplogs.get('FastDAC', None)
        if fastdac_dict:
            self._set_fastdac(fastdac_dict)

    @with_hdf_write
    def _init_awg(self):
        group = self.hdf.group
        sweeplogs = self.sweeplogs
        awg_dict = dictor(sweeplogs, 'FastDAC.AWG', None)
        if awg_dict:
            InitLogs.set_awg(group, awg_dict)

    @with_hdf_write
    def _init_temps(self):
        group = self.hdf.group
        sweeplogs = self.sweeplogs
        temp_dict = sweeplogs.get('Temperatures', None)
        if temp_dict:
            InitLogs.set_temps(group, temp_dict)

    @with_hdf_write
    def _init_mags(self):
        logger.warning(f'Need to make _init_mags more permanent...')  # TODO: Need to improve getting mags from sweeplogs
        group = self.hdf.group
        sweeplogs = self.sweeplogs
        mag_dict = sweeplogs.get('LS625 Magnet Supply', None)
        if mag_dict:
            self._set_mags(mag_dict)

    @with_hdf_write
    def _init_other(self):
        sweeplogs = self.sweeplogs
        other_name_paths = {
            'comments': 'comment',
            'xlabel': 'axis_labels.x',
            'ylabel': 'axis_labels.y',
            'sampling_freq': 'FastDAC.SamplingFreq',
            'measure_freq': 'FastDAC.MeasureFreq',
            'time_completed': 'time_completed',
            'time_elapsed': 'time_elapsed',
        }
        for name, path in other_name_paths.items():
            val = dictor(sweeplogs, path, default=None)
            if val is not None:
                self.set_group_attr(name, val)

        # Try other things (make sure that if they fail it doesn't stop everything!
        try:
            x = self.dat.Data.x_array
            sweeprate = CU.get_sweeprate(dictor(sweeplogs, 'FastDAC.MeasureFreq', checknone=True), x)
            self.set_group_attr('sweeprate', sweeprate)
        except Exception as e:
            logger.warning(f'When trying to get sweeprate, {e} was raised.')
            pass



    @with_hdf_write
    def _set_mags(self, mag_dict: dict):  # TODO: Make this more general... the whole get mags will only read 1 mag because they are duplicated in the sweeplogs
        group = self.hdf.group
        mag = _get_mag_field(mag_dict)
        mags_group = group.require_group(f'Magnets')  # Make sure there is an srss group
        HDU.set_attr(mags_group, mag.name, mag)  # Save in srss group

    def _set_babydac(self, babydac_dict):
        bds = _dac_logs_to_dict(babydac_dict)
        self.set_group_attr('BabyDACs', bds)

    @with_hdf_write
    def _set_fastdac(self, fastdac_dict):
        group = self.hdf.get(self.group_name)
        fds = _dac_logs_to_dict(fastdac_dict)
        # self.set_group_attr('FastDACs', fds)
        additional_attrs = {}
        for name, k in zip(['sampling_freq', 'measure_freq', 'visa_address'], ['SamplingFreq', 'MeasureFreq', 'visa_address']):
            additional_attrs[name] = dictor(fastdac_dict, k, None)
        fastdac = FastDac(**additional_attrs, dacs=fds)
        fastdac.save_to_hdf(group, name='FastDACs')


def _dac_logs_to_dict(dac_dict) -> dict:
    dacs = {k: v for k, v in dac_dict.items() if k[:3] == 'DAC'}
    names = [re.search('(?<={).*(?=})', k)[0] for k in dacs.keys()]
    nums = [int(re.search('\d+', k)[0]) for k in dacs.keys()]

    # Make sure they are in order
    dacs = dict(sorted(zip(nums, dacs.values())))
    names = dict(sorted(zip(nums, names)))

    return _dac_dict(dacs, names)


def _get_mag_field(mag_dict: dict) -> Magnet:  # TEMPORARY
    field = mag_dict['field mT']
    rate = mag_dict['rate mT/min']
    variable_name = mag_dict['variable name']
    mag = Magnet(variable_name, field, rate)
    return mag


# class OldLogs(DatAttribute):
#     version = '1.0'
#     group_name = 'Logs'
#
#     def __init__(self, hdf):
#         super().__init__(hdf)
#         self.full_sweeplogs = None
#
#         self.Babydac: BABYDACtuple = None
#         self.Fastdac: FASTDACtuple = None
#         self.AWG: AWGtuple = None
#
#         self.comments = None
#         self.filenum = None
#         self.x_label = None
#         self.y_label = None
#
#         self.time_completed = None
#         self.time_elapsed = None
#
#         self.part_of = None
#
#         self.dim = None
#         self.temps = None
#         self.get_from_HDF()
#
#     @property
#     def fds(self):
#         return _dac_dict(self.Fastdac.dacs, self.Fastdac.dacnames) if self.Fastdac else None
#
#     @property
#     def bds(self):
#         return _dac_dict(self.Babydac.dacs, self.Babydac.dacnames) if self.Babydac else None
#
#     @property
#     def sweeprate(self):
#         sweeprate = None
#         measure_freq = self.Fastdac.measure_freq if self.Fastdac else None
#         if measure_freq:
#             data_group = self.hdf.get('Data')
#             if data_group:
#                 x_array = data_group.get('Exp_x_array')
#                 if x_array:
#                     sweeprate = CU.get_sweeprate(measure_freq, x_array)
#         return sweeprate
#
#     def update_HDF(self):
#         logger.warning('Calling update_HDF on Logs attribute has no effect')
#         pass
#
#     def _check_default_group_attrs(self):
#         super()._check_default_group_attrs()
#
#     def get_from_HDF(self):
#         group = self.group
#
#         # Get full copy of sweeplogs
#         self.full_sweeplogs = HDU.get_attr(group, 'Full sweeplogs', None)
#
#         # Get top level attrs
#         for k, v in group.attrs.items():
#             if k in EXPECTED_TOP_ATTRS:
#                 setattr(self, k, v)
#             elif k in ['description']:  # HDF only attr
#                 pass
#             else:
#                 logger.info(f'Attr [{k}] in Logs group attrs unexpectedly')
#
#         # Get instr attrs
#         fdac_json = HDU.get_attr(group, 'FastDACs', None)
#         if fdac_json:
#             self._set_fdacs(fdac_json)
#
#         bdac_json = HDU.get_attr(group, 'BabyDACs', None)
#         if bdac_json:
#             self._set_bdacs(bdac_json)
#
#         awg_tuple = HDU.get_attr(group, 'AWG', None)
#         if awg_tuple:
#             self.AWG = awg_tuple
#
#         srss_group = group.get('srss', None)
#         if srss_group:
#             for key in srss_group.keys():
#                 if isinstance(srss_group[key], h5py.Group) and srss_group[key].attrs.get('description',
#                                                                                          None) == 'NamedTuple':
#                     setattr(self, key, HDU.get_attr(srss_group, key))
#
#         mags_group = group.get('mags', None)
#         if mags_group:
#             for key in mags_group.keys():
#                 if isinstance(mags_group[key], h5py.Group) and mags_group[key].attrs.get('description', None) == 'dataclass':
#                     setattr(self, key, HDU.get_attr(mags_group, key))
#
#         temp_tuple = HDU.get_attr(group, 'Temperatures', None)
#         if temp_tuple:
#             self.temps = temp_tuple
#
#     def _set_bdacs(self, bdac_json):
#         """Set values from BabyDAC json"""
#         """dac dict should be stored in format:
#                             visa_address: ...
#                     """  # TODO: Fill this in
#         dacs = {k: v for k, v in bdac_json.items() if k[:3] == 'DAC'}
#         nums = [int(re.search('\d+', k)[0]) for k in dacs.keys()]
#         names = [re.search('(?<={).*(?=})', k)[0] for k in dacs.keys()]
#         dacs = dict(zip(nums, dacs.values()))
#         dacnames = dict(zip(nums, names))
#         self.Babydac = BABYDACtuple(dacs=dacs, dacnames=dacnames)
#
#
#     def _set_fdacs(self, fdac_json):
#         """Set values from FastDAC json"""  # TODO: Make work for more than one fastdac
#         """fdac dict should be stored in format:
#                                 visa_address: ...
#                                 SamplingFreq:
#                                 DAC#{<name>}: <val>
#                                 ADC#: <val>
#
#                                 ADCs not currently required
#                                 """
#         fdacs = {k: v for k, v in fdac_json.items() if k[:3] == 'DAC'}
#         nums = [int(re.search('\d+', k)[0]) for k in fdacs.keys()]
#         names = [re.search('(?<={).*(?=})', k)[0] for k in fdacs.keys()]
#
#         dacs = dict(zip(nums, fdacs.values()))
#         dacnames = dict(zip(nums, names))
#         sample_freq = dictor(fdac_json, 'SamplingFreq', None)
#         measure_freq = dictor(fdac_json, 'MeasureFreq', None)
#         visa_address = dictor(fdac_json, 'visa_address', None)
#         self.Fastdac = FASTDACtuple(dacs=dacs, dacnames=dacnames, sample_freq=sample_freq, measure_freq=measure_freq,
#                                     visa_address=visa_address)


def _dac_dict(dacs, names) -> dict:
    return {names[k] if names[k] != '' else f'DAC{k}': dacs[k] for k in dacs}


class SRStuple(NamedTuple):
    gpib: int
    out: int
    tc: float
    freq: float
    phase: float
    sens: float
    harm: int
    CH1readout: int


class Magnet(NamedTuple):
    name: str
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

    # def __hash__(self):
    #     return hash((sorted(frozenset(self.outputs.items())), self.wave_len, self.num_adcs, self.samplingFreq, self.measureFreq, self.num_cycles,
    #                  self.num_steps))
    #
    # def __eq__(self, other):
    #     if isinstance(other, self.__class__):
    #         return hash(other) == hash(self)
    #     return False


class FASTDACtuple(NamedTuple):
    dacs: dict
    dacnames: dict
    sample_freq: float
    measure_freq: float
    visa_address: str


# class BABYDACtuple(NamedTuple):
#     dacs: dict
#     dacnames: dict


class InitLogs(object):
    """Class to contain all functions required for setting up Logs in HDF """
    BABYDAC_KEYS = ['com_port',
                    'DAC#{<name>}']  # TODO: not currently using DAC#{}, should figure out a way to check this
    FASTDAC_KEYS = ['SamplingFreq', 'MeasureFreq', 'visa_address', 'AWG', 'numADCs', 'DAC#{<name>}', 'ADC#']


    @staticmethod
    def check_key(k, expected_keys):
        if k in expected_keys:
            return
        elif k[0:3] in ['DAC', 'ADC']:  # TODO: This should be checked better
            return
        else:
            logger.warning(f'Unexpected key in logs: k = {k}, Expected = {expected_keys}')
            return

    @staticmethod
    def set_babydac(group, babydac_json):
        """Puts info into Dat HDF"""
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
    def set_awg(group, awg_json):
        """Put info into Dat HDF

        Args:
            group (h5py.Group): Group to put AWG NamedTuple in
            awg_json (dict): From standardized Exp sweeplogs

        Returns:
            None
        """

        if awg_json is not None:
            # Simplify and shorten names
            awg_data = awg_from_json(awg_json)

            # Store in NamedTuple
            ntuple = data_to_NamedTuple(awg_data, AWGtuple)
            HDU.set_attr(group, 'AWG', ntuple)
        else:
            logger.info(f'No "AWG" added')

    @staticmethod
    def set_srss(group, json):
        """Sets SRS values in Dat HDF from either full sweeplogs or minimally json which contains SRS_{#} keys"""
        srs_ids = [key[4] for key in json.keys() if key[:3] == 'SRS']

        for num in srs_ids:
            if f'SRS_{num}' in json.keys():
                srs_data = srs_from_json(json, num)  # Converts to my standard
                ntuple = data_to_NamedTuple(srs_data, SRStuple)  # Puts data into named tuple
                srs_group = group.require_group(f'srss')  # Make sure there is an srss group
                HDU.set_attr(srs_group, f'srs{num}', ntuple)  # Save in srss group
            else:
                logger.error(f'No "SRS_{num}" found in json')  # Should not get to here

    @staticmethod
    def set_mags(group, json):
        """Sets mags"""
        raise NotImplementedError('Need to do this!!')
        # TODO: FIXME: DO THIS!

    @staticmethod
    def set_temps(group, temp_json):
        """Sets Temperatures in DatHDF from temperature part of sweeplogs"""
        if temp_json:
            temp_data = temp_from_json(temp_json)
            ntuple = data_to_NamedTuple(temp_data, TEMPtuple)
            HDU.set_attr(group, 'Temperatures', ntuple)
        else:
            logger.warning('No "Temperatures" added')

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
        group.attrs['part_of'] = get_part(dictor(json, 'comment', ''))


def get_part(comments):
    """
    If comments contain 'part#of#' this will return a tuple of (a, b) where it is part a of b
    Args:
        comments (str): Sweeplog comments (where info on part#of# should be found)

    Returns:
        Tuple[int, int]: (a, b) -- part a of b
    """
    if check_if_partial(comments):
        comments = comments.split(',')
        comments = [com.strip() for com in comments]
        part_comment = [com for com in comments if re.match('part*', com)][0]
        part_num = int(re.search('(?<=part)\d+', part_comment).group(0))
        of_num = int(re.search('(?<=of)\d+', part_comment).group(0))
        return part_num, of_num
    else:
        return 1, 1


def check_if_partial(comments):
    """
    Checks if comments contain info about Dat being a part of a series of dats (i.e. two part entropy scans where first
    part is wide and second part is narrow with more repeats)

    Args:
        comments (string): Sweeplogs comments (where info on part#of# should be found)
    Returns:
        bool: True or False
    """
    assert type(comments) == str
    comments = comments.split(',')
    comments = [com.strip() for com in comments]
    part_comment = [com for com in comments if re.match('part*', com)]
    if part_comment:
        return True
    else:
        return False


def check_key(k, expected_keys):
    """Used to check if k is an expected key (with some fudging for DACs at the moment)"""
    if k in expected_keys:
        return
    elif k[0:3] in ['DAC', 'ADC']:  # TODO: This should be checked better
        return
    else:
        logger.warning(f'Unexpected key in logs: k = {k}, Expected = {expected_keys}')
        return


def awg_from_json(awg_json):
    """Converts from standardized exp json to my dictionary of values (to be put into AWG NamedTuple)

    Args:
        awg_json (dict): The AWG json from exp sweep_logs in standard form

    Returns:
        dict: AWG data in a dict with my keys (in AWG NamedTuple)

    """

    AWG_KEYS = ['AW_Waves', 'AW_Dacs', 'waveLen', 'numADCs', 'samplingFreq', 'measureFreq', 'numWaves', 'numCycles',
                'numSteps']
    if awg_json is not None:
        # Check keys make sense
        for k, v in awg_json.items():
            check_key(k, AWG_KEYS)

        d = {}
        waves = dictor(awg_json, 'AW_Waves', '')
        dacs = dictor(awg_json, 'AW_Dacs', '')
        d['outputs'] = {int(k): [int(val) for val in list(v.strip())] for k, v in zip(waves.split(','), dacs.split(','))}  # e.g. {0: [1,2], 1: [3]}
        d['wave_len'] = dictor(awg_json, 'waveLen')
        d['num_adcs'] = dictor(awg_json, 'numADCs')
        d['samplingFreq'] = dictor(awg_json, 'samplingFreq')
        d['measureFreq'] = dictor(awg_json, 'measureFreq')
        d['num_cycles'] = dictor(awg_json, 'numCycles')
        d['num_steps'] = dictor(awg_json, 'numSteps')
        return d
    else:
        return None


def srs_from_json(srss_json, id):
    """Converts from standardized exp json to my dictionary of values (to be put into SRS NamedTuple)

    Args:
        srss_json (dict): srss part of json (i.e. SRS_1: ... , SRS_2: ... etc)
        id (int): number of SRS
    Returns:
        dict: SRS data in a dict with my keys
    """
    if 'SRS_' + str(id) in srss_json.keys():
        srsdict = srss_json['SRS_' + str(id)]
        srsdata = {'gpib': srsdict['gpib_address'],
                   'out': srsdict['amplitude V'],
                   'tc': srsdict['time_const ms'],
                   'freq': srsdict['frequency Hz'],
                   'phase': srsdict['phase deg'],
                   'sens': srsdict['sensitivity V'],
                   'harm': srsdict['harmonic'],
                   'CH1readout': srsdict.get('CH1readout', None)
                   }
    else:
        srsdata = None
    return srsdata


def mag_from_json(jsondict, id, mag_type='ls625'):
    if 'LS625 Magnet Supply' in jsondict.keys():  # FIXME: This probably only works if there is 1 magnet ONLY!
        mag_dict = jsondict['LS625 Magnet Supply']  #  FIXME: Might just be able to pop entry out then look again
        magname = mag_dict.get('variable name', None)  # Will get 'magy', 'magx' etc
        if magname[-1:] == id:  # compare last letter
            mag_data = {'field': mag_dict['field mT'],
                        'rate': mag_dict['rate mT/min']
                        }
        else:
            mag_data = None
    else:
        mag_data = None
    return mag_data


def temp_from_json(tempdict, fridge='ls370'):
    """Converts from standardized exp json to my dictionary of values (to be put into Temp NamedTuple)

    Args:
        jsondict ():
        fridge ():

    Returns:

    """
    if tempdict:
        tempdata = {'mc': tempdict.get('MC K', None),
                    'still': tempdict.get('Still K', None),
                    'fourk': tempdict.get('4K Plate K', None),
                    'mag': tempdict.get('Magnet K', None),
                    'fiftyk': tempdict.get('50K Plate K', None)}
    else:
        tempdata = None
    return tempdata


def replace_in_json(jsonstr: str, jsonsubs: dict) -> dict:
    if jsonsubs is not None:
        for pattern, repl in jsonsubs.items():
            jsonstr = re.sub(pattern, repl, jsonstr)
    try:
        jsondata = json.loads(jsonstr)
    except json.decoder.JSONDecodeError as e:
        print(jsonstr)
        raise e
    return jsondata