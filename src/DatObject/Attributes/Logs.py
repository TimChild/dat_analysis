from typing import NamedTuple
import re
import h5py
from src.DatObject.Attributes.DatAttribute import DatAttribute
import logging
from dictor import dictor
import src.HDF_Util as HDU
import src.CoreUtil as CU
from dataclasses import dataclass
logger = logging.getLogger(__name__)

'''
Required Dat attribute
    Represents all basic logging functionality from SRSs, magnets, temperature probes, and anything else of interest
'''

EXPECTED_TOP_ATTRS = ['version', 'comments', 'filenum', 'x_label', 'y_label', 'current_config', 'time_completed',
                      'time_elapsed', 'part_of']


class NewLogs(DatAttribute):
    version = '1.0'
    group_name = 'Logs'

    def __init__(self, hdf):
        super().__init__(hdf)
        self.full_sweeplogs = None

        self.Babydac: BABYDACtuple = None
        self.Fastdac: FASTDACtuple = None
        self.AWG: AWGtuple = None

        self.comments = None
        self.filenum = None
        self.x_label = None
        self.y_label = None

        self.time_completed = None
        self.time_elapsed = None

        self.part_of = None

        self.dim = None
        self.temps = None
        self.get_from_HDF()

    @property
    def fds(self):
        return _dac_dict(self.Fastdac.dacs, self.Fastdac.dacnames) if self.Fastdac else None

    @property
    def bds(self):
        return _dac_dict(self.Babydac.dacs, self.Babydac.dacnames) if self.Babydac else None

    @property
    def sweeprate(self):
        sweeprate = None
        measure_freq = self.Fastdac.measure_freq if self.Fastdac else None
        if measure_freq:
            data_group = self.hdf.get('Data')
            if data_group:
                x_array = data_group.get('Exp_x_array')
                if x_array:
                    sweeprate = CU.get_sweeprate(measure_freq, x_array)
        return sweeprate

    def update_HDF(self):
        logger.warning('Calling update_HDF on Logs attribute has no effect')
        pass

    def _set_default_group_attrs(self):
        super()._set_default_group_attrs()

    def get_from_HDF(self):
        group = self.group

        # Get full copy of sweeplogs
        self.full_sweeplogs = HDU.get_attr(group, 'Full sweeplogs', None)

        # Get top level attrs
        for k, v in group.attrs.items():
            if k in EXPECTED_TOP_ATTRS:
                setattr(self, k, v)
            elif k in ['description']:  # HDF only attr
                pass
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

        mags_group = group.get('mags', None)
        if mags_group:
            for key in mags_group.keys():
                if isinstance(mags_group[key], h5py.Group) and mags_group[key].attrs.get('description', None) == 'dataclass':
                    setattr(self, key, HDU.get_attr(mags_group, key))

        temp_tuple = HDU.get_attr(group, 'Temperatures', None)
        if temp_tuple:
            self.temps = temp_tuple

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


def _dac_dict(dacs, names):
    return {names[k] if names[k] != '' else f'DAC{k}': dacs[k] for k in dacs.keys()}


class SRStuple(NamedTuple):
    gpib: int
    out: int
    tc: float
    freq: float
    phase: float
    sens: float
    harm: int
    CH1readout: int


@dataclass
class MAGs:
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


class FASTDACtuple(NamedTuple):
    dacs: dict
    dacnames: dict
    sample_freq: float
    measure_freq: float
    visa_address: str


class BABYDACtuple(NamedTuple):
    dacs: dict
    dacnames: dict


