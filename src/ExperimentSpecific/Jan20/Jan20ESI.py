from src.Configs.ConfigBase import ConfigBase
from src.DFcode.SetupDF import SetupDF
from src.ExperimentSpecific.BaseClasses import ExperimentSpecificInterface
from src.ExperimentSpecific.Jan20 import Jan20Config
from src.DatBuilder import Util
import src.CoreUtil as CU
from dictor import dictor
import h5py
import logging

logger = logging.getLogger(__name__)
# TODO: Update this to the new way of doing things. Probably need to update all of SetupDF!


class JanESI(ExperimentSpecificInterface):
    def set_setupdf(self) -> SetupDF:
        self.setupdf = SetupDF(config=Jan20Config.JanConfig())
        return self.setupdf  # Just to stop type hints

    def set_Config(self) -> ConfigBase:
        self.Config = Jan20Config.JanConfig()
        return self.Config  # Just to stop type hints

    def get_sweeplogs(self) -> dict:
        sweep_logs = super().get_sweeplogs()
        # Then need to change the format of BabyDAC and FastDAC
        bdacs_json = dictor(sweep_logs, 'BabyDAC', None)
        if bdacs_json is not None:
            sweep_logs['BabyDAC'] = convert_babydac_json(bdacs_json)

        fdacs_json = dictor(sweep_logs, 'FastDAC', None)
        if fdacs_json is not None:
            hdf = self.get_exp_dat_hdf()
            num_adc = get_num_adc_from_hdf(hdf)
            hdf.close()
            sweep_logs['FastDAC'] = convert_fastdac_json(fdacs_json, num_adc)
        return sweep_logs


def get_num_adc_from_hdf(hdf:h5py.Group):
    num = 0
    for key in hdf.keys():
        if key[:3] == 'ADC':
            num += 1
    if num == 0:
        logger.warning(f'Counted ZERO ADC# waves in {hdf.name}, returning 1 to prevent errors')
        num = 1
    return num


def convert_fastdac_json(fastdac_json: dict, num_adc) -> dict:
    """Was in "CH0": val..., "CH0name": val...
    but needs to be in "DAC{name}: val" format now"""
    fdacs = {int(key[2:]): fastdac_json[key] for key in fastdac_json if key[-4:] not in ['name', 'Keys', 'Freq']}
    fdacnames = {int(key[2:-4]): fastdac_json[key] for key in fastdac_json if key[-4:] == 'name'}
    fdacfreq = fastdac_json['SamplingFreq']
    names = _make_new_dac_names(fdacs, fdacnames)
    new_dict = _make_new_dac_dict(fdacs, names)
    new_dict['SamplingFreq'] = fdacfreq
    new_dict['MeasureFreq'] = fdacfreq/num_adc
    try:
        new_dict['visa_address'] = dictor(fastdac_json, 'fdacKeys', '').split(',')[-2][6:]
    except IndexError:
        new_dict['visa_address'] = None
    return new_dict


def convert_babydac_json(babydac_json: dict) -> dict:
    """Was in "CH0": val..., "CH0name": val...
    but needs to be in "DAC{name}: val" format now"""
    dacs = {int(key[2:]): babydac_json[key] for key in babydac_json if key[-4:] not in ['name', 'port']}
    dacnames = {int(key[2:-4]): babydac_json[key] for key in babydac_json if key[-4:] == 'name'}

    names = _make_new_dac_names(dacs, dacnames)
    new_dict = _make_new_dac_dict(dacs, names)
    return new_dict


def _make_new_dac_names(dacs, dacnames):
    names = {i: '{}' for i in range(len(dacs))}
    for k, v in dacnames.items():
        names[k] = f'{{{v}}}'
    return names


def _make_new_dac_dict(dacs, new_dacnames):
    new_dict = {}
    for (i, n), (j, v) in zip(new_dacnames.items(), dacs.items()):
        assert i == j
        new_dict[f'DAC{i}{n}'] = v
    return new_dict