import os

import h5py
from dictor import dictor

import dat_analysis.data_standardize
from dat_analysis.DFcode.SetupDF import SetupDF

from dat_analysis.data_standardize.base_classes import Exp2HDF
from dat_analysis.data_standardize.exp_config import ExpConfigBase
import logging
logger = logging.getLogger(__name__)


class JanExpConfig(ExpConfigBase):
    def synchronize_data_batch_file(self) -> str:
        pass

    dir_name = 'Nik_entropy_v2'

    def __init__(self):
        super().__init__()

    def set_directories(self):
        exp_folder = os.path.join(self.main_folder_path, self.dir_name)
        hdfdir, ddir, dfsetupdir, dfbackupdir = self.get_expected_sub_dir_paths(exp_folder)
        self.Directories.set_dirs(hdfdir, ddir, dfsetupdir, dfbackupdir)

    def get_sweeplogs_json_subs(self, datnum):
        return [('"comment": "{"gpib_address":4, "units":"VOLT", "range":.1.000000E.0., "resolution":...000000E-0.}"',
                '"comment": "replaced to fix json"'), (":\+", ':'), ('\r', '')]

    def get_dattypes_list(self):
        return ['none', 'i_sense', 'entropy', 'transition', 'pinch', 'dot tuning', 'dcbias', 'lockin theta']

    def get_exp_names_dict(self):
        d = dict(x_array=['x_array'], y_array=['y_array'],
                 i_sense=['i_sense', 'cscurrent', 'cscurrent_2d'],
                 entx=['entropy_x_2d', 'entropy_x'],
                 enty=['entropy_y_2d', 'entropy_y'])  # took out entx, enty because not in SetupDF
        return d


dir_name = 'Nik_entropy_v2'  # Name of folder inside main data folder specified by dat_analysis.config.main_data_path


class JanESI(Exp2HDF):
    def set_setupdf(self) -> SetupDF:
        self.setupdf = SetupDF(config=dat_analysis.DataStandardize.ExpSpecific.Jan20.JanExpConfig())
        return self.setupdf  # Just to stop type hints

    def set_ExpConfig(self) -> ExpConfigBase:
        self.Config = dat_analysis.DataStandardize.ExpSpecific.Jan20.JanExpConfig()
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








# import os
#
# import dat_analysis.Builders.Util
# import dat_analysis.data_standardize.Exp_to_standard
#
# path_replace = ('work\\Fridge Measurements with PyDatAnalysis', 'work\\Fridge_Measurements_and_Devices\\Fridge Measurements with PyDatAnalysis')
#
# instruments = {'srs': 'srs830', 'dmm': 'hp34401a', 'dac': 'babydac', 'fastdac': 'fastdac', 'magnet':'ls625', 'fridge':'ls370'}
# instrument_num = {'srs': 3, 'dmm': 1, 'dac': 16, 'fastdac': 8, 'magnet': 3}
#
# dat_types_list = ['none', 'i_sense', 'entropy', 'transition', 'pinch', 'dot tuning', 'dcbias', 'lockin theta']
#
# ### Path to all Data (e.g. dats, dataframes, pickles etc). Hopefully this will allow moving out of project without
# #  losing access to everything
# abspath = os.path.abspath('../..').split('PyDatAnalysis')[0]
#
# wavenames = ['x_array', 'y_array', 'i_sense', 'entx', 'enty']
# raw_wavenames = [f'ADC{i:d}' for i in range(4)] + [f'ADC{i:d}_2d' for i in range(4)] + ['g1x', 'g1y', 'g2x', 'g2y']
#
# i_sense_keys = ['i_sense', 'cscurrent', 'cscurrent_2d']
# entropy_x_keys = ['entx', 'entropy_x_2d', 'entropy_x']
# entropy_y_keys = ['enty', 'entropy_y_2d', 'entropy_y']
# li_theta_x_keys = ['g3x']
# li_theta_y_keys = ['g3y']
#
# json_subs = [('"comment": "{"gpib_address":4, "units":"VOLT", "range":.1.000000E.0., "resolution":...000000E-0.}"',
#                 '"comment": "replaced to fix json"'),
#              (":\+", ':'),
#              ('\r', '')]
#
#
# DC_HQPC_current_bias_resistance = 10e6  # Resistor inline with DC bias for HQPC
#
#
#
#
# # def add_magy(dats):
# #     i = 0
# #     from dat_analysis.DFcode.DatDF import DatDF
# #     datdf = DatDF()
# #     for dat in dats:
# #         if dat.Logs.magy is None:
# #             for string in dat.Logs.comments.split(','):
# #                 if string[:4] == 'magy':
# #                     print(f'Updating Dat{dat.datnum}: Magy = {string[5:-2]}mT')
# #                     dat.Logs.magy = float(string[5:-2])
# #                     datdf.update_dat(dat)
# #                     i+=1
# #         if dat.Logs.magy is None:
# #             print(f'dat{dat.datnum} has no magy in comments')
# #     if i != 0:
# #         print(f'saving {i} updates to df')
# #         datdf.save()
#
#
# def add_mag_to_logs(dats):
#     import h5py
#     import dat_analysis.Main_Config as cfg
#     import dat_analysis.DFcode.DatDF as DF
#     datdf = DF.DatDF()
#     i =0
#     for dat in dats:
#         if dat.Logs.magy is None:
#             hdf = h5py.File(dat.Data.hdfpath, 'r')
#             sweeplogs = hdf['metadata'].attrs['sweep_logs']
#             sweeplogs = dat_analysis.Builders.Util.metadata_to_JSON(sweeplogs)
#             mags = {'mag' + id: dat_analysis.data_standardize.Exp_to_standard.mag_from_json(sweeplogs, id, mag_type=instruments['magnet']) for id in ['x', 'y', 'z']}
#             dat.Logs.add_mags(mags)
#             dat.Instruments.add_mags(mags)
#             dat.Instruments.field_y = dat.Instruments.magy.field
#             cfg.yes_to_all = True
#             datdf.update_dat(dat)
#             cfg.yes_to_all = False
#             i+=1
#     if i != 0:
#         print(f'saving {i} updates to df')
#         datdf.save()