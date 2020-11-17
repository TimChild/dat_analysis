import json
import re
import threading
import time
from typing import TYPE_CHECKING
import logging
import numpy as np
from dictor import dictor

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


###############################################


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


def metadata_to_JSON(data: str, config, datnum=None) -> dict:
    jsonsubs = config.json_subs  # Get experiment specific json subs from config
    if callable(
            jsonsubs):  # Cheeky way to be able to get datnum specific jsonsubs by returning a function in the first place
        jsonsubs = jsonsubs(datnum)
    return replace_in_json(data, jsonsubs)


def replace_in_json(jsonstr, jsonsubs):
    if jsonsubs is not None:
        for pattern_repl in jsonsubs:
            jsonstr = re.sub(pattern_repl[0], pattern_repl[1], jsonstr)
    try:
        jsondata = json.loads(jsonstr)
    except json.decoder.JSONDecodeError as e:
        print(jsonstr)
        raise e
    return jsondata


# def get_dattypes(dattypes, comments, dat_types_list):
#     """Will return dattypes from comments if they are in dat_types_list and also adds dattype dependencies (i.e.
#     'transition' if a DCbias or Entropy type)"""
#     if dattypes is None:  # Will return basic dat only
#         dattypes = {'none_given'}
#         for key in dat_types_list:
#             if comments is not None:
#                 if key in [val.strip() for val in comments.split(',')]:
#                     dattypes.add(key)
#     if 'dcbias' in dattypes:
#         dattypes.add('transition')
#     if 'entropy' in dattypes:
#         dattypes.add('transition')
#     if 'square entropy' in dattypes:
#         dattypes.add('transition')
#     return dattypes


# def get_value_from_setupdf(datnum, name, setupdf):
#     setupdata = setupdf.get_valid_row(datnum)
#     if name in setupdata.keys() and setupdata[name] is not None:
#         value = setupdata[name]
#     else:
#         logger.warning(f'[{name}] not found in setupdf')
#         value = 1
#     return value


# def _get_setup_dict_entry(datnum, setupdf, exp_names=None):
#     """Returns setup_dict_entry in form [<possible names>, <multiplier per name>, <offset per name>]
#     Intended for a full setup_dict which would be {<standard_name>: <setup_dict_entry>}"""
#     setupdata = setupdf.get_valid_row(datnum)
#     for name in exp_names:
#         if name not in setupdata.keys():
#             logger.warning(f'[{name}] not found in setupDF')
#     multipliers = [setupdata.get(name, 1) for name in exp_names]
#     offsets = [setupdata.get(name + '_offset', 0) for name in exp_names]
#     return [exp_names, multipliers, offsets]
#
#
# def get_data_setup_dict(datnum, dattypes, exp_names_dict, srss_json=None):
#     setup_dict = dict()
#     setup_dict['x_array'] = _get_setup_dict_entry(datnum, setupdf, exp_names=exp_names_dict['x_array'])
#     setup_dict['y_array'] = _get_setup_dict_entry(datnum, setupdf, exp_names=exp_names_dict['y_array'])
#
#     if {'i_sense', 'transition', 'entropy', 'dcbias'} & set(dattypes):  # If there is overlap between lists then...
#         setup_dict['i_sense'] = _get_setup_dict_entry(datnum, setupdf, exp_names_dict['i_sense'])
#
#     if 'entropy' in dattypes:
#         entx_setup = _get_setup_dict_entry(datnum, setupdf, exp_names_dict['entx'])
#         enty_setup = _get_setup_dict_entry(datnum, setupdf, exp_names_dict['enty'])
#         current_amplification = get_value_from_setupdf(datnum, 'ca0amp', setupdf)
#         srs = get_value_from_setupdf(datnum, 'entropy_srs', setupdf)
#         if srs[:3] == 'srs':
#             if srss_json is None:
#                 raise ValueError(f"Need to pass in a json (probably whole sweeplog) with the SRS_# dicts to get srs_sens for {srs}")
#             key = f'SRS_{srs[-1]}'
#             multiplier = dictor(srss_json, f'{key}.sensitivity V', checknone=True) / 10 * 1e-3 / current_amplification * 1e9  # /10 because 10V range of output, 1e-3 to go to V, 1e9 to go to nA
#         else:
#             multiplier = 1e9 / current_amplification  # 1e9 to nA, /current_amp to current in A.
#             logger.info(
#                 f'Not using "srs_sens" for entropy signal for dat{datnum} with setupdf config=[{setupdf.config_name}]')
#         if entx_setup is not None:
#             entx_setup[1] = list(np.array(entx_setup[1]) * multiplier)  # Change multipliers
#         if enty_setup is not None:
#             enty_setup[1] = list(np.array(enty_setup[1]) * multiplier)  # Change multipliers
#         setup_dict['entx'] = entx_setup
#         setup_dict['enty'] = enty_setup
#     return setup_dict


# def convert_sweeplogs(exp_sweep_logs):
#     """Converts standardized exp sweep_logs into my sweeplogs
#     i.e. Expects sweeplogs in a standard format which should be close to what is given by experiment
#     and then converts it to my standard which should not change"""
#     new_logs = {}
#     esl = exp_sweep_logs[:]  # Copy so I can delete things from it to compare at end
#     def move_value(name, new_name=None, default=None):
#         value = dictor(esl, name, default)
#
#         if value is not None:
#             new_name = new_name if new_name else name
#             new_logs[new_name] = value
#
#         # Remove from esl so I can check if any are left at end
#         if name in esl.keys():
#             del esl[name]
#
#     move_value('comment', default='')
#     move_value('filenum')
#     move_value('axis_labels')  # Which should be a dict with x:, y:
#     move_value('current_config')
#     move_value('time_completed', default='')
#     move_value('time_elapsed', default=np.nan)
#     move_value('BabyDAC')
#
#     if 'FastDAC 1' in esl.keys():
#         fd = dictor(esl, 'FastDAC')
#     elif 'FastDAC' in esl.keys():
#         fd = dictor(esl, 'FastDAC')
#     else:
#         fd = None
#     if fd is not None:
#         if 'AWG' in fd.keys():
#             new_logs['AWG'] = dictor(fd, 'AWG')
#             del fd['AWG']
#
#
def wait_for(datnum, ESI_class=None):
    def _wait_fn(num):
        esi = ESI_class(num)
        while True:
            found = esi._check_data_exists(suppress_output=True)
            if found:
                print(f'Dat{num} is ready!!')
                break
            else:
                time.sleep(10)

    if ESI_class is None:
        from src.DatObject.Make_Dat import default_Exp2HDF
        ESI_class = default_Exp2HDF

    x = threading.Thread(target=_wait_fn, args=(datnum,))
    x.start()
    print(f'A thread is waiting on dat{datnum} to appear')
    return x


def clean_basic_sweeplogs(sweeplogs: dict) -> dict:
    if 'BF Small' in sweeplogs.keys():  # TODO: Move to all exp specific instead of Base
        sweeplogs['Temperatures'] = sweeplogs['BF Small']
        del sweeplogs['BF Small']
    return sweeplogs
