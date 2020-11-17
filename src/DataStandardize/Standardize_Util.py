import threading
import time
from typing import TYPE_CHECKING
import logging
from deprecation import deprecated

# from DatObject.Attributes.Logs import replace_in_json

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


###############################################

# @deprecated('nov20')
# def metadata_to_JSON(data: str, config, datnum=None) -> dict:
#     jsonsubs = config.json_subs  # Get experiment specific json subs from config
#     if callable(
#             jsonsubs):  # Cheeky way to be able to get datnum specific jsonsubs by returning a function in the first place
#         jsonsubs = jsonsubs(datnum)
#     return replace_in_json(data, jsonsubs)
#

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


# def clean_basic_sweeplogs(sweeplogs: dict) -> dict:
#     if 'BF Small' in sweeplogs.keys():  # TODO: Move to all exp specific instead of Base
#         sweeplogs['Temperatures'] = sweeplogs['BF Small']
#         del sweeplogs['BF Small']
#     return sweeplogs
