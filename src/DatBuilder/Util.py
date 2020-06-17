import json
import re
from typing import NamedTuple, Union, List, Tuple
import h5py
from src import CoreUtil as CU
import logging
from src.Configs import Main_Config as cfg
from src.DFcode import SetupDF as SF

logger = logging.getLogger(__name__)


def match_name_in_group(names, data_group):
    """
    Returns the first name from exp_names which is a dataset in exp_hdf

    @param names: list of expected names in exp_dataset
    @type names: Union[str, list]
    @param data_group: The experiment hdf (or group) to look for datasets in
    @type data_group: Union[h5py.File, h5py.Group]
    @return: First name which is a dataset or None if not found
    @rtype: Union[str, None]

    """
    names = CU.ensure_list(names)
    for i, name in enumerate(names):
        if name in data_group.keys() and isinstance(data_group[name], h5py.Dataset):
            return name, i
    logger.warning(f'[{names}] not found in [{data_group.name}]')
    return None


def metadata_to_JSON(data: str, config=None, datnum=None) -> dict:
    if config is None:
        config = cfg.current_config
    jsonsubs = config.json_subs  # Get experiment specific json subs from config
    if callable(
            jsonsubs):  # Cheeky way to be able to get datnum specific jsonsubs by returning a function in the first place
        jsonsubs = jsonsubs(datnum)
    if jsonsubs is not None:
        for pattern_repl in jsonsubs:
            data = re.sub(pattern_repl[0], pattern_repl[1], data)
    try:
        jsondata = json.loads(data)
    except json.decoder.JSONDecodeError as e:
        print(data)
        raise e
    return jsondata


def data_to_NamedTuple(data: dict, named_tuple) -> NamedTuple:
    """Given dict of key: data and a named_tuple with the same keys, it returns the filled NamedTuple
    If data is not stored then a cfg._warning string is set"""
    tuple_dict = named_tuple.__annotations__  # Get ordered dict of keys of namedtuple
    for key in tuple_dict.keys():  # Set all values to None so they will default to that if not entered
        tuple_dict[key] = None
    for key in set(data.keys()) & set(tuple_dict.keys()):  # Enter valid keys values
        tuple_dict[key] = data[key]
    if set(data.keys()) - set(tuple_dict.keys()):  # If there is something left behind
        cfg.warning = f'data keys not stored: {set(data.keys()) - set(tuple_dict.keys())}'
        logger.warning(f'The following data is not being stored: {set(data.keys()) - set(tuple_dict.keys())}')
    else:
        cfg.warning = None
    ntuple = named_tuple(**tuple_dict)
    return ntuple


def get_value_from_setupdf(datnum, name, setupdf):
    setupdata = setupdf.get_valid_row(datnum)
    if name in setupdata.keys() and setupdata[name] is not None:
        value = setupdata[name]
    else:
        logger.warning(f'[{name}] not found in setupdf')
        value = 1
    return value


def _get_setup_dict_entry(datnum, standard_name, setupdf=None, config=None):
    """Returns setup_dict_entry in form [<possible names>, <multiplier per name>, <offset per name>]
    Intended for a full setup_dict which would be {<standard_name>: <setup_dict_entry>}"""
    setupdf = setupdf if setupdf else SF.SetupDF()
    config = config if config else cfg.current_config
    setupdata = setupdf.get_valid_row(datnum)
    if hasattr(config, f'{standard_name}_keys'):
        exp_names = getattr(config, f'{standard_name}_keys')
        for name in exp_names:
            if name not in setupdata.keys():
                logger.warning(f'[{name}] not found in setupDF')
        multipliers = [setupdata.get(name, None) for name in exp_names]
        offsets = [setupdata.get(name + '_offset', None) for name in exp_names]
        return [exp_names, multipliers, offsets]
    else:
        logger.warning(f'{standard_name}_keys not found in [{config.__name__}]')
        return None


def get_data_setup_dict(dat_builder, dattypes, setupdf, config):
    datnum = dat_builder.datnum
    setup_dict = dict()
    setup_dict['x_array'] = _get_setup_dict_entry(datnum, 'x_array', setupdf, config)
    setup_dict['y_array'] = _get_setup_dict_entry(datnum, 'y_array', setupdf, config)  # TODO: what if no y_array?

    if {'i_sense', 'transition', 'entropy', 'dcbias'} & set(dattypes):  # If there is overlap between lists then...
        setup_dict['i_sense'] = _get_setup_dict_entry(datnum, 'i_sense', setupdf, config)

    if 'entropy' in dattypes:
        entx_setup = _get_setup_dict_entry(datnum, 'entx', setupdf, config)
        enty_setup = _get_setup_dict_entry(datnum, 'enty', setupdf, config)
        current_amplification = get_value_from_setupdf(datnum, 'ca0amp', setupdf)
        srs = get_value_from_setupdf(datnum, 'entropy_srs', setupdf)
        if srs[:3] == 'srs':
            multiplier = getattr(getattr(dat_builder.Instruments, srs),
                                 'sens') / 10 * 1e-3 / current_amplification * 1e9  # /10 because 10V range of output, 1e-3 to go to V, 1e9 to go to nA
        else:
            multiplier = 1e9 / current_amplification  # 1e9 to nA, /current_amp to current in A.
            logger.info(
                f'Not using "srs_sens" for entropy signal for dat{datnum} with setupdf config=[{setupdf.config_name}]')
        if entx_setup is not None:
            entx_setup[1] = entx_setup[1] * multiplier
        if enty_setup is not None:
            enty_setup[1] = enty_setup[1] * multiplier
        setup_dict['entx'] = entx_setup
        setup_dict['enty'] = enty_setup
    return setup_dict


def get_dattypes(dattypes, comments, dat_types_list):
    """Will return dattypes from comments if they are in dat_types_list and also adds dattype dependencies (i.e.
    'transition' if a DCbias or Entropy type)"""
    if dattypes is None:  # Will return basic dat only
        dattypes = {'none_given'}
        for key in dat_types_list:
            if comments is not None:
                if key in [val.strip() for val in comments.split(',')]:
                    dattypes.add(key)
    if 'dcbias' in dattypes:
        dattypes.add('transition')
    if 'entropy' in dattypes:
        dattypes.add('transition')
    return dattypes
