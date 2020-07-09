from collections import namedtuple
from typing import NamedTuple

import os
import h5py
import numpy as np
import lmfit as lm
import json
import ast
import datetime
from dateutil import parser
import logging

from src import CoreUtil as CU

logger = logging.getLogger(__name__)

ALLOWED_TYPES = (int, float, complex, str, bool, np.ndarray)


def get_dat_hdf_path(dat_id, hdfdir_path, overwrite=False):
    file_path = os.path.join(hdfdir_path, dat_id + '.h5')
    if os.path.exists(file_path):
        if overwrite is True:
            os.remove(file_path)
        else:
            raise FileExistsError(
                f'HDF file already exists for {dat_id} at {hdfdir_path}. Use "overwrite=True" to overwrite')
    if not os.path.exists(file_path):  # make empty file then return path
        hdfdir_path, _ = os.path.split(file_path)
        os.makedirs(hdfdir_path, exist_ok=True)  # Ensure directory exists
        f = h5py.File(file_path, 'w')  # Init a HDF file
        f.close()
    return file_path


PARAM_KEYS = {'name', 'value', 'vary', 'min', 'max', 'expr', 'brute_step'}
ADDITIONAL_PARAM_KEYS = {'init_value', 'stderr'}


def params_to_HDF(params: lm.Parameters, group: h5py.Group):
    group.attrs['description'] = "Single Parameters of fit"
    all_par_values = ''
    for key in params.keys():
        par = params[key]
        par_group = group.require_group(key)
        par_group.attrs['description'] = "Single Param"
        for par_key in PARAM_KEYS | ADDITIONAL_PARAM_KEYS:
            attr_val = getattr(par, par_key, np.nan)
            attr_val = attr_val if attr_val is not None else np.nan
            par_group.attrs[par_key] = attr_val
        # par_group.attrs['init_value'] = getattr(par, 'init_value', np.nan)
        # par_group.attrs['stderr'] = getattr(par, 'stderr', np.nan)

        #  For HDF only. If stderr is None, then fit failed
        if getattr(par, 'stderr', None) is None:
            all_par_values += f'{key}=None'
        else:
            all_par_values += f'{key}={par.value:.3g}, '
    logger.debug(f'Saving best_values as: {all_par_values}')
    group.attrs['best_values'] = all_par_values  # For viewing in HDF only
    pass


def params_from_HDF(group) -> lm.Parameters:
    params = lm.Parameters()
    for key in group.keys():
        if isinstance(group[key], h5py.Group) and group[key].attrs.get('description', None) == 'Single Param':
            par_group = group[key]
            par_vals = {par_key: par_group.attrs.get(par_key, None) for par_key in PARAM_KEYS}
            par_vals = {key: v if not (isinstance(v, float) and np.isnan(v)) else None for key, v in par_vals.items()}
            params.add(**par_vals)  # create par
            par = params[key]  # Get single par

            par.stderr = par_group.attrs.get('stderr', np.nan)
            par.stderr = None if np.isnan(par.stderr) else par.stderr  # Replace NaN with None if thats what it was

            # Because the saved value was actually final value, but inits into init_val I need to switch them.
            # If stderr is None, fit previously failed so store None for final Value instead.
            par.value = par.init_value if par.stderr is not None else None

            par.init_value = par_group.attrs.get('init_value', np.nan)  # I save init_value separately
            par.init_value = None if np.isnan(par.init_value) else par.init_value  # Replace NaN with None

            # for par_key in PARAM_KEYS | ADDITIONAL_PARAM_KEYS:
            #     if getattr(par, par_key) == np.nan:  # How I store None in HDF
            #         setattr(par, par_key, None)
    return params


def set_data(group, name, data):
    ds = group.get(name, None)
    if ds is not None:
        # TODO: Do something better here to make sure I'm not just needlessly rewriting data
        logger.info(f'Removing dataset {ds.name} with shape {ds.shape} to '
                    f'replace with data of shape {data.shape}')
        del group[name]
    group[name] = data


def set_attr(group: h5py.Group, name: str, value):
    """Saves many types of value to the group under the given name which can be used to get it back from HDF"""
    assert isinstance(group, h5py.Group)
    if type(value) in ALLOWED_TYPES:
        if isinstance(value, np.ndarray) and value.size > 30:
            raise ValueError(f'Trying to add array of size {value.size} as an attr. Save as a dataset instead')
        group.attrs[name] = value

    elif type(value) == dict:
        if len(value) < 5:
            d_str = json.dumps(value)
            group.attrs[name] = d_str
        else:
            dict_group = group.require_group(name)
            save_dict_to_hdf_group(dict_group, value)

    elif type(value) == set:
        group.attrs[name] = str(value)

    elif isinstance(value, datetime.date):
        group.attrs[name] = str(value)

    elif _isnamedtupleinstance(value):
        ntg = group.require_group(name)
        save_namedtuple_to_group(value, ntg)
    elif value is None:
        group.attrs[name] = np.nan
    else:
        raise TypeError(
            f'type: {type(value)} not allowed in attrs for group, key, value: {group.name}, {name}, {value}')


def get_attr(group: h5py.Group, name, default=None, check_exists=False):
    """Inverse of set_attr. Gets many different types of values stored by set_attrs"""
    assert isinstance(group, h5py.Group)
    attr = group.attrs.get(name, None)
    if attr is not None:
        # Shouldn't need to check for int/float because they are stored correctly in HDF
        # try:  # See if it was an int
        #     i = int(attr)
        #     return i
        # except ValueError:
        #     pass
        # try:  # See if it was a float
        #     f = float(attr)
        #     return f
        # except ValueError:
        #     pass
        try:  # See if it was a dict that was saved
            d = json.loads(attr)  # get back to dict
            d = _convert_keys_to_int(d)  # Make keys integers again as they are stored as str in JSON
            return d
        except (TypeError, json.JSONDecodeError, AssertionError) as e:
            pass
        try:
            s = ast.literal_eval(attr)  # get back to set
            if type(s) == set:
                return s
        except (ValueError, SyntaxError):
            pass
        try:
            dt = parser.parse(attr)  # get back to datetime
            if datetime.datetime(2017, 1, 1) < dt < datetime.datetime(2023, 1, 1):
                return dt
            else:  # Probably not supposed to be a datetime
                pass
        except (TypeError, ValueError):
            pass
        return attr

    g = group.get(name, None)
    if g is not None:
        if isinstance(g, h5py.Group):
            if g.attrs.get('description') == 'simple dictionary':
                attr = load_dict_from_hdf_group(g)  # TODO: Want to see how loading full sweeplogs works
                return attr
            if g.attrs.get('description') == 'NamedTuple':
                attr = load_group_to_namedtuple(g)
                return attr
    if check_exists is True:
        raise KeyError(f'{name} is not an attr that can be loaded by get_attr in group {group.name}')
    else:
        return default


def _convert_keys_to_int(d: dict):
    """Converts any keys which are strings of int's to int"""
    assert isinstance(d, dict)
    new_dict = {}
    for k, v in d.items():
        try:
            new_key = int(k)
        except ValueError:
            new_key = k
        if type(v) == dict:
            v = _convert_keys_to_int(v)
        new_dict[new_key] = v
    return new_dict


def save_dict_to_hdf_group(group: h5py.Group, dictionary: dict):
    """
    Saves dictionary to a group (each entry can contain more dictionaries etc)
    @param group:
    @type group:
    @param dictionary:
    @type dictionary:
    @return:
    @rtype:
    """
    group.attrs['description'] = 'simple dictionary'
    for k, v in dictionary.items():
        set_attr(group, k, v)


def load_dict_from_hdf_group(group: h5py.Group):
    """Inverse of save_simple_dict_to_hdf returning to same form"""
    d = {}
    for k, v in group.attrs.items():
        if k != 'description':
            d[k] = get_attr(group, k, None)
    for k, g in group.items():
        if isinstance(g, h5py.Group) and g.attrs.get('description') == 'simple dictionary':
            d[k] = load_dict_from_hdf_group(g)
    return d


def save_namedtuple_to_group(ntuple: NamedTuple, group: h5py.Group):
    """Saves named tuple inside group given"""
    group.attrs['description'] = 'NamedTuple'
    group.attrs['NT_name'] = ntuple.__class__.__name__
    for key, val in ntuple.__annotations__.items():
        set_attr(group, key, val)  # Store as attrs of group in HDF


def load_group_to_namedtuple(group: h5py.Group):
    """Returns namedtuple with name of group and key: values of group attrs
    e.g. srs1 group which has gpib: 1... will be returned as an srs1 namedtuple with .gpib etc
    """
    # Check it was stored as a namedTuple
    if group.attrs.get('description', None) != 'NamedTuple':
        raise ValueError(
            f'Trying to load_group_to_named_tuple which has description: {group.attrs.get("description", None)}')

    # Get the name of the NamedTuple either through the stored name or the group name
    name = group.attrs.get('NT_name', None)
    if name is None:
        logger.warning('Did not find "name" attribute for NamedTuple, using folder name instead')
        name = group.name.split('/')[-1]

    # d = {key: val for key, val in group.attrs.items()}
    d = {key: get_attr(group, key) for key in group.attrs.keys()}

    # Remove HDF only descriptors
    for k in ['description', 'NT_name']:
        if k in d.keys():
            del d[k]

    # Make the NamedTuple
    ntuple = namedtuple(name, d.keys())
    filled_tuple = ntuple(**d)  # Put values into tuple
    return filled_tuple


# TODO: Move to better place
def _isnamedtupleinstance(x):
    """https://stackoverflow.com/questions/2166818/how-to-check-if-an-object-is-an-instance-of-a-namedtuple"""
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple: return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple): return False
    return all(type(n) == str for n in f)


def check_group_attr_overlap(group: h5py.Group, make_unique=False, exceptions=None):
    """Checks if there are any keys in a group which are the same as keys of attrs in that group"""
    group_keys = group.keys()
    attr_keys = group.attrs.keys()
    exception_keys = set(exceptions) if exceptions else set()
    if keys := (set(group_keys) & set(attr_keys)) - exception_keys:
        if make_unique is True:
            logger.info(f'In group [{group.name}], changing {keys} in attrs to make unique from group keys')
            for key in keys:
                setattr(group.attrs, f'{key}_attr', group.attrs[key])
                del group.attrs[key]
        else:
            logger.warning(f'Keys: {keys} are keys in both the group and group attrs of {group.name}. No changes made.')


def match_name_in_group(names, data_group):
    """
    Returns the first name from names which is a dataset in data_group

    @param names: list of potential names in data_group
    @type names: Union[str, list]
    @param data_group: The group (or hdf) to look for datasets in
    @type data_group: Union[h5py.File, h5py.Group]
    @return: First name which is a dataset or None if not found
    @rtype: Union[str, None]

    """
    names = CU.ensure_list(names)
    for i, name in enumerate(names):
        if name in data_group.keys() and isinstance(data_group[name], h5py.Dataset):
            return name, i
    logger.warning(f'[{names}] not found in [{data_group.name}]')
    return None, None
