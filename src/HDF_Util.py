from collections import namedtuple
from typing import NamedTuple, Union, Optional

import os
import h5py
import numpy as np
import lmfit as lm
import json
import ast
import datetime
from dateutil import parser
import logging
from dataclasses import is_dataclass, asdict, dataclass
from inspect import getsource

from src import CoreUtil as CU

logger = logging.getLogger(__name__)

ALLOWED_TYPES = (int, float, complex, str, bool, list, tuple, np.bool_, np.ndarray, np.number, type(None))


def allowed(value):
    if isinstance(value, ALLOWED_TYPES) or is_dataclass(value):
        return True
    elif hasattr(value, 'save_to_hdf'):
        return True
    elif type(value) in [dict, set]:
        return True
    elif isinstance(value, datetime.date):
        return True
    else:
        return False


def sanitize(val):
    if type(val) == list:
        if None in val:
            val = [v if v is not None else 'None' for v in val]
    if type(val) == tuple:
        if any(map(lambda x: x is None, val)):
            val = tuple([v if v is not None else 'None' for v in val])
    return val


def desanitize(val):
    if type(val) == list:
        if np.nan in val:
            val = [v if v != 'None' else None for v in val]
    if type(val) == tuple:
        if None in val:
            val = tuple([v if v != 'None' else None for v in val])
    return val


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


def set_data(group: h5py.Group, name: str, data: Union[np.ndarray, h5py.Dataset]):
    """
    Creates a dataset in Group with Name for data that is either an np.ndarray or a h5py.Dataset already (if using a
    dataset, it will only create a link to that dataset which is good for saving storage space, but don't do this
    if you intend to change the data)

    Args:
        group (h5py.Group): HDF group to store dataset in
        name (str): Name with which to store the dataset (will overwrite existing datasets)
        data (Union[np.ndarray, h5py.Dataset]): Data to be stored, can be np.ndarray or h5py.Dataset

    Returns:
        None
    """
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
    if isinstance(value, ALLOWED_TYPES) and not _isnamedtupleinstance(value) and value is not None:  # named tuples subclass from tuple...
        value = sanitize(value)
        if isinstance(value, np.ndarray) and value.size > 500:
            # raise ValueError(f'Trying to add {name} which is an array of size {value.size} as an attr. Save as a dataset instead')
            set_data(group, name, value)
        else:
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
    elif hasattr(value, 'save_to_hdf'):
        # g = group.require_group(name)
        value.save_to_hdf(group, name)
    elif _isnamedtupleinstance(value):
        ntg = group.require_group(name)
        save_namedtuple_to_group(value, ntg)
    elif is_dataclass(value):
        dcg = group.require_group(name)
        save_dataclass_to_group(value, dcg)
    elif value is None:
        group.attrs[name] = 'None'
    else:
        raise TypeError(
            f'type: {type(value)} not allowed in attrs for group, key, value: {group.name}, {name}, {value}')


def get_attr(group: h5py.Group, name, default=None, check_exists=False):
    """Inverse of set_attr. Gets many different types of values stored by set_attrs"""
    assert isinstance(group, h5py.Group)
    attr = group.attrs.get(name, None)
    if attr is not None:
        if isinstance(attr, str) and attr == 'None':
            attr = None
        if isinstance(attr, h5py.Dataset):
            attr = attr[:]  # Only small here, and works better with dataclasses to have real array not h5py dataset
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
        if type(attr) == list:
            attr = desanitize(attr)
        return attr

    g = group.get(name, None)
    if g is not None:
        if isinstance(g, h5py.Group):
            description = g.attrs.get('description')
            if description == 'simple dictionary':
                attr = load_dict_from_hdf_group(g)  # TODO: Want to see how loading full sweeplogs works
                return attr
            if description == 'NamedTuple':
                attr = load_group_to_namedtuple(g)
                return attr
            if description == 'dataclass':
                attr = load_group_to_dataclass(g)
                return attr
            if description == 'FitInfo':
                from src.DatObject.Attributes.DatAttribute import FitInfo
                attr = FitInfo.from_hdf(group, name)
                return attr
        elif isinstance(g, h5py.Dataset):
            return g
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


def set_list(group, name, list_):
    """
    Saves list as a h5py group inside 'group'
    Args:
        group (h5py.Group):
        name (str): Name of list
        list_ (list):

    Returns:
        (None):
    """
    lg = group.require_group(name)
    lg.attrs['description'] = 'list'
    for i, v in enumerate(list_):
        if isinstance(v, (np.ndarray, h5py.Dataset)):
            set_data(lg, str(i), v)
        else:
            set_attr(lg, str(i), v)


def get_list(group, name):
    """
    Inverse of set_list
    Args:
        group (h5py.Group):
        name (str): name of list in group

    Returns:
        (list):
    """
    lg = group.get(name)
    assert isinstance(lg, h5py.Group)
    assert lg.attrs.get('description') == 'list'
    all_keys = set(lg.keys()).union(lg.attrs.keys()) - {'description'}
    vals = dict()
    for k in lg.keys():
        vals[k] = get_attr(lg, k)
        if vals[k] is None:  # For getting datasets, but will default to None if it doesn't exist
            v = lg.get(k, None)
            vals[k] = v if v is None else v[:]
    return [vals[k] for k in sorted(vals)]  # Returns list in original order


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


def save_dataclass_to_group(dataclass, group: h5py.Group):
    """Saves dataclass inside group given"""
    assert is_dataclass(dataclass)
    group.attrs['description'] = 'dataclass'
    dc_name = dataclass.__class__.__name__
    if 'DC_name' in group.attrs.keys() and (n := group.attrs['DC_name']) != dc_name:
        raise TypeError(f'Trying to store dataclass with name {dc_name} where a dataclass with name {n} '
                        f'already exists')
    elif 'DC_name' not in group.attrs.keys():
        group.attrs['DC_name'] = dc_name
        group.attrs['DC_class'] = getsource(dataclass.__class__)

    # for key, val in asdict(dataclass).items():  # This tries to be too clever and turn everything into dicts, which does not work for lm.Parameters and I don't know what else
    for k in dataclass.__annotations__:
        v = getattr(dataclass, k)
        set_attr(group, k, v)


def load_group_to_dataclass(group: h5py.Group):
    """Returns dataclass as stored"""
    if group.attrs.get('description', None) != 'dataclass':
        raise ValueError(f'Trying to load_group_to_dataclass which has description: '
                         f'{group.attrs.get("description", None)}')
    DC_name = group.attrs.get('DC_name')
    dataclass_ = get_func(DC_name, group.attrs.get('DC_class'), is_a_dataclass=True, exec_code=True)
    d = {key: get_attr(group, key) for key in list(group.attrs.keys())+list(group.keys()) if key not in ['DC_class', 'description', 'DC_name']}
    return dataclass_(**d)


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


def get_func(func_name, func_code, is_a_dataclass=False, exec_code=True):
    """Cheeky way to get a function or class stored in an HDF file.
    I at least check that I'm not overwriting something, but still should be careful here"""
    from src.Scripts.SquareEntropyAnalysis import EA_data, EA_datas, EA_values, EA_params, \
        EA_value  # FIXME: Need to find a better way of doing this... Problem is that global namespaces is this module only, so can't see these even though they are imported at runtime.
    if func_name not in list(globals().keys()) + list(locals().keys()):
        if exec_code:
            logger.info(f'Executing: {func_code}')
            if is_a_dataclass:
                prepend = 'from __future__ import annotations\n@dataclass\n'
            else:
                prepend = ''
            from typing import List, Union, Optional, Tuple, Dict
            from dataclasses import field, dataclass
            d = dict(locals(), **globals())
            exec(prepend+func_code, d, d)  # Should be careful about this! Just running whatever code is stored in HDF
            globals()[func_name] = d[func_name]  # So don't do this again next time
        else:
            raise LookupError(f'{func_name} not found in global namespace, must be imported first!')  # FIXME: This doesn't work well because globals() here is only of this module, not the true global namespace... Not an easy workaround for this either.
    else:
        logger.debug(f'Func {func_name} already exists so not running self.func_code')
        if func_name in locals().keys():
            globals()[func_name] = locals()[func_name]
    func = globals()[func_name]  # Should find the function which already exists or was executed above
    assert callable(func)
    return func


class MyDataset(h5py.Dataset):
    def __init__(self, dataset: h5py.Dataset):
        super().__init__(dataset.id)

    @property
    def axis_label(self) -> str:
        title = self.attrs.get('axis_label', 'Not set')
        return title

    @axis_label.setter
    def axis_label(self, value: str):
        self.attrs['axis_label'] = value

    @property
    def units(self):
        return self.attrs.get('units', 'Not set')

    @units.setter
    def units(self, value: str):
        self.attrs['units'] = value

    @property
    def label(self):
        return self.attrs.get('label', 'Not set')

    @label.setter
    def label(self, value: str):
        self.attrs['label'] = value

    @property
    def bad_rows(self):
        bad_rows = self.attrs.get('bad_rows', None)
        if isinstance(bad_rows, str):
            if bad_rows.lower() == 'none':
                bad_rows = None
        return bad_rows

    @bad_rows.setter
    def bad_rows(self, value: list):
        if value is None:
            self.attrs['bad_rows'] = 'none'
        else:
            assert isinstance(value, (list, tuple, np.ndarray))
            self.attrs['bad_rows'] = value

    @property
    def good_rows(self):
        bad_rows = self.bad_rows
        if bad_rows is not None:
            good_rows = np.s_[list(set(range(self.shape[0]))-set(bad_rows))]
        else:
            good_rows = np.s_[:]
        return good_rows


if __name__ == '__main__':
    from src.DatObject.Make_Dat import DatHandler as DH
    dat = DH.get_dat(7582)


