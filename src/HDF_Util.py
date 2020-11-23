from __future__ import annotations
import functools
from collections import namedtuple
from typing import NamedTuple, Union, Optional, Type, TYPE_CHECKING, Any, List

from deprecation import deprecated
import os
import h5py
import numpy as np
import lmfit as lm
import json
import ast
import datetime
from dateutil import parser
import logging
from dataclasses import is_dataclass, asdict, dataclass, field
from inspect import getsource
import sys
from src import CoreUtil as CU

if TYPE_CHECKING:
    from src.DatObject.Attributes.DatAttribute import DatDataclassTemplate
    from src.DatObject.DatHDF import DatHDF
    from src.DatObject.Attributes.DatAttribute import DatAttribute

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
    elif isinstance(value, h5py.SoftLink):
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


def init_hdf_id(dat_id, hdfdir_path, overwrite=False):
    """Makes sure HDF folder exists, and creates an empty HDF there (will only overwrite if overwrite=True)"""
    file_path = os.path.join(hdfdir_path, dat_id + '.h5')
    file_path = init_hdf_path(file_path, overwrite=overwrite)
    return file_path


def init_hdf_path(path, overwrite=False):
    """Makes sure HDF folder exists, and creates an empty HDF there (will only overwrite if overwrite=True)"""
    if os.path.exists(path):
        if overwrite is True:
            os.remove(path)
        else:
            raise FileExistsError(
                f'HDF file already exists for {path}. Use "overwrite=True" to overwrite')
    else:
        hdfdir_path, _ = os.path.split(path)
        os.makedirs(hdfdir_path, exist_ok=True)  # Ensure directory exists
        f = h5py.File(path, 'w')  # Init a HDF file
        f.close()
    return path


def check_hdf_path(path: str) -> str:
    """Just checks if HDF exists at path and returns same path. Used for loading"""
    if not os.path.exists(path):
        raise FileNotFoundError(f'No HDF found at {path}')
    return path


def check_hdf_id(dat_id: str, hdfdir_path: str) -> str:
    """Just checks if HDF exists at path and returns same path. Used for loading"""
    path = os.path.join(hdfdir_path, dat_id, '.h5')
    path = check_hdf_path(path)
    return path


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


def get_data(group: h5py.Group, name:str) -> np.ndarray:
    """
    Gets data in array form. This is mostly so that I can change the way I get data later
    # TODO: Maybe make this only get data that isn't marked as bad or something in the future?

    Args:
        group (): Group that data is in
        name (): Name of data

    Returns:
        np.ndarray
    """
    return get_dataset(group, name)[:]


def get_dataset(group: h5py.Group, name:str) -> h5py.Dataset:
    """
    Gets dataset from group and returns the open Dataset

    Note: Remember to get data from dataset BEFORE closing hdf file (i.e. data = ds[:])

    Args:
        group (): Group which Dataset is in
        name (): Name of Dataset

    Returns:
        h5py.Dataset
    """
    if name in group.keys():
        ds = group.get(name)
    else:
        raise FileNotFoundError(f'{name} does not exist in {group.name}')
    return ds


def set_data(group: h5py.Group, name: str, data: Union[np.ndarray, h5py.Dataset], dtype=None):
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
    group.create_dataset(name, data.shape, dtype, data, maxshape=data.shape)
    # maxshape allows for dataset to be resized (smaller) later, which might be useful for getting rid of bad data


def link_data(from_group: h5py.Group, to_group: h5py.Group, from_name: str, to_name: Optional[str] = None):
    """
    Links named data from one group to another (with option to store as a different name in dest group).

    Args:
        from_group (): Group to link data from
        to_group (): Group to link data to
        from_name (): Name of data to link
        to_name (): Optional new name of data, otherwise uses same name

    Returns:
        None:
    """
    to_name = to_name if to_name else from_name
    if to_name not in to_group.keys():
        ds = from_group.get(from_name, None)
        if ds is not None:
            assert isinstance(ds, h5py.Dataset)
            to_group[to_name] = ds
        else:
            raise FileNotFoundError(f'{from_name} not found in {from_group.name}')
    else:
        raise FileExistsError(f'{to_name} already exits in {to_group.name}')


def set_attr(group: h5py.Group, name: str, value, dataclass: Optional[Type[DatDataclassTemplate]] = None):
    """Saves many types of value to the group under the given name which can be used to get it back from HDF"""
    assert isinstance(group, h5py.Group)
    if dataclass:
        assert isinstance(value, dataclass)
        value.save_to_hdf(group, name)
    elif isinstance(value, ALLOWED_TYPES) and not _isnamedtupleinstance(value) and value is not None:  # named tuples subclass from tuple...
        value = sanitize(value)
        if isinstance(value, np.ndarray) and value.size > 500:
            set_data(group, name, value)
        else:
            group.attrs[name] = value
    elif isinstance(value, h5py.SoftLink):
        if name in group:
            del group[name]
        group[name] = value
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


def get_attr(group: h5py.Group, name, default=None, check_exists=False, dataclass: Type[DatDataclassTemplate] = None):
    """
    Inverse of set_attr. Gets many different types of values stored by set_attrs

    Args:
        group ():
        name ():
        default ():
        check_exists ():
        dataclass (): Optional DatDataclass which can be used to load the information back into dataclass form

    Returns:

    """
    """Inverse of set_attr. Gets many different types of values stored by set_attrs
    
    
    
            dataclass (): Optional DatDataclass which can be used to load the information back into dataclass form 
    """
    assert isinstance(group, h5py.Group)
    if dataclass:
        return dataclass.from_hdf(group, name)  # TODO: Might need to allow name to default to None here

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
            link = group.get(name, getlink=True)
            if isinstance(link, h5py.SoftLink):  # If stored as a SoftLink, only return as a SoftLink)
                return link
            else:
                return g
    if check_exists is True:
        raise KeyError(f'{name} does not exist or is not an attr that can be loaded by get_attr in group {group.name}')
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


@deprecated
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


@dataclass
class HDFContainer:
    """For storing a possibly open HDF along with the filepath required to open it (i.e. so that an open HDF can
    be passed around, and if it needs to be reopened in a different mode, it can be)"""
    hdf: h5py.File  # Open/Closed HDF
    hdf_path: str  # Path to the open/closed HDF (so that it can be reopened in required mode)
    group: h5py.Group = field(default=False)  # False will look the same as a closed HDF group
    group_name: str = field(default=None)

    def __post_init__(self):
        if self.group and not self.group_name:  # Just check that if a group is passed, then group name is also passed
            raise ValueError(f'group: {self.group}, group_name: {self.group_name}. group_name must not be None if group'
                             f'is passed in')

    @classmethod
    def from_path(cls, path, mode='r'):
        """Initialize just from path, only change read_mode for creating etc
        Note: The HDF is closed on purpose before leaving this function!"""
        hdf = h5py.File(path, mode)
        hdf.close()
        inst = cls(hdf=hdf, hdf_path=path)
        return inst

    @classmethod
    def from_hdf(cls, hdf: h5py.File):
        """Initialize from OPEN hdf (has to be open to read filename).
        Note: This will close the file as the aim of this is to keep them closed as much as possible"""
        inst = cls(hdf=hdf, hdf_path=hdf.filename)
        hdf.close()
        return inst

    def get(self, *args, **kwargs):
        """Makes HDFContainer act like HDF for most most get calls. Will warn if HDF was not already open as this should
        be handled before making calls.
        """
        if self.hdf:
            return self.hdf.get(*args, **kwargs)
        else:
            logger.warning(f'Trying to get value from closed HDF, this should handled with wrappers')
            with h5py.File(self.hdf_path, 'r') as f:
                return f.get(*args, **kwargs)

    def __getattr__(self, item):
        """Default to trying to apply action to the h5py.File"""
        return getattr(self.hdf, item)


def _with_dat_hdf(func, mode='read'):
    """Assuming being called within a Dat object (i.e. self.hdf and self.hdf_path exist)
    Ensures that the HDF is open in write mode before calling function, and then closes at the end"""
    READ = tuple('r')
    WRITE = tuple(('r+', 'w', 'w+', 'a'))
    if mode == 'read':
        MODES = READ
    elif mode == 'write':
        MODES = WRITE + READ
    else:
        raise ValueError

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        obj = args[0]  # Should be a 'self' argument which has self.hdf for holding HDFContainer
        container = _get_obj_hdf_container(obj)
        f = container.hdf

        opened = False  # Whether this wrapper has done the opening (and is responsible for closing)
        set_write = False  # Whether this wrapper changed file from Read to Write and should change back after
        if not f:
            container.hdf = h5py.File(container.hdf_path, MODES[0])
            opened = True
            prev_group_name, prev_group = None, None
        elif mode == 'write' and f.mode in READ:
            container.hdf.close()
            container.hdf = h5py.File(container.hdf_path, WRITE[0])
            set_write = True
            prev_group_name, prev_group = container.group_name, container.group
        else:
            prev_group_name, prev_group = container.group_name, container.group

        _set_container_group(obj)  # Sets obj.container.group_name and .group to current group
        try:
            ret = func(*args, **kwargs)
        except:  # Catch ANY exception (because I need to close the file no matter what)
            # print(sys.exc_info()[0])
            container.hdf.close()
            raise

        if opened:
            container.hdf.close()  # Assumes self.container attribute not being overwritten in any deeper function call!
        else:
            if set_write:  # Put back into Read mode to minimize time HDF is locked in write mode by any process
                container.hdf.close()
                container.hdf = h5py.File(container.hdf_path, READ[0])
            _set_container_group(obj, prev_group_name, prev_group)  # Return to previous group/group_name
        return ret
    return wrapper


def with_hdf_read(func):
    return _with_dat_hdf(func, mode='read')


def with_hdf_write(func):
    return _with_dat_hdf(func, mode='write')


def _get_obj_hdf_container(obj):
    if not hasattr(obj, 'hdf'):
        raise RuntimeError(f'Did not find "self.hdf" for object: {obj}')
    container: HDFContainer = getattr(obj, 'hdf')
    if not isinstance(container, HDFContainer):
        raise TypeError(f'got type {container}. HDF should be stored in an HDFContainer not as a plain HDF. '
                        f'Use HDU.HDFContainer (because need to ensure that a path is present along with HDF)')
    return container


def _set_container_group(obj: Union[DatAttribute, DatHDF],
                         group_name: Optional[str] = None,
                         group: Optional[h5py.Group] = None) -> HDFContainer:
    """
    Sets the group and group_name in HDFContainer so that the current group is accessible through self.hdf.group
    Args:
        obj (): Any object which has a self.group_name and self.hdf (self.hdf: HDFContainer)
        group_name (str): Optional group_name to set  (for setting back to previous setting)
        group (h5py.Group): Optional group to set   (for setting back to previous setting)
    Returns:
        (HDFContainer): Returns the modified HDFContainer (although it is modified in place)
    """
    container = _get_obj_hdf_container(obj)
    if group_name or group:  # If passing in group to set
        if group and group_name:
            if group_name not in group.name:
                raise RuntimeError(f'{group.name} != {group_name}')
        if group:
            container.group = group
            container.group_name = group.name
        elif group_name:
            container.group = container.hdf.get(group_name)
            container.group_name = group_name
        else:
            raise NotImplementedError(f'Should not reach this')
    else:  # Infer from self.group_name
        group_name = getattr(obj, 'group_name', None)
        if group_name:
            container.group_name = group_name
            container.group = container.hdf.get(group_name)
    return container


def ensure_hdf_container(possible_hdf: Union[h5py.File, HDFContainer]):
    """
    If HDFContainer passed in, it gets passed back unchanged.
    An open h5py.File can be used to initialize the HDFContainer (NOTE: it will close the h5py.File)
    Otherwise an error is thrown
    Args:
        possible_hdf (): Either HDFContainer or open h5py.File

    Returns:
        HDFContainer: container of hdf and filepath to hdf
    """
    if isinstance(possible_hdf, HDFContainer):
        return possible_hdf
    elif isinstance(possible_hdf, h5py.File):
        if possible_hdf:
            return HDFContainer.from_hdf(possible_hdf)
        else:
            raise ValueError(f'h5py.File passed in was closed so path could not be read: {possible_hdf}')
    else:
        raise TypeError(f'{possible_hdf} is not an HDFContainer or h5py.File')


def is_Group(parent_group, key):
    class_ = parent_group.get(key, getclass=True)
    if class_ is h5py.Group:
        return True
    else:
        return False


def is_Dataset(parent_group, key):
    class_ = parent_group.get(key, getclass=True)
    if class_ is h5py.Dataset:
        return True
    else:
        return False


def is_DataDescriptor(group):
    if 'data_link' in group.keys():  # Check the group is a DataDescriptor
        return True
    else:
        return False


def find_all_groups_names_with_attr(parent_group: h5py.Group, attr_name: str, attr_value: Optional[Any] = None):
    """
    Returns list of group_names for all groups which contain the specified attr_name with Optional attr_value

    Args:
        parent_group (h5py.Group): Group to recursively look inside of
        attr_name (): Name of attribute to be checking in each group
        attr_value (): Optional value of attribute to compare to

    Returns:
        (List[str]): List of group_names which contain specified attr_name [equal to att_value]
    """
    _DEFAULTED = object()
    group_names = []

    def find_single(name, obj: Union[h5py.Dataset, h5py.Group]):
        """h5py.visititems() requires a function which takes either a Dataset or Group and returns
        None or value"""
        if obj.name not in group_names and isinstance(obj, h5py.Group):
            val = get_attr(obj, attr_name, _DEFAULTED)
            if val is not _DEFAULTED:
                if attr_value is None or val == attr_value:
                    return obj.name  # Full path
        return None

    while True:
        name = parent_group.visititems(find_single)
        if not name:
            break  # Break out if not found anywhere
        else:
            group_names.append(name)
    return group_names


def find_data_paths(parent_group: h5py.Group, data_name: str, first_only: bool = False) -> List[str]:
    """
    Returns list of data_paths to data with 'data_name'. If first_only is True, then will return the first found
    matching data_path as a string
    Args:
        parent_group (): Group to look for data in
        data_name ():
        first_only ():

    Returns:
        (List[str]): either list of paths to named data or single path if first_only == True
    """
    data_paths = []

    def find_single(name, obj: Union[h5py.Group, h5py.Dataset]):
        """h5py.visititems() requires a function which takes either a Dataset or Group and returns
        None or value"""
        if isinstance(obj, h5py.Dataset):
            if obj.name not in data_paths and name == data_name:
                return obj.name  # Full path
        return None

    if first_only:
        return [parent_group.visititems(find_single)]
    else:
        while True:
            path = parent_group.visititems(find_single)
            if not path:
                break  # Break out if not found anywhere
            else:
                data_paths.append(path)
        return data_paths


