from __future__ import annotations

import abc
import threading
import functools
from collections import namedtuple
from typing import NamedTuple, Union, Optional, Type, TYPE_CHECKING, Any, List, Tuple, Callable, Dict, TypeVar
import copy

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
from dataclasses import is_dataclass, dataclass, field
from inspect import getsource
from src import CoreUtil as CU
import time

if TYPE_CHECKING:
    from src.DatObject.DatHDF import DatHDF
    from src.DatObject.Attributes.DatAttribute import DatAttribute, logger

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
    """
    Saves params to group (i.e. create group to save params in first)
    Args:
        params (): lm.Parameters to save
        group (): Group to save them in

    Returns:

    """
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

        #  For HDF only. If stderr is None, then fit failed to calculate uncertainties (fit may have been successful)
        if getattr(par, 'stderr', None) is None:
            all_par_values += f'{key}=None'
        else:
            all_par_values += f'{key}={par.value:.5g}, '
    logger.debug(f'Saving best_values as: {all_par_values}')
    group.attrs['best_values'] = all_par_values  # For viewing in HDF only
    pass


def params_from_HDF(group, initial=False) -> lm.Parameters:
    params = lm.Parameters()
    for key in group.keys():
        if isinstance(group[key], h5py.Group) and group[key].attrs.get('description', None) == 'Single Param':
            par_group = group[key]
            par_vals = {par_key: par_group.attrs.get(par_key, None) for par_key in PARAM_KEYS}
            par_vals = {key: v if not (isinstance(v, float) and np.isnan(v)) else None for key, v in par_vals.items()}
            params.add(**par_vals)  # create par
            par = params[key]  # Get single par

            if not initial:
                par.stderr = par_group.attrs.get('stderr', np.nan)
                par.stderr = None if np.isnan(par.stderr) else par.stderr  # Replace NaN with None if that's what it was

                # Because the saved value was actually final value, but inits into init_val I need to switch them.
                par.value = par.init_value
                par.init_value = par_group.attrs.get('init_value',
                                                     np.nan)  # I save init_value separately  # TODO: Don't think this is true any more 23/11
                par.init_value = None if np.isnan(par.init_value) else par.init_value  # Replace NaN with None

    return params


def get_data(group: h5py.Group, name: str) -> np.ndarray:
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


def get_dataset(group: h5py.Group, name: str) -> h5py.Dataset:
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
    group.create_dataset(name, data.shape, dtype, data)  #, maxshape=data.shape)  # 29mar21 commented out because not using and worried this is increasing file size
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


@dataclass
class DatDataclassTemplate(abc.ABC):
    """
    This provides some useful methods, and requires the necessary overrides s.t. a dataclass with most types of info
    can be saved in a HDF file and be loaded from an HDF file

    Any Dataclasses which are going to be stored in HDF should inherit from this. This provides a method to save to
    hdf as well as a class method to load back from the hdf.

    NOTE: Also make sure parent_group is CLOSED after calling save_to_hdf or from_hdf !!!

    e.g.
        @dataclass
        class Test(DatDataClassTemplate):
            a: int
            b: str = field(default = "Won't show in repr", repr = False)

        d = Test(1)
        d.save_to_hdf(group)  # Save in group with default dataclass name (Test)
        e = Test.from_hdf(group)  # Load from group (will have the same settings as initial dataclass)
    """

    def save_to_hdf(self, parent_group: h5py.Group, name: Optional[str] = None):
        """
        Default way to save all info from Dataclass to HDF in a way in which it can be loaded back again. Override this
        to save more complex dataclasses.
        Make sure if you override this that you override "from_hdf" in order to get the data back again.

        Args:
            parent_group (h5py.Group): The group in which the dataclass should be saved (i.e. it will create it's own group in
                here)
            name (Optional[str]): Optional specific name to store dataclass with, otherwise defaults to Dataclass name

        Returns:
            None
        """
        if name is None:
            name = self._default_name()
        name = name.replace('/', '-')  # '/' makes nested subgroups in HDF
        if name in parent_group.keys():
            if is_Group(parent_group, name):
                logger.debug(f'Deleting contents of {parent_group.name}/{name}')
                del parent_group[name]
            else:
                raise FileExistsError(f'{parent_group.get(name).name} exists in path where dataclass was asked to save')
        dc_group = parent_group.require_group(name)
        self._save_standard_attrs(dc_group, ignore_keys=self.ignore_keys_for_hdf())

        # Save any additional things
        self.additional_save_to_hdf(dc_group)

        return dc_group  # For making overriding easier (i.e. can add more to group after calling super().save_to_hdf())

    def additional_save_to_hdf(self, dc_group: h5py.Group):
        """
        Override to save any additional things to HDF which require special saving (e.g. other Dataclasses)
        Note: Don't forget to implement additional_load_from_hdf() and to add key(s) to ignore_keys_for_hdf
        Args:
            dc_group (): The group in which the rest of the dataclass is being stored in

        Returns:

        """
        pass

    @staticmethod
    def additional_load_from_hdf(dc_group: h5py.Group) -> Dict[str, Any]:
        """
        Override to load any additional things from HDF which require special loading (e.g. other Dataclasses)
        Note: Don't forget to implement additional_save_to_hdf() and to add key(s) to ignore_keys_for_hdf
        Note: @staticmethod because must be called before class is instantiated
        Args:
            dc_group (): The group in which the rest of the dataclass is being stored in

        Returns:
            (Dict[str, Any]): Returns a dict where keys are names in dataclass
        """
        return {}

    @staticmethod
    def ignore_keys_for_hdf() -> Optional[Union[str, List[str]]]:
        """Override this to ignore specific dataclass keys when saving to HDF or loading from HDF
        Note: To save or load additional things, override additional_save_to_hdf and additional_load_from_hdf
        """
        return None

    @classmethod
    def from_hdf(cls: Type[T], parent_group: h5py.Group, name: Optional[str] = None) -> T:
        """
        Should get back all data saved to HDF with "save_to_hdf" and initialize the dataclass and return that instance
        Remember to override this when overriding "save_to_hdf"

        Args:
            parent_group (h5py.Group): The group in which the saved data should be found (i.e. it will be a sub group in this
                group)
            name (Optional[str]): Optional specific name to look for if saved with a specific name, otherwise defaults
                to the name of the Dataclass

        Returns:
            (T): Returns and instance of cls. Should have the correct typing. Or will return None with a warning if
            no data is found.
        """
        if name is None:
            name = cls._default_name()
        name = name.replace('/', '-')  # Because I make this substitution on saving, also make it for loading.
        dc_group = parent_group.get(name)

        if dc_group is None:
            raise NotFoundInHdfError(f'No {name} group in {parent_group.name}')

        # Get standard things from HDF
        d = cls._get_standard_attrs_dict(dc_group)

        # Add additional things from HDF
        d = dict(**d, **cls.additional_load_from_hdf(dc_group))

        d = {k: v if not isinstance(v, h5py.Dataset) else v[:] for k, v in
             d.items()}  # Load all data into memory here if necessary
        inst = cls(**d)
        return inst

    def _save_standard_attrs(self, group: h5py.Group, ignore_keys: Optional[Union[str, List[str]]] = None):
        ignore_keys = CU.ensure_set(ignore_keys)
        for k in set(self.__annotations__) - ignore_keys:
            val = getattr(self, k)
            if isinstance(val, (np.ndarray,
                                h5py.Dataset)) and val.size > 1000:  # Pretty much anything that is an array should be saved as a dataset
                set_data(group, k, val)
            elif allowed(val):
                set_attr(group, k, val)
            else:
                logger.warning(
                    f'{self.__class__.__name__}.{k} = {val} which has type {type(val)} (where type {self.__annotations__[k]} was expected) which is not able to be saved automatically. Override "save_to_hdf" and "from_hdf" in order to save and load this variable')

    @classmethod
    def _get_standard_attrs_dict(cls, group: h5py.Group, keys=None) -> dict:
        assert isinstance(group, h5py.Group)
        d = dict()
        if keys is None:
            keys = cls.__annotations__
        ignore_keys = cls.ignore_keys_for_hdf()
        if ignore_keys is None:
            ignore_keys = []
        for k in keys:
            if k not in ignore_keys:
                d[k] = get_attr(group, k, None)
        return d

    @classmethod
    def _default_name(cls):
        return cls.__name__


def set_attr(group: h5py.Group, name: str, value, dataclass: Optional[Type[DatDataclassTemplate]] = None):
    """Saves many types of value to the group under the given name which can be used to get it back from HDF"""
    if not isinstance(group, h5py.Group):
        raise TypeError(f'{group} is not a h5py.Group: Trying to set {name} with {value}. dataclass={dataclass}')
    if dataclass:
        assert isinstance(value, dataclass)
        value.save_to_hdf(group, name)
    elif is_dataclass(value) and not dataclass:
        raise ValueError(f'Dataclass must be passed in when saving a dataclass (i.e. the class should be passed in)')
    elif _is_list_of_arrays(value):
        save_list_of_arrays(group, name, value)
    elif isinstance(value, ALLOWED_TYPES) and not _isnamedtupleinstance(
            value) and value is not None:  # named tuples subclass from tuple...
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
            d_str = CU.json_dumps(value)
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
    # elif is_dataclass(value):
    #     dcg = group.require_group(name)
    #     save_dataclass_to_group(value, dcg)
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
    if not isinstance(group, h5py.Group):
        raise TypeError(f'{group} is not an h5py.Group')
    if dataclass:
        return dataclass.from_hdf(group, name)  # TODO: Might need to allow name to default to None here

    attr = group.attrs.get(name, None)
    if attr is not None:
        if isinstance(attr, str) and attr == 'None':
            return None
        if isinstance(attr, h5py.Dataset):
            return attr[:]  # Only small here, and works better with dataclasses to have real array not h5py dataset
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
            if isinstance(attr, str) and attr.count(':') > 1:  # If it's a datetime its likely in form (March 1, 2021 09:00:00')
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
            # if description == 'dataclass':
            #     attr = load_group_to_dataclass(g)
            #     return attr
            if description == 'FitInfo':
                from src.AnalysisTools.general_fitting import FitInfo
                attr = FitInfo.from_hdf(group, name)
                return attr
            if description == 'List of arrays':
                return load_list_of_arrays(g)
        elif isinstance(g, h5py.Dataset):
            link = group.get(name, getlink=True)
            if isinstance(link, h5py.SoftLink):  # If stored as a SoftLink, only return as a SoftLink)
                return link
            else:
                return g
    if check_exists is True:
        raise NotFoundInHdfError(
            f'{name} does not exist or is not an attr that can be loaded by get_attr in group {group.name}')
    else:
        return default


def _is_list_of_arrays(value: Any) -> bool:
    if isinstance(value, list):
        if all([isinstance(v, np.ndarray) for v in value]):
            return True
    return False


def save_list_of_arrays(group: h5py.Group, name: str, arrays: List[np.ndarray]):
    list_group = group.require_group(name)
    list_group.attrs['description'] = 'List of arrays'
    for i, arr in enumerate(arrays):
        set_data(list_group, str(i), arr)


def load_list_of_arrays(group):
    ret_dict = {int(k): get_data(group, k) for k in group}
    return [ret_dict[k] for k in sorted(ret_dict)]  # Make sure it comes back in same order as saved!


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
    for key, val in ntuple._asdict().items():
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
    d = {key: get_attr(group, key) for key in list(group.attrs.keys()) + list(group.keys()) if
         key not in ['DC_class', 'description', 'DC_name']}
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
    # from src.Scripts.SquareEntropyAnalysis import EA_data, EA_datas, EA_values, EA_params, \
    #     EA_value  # FIXME: Need to find a better way of doing this... Problem is that global namespaces is this module only, so can't see these even though they are imported at runtime.
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
            exec(prepend + func_code, d, d)  # Should be careful about this! Just running whatever code is stored in HDF
            globals()[func_name] = d[func_name]  # So don't do this again next time
        else:
            raise LookupError(
                f'{func_name} not found in global namespace, must be imported first!')  # FIXME: This doesn't work well because globals() here is only of this module, not the true global namespace... Not an easy workaround for this either.
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
            good_rows = np.s_[list(set(range(self.shape[0])) - set(bad_rows))]
        else:
            good_rows = np.s_[:]
        return good_rows


class ThreadID:
    def __init__(self, target_mode: str):
        self.id = threading.get_ident()
        self.target_mode = target_mode
        self.current_status = None


class ThreadQueue:
    def __init__(self):
        """Need to make sure this is only called once per process (i.e. threadlock wherever this is being created, and
        check if it already exists first)"""
        self._lock = threading.Lock()
        self.queue = []

    def put(self, entry: ThreadID):
        """
        Put new thread into waiting queue
        Args:
            entry (ThreadID): ThreadID object to add to queue
        Returns:

        """
        with self._lock:
            self.queue.append(entry)

    def get_next(self):
        with self._lock:
            return self.queue.pop(0)


READ = tuple('r')
WRITE = tuple(('r+', 'w', 'w+', 'a'))
_NOT_SET = object()


class ThreadManager:
    def __init__(self):
        self.waiting_write_threads = 0
        self.waiting_read_threads = 0

        self.active_write_thread: Optional[Tuple[int, datetime.datetime]] = None  # Only ever 1 write thread
        self.active_read_threads: Dict[int, datetime.datetime] = dict()  # Dict of {thread_id: datetime}
        self.stashed_read_threads: Dict[int, datetime.datetime] = dict()  # Dict of {thread_id: datetime}

        self.lock = threading.RLock()
        self.read_condition = threading.Condition(self.lock)  # Intended to release all at once
        self.write_condition = threading.Condition(self.lock)  # Intended to only release 1 at a time
        self.manager_condition = threading.Condition(self.lock)  # Manages the read and write condition
        self.manager_event = threading.Event()  # Triggers manager to look for something

        self.manager_thread = None
        self.start_scheduler()

    def get_condition(self, requested_mode: str):
        if requested_mode == 'r':
            condition = self.read_condition
        elif requested_mode == 'w':
            condition = self.write_condition
        else:
            raise NotImplementedError(f'{requested_mode} not recognized')
        return condition

    def add_to_queue(self, requested_mode: str):
        """Add thread into either 'r' or 'w' queue and then get manager to figure out if anything should start"""
        with self.lock:
            self.start_scheduler()
            # thread_id = threading.get_ident()
            #
            # # TODO: Do I need to do this check here?
            # if thread_id in self.active_read_threads:
            #     self.stash_active_read()

            if requested_mode == 'r':
                self.waiting_read_threads += 1
            elif requested_mode == 'w':
                self.waiting_write_threads += 1
            self.manager_event.set()
            return self.waiting_read_threads, self.waiting_write_threads

    def stash_active_read(self):
        """Stashes current active read thread so that it can become a waiting write thread without waiting for itself"""
        with self.lock:
            thread_id = threading.get_ident()
            if thread_id in self.active_read_threads:
                if thread_id in self.stashed_read_threads:
                    raise RuntimeError(f'{thread_id} has already been stashed before. Should not be trying to stash '
                                       f'again')
                self.stashed_read_threads[thread_id] = self.active_read_threads.pop(thread_id)
            elif self.active_write_thread and thread_id == self.active_write_thread[0]:
                raise RuntimeError(f'{thread_id} is trying attempting to be stashed as a write thread. This should '
                                   f'not happen')
            else:
                raise KeyError(f'{thread_id} is not an active thread')

    def set_active(self, mode: str):
        """Adds current thread to active list"""
        with self.lock:
            thread_id = threading.get_ident()
            if mode == 'r':
                self.active_read_threads[thread_id] = datetime.datetime.now()
            elif mode == 'w':
                self.active_write_thread = (thread_id, datetime.datetime.now())
            else:
                raise NotImplementedError
            # No updates to do upon adding threads (no self.manager_condition.notify())

    def remove_active(self):
        """Removes current thread from active lists, and restores stashed read status if present
         (i.e. if finishing the initial call to write part and returning back to read only)"""
        with self.lock:
            self.start_scheduler()
            thread_id = threading.get_ident()
            if thread_id in self.active_read_threads:
                self.active_read_threads.pop(thread_id)
            elif self.active_write_thread and self.active_write_thread[0] == thread_id:
                self.active_write_thread = None
                if thread_id in self.stashed_read_threads:
                    self.active_read_threads[thread_id] = self.stashed_read_threads.pop(thread_id)
            else:
                raise RuntimeError(f'{thread_id} not recognized')

            self.manager_event.set()
            return True

    def get_active_threads(self) -> Dict[int, datetime.datetime]:
        """Returns a dict of all current running threads"""
        with self.lock:
            active = copy.copy(self.active_read_threads)
            if self.active_write_thread:
                id_, date = self.active_write_thread
                active[id_] = date
            return active

    def get_stashed_read_threads(self) -> Dict[int, datetime.datetime]:
        """Returns a dict of all current stashed read threads (i.e. threads that were in read but wanted to go to write mode"""
        with self.lock:
            stashed = copy.copy(self.stashed_read_threads)
            return stashed

    def start_scheduler(self):
        """Starts the condition watcher thread which will let threads know when they can have access to the HDF"""
        def watcher():
            while True:
                triggered = self.manager_event.wait(timeout=10)  # Only check when updates to active_threads
                with self.manager_condition:  # Only aquire the manager lock here (waiting for event blocks otherwise)
                    if self.manager_event.is_set():
                        self.manager_event.clear()
                        self.notify_relevant_threads()
                    else:
                        # logger.error(f'{threading.get_ident()} manager thread timed out')
                        if triggered:
                            logger.error(f'{threading.get_ident()} manager thread timed out '
                                         f'(triggered = {triggered} <- should be False)')
                        self.manager_thread = None
                        break  # Timed out, let thread end

        with self.lock:
            if self.manager_thread is None:
                self.manager_thread = threading.Thread(target=watcher)
                self.manager_thread.start()
            return self.manager_thread

    def notify_relevant_threads(self):
        """Sends a notification either to lots of read threads, or one write thread"""
        with self.lock:
            if self.waiting_write_threads > 0:  # If there is a write thread waiting
                if not self.active_read_threads and not self.active_write_thread:
                    self.write_condition.notify()  # Set one thread running in write mode
                    self.waiting_write_threads -= 1
                    return True
                else:
                    return False  # Don't want to start any read or write threads,
                    # they should queue up and wait for current threads to finish

            if self.waiting_read_threads > 0:
                if self.active_write_thread is None:
                    self.read_condition.notify_all()
                    self.waiting_read_threads = 0
                    return True
                else:
                    return False
            return False  # Nothing to notify


@dataclass
class HDFContainer:
    hdf: h5py.File
    hdf_path: str

    def __post_init__(self):
        self.thread_manager = ThreadManager()

        self._groups = {}  # {<thread_id>: <group>}
        self._group_names = {}  # {<thread_id>: <group_name>}
        self._lock = threading.RLock()

    @property
    def group(self) -> h5py.Group:
        """Use this to get the threadsafe group object
        Examples:
            class SomeDatAttr(DatAttribute):
                group_name = 'TestName'

                @with_hdf_read
                def do_something(self, a, b):
                    group = self.hdf.group  # Group will point to /TestName in HDF (Threadsafe)
        """
        thread_id = threading.get_ident()
        with self._lock:  # TODO: Is it necessary to thread lock a lookup?
            group = self._groups.get(thread_id, False)  # Default to False to look like a Closed Group.
        return group

    @group.setter
    def group(self, value):
        assert isinstance(value, (type(None), h5py.Group))
        thread_id = threading.get_ident()
        with self._lock:
            self._groups[thread_id] = value

    @group.deleter
    def group(self):
        thread_id = threading.get_ident()
        with self._lock:
            del self._groups[thread_id]

    @property
    def group_name(self) -> str:
        """Use this to get the threadsafe group_name (mostly to be used in with_hdf_read/write)
        Examples:
            class SomeDatAttr(DatAttribute):
                group_name = 'TestName'

                @with_hdf_read
                def do_something(self, a, b):
                    group_name = self.hdf.group_name  # gets '/TestName'
        """
        thread_id = threading.get_ident()
        with self._lock:
            group_name = self._group_names.get(thread_id, None)
        return group_name

    @group_name.setter
    def group_name(self, value):
        assert isinstance(value, (type(None), str))
        thread_id = threading.get_ident()
        with self._lock:
            self._group_names[thread_id] = value

    @group_name.deleter
    def group_name(self):
        thread_id = threading.get_ident()
        with self._lock:
            del self._group_names[thread_id]

    @classmethod
    def from_path(cls, path, mode='r'):
        """Initialize just from path, only change read_mode for creating etc
        Note: The HDF is closed on purpose before leaving this function!"""
        logger.debug(f'initializing from path')
        hdf = h5py.File(path, mode)
        hdf.close()
        inst = cls(hdf=hdf, hdf_path=path)
        return inst

    @classmethod
    def from_hdf(cls, hdf: h5py.File):
        """Initialize from OPEN hdf (has to be open to read filename).
        Note: This will close the file as the aim of this is to keep them closed as much as possible"""
        logger.debug(f'initializing from hdf')
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
            logger.warning(f'Trying to get value from closed HDF, this should be handled with wrappers')
            with h5py.File(self.hdf_path, 'r') as f:
                return f.get(*args, **kwargs)

    def set_group(self, group_name: str):
        if group_name:
            self.group = self.hdf.get(group_name)
            self.group_name = group_name

    def __getattr__(self, item):
        """Default to trying to apply action to the h5py.File"""
        return getattr(self.hdf, item)

    def function_wrapper(self, func: Callable, mode_: str, group_name: str):
        """
            Wraps 'func' s.t. HDF is opened in read or write in synchronous way. (i.e. write single threaded, read multi threaded)

            Single Thread Behaviour:
                  Opens HDF if necessary, or switches from 'r' to 'w' if necessary. Does not switch from 'w' to 'r' to prevent
                  unnecessary switches.
                  Calls function with args
                  Returns HDF to starting state whether or not an Exception is raised, then Raises exception further.
                      If exception is not caught, it will eventually reach the call which opened the HDF and close anyway.
                      If caught somewhere along the way, HDF will remain in correct state
                  If no exception raised, returns whatever the function returned

            Multi Thread Behaviour:
                Very similar to Single Thread with some exceptions:
                1. If a new thread wants to use HDF in 'r' and other threads have 'r' only, then it STARTS immediately.
                2. If a new thread wants to use HDF and another thread already has 'w' mode, then joins
                    'r' or 'w' QUEUE respectively.
                4. If a thread wants to change to 'w' mode, it joins the 'w' queue and waits for all existing threads to
                    finish or also join the 'w' queue, and any new thread is forced to join a 'r' or 'w' queue.

            Note: There should only ever be 1 thread in 'w' mode at a time.

            Args:
                mode_ (): Required HDF mode (read or write)
                func (): Function to call
                group_name (): Name of group to set up in hdf.group

            Returns:
                (Any): Return of function

            Examples:
                Case 1: First and single call to _read_
                    1. HDF Opened in 'r'
                    2. Function called
                    3. HDF Closed
                    4. Return functions return

                Case 2: Open in _read_ then _write_ something *inside* _read_ function
                    1. HDF Opened in 'r'
                    2. _read_ function called
                        2a. HDF already open in 'r', but needs to switch to 'w'. Note: No other threads so can switch to 'w'
                        2b. _write_ function called
                        2c. Switch HDF back into 'r'. Note: Does not close because still active
                        2d. Return _write_ functions return
                    3. HDF closed
                    4. Return _read_functions return

                Case 3: Open in _write_ then call _read_ *inside* _write_ function but an unexpected EXCEPTION is raised before
                    getting to another _read_ function which is inside _write_.
                    1. HDF Opened in 'w'
                    2. _write_ function called
                        2a. HDF already open in 'w', only need 'r', but no need to switch (and shouldn't for efficiency)
                        2b. _read_ function called
                        2c. EXCEPTION raised
                        2d. CATCH exception, but make no change to HDF state because no change made at beginning of this call
                        2e. RAISE exception
                        --- (next _read_ function not reached because exception raised and not caught within _write_ function)
                    3. CATCH exception, and because this opened HDF, Close HDF
                    4. RAISE exception

                Case 4: Open in _write_, _read_ something *inside* _write_ function, then _read_ something else that is expected to raise and EXCEPTION, followed by another _read_ function
                    1. HDF Opened in 'w'
                    2. _write_ function called
                        2a. HDF already open in 'w', only need 'r', but no need to switch. Note: Should stay in 'w' to
                            prevent unnecessary switching
                        2b. _read_ function called
                        2c. Leave HDF in 'w'
                        2d. Return _read_ functions return
                        ----- (next _read_ function inside of _write_ which expects a possible EXCEPTION)
                        2e. same as 2a
                        2f. _read_ function called -- EXCEPTION Raised
                        2g. CATCH exception, but make no change because HDF already in starting state (i.e. prior to this _read_ call)
                        2h. RAISE exception
                        2i. EXCEPTION caught in _write_ function and handled
                        ----- (last _read_ function, still reached because _write_ function handled exception from 2nd _read_)
                        2jklm. same as 2abcd
                    3. HDF CLosed
                    4. Return _write_ functions return

                Case 5: 1st Thread Opens in _read_, 2nd Thread wants to _read_ (with another _read_ inside), BUT 1st thread raises unexpected EXCEPTION
                    A1. HDF Opened in 'r' by 1st Thread (A)
                    B1. HDF already open in 'r' for 2nd Thread (B)
                    A2. A function called but RAISES EXCEPTION
                    A3. CATCH exception, does not close HDF because still active threads
                    A4. Removes self from active and raises exception
                    B2. Bs inner _read_ called
                        B2a. HDF already open in 'r'
                        ...
                        B2x. Return functions return
                    B3. Closes HDF because no other threads active
                    B4. Return functions return


        """
        def get_starting_state() -> Tuple[bool, str]:
            """Returns whether HDF is currently open, and if so what state it is in ('r' or 'w')"""
            if self.hdf:
                open_ = True
                hdf_mode = self.hdf.mode
                if hdf_mode in READ:
                    s_mode = 'r'
                elif hdf_mode in WRITE:
                    s_mode = 'w'
                else:
                    raise ValueError(f'{hdf_mode} not recognized as READ or WRITE: {READ, WRITE}')
            else:
                open_ = False
                s_mode = None
            return open_, s_mode

        def set_final_state(cv: threading.Condition, was_active_setter: bool):
            with cv:
                # Mark as done
                if was_active_setter:
                    self.thread_manager.remove_active()  # Remove this thread as an active thread
                    # (if write, will also check to restore read state if stashed)

                # If no others left, then close hdf file
                if not self.thread_manager.get_active_threads():  # and not self.thread_manager.get_stashed_read_threads():
                    self.hdf.close()

        def call_func(*args, **kwargs):
            self.set_group(group_name=group_name)
            return func(*args, **kwargs)

        def wrapper(*args, **kwargs):
            mode = mode_
            if mode == 'read':
                mode = 'r'
            elif mode == 'write':
                mode = 'w'
            else:
                raise ValueError(f'{mode} is not a valid option. Use "read" or "write"')

            set_active = False
            condition = self.thread_manager.get_condition(requested_mode=mode)
            with condition:  # All will enter this, and only one will leave at a time (e.g. many sequential 'r' modes)
                thread_id = threading.get_ident()
                # If not already active, join queue
                # If already active reader and trying to read, carry on,
                # If already active reader, but want to switch to writing then stash read status (remove from active readers) and join write queue, then at end of write, restore as active reader
                # If already active writer and trying to write or read, carry on (and stay as active writer)
                if thread_id in self.thread_manager.active_read_threads:
                    if mode == 'r':
                        pass
                    elif mode == 'w':
                        self.thread_manager.stash_active_read()  # Remove from active readers, but remember that this should be restored as reader after writing is finished
                        self.thread_manager.add_to_queue(requested_mode='w')
                        condition.wait()
                        self.thread_manager.set_active(mode='w')
                        set_active = True
                    else:
                        raise NotImplementedError
                elif self.thread_manager.active_write_thread and thread_id == self.thread_manager.active_write_thread[0]:
                    pass  # Already active writer, so can write or read freely (and should stay as active writer)
                else:  # A new thread, so join queue
                    self.thread_manager.add_to_queue(
                        requested_mode=mode)  # Add this thread to the queue, will either join 'r' or 'w' queue
                    condition.wait()
                    self.thread_manager.set_active(mode=mode)
                    set_active = True

                initial_open, initial_mode = get_starting_state()
                if initial_open is False:
                    self.hdf = h5py.File(self.hdf_path, 'r')
                    # Easy to just have this be the default start state (even if opened in 'w' immediately after)

            prev_group_name = self.group_name
            try:
                if mode == 'w':
                    if self.hdf.mode not in WRITE:  # Need to switch to write mode
                        try:
                            self.hdf.close()
                            with h5py.File(self.hdf_path, 'r+') as f:
                                self.hdf = f
                                ret = call_func(*args, **kwargs)  # Call the function
                        finally:
                            self.hdf = h5py.File(self.hdf_path, 'r')
                            self.set_group(prev_group_name)  # So when returning, dat.hdf.group isn't closed
                            # Easy to just have this be the default end state (even if about to be closed immediately after)
                    else:  # Already in write mode, so don't close hdf which was opened in 'with' by starting write thread
                        ret = call_func(*args, **kwargs)
                elif mode == 'r':
                    ret = call_func(*args, **kwargs)
                else:
                    raise NotImplementedError
            finally:
                self.set_group(prev_group_name)  # Restore self.hdf.group/group_name to what it was before this call
                set_final_state(condition, was_active_setter=set_active)  # Hold lock while making sure hdf is closed properly
            return ret
        return wrapper


# @dataclass
# class HDFContainer_OLD:
#     """For storing a possibly open HDF along with the filepath required to open it (i.e. so that an open HDF can
#     be passed around, and if it needs to be reopened in a different mode, it can be)"""
#     hdf: h5py.File  # Open/Closed HDF
#     hdf_path: str  # Path to the open/closed HDF (so that it can be reopened in required mode)
#
#     # group: h5py.Group = field(default=False)  # False will look the same as a closed HDF group
#     # group_name: str = field(default=None)
#
#     def __post_init__(self):
#         # if self.group and not self.group_name:  # Just check that if a group is passed, then group name is also passed
#         #     raise ValueError(f'group: {self.group}, group_name: {self.group_name}. group_name must not be None if group'
#         #                      f'is passed in')
#         self._threads = {}  # {<thread_id>: <read/write>}
#         self._groups = {}  # {<thread_id>: <group>}
#         self._group_names = {}  # {<thread_id>: <group_name>}
#         self._setup_lock = threading.Lock()
#         self._close_lock = threading.Lock()
#         self._lock = threading.RLock()
#
#     @property
#     def thread(self):
#         """
#         Use this to get the state of the current thread (i.e. None, read, write)
#         """
#         thread_id = threading.get_ident()
#         with self._lock:
#             thread = self._threads.get(thread_id, _NOT_SET)  #
#         return thread
#
#     @thread.setter
#     def thread(self, value):
#         thread_id = threading.get_ident()
#         with self._lock:
#             self._threads[thread_id] = value
#
#     @thread.deleter
#     def thread(self):
#         thread_id = threading.get_ident()
#         with self._lock:
#             self._threads.pop(thread_id)
#
#     @property
#     def group(self) -> h5py.Group:
#         """Use this to get the threadsafe group object
#         Examples:
#             class SomeDatAttr(DatAttribute):
#                 group_name = 'TestName'
#
#                 @with_hdf_read
#                 def do_something(self, a, b):
#                     group = self.hdf.group  # Group will point to /TestName in HDF (Threadsafe)
#         """
#         thread_id = threading.get_ident()
#         with self._lock:
#             group = self._groups.get(thread_id, False)  # Default to False to look like a Closed Group.
#         return group
#
#     @group.setter
#     def group(self, value):
#         assert isinstance(value, (type(None), h5py.Group))
#         thread_id = threading.get_ident()
#         with self._lock:
#             self._groups[thread_id] = value
#
#     @group.deleter
#     def group(self):
#         thread_id = threading.get_ident()
#         with self._lock:
#             del self._groups[thread_id]
#
#     @property
#     def group_name(self) -> str:
#         """Use this to get the threadsafe group_name (mostly to be used in with_hdf_read/write)
#         Examples:
#             class SomeDatAttr(DatAttribute):
#                 group_name = 'TestName'
#
#                 @with_hdf_read
#                 def do_something(self, a, b):
#                     group_name = self.hdf.group_name  # gets '/TestName'
#         """
#         thread_id = threading.get_ident()
#         with self._lock:
#             group_name = self._group_names.get(thread_id, None)
#         return group_name
#
#     @group_name.setter
#     def group_name(self, value):
#         assert isinstance(value, (type(None), str))
#         thread_id = threading.get_ident()
#         with self._lock:
#             self._group_names[thread_id] = value
#
#     @group_name.deleter
#     def group_name(self):
#         thread_id = threading.get_ident()
#         with self._lock:
#             del self._group_names[thread_id]
#
#     @classmethod
#     def from_path(cls, path, mode='r'):
#         """Initialize just from path, only change read_mode for creating etc
#         Note: The HDF is closed on purpose before leaving this function!"""
#         logger.debug(f'initializing from path')
#         hdf = h5py.File(path, mode)
#         hdf.close()
#         inst = cls(hdf=hdf, hdf_path=path)
#         return inst
#
#     @classmethod
#     def from_hdf(cls, hdf: h5py.File):
#         """Initialize from OPEN hdf (has to be open to read filename).
#         Note: This will close the file as the aim of this is to keep them closed as much as possible"""
#         logger.debug(f'initializing from hdf')
#         inst = cls(hdf=hdf, hdf_path=hdf.filename)
#         hdf.close()
#         return inst
#
#     def _other_threads(self) -> dict:
#         """Returns a dict of any other threads currently running. e.g. {<other_thread_id>: 'read', ...}"""
#         logger.debug(f'Going into RLocking _other_threads')
#         with self._lock:
#             logger.debug(f'RLocking _other_threads')
#             thread_id = threading.get_ident()
#             other_threads = {k: v for k, v in self._threads.items() if k != thread_id}
#             logger.debug(f'Releasing Rlocking')
#         return other_threads
#
#     def setup_hdf_state(self, mode) -> Tuple[bool, bool, Optional[str]]:
#         def condition(mode_, other_threads):
#             if mode_ == 'read':
#                 if 'write' not in other_threads.values():  # I.e. write mode has finished
#                     return True
#             elif mode_ == 'write':
#                 if all([v == 'waiting' for v in
#                         other_threads.values()]) or other_threads == {}:  # I.e. all other threads are waiting or have exited
#                     return True
#             # logger.debug(f'mode: {mode}, other_threads = {other_threads}')
#             return False
#
#         opened, set_write, prev_group_name = False, False, None
#         with self._lock:
#             logger.debug(f'Going to check if in write before main setup')
#             if self.thread == 'write':  # There should only ever be one of these
#                 logger.debug(f'I am in write mode')
#                 # assert 'write' not in self._other_threads().values()
#                 # assert self.hdf.mode in WRITE
#                 return opened, set_write, self.group_name
#             else:
#                 pass  # Go onto the more complicated checking process
#
#         entering_status = self.thread
#         if entering_status is _NOT_SET:
#             logger.debug(f'setting status to None')
#             entering_status = None
#         self.thread = 'waiting'
#         with self._setup_lock:  # Only one thread should be setting up a state
#             with self._close_lock:  # Don't let any threads close while checking setup of state
#                 logger.debug(f'Starting double locked zone')
#                 self.thread = entering_status
#                 prev_group_name = self.group_name
#                 f = self.hdf
#                 if not f:
#                     opened = True
#                     if mode == 'read':
#                         self.hdf = h5py.File(self.hdf_path, READ[0])
#                         self.thread = 'read'
#                     elif mode == 'write':
#                         self.hdf = h5py.File(self.hdf_path, WRITE[0])
#                         set_write = True
#                         self.thread = 'write'
#                     else:
#                         raise ValueError(f'{mode} not in ["read", "write"]')
#                     return opened, set_write, prev_group_name
#                 else:  # File is open already
#                     if mode == 'read' and (
#                             self.hdf.mode in READ or self.thread == 'write'):  # Carry on reading if current thread is in write or start reading if hdf in read mode (do not start if another thread is in write)
#                         if self.thread is None:
#                             opened = True  # Even though this isn't opening, this thread/wrapper should check to close at end
#                             self.thread = 'read'
#                         elif self.thread == 'read':
#                             pass  # HDF is in READ and thread is in read so that's OK
#                         elif self.thread == 'write' and self.hdf.mode in READ:
#                             raise RuntimeError(f'thread thinks HDF should be in write already, but it is in READ')
#                         return opened, set_write, prev_group_name
#                     elif mode == 'write' and condition('write', self._other_threads()):
#                         if self.thread == 'write' or self.thread is None:
#                             logger.debug(f'My mode before this was {self.thread}')
#                             self.thread = 'write'
#                             assert self.hdf.mode in WRITE  # This should not fail if everything else is being tidied up properly
#                         elif self.thread == 'read':
#                             assert self.hdf.mode in READ  # This should not fail if everything else is being tided up properly
#                             self.hdf.close()
#                             self.hdf = h5py.File(self.hdf_path, WRITE[0])
#                             self.thread = 'write'
#                             set_write = True
#                         else:
#                             raise RuntimeError(
#                                 f'No other threads, but files is open and current thread is in {self.thread} mode which is not in "write" mode, something must be wrong')
#                         return opened, set_write, prev_group_name
#                     else:
#                         # File is open and either: trying to 'read' but file is in 'write' mode on another thread OR trying to get 'write' mode, but other threads exist
#                         # Either way, I need to free up self._close_lock and wait for others to close
#                         pass
#
#             # with self._lock:
#             #     self._thread_queue[threading.get_ident()] = 'waiting'
#             # This is where we wait for other threads to finish (and have released self._close_lock)
#
#             i = 0
#             while True:
#                 i += 1
#                 # logger.debug(f'waiting with mode: {mode}')
#                 if condition(mode, self._other_threads()):
#                     logger.debug(f'breaking free in mode {mode}')
#                     break
#                 else:
#                     time.sleep(0.01)
#                     if not (i + 1) % 300:
#                         logger.debug(f'Thread: {threading.get_ident()} with mode {mode} is removing other threads!!!!!'
#                                      f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#                         with self._lock:
#                             logger.debug(f'self._threads = {self._threads}')
#                             self._threads = {k: v for k, v in self._threads.items() if v == 'waiting'}
#                             # for k in list(self._other_threads()):
#                             #     self._threads.pop(k)
#                         break
#
#             # TODO: Could a finishing_hdf_state running here have bad consequences?
#             logger.debug(f'about to enter close_lock in setup mode: {mode}')
#             with self._close_lock:  # Stop things from changing again while I finish setting up and also wait for a previous write thread to switch back to read?
#                 # Now check current state, and set up as necessary
#                 logger.debug(f'in the final part of setup in mode: {mode}')
#                 f = self.hdf
#                 if not f:
#                     if mode == 'write':
#                         self.hdf = h5py.File(self.hdf_path, WRITE[0])
#                         self.thread = 'write'
#                         opened, set_write = True, True
#                     elif mode == 'read':
#                         self.hdf = h5py.File(self.hdf_path, READ[0])
#                         self.thread = 'read'
#                         opened = True
#                     else:
#                         logger.debug(f'Failing out of end of setup')
#                         raise ValueError(f'{mode} not supported')
#                     logger.debug(f'Returning out of end of setup')
#                     return opened, set_write, prev_group_name
#                 else:
#                     if f.mode in READ:
#                         if mode == 'write':
#                             logger.debug(f'changing to write mode')
#                             self.hdf.close()
#                             self.hdf = h5py.File(self.hdf_path, WRITE[0])
#                             self.thread = 'write'
#                             set_write = True
#                             logger.debug(f'in write mode now for for {self.thread}')
#                         elif mode == 'read' and (self.thread is None or self.thread == 'read'):
#                             self.thread = 'read'
#                         elif mode == 'read' and self.thread == 'write':
#                             logger.debug(f'Failing out of end of setup')
#                             raise RuntimeError(f'File is in read mode, but thread thinks it should be in write already')
#                         else:
#                             logger.debug(f'Failing out of end of setup')
#                             raise ValueError(f'{mode} not supported')
#                         logger.debug(f'Returning out of end of setup')
#                         return opened, set_write, prev_group_name
#                     elif f.mode in WRITE:
#                         filename = f.filename
#                         f.close()
#                         logger.debug(f'Failing out of end of setup')
#                         raise RuntimeError(
#                             f'A writing thread had to wait for other processes to finish or wait, but then '
#                             f'found hdf ({filename}) in WRITE mode which shouldn\'t be possible because '
#                             f'write threads should never have to wait once they are running (everything else'
#                             f'should be waiting for the write thread to finish)')
#         logger.debug(f'Failing out of end of setup')
#         raise RuntimeError(f'Should not get to here')
#
#     def finish_hdf_state(self, opened, set_write, prev_group_name, error_close=False):
#         with self._close_lock:  # Only one should close at a time, and this is also locked in beginning of setup
#             logger.debug(f'Finishing xxxxxxxxxxxxxxxxxxxxx')
#             if error_close:
#                 if self._other_threads() == {} and self.hdf:
#                     self.hdf.close()
#                 if threading.get_ident() in self._threads:
#                     del self.thread
#                     logger.debug(f'getting out of error')
#                 return
#
#             if not opened and not set_write:
#                 self.set_group(prev_group_name)
#                 logger.debug(f'not changing hdf')
#             elif opened:
#                 if self._other_threads() == {}:
#                     self.hdf.close()
#                     logger.debug(f'closed hdf')
#                 logger.debug(f'deleting {self.thread}')
#                 del self.thread  # Remove this thread from self._threads
#             elif set_write:
#                 # assert self._other_threads() == {}  # Should only ever be one thread writing
#                 logger.debug(f'returning to read mode')
#                 self.hdf.close()
#                 self.hdf = h5py.File(self.hdf_path, READ[0])
#                 self.set_group(prev_group_name)
#                 self.thread = 'read'
#                 logger.debug(f'returned to read mode for {self.thread}')
#             else:
#                 raise RuntimeError(f'Should not reach this I think...')
#
#     def set_group(self, group_name: str):
#         if group_name:
#             self.group = self.hdf.get(group_name)
#             self.group_name = group_name
#
#     def get(self, *args, **kwargs):
#         """Makes HDFContainer act like HDF for most most get calls. Will warn if HDF was not already open as this should
#         be handled before making calls.
#         """
#         if self.hdf:
#             return self.hdf.get(*args, **kwargs)
#         else:
#             logger.warning(f'Trying to get value from closed HDF, this should handled with wrappers')
#             with h5py.File(self.hdf_path, 'r') as f:
#                 return f.get(*args, **kwargs)
#
#     def __getattr__(self, item):
#         """Default to trying to apply action to the h5py.File"""
#         return getattr(self.hdf, item)


def _with_dat_hdf(func, mode_='read'):
    """Assuming being called within a Dat object (i.e. self.hdf and self.hdf_path exist)
    Ensures that the HDF is open in write mode before calling function, and then closes at the end"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        container = _get_obj_hdf_container(self)
        wrapped_func = container.function_wrapper(func, mode_=mode_, group_name=getattr(self, 'group_name', None))  # TODO: Just need to make sure container.set_group is called!
        ret = wrapped_func(*args, **kwargs)
        # ret = func(*args, **kwargs)
        return ret

        # opened, set_write, prev_group_name = container.setup_hdf_state(mode)  # Threadsafe setup into read/write mode
        # try:
        #     container.set_group(getattr(self, 'group_name', None))
        #     ret = func(self, *args, **kwargs)
        # except:  # TODO: Is it necessary to do this? What happens if fail and don't close HDF, does it mess with other processes?
        #     # WARNING!!!: Any try catch statements which go through a @with_hdf_... will CLOSE the HDF if they fail
        #     # BEFORE reaching the catch part!!!
        #     container.finish_hdf_state(False, False, '', error_close=True)  # Just make sure this thread closes if
        #     # necessary
        #     raise
        #
        # container.finish_hdf_state(opened, set_write, prev_group_name)
        # return ret

    return wrapper


def with_hdf_read(func):
    return _with_dat_hdf(func, mode_='read')


def with_hdf_write(func):
    return _with_dat_hdf(func, mode_='write')


def _get_obj_hdf_container(obj) -> HDFContainer:
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


def find_all_groups_names_with_attr(parent_group: h5py.Group, attr_name: str, attr_value: Optional[Any] = None,
                                    find_nested=False,
                                    find_recursive=True) -> List[str]:
    """
    Returns list of group_names for all groups which contain the specified attr_name with Optional attr_value

    Args:
        parent_group (h5py.Group): Group to recursively look inside of
        attr_name (): Name of attribute to be checking in each group
        attr_value (): Optional value of attribute to compare to
        find_nested (): Whether to carry on looking in sub groups of groups which DO already meet criteria (MUCH SLOWER TO DO SO)
        find_recursive (): Whether to look inside sub groups which DO NOT meet criteria

    Returns:
        (List[str]): List of group_names which contain specified attr_name [equal to att_value]
    """
    if find_nested is False:
        return _find_all_group_paths_fast(parent_group, attr_name, attr_value, find_recursive)
    else:
        return _find_all_group_paths_visit_all(parent_group, attr_name, attr_value)


def _find_all_group_paths_visit_all(parent_group: h5py.Group, attr_name: str, attr_value: Optional[Any]) -> List[str]:
    """
    Thorough but slow way to recursively search through all children of a group. (use 'find_all_groups_names_with_attr')
    with find_nested = True
    Args:
        parent_group ():
        attr_name ():
        attr_value ():

    Returns:

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


def _find_all_group_paths_fast(parent_group: h5py.Group, attr_name: str, attr_value: Optional[Any],
                               find_recursive=True) -> List[str]:
    """
    Fast way to search through children of group. Stops searching any route once criteria is met (i.e. will not go into
    subgroups of a group which already meets criteria). (use 'find_all_groups_names_with_attr' with find_nested = False)

    Args:
        parent_group ():
        attr_name ():
        attr_value ():
        find_recursive (): Whether to look inside of subgroups which DO NOT currently meet criteria (slower)

    Returns:

    """
    paths = []
    _DEFAULTED = object()
    for k in parent_group.keys():
        if is_Group(parent_group, k):
            g = parent_group.get(k)
            val = g.attrs.get(attr_name, _DEFAULTED)
            if (attr_value is None and val != _DEFAULTED) or val == attr_value:
                paths.append(g.name)
            elif find_recursive:
                paths.extend(_find_all_group_paths_fast(g, attr_name,
                                                            attr_value))  # Recursively search deeper until finding attr_name then go no further
    return paths


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


class NotFoundInHdfError(Exception):
    """Raise when something not found in HDF file"""
    pass


T = TypeVar('T', bound='DatDataclassTemplate')  # Required in order to make subclasses return their own subclass