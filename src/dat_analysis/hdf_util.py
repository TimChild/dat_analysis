from __future__ import annotations

import abc
import dataclasses
import threading
import functools
from collections import namedtuple
from typing import NamedTuple, Union, Optional, Type, TYPE_CHECKING, Any, List, Tuple, Callable, Dict, TypeVar

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
from dataclasses import is_dataclass, dataclass, field, fields
from inspect import getsource
from . import core_util as CU
import time
from .hdf_file_handler import HDFFileHandler

if TYPE_CHECKING:
    pass

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
        # val = [f'i_{v}' if isinstance(v, int) else v for v in val]
    if type(val) == tuple:
        if any(map(lambda x: x is None, val)):
            val = tuple([v if v is not None else 'None' for v in val])
    return val


def desanitize(val):
    if type(val) == list:
        if np.nan in val:
            val = [v if v != 'None' else None for v in val]
        # val = [int(v[2:]) if isinstance(v, str) and v.startswith('i_') else v for v in val]
    if type(val) == tuple:
        if None in val:
            val = tuple([v if v != 'None' else None for v in val])
    return val


@deprecated(deprecated_in='3.0.0')
def init_hdf_id(dat_id, hdfdir_path, overwrite=False):
    """Makes sure HDF folder exists, and creates an empty HDF there (will only overwrite if overwrite=True)"""
    file_path = os.path.join(hdfdir_path, dat_id + '.h5')
    file_path = init_hdf_path(file_path, overwrite=overwrite)
    return file_path


@deprecated(deprecated_in='3.0.0')
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
        filehandler = HDFFileHandler(path, 'w')  # Use this to safely interact with other threads/processes
        hdf = filehandler.new()
        filehandler.previous()
        # f = h5py.File(path, 'w')  # Init a HDF file
        # f.close()
    return path


@deprecated(deprecated_in='3.0.0')
def check_hdf_path(path: str) -> str:
    """Just checks if HDF exists at path and returns same path. Used for loading"""
    if not os.path.exists(path):
        raise FileNotFoundError(f'No HDF found at {path}')
    return path


@deprecated(deprecated_in='3.0.0')
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


T = TypeVar('T', bound='HDFStoreableDataclass')  # Required in order to make subclasses return their own subclass


@dataclass
class HDFStoreableDataclass(abc.ABC):
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
    def __getitem__(self, key):
        """Allows attributes to be accessed as if it is a dict"""
        return self.__dict__[key]

    def keys(self):
        return self.__dict__.keys()

    def asdict(self):
        return dataclasses.asdict(self)

    @classmethod
    def _default_name(cls):
        return cls.__name__

    def save_to_hdf(self, parent_group: h5py.Group, name: Optional[str] = None):
        """
        Default way to save all info from Dataclass to HDF in a way in which it can be loaded back again.

        Note: Override ignore_keys_for_hdf() and additional_save/load_to/from_hdf() to save non-standard attributes

        Only override this method if you really need to save a very complex dataclass

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

    @staticmethod
    def ignore_keys_for_hdf() -> Optional[Union[str, List[str]]]:
        """Override this to ignore specific dataclass keys when saving to HDF or loading from HDF
        Note: To save or load additional things, override additional_save_to_hdf and additional_load_from_hdf
        """
        return None

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

    @classmethod
    def from_hdf(cls: Type[T], parent_group: h5py.Group, name: Optional[str] = None) -> T:
        """
        Should get back all data saved to HDF with "save_to_hdf" and initialize the dataclass and return that instance.
        Remember to override this when overriding "save_to_hdf"

        Args:
            parent_group (h5py.Group): The group in which the saved data should be found (i.e. it will be a subgroup in this
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
        # for k in set(self.__annotations__) - ignore_keys:  # TODO: Use fields() instead of .__annotations__? https://stackoverflow.com/questions/57601705/annotations-doesnt-return-fields-from-a-parent-dataclass
        for k in set([f.name for f in fields(self)]) - ignore_keys:  # TODO: Use fields() instead of .__annotations__? https://stackoverflow.com/questions/57601705/annotations-doesnt-return-fields-from-a-parent-dataclass
            val = getattr(self, k)
            if isinstance(val, (np.ndarray,
                                h5py.Dataset)) and val.size > 1000:  # Pretty much anything that is an array should be saved as a dataset
                set_data(group, k, val)
            elif allowed(val):
                set_attr(group, k, val)
            else:
                expected_field = [f for f in fields(self) if f.name == k]
                logger.warning(
                    f'{self.__class__.__name__}.{k} = {val} which has type {type(val)} (where type {expected_field.type} was expected) which is not able to be saved automatically. Override "save_to_hdf" and "from_hdf" in order to save and load this variable')

    @classmethod
    def _get_standard_attrs_dict(cls, group: h5py.Group, keys=None) -> dict:
        assert isinstance(group, h5py.Group)
        d = dict()
        if keys is None:
            # keys = cls.__annotations__
            keys = [f.name for f in fields(cls)]
        ignore_keys = cls.ignore_keys_for_hdf()
        if ignore_keys is None:
            ignore_keys = []
        for k in keys:
            if k not in ignore_keys:
                d[k] = get_attr(group, k, None)
        return d


def set_attr(group: h5py.Group, name: str, value, dataclass: Optional[Type[HDFStoreableDataclass]] = None):
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
            group.attrs[str(name)] = value
    elif isinstance(value, h5py.SoftLink):
        if name in group:
            del group[name]
        group[name] = value
    elif type(value) == dict:
        # if len(value) < 5:  # Dangerous if dict contains large datasets
        #     d_str = CU.json_dumps(value)
        #     group.attrs[name] = d_str
        # else:
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


def get_attr(group: h5py.Group, name, default=None, check_exists=False, dataclass: Type[HDFStoreableDataclass] = None):
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
                attr = load_dict_from_hdf_group(g)
                return attr
            if description == 'NamedTuple':
                attr = load_group_to_namedtuple(g)
                return attr
            # if description == 'dataclass':
            #     attr = load_group_to_dataclass(g)
            #     return attr
            if description == 'FitInfo':
                from .analysis_tools.general_fitting import FitInfo
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
        else:
            d[k] = get_attr(group, k)
            if isinstance(d[k], h5py.Dataset):  # Don't want to leave Dataset pointers in a dictionary (copy data out)
                d[k] = d[k][:]
    d = _convert_keys_to_int(d)  # int keys aren't supported in HDF so stored as str, but probably want int back.
    return d


def save_namedtuple_to_group(ntuple: NamedTuple, group: h5py.Group):
    """Saves named tuple inside group given"""
    group.attrs['description'] = 'NamedTuple'
    group.attrs['NT_name'] = ntuple.__class__.__name__
    for key, val in ntuple._asdict().items():
        set_attr(group, key, val)  # Store as attrs of group in HDF


@deprecated(deprecated_in='3.0.0', details='use HDFStoreableDataclass instead, and then .save_to_hdf method')
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
    d = {key: get_attr(group, key) for key in list(group.attrs.keys())+list(group.keys())}

    # Remove HDF only descriptors
    for k in ['description', 'NT_name']:
        if k in d.keys():
            del d[k]

    # Make the NamedTuple
    ntuple = namedtuple(name, d.keys())
    filled_tuple = ntuple(**d)  # Put values into tuple
    return filled_tuple


def _isnamedtupleinstance(x):
    """https://stackoverflow.com/questions/2166818/how-to-check-if-an-object-is-an-instance-of-a-namedtuple"""
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple: return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple): return False
    return all(type(n) == str for n in f)


@deprecated(deprecated_in='3.0.0')
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


@deprecated(deprecated_in='3.0.0')
def match_name_in_group(names, data_group: Union[h5py.File, h5py.Group]):
    """
    Returns the first name from names which is a dataset in data_group
    Args:
        names (): list of potential names in data_group
        data_group (): The group (or hdf) to look for datasets in

    Returns:
        First name which is a dataset or None if not found
    """
    names = CU.ensure_list(names)
    for i, name in enumerate(names):
        if name in data_group.keys() and isinstance(data_group[name], h5py.Dataset):
            return name, i
    logger.warning(f'[{names}] not found in [{data_group.name}]')
    return None, None


@deprecated(deprecated_in='3.0.0', details='Only used in old dat object, no longer using.')
@dataclass
class HDFContainer:
    hdf: h5py.File
    hdf_path: str

    def __post_init__(self):
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
        with HDFFileHandler(path, mode) as hdf:
            inst = cls(hdf=hdf, hdf_path=path)
        return inst

    @classmethod
    def from_hdf(cls, hdf: h5py.File):
        """Initialize from OPEN hdf (has to be open to read filename).
        Note: This will close the file as the aim of this is to keep them closed as much as possible"""
        logger.debug(f'initializing from hdf')
        inst = cls(hdf=hdf, hdf_path=hdf.filename)
        return inst

    def get(self, *args, **kwargs):
        """Makes HDFContainer act like HDF for most most get calls.
        """
        return self.hdf.get(*args, **kwargs)
        # if self.hdf:
        #     return self.hdf.get(*args, **kwargs)
        # else:
        #     raise OSError()
        #     logger.warning(f'Trying to get value from closed HDF, this should be handled with wrappers')
        #     with h5py.File(self.hdf_path, 'r') as f:
        #         return f.get(*args, **kwargs)

    def set_group(self, group_name: str):
        if group_name:
            self.group = self.hdf.get(group_name)
            self.group_name = group_name

    def __getattr__(self, item):
        """Default to trying to apply action to the h5py.File"""
        return getattr(self.hdf, item)


@deprecated(deprecated_in='3.0.0', details='Only used in old dat object, no longer using.')
def _with_dat_hdf(func, mode_='read'):
    """Assuming being called within a Dat object (i.e. self.hdf and self.hdf_path exist)
    Ensures that the HDF is open in correct mode before calling function, and then closes at the end"""
    assert mode_ in ['read', 'write']

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        group_name = getattr(self, 'group_name', None)
        container = _get_obj_hdf_container(self)
        previous_group_name = container.group_name

        mode = 'r+' if mode_ == 'write' else 'r'
        filemanager = HDFFileHandler(container.hdf_path, mode)
        new_f = filemanager.new()
        try:
            container.hdf = new_f
            container.set_group(group_name)
            ret = func(*args, **kwargs)
        finally:
            container.hdf = filemanager.previous()
            # if not container.hdf:
            #     logger.error(f'HDF was closed before reaching filemanager.previous() -- Need to fix this')
            #     # I think it is probably a problem to not call .previous() in case there are still functions in the
            #     # stack which are expecting an open HDF file... Probably the answer is to check for a closed file in
            #     # the .previous() and open again if necessary. Either that, or at least pop any more records for that
            #     # file out of the filemanager._open_files or equivalent.
            # if container.hdf:  # Only revert state of HDF if HDF is still open
            #     container.hdf = filemanager.previous()
            if container.hdf:  # If still an open file after .previous() call, then reset the target group
                container.set_group(previous_group_name)
        return ret
    return wrapper


@deprecated(deprecated_in='3.0.0', details='Only used in old dat object, no longer using.')
def with_hdf_read(func):
    return _with_dat_hdf(func, mode_='read')


@deprecated(deprecated_in='3.0.0', details='Only used in old dat object, no longer using.')
def with_hdf_write(func):
    return _with_dat_hdf(func, mode_='write')


@deprecated(deprecated_in='3.0.0', details='Only used in old dat object, no longer using.')
def _get_obj_hdf_container(obj) -> HDFContainer:
    if not hasattr(obj, 'hdf'):
        raise RuntimeError(f'Did not find "self.hdf" for object: {obj}')
    container: HDFContainer = getattr(obj, 'hdf')
    if not isinstance(container, HDFContainer):
        raise TypeError(f'got type {container}. HDF should be stored in an HDFContainer not as a plain HDF. '
                        f'Use HDU.HDFContainer (because need to ensure that a path is present along with HDF)')
    return container


@deprecated(deprecated_in='3.0.0', details='Only used in old dat object, no longer using.')
def _set_container_group(obj: Any,
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


@deprecated(deprecated_in='3.0.0', details='Only used in old dat object, no longer using.')
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


@deprecated(deprecated_in='3.0.0', details='Only used in old dat object, no longer using.')
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


def _file_free(filepath) -> bool:
    """Works on Windows only... Checks if file is open by any process"""
    try:
        os.rename(filepath, filepath)
        return True
    except FileNotFoundError as e:
        return True
    except PermissionError:
        return False


def wait_until_file_free(filepath, timeout=30):
    """Wait until file is free from any external processes (on Windows only)"""
    start = time.time()
    while time.time() - start < timeout:
        if _file_free(filepath):
            logging.debug('file is free')
            return True
        else:
            time.sleep(0.1)
    raise TimeoutError(f'File {filepath} not accessible within timeout of {timeout}s')
