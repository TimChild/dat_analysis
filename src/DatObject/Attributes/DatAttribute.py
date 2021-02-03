from __future__ import annotations
import pandas as pd
import os
from hashlib import md5
import inspect
from typing import Union, List, Optional, TypeVar, Type, Callable, Any, Dict, Tuple
from src.HDF_Util import is_DataDescriptor, NotFoundInHdfError, is_Group
from src import CoreUtil as CU
import abc
import h5py
import logging
import numpy as np
import lmfit as lm
from dataclasses import dataclass, field, InitVar
from src.HDF_Util import params_from_HDF, params_to_HDF, with_hdf_read, with_hdf_write
import src.HDF_Util as HDU
from functools import lru_cache, partial
from src.CoreUtil import MyLRU, my_partial
from deprecation import deprecated
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.DatObject.DatHDF import DatHDF
    from src.DatObject.Attributes.Data import Data

logger = logging.getLogger(__name__)

FIT_NUM_BINS = 1000  # TODO: This should be somewhere else (use in FitInfo)


def update_meta(self, other):
    """https://code.activestate.com/recipes/408713-late-binding-properties-allowing-subclasses-to-ove/"""
    self.__name__ = other.__name__
    self.__doc__ = other.__doc__
    self.__dict__.update(other.__dict__)
    return self


class LateBindingProperty (property):
    """https://code.activestate.com/recipes/408713-late-binding-properties-allowing-subclasses-to-ove/

    """

    def __new__(cls, fget=None, fset=None, fdel=None, doc=None):

        if fget is not None:
            def __get__(obj, objtype=None, name=fget.__name__):
                fget = getattr(obj, name)
                return fget()

            fget = update_meta(__get__, fget)

        if fset is not None:
            def __set__(obj, value, name=fset.__name__):
                fset = getattr(obj, name)
                return fset(value)

            fset = update_meta(__set__, fset)

        if fdel is not None:
            def __delete__(obj, name=fdel.__name__):
                fdel = getattr(obj, name)
                return fdel()

            fdel = update_meta(__delete__, fdel)

        return property(fget, fset, fdel, doc)


class DatAttribute(abc.ABC):
    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Should returns something like '1.0.0 (major.feature_breaking.feature[.bug])'

        Note does not need to be a whole property to override.. can just be "version = '1.0.0'"
        at the top of the class"""
        # FIXME: This needs to be a bit more clever... Should read version from HDF if exists, otherwise should set in HDF based on class version
        return ''

    @property
    @abc.abstractmethod
    def group_name(self) -> str:
        """Should return name of group in HDF (i.e. 'Transition')

        Note: does not need to be a whole property override, can just be a class variable"""
        return ''

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Should be a short description of what this DatAttribute does (for human reading only)

        Note: Does not need to be a whole property, can just be a class variable
        """
        return ''

    def __init__(self, dat: DatHDF):
        self.dat: DatHDF = dat  # Save pointer to parent DatHDF object
        self.hdf: HDU.HDFContainer = dat.hdf
        self.check_init()  # Ensures in minimally initialized state

    @property
    def initialized(self):
        return self._get_initialized_state()

    @initialized.setter
    def initialized(self, value):
        assert isinstance(value, bool)
        self.set_group_attr('initialized', value)

    def set_group_attr(self, name: str, value, group_name: str = None,
                       DataClass: Optional[Type[DatDataclassTemplate]] = None):
        """
        Can be used to store attributes in HDF group, optionally in a named group
        Args:
            name (str): Name of attribute to store in HDF
            value (any): Any HDU.allowed() value to store in HDF
            group_name (str): Optional full path to the group in which the value should be stored
            DataClass (Optional[Type[DatDataclassTemplate]]): If storing a dataclass instance, pass in the dataclass Class
        """
        if HDU.allowed(value):
            group_name = group_name if group_name else self.group_name
            self._set_attr(group_name, name, value, DataClass=DataClass)
        else:
            raise TypeError(f'{value} not allowed in HDF')

    @with_hdf_read
    def get_group_attr(self, name: str, default=None, check_exists: bool = False, group_name: Optional[str] = None,
                       DataClass: Optional[Type[DatDataclassTemplate]] = None):
        """
        Used to get a value from the HDF group, optionally from named group
        Args:
            name (str): Name of attribute in HDF
            default (any): Value to default to if not found
            check_exists (bool): Whether to raise an error if not found
            group_name (Optional[str]): Optional full path to the group in which the value is stored (the parent group
                to the value, even if the value is stored as a group itself)
            DataClass (): Optional DatDataclass which can be used to load the information back into dataclass form

        Returns:
            any
        """
        group_name = group_name if group_name else self.group_name
        group = self.hdf.get(group_name)
        return HDU.get_attr(group, name, default=default, check_exists=check_exists, dataclass=DataClass)

    @with_hdf_write
    def _set_attr(self, group_name, name, value, DataClass: Optional[Type[DatDataclassTemplate]] = None):
        group = self.hdf.get(group_name)
        HDU.set_attr(group, name, value, dataclass=DataClass)

    @with_hdf_read
    def check_init(self):
        group = self.hdf.get(self.group_name, None)
        if group is None:
            self._create_group(self.group_name)
            group = self.hdf.get(self.group_name)
        if group.attrs.get('initialized', False) is False:
            self._initialize()

    @with_hdf_write
    def _initialize(self):
        self.initialize_minimum()
        self._write_default_group_attrs()
        assert self.initialized == True
        self.set_group_attr('date_initialized', str(CU.time_now()))

    @with_hdf_read
    def _get_initialized_state(self):
        group = self.hdf.get(self.group_name)
        return HDU.get_attr(group, 'initialized', False)

    @abc.abstractmethod
    @with_hdf_write
    def initialize_minimum(self):
        """Override this to do whatever needs to be done to MINIMALLY initialize the HDF
        i.e. this should be as fast as possible, leaving any intensive stuff to be done lazily or in a separate
        call
        Should not be called directly as it will be called as part of DatAttribute init
        (which also does other group attrs)

        Note: Don't forget to set self.initialized = True
        """
        # Do stuff
        self.initialized = True  # Then set this True
        pass

    @with_hdf_write
    def initialize_maximum(self):
        """Use this to run all initialization (i.e. all intensive time consuming stuff). Ideally this will be run
        in the background at some point"""
        logger.warning(f'No "initialize_max" implemented for {self.__class__}')
        pass

    @with_hdf_write
    def _create_group(self, group_name):
        if group_name not in self.hdf.keys():
            self.hdf.create_group(group_name)

    @with_hdf_read
    def _check_default_group_attrs(self):
        """Set default attributes of group if not already existing
        e.g. upon creation of new dat, add description of group in attrs"""
        group = self.hdf.get(self.group_name)
        if {'version', 'description'} - set(group.attrs.keys()):
            self._write_default_group_attrs()

    @with_hdf_write
    def _write_default_group_attrs(self):
        """Writes the default group attrs"""
        group = self.hdf.get(self.group_name)
        version = self.version
        description = self.description
        group.attrs['version'] = version
        group.attrs['description'] = description

    def property_prop(self, attr_name: str, group_name: Optional[str] = None,
                      dataclass: Type[DatDataclassTemplate] = None) -> Any:
        """
        Use this to help make shorthand properties for getting attrs from HDF group
        Args:
            attr_name (): Name of attribute to look for
            group_name (): Optional full group path to look for attribute in
            dataclass (): If loading something that was saved as a dataclass, must load with the same dataclass

        Returns:
            Whatever was stored in the HDF under the attr_name

        Examples:
            sweeplogs: dict = property(my_partial(DatAttribute.property_prop, 'sweeplogs', arg_start=1),)
        """
        private_key = self._get_private_key(attr_name)
        if not getattr(self, private_key, None):
            setattr(self, private_key,
                    self.get_group_attr(attr_name, check_exists=True, group_name=group_name, DataClass=dataclass))
        return getattr(self, private_key)

    def property_set(self, attr_name: str, value: Any, group_name: Optional[str] = None):
        """Use this to help make shorthand properties for setting attrs in HDF group
        Note: Don't use this if you want the attribute to be read only
        """
        private_key = self._get_private_key(attr_name)
        if HDU.allowed(value):
            self.set_group_attr(attr_name, value, group_name=group_name)
            setattr(self, private_key, value)
        else:
            raise TypeError(f'{value} with type {type(value)} not allowed in HDF according to HDU.allowed()')

    def property_del(self, attr_name: str, group_name: Optional[str] = None):
        """Use this to help make shorthand properties for deleting attrs in HDF group
        Note: Don't use this if you dont want the attribute to be deletable"""
        private_key = self._get_private_key(attr_name)
        if getattr(self, private_key, None):
            delattr(self, private_key)
        raise NotImplementedError(f"Haven't implemented removing attributes from HDF yet")  # TODO: implement this

    @staticmethod
    def _get_private_key(attr_name):
        return '_' + attr_name

    # @abc.abstractmethod
    # @with_hdf_read
    # def get_from_HDF(self):
    #     """Should be able to run this to get all data from HDF into expected attrs of DatAttr (remember to use
    #     context manager to open HDF)"""
    #     pass

    # @abc.abstractmethod
    # @with_hdf_write
    # def update_HDF(self):
    #     """Should be able to run this to set all data in HDF s.t. loading would return to current state"""
    #     group = self.hdf.get(self.group_name)
    #     group.attrs['version'] = self.__class__.version

    def clear_caches(self):
        """Should clear out any caches (or equivalent of caches)
        e.g. self.cached_method.clear_cache() if using @functools.LRU_cache, or del self._<manual_cache>"""
        logger.warning(f'Clear cache has not been overwritten for {self.__class__} so has no effect')
        pass


T = TypeVar('T', bound='DatDataclassTemplate')  # Required in order to make subclasses return their own subclass


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

        d = {k: v if not isinstance(v, h5py.Dataset) else v[:] for k, v in d.items()}  # Load all data into memory here if necessary
        inst = cls(**d)
        return inst

    def _save_standard_attrs(self, group: h5py.Group, ignore_keys: Optional[Union[str, List[str]]] = None):
        ignore_keys = CU.ensure_set(ignore_keys)
        for k in set(self.__annotations__) - ignore_keys:
            val = getattr(self, k)
            if isinstance(val, (np.ndarray, h5py.Dataset)) and val.size > 1000:
                HDU.set_data(group, k, val)
            elif HDU.allowed(val):
                HDU.set_attr(group, k, val)
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
                d[k] = HDU.get_attr(group, k, None)
        return d

    @classmethod
    def _default_name(cls):
        return cls.__name__


class Values(object):
    """Object to store Init/Best values in and stores Keys of those values in self.keys"""

    def __init__(self):
        self.keys = []

    def __getattr__(self, item):
        if item.startswith('__') or item.startswith(
                '_') or item == 'keys':  # So don't complain about things like __len__
            return super().__getattribute__(item)  # Come's here looking for Ipython variables
        else:
            if item in self.keys:
                return super().__getattribute__(item)
            else:
                msg = f'{item} does not exist. Valid keys are {self.keys}'
                print(msg)
                logger.warning(msg)
                return None

    def get(self, item, default=None):
        if item in self.keys:
            val = self.__getattr__(item)
        else:
            val = default
        return val

    def __setattr__(self, key, value):
        if key.startswith('__') or key.startswith('_') or key == 'keys' or not isinstance(value, (
                np.number, float, int, type(None))):  # So don't complain about
            # things like __len__ and don't keep key of random things attached to class
            super().__setattr__(key, value)
        else:  # probably is something I want the key of
            self.keys.append(key)
            super().__setattr__(key, value)

    def __repr__(self):
        string = ''
        for key in self.keys:
            v = getattr(self, key)
            if v is not None:
                string += f'{key}={self.__getattr__(key):.5g}\n'
            else:
                string += f'{key}=None\n'
        return string

    def to_df(self):
        df = pd.DataFrame(data=[[self.get(k) for k in self.keys]], columns=[k for k in self.keys])
        return df


@dataclass
class NewFitInfo(DatDataclassTemplate):
    params: lm.Parameters
    func_name: str
    func_code: str
    fit_report: str

    model: lm.Model
    best_values: Values
    init_values: Values


@dataclass
class FitInfo(DatDataclassTemplate):
    params: Union[lm.Parameters, None] = None
    init_params: lm.Parameters = None
    func_name: Union[str, None] = None
    func_code: Union[str, None] = None
    fit_report: Union[str, None] = None
    model: Union[lm.Model, None] = None
    best_values: Union[Values, None] = None
    init_values: Union[Values, None] = None
    hash: int = None

    # Will only exist when set from fit, or after recalculate_fit
    fit_result: Union[lm.model.ModelResult, None] = None

    def init_from_fit(self, fit: lm.model.ModelResult, hash_: Optional[int] = None):
        """Init values from fit result"""
        if fit is None:
            logger.warning(f'Got None for fit to initialize from. Not doing anything.')
            return None
        assert isinstance(fit, lm.model.ModelResult)
        self.params = fit.params
        self.init_params = fit.init_params
        self.func_name = fit.model.func.__name__

        #  Can't get source code when running from deepcopy (and maybe other things will break this)
        try:
            func_code = inspect.getsource(fit.model.func)
        except OSError:
            if self.func_code is not None:
                func_code = '[WARNING: might not be correct as fit was re run and could not get source code' + self.func_code
            else:
                logger.warning('Failed to get source func_code and no existing func_code')
                func_code = 'Failed to get source code due to OSError'
        self.func_code = func_code

        self.fit_report = fit.fit_report()
        self.model = fit.model
        self.best_values = Values()
        self.init_values = Values()
        for key in self.params.keys():
            par = self.params[key]
            self.best_values.__setattr__(par.name, par.value)
            self.init_values.__setattr__(par.name, par.init_value)
        self.hash = hash_
        self.fit_result = fit

    def init_from_hdf(self, group: h5py.Group):
        """Init values from HDF file"""
        self.params = params_from_HDF(group)
        self.init_params = params_from_HDF(group.get('init_params'), initial = True)
        self.func_name = group.attrs.get('func_name', None)
        self.func_code = group.attrs.get('func_code', None)
        self.fit_report = group.attrs.get('fit_report', None)
        self.model = lm.models.Model(self._get_func())
        self.best_values = Values()
        self.init_values = Values()
        for key in self.params.keys():
            par = self.params[key]
            self.best_values.__setattr__(par.name, par.value)
            self.init_values.__setattr__(par.name, par.init_value)

        temp_hash = group.attrs.get('hash')
        if temp_hash is not None:
            self.hash = int(temp_hash)
        else:
            self.hash = None
        self.fit_result = None

    def save_to_hdf(self, parent_group: h5py.Group, name: Optional[str] = None):
        if name is None:
            name = self._default_name()
        parent_group = parent_group.require_group(name)

        if self.params is None:
            logger.warning(f'No params to save for {self.func_name} fit. Not doing anything')
            return None
        params_to_HDF(self.params, parent_group)
        params_to_HDF(self.init_params, parent_group.require_group('init_params'))
        parent_group.attrs['description'] = 'FitInfo'  # Overwrites what params_to_HDF sets
        parent_group.attrs['func_name'] = self.func_name
        parent_group.attrs['func_code'] = self.func_code
        parent_group.attrs['fit_report'] = self.fit_report
        if self.hash is not None:
            parent_group.attrs['hash'] = int(self.hash)

    def _get_func(self):
        """Cheeky way to get the function which was used for fitting (stored as text in HDF so can be executed here)
        Definitely not ideal, so I at least check that I'm not overwriting something, but still should be careful here"""
        return HDU.get_func(self.func_name, self.func_code)

    def eval_fit(self, x: np.ndarray):
        """Return best fit for x array using params"""
        return self.model.eval(self.params, x=x)

    def eval_init(self, x: np.ndarray):
        """Return init fit for x array using params"""
        init_pars = CU.edit_params(self.params, list(self.params.keys()),
                                   [par.init_value for par in self.params.values()])
        return self.model.eval(init_pars, x=x)

    def recalculate_fit(self, x: np.ndarray, data: np.ndarray, auto_bin=False):
        """Fit to data with x array and update self"""
        assert data.ndim == 1
        data, x = CU.remove_nans(data, x)
        if auto_bin is True and len(data) > FIT_NUM_BINS:
            logger.info(f'Binning data of len {len(data)} into {FIT_NUM_BINS} before fitting')
            x, data = CU.bin_data([x, data], round(len(data) / FIT_NUM_BINS))
        fit = self.model.fit(data.astype(np.float32), self.params, x=x, nan_policy='omit')
        self.init_from_fit(fit, self.hash)

    def edit_params(self, param_names=None, values=None, varys=None, mins=None, maxs=None):
        self.params = CU.edit_params(self.params, param_names, values, varys, mins, maxs)

    def __hash__(self):
        if self.hash is None:
            raise AttributeError(f'hash value stored as None so hashing not supported')
        return int(self.hash)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return hash(other) == hash(self)
        return False

    @classmethod
    def from_fit(cls, fit, hash_: Optional[int] = None):
        """Use FitIdentifier to generate hash (Should be done before binning data to be able to check if
        matches before doing expensive processing)"""
        inst = cls()
        inst.init_from_fit(fit, hash_)
        return inst

    @classmethod
    def from_hdf(cls, parent_group: h5py.Group, name: str = None):
        if name is None:
            name = cls._default_name()
        fg = parent_group.get(name)
        if fg is None:
            raise NotFoundInHdfError(f'{name} not found in {parent_group.name}')
        inst = cls()
        inst.init_from_hdf(fg)
        return inst


class DatAttributeWithData(DatAttribute, abc.ABC):
    def get_data(self, key: str):
        """Get data array with given name (will look through names of DataDescriptors specific to FittingAttribute
        first, then DataDescriptors in Data, then data in Data, then data in Experiment Copy

        Examples:
            # in sub group
            def get_data(self, key: str):
                return super().get_data(key)

            # Then setting a property
            x: np.ndarray = property(my_partial(get_data, 'x', arg_start=1))
        """
        data = self._get_data(key)  # To make it easy to override the behaviour of default properties
        return data

    def _get_data(self, key: str):
        """Get data by looking for
        descriptor in current group > descriptor in Data > data in Data > data in Experiment Copy"""
        D: Data = self.dat.Data
        return D.get_data(key, data_group_name=self.group_name)

    def set_data(self, key: str, value: np.ndarray):
        """Set data in Data group with attribute which says it was saved from <group_name>

        Examples:
            # in sub group
            def set_data(self, key: str, value: np.ndarray):
                super().set_data(key, value)

            # Then setting a property
            x: np.ndarray = property(my_partial(get_data, 'x', arg_start=1), my_partial(set_data, 'x', arg_start=1))
        """
        self._set_data(key, value)  # To make it easy to override the behaviour for default properties

    def _set_data(self, key: str, value: np.ndarray, descriptor: Optional[DataDescriptor] = None):
        """Uses Data to handle setting data"""
        D: Data = self.dat.Data
        D.set_data(data=value, name=key, descriptor=descriptor, data_group_name=self.group_name)

    @property
    def specific_data_descriptors_keys(self) -> Dict[str, DataDescriptor]:
        """Data Descriptors specific to the DatAttribute subclassed from this ONLY (dat.Data has ALL descriptors)"""
        D: Data = self.dat.Data
        all_descriptors = D.data_descriptors
        specific_descriptors = {k.split('/')[-1]: v for k, v in all_descriptors.items() if self.group_name in k}
        return specific_descriptors

    def set_data_descriptor(self, descriptor: DataDescriptor, name: Optional[str]):
        """Set a DataDescriptor specific to the DatAttribute subclassed from this"""
        D: Data = self.dat.Data
        D.set_data_descriptor(descriptor, name=name, data_group_name=self.group_name)

    def get_descriptor(self, name: str, filled: bool = False):
        """Gets either an existing DataDescriptor for 'name' or a default DataDescriptor for 'name' prioritizing
        DatAttribute group.
        I.e. useful to use this to get the current DataDescriptor for something and then just modify from there
        """
        D: Data = self.dat.Data
        return D.get_data_descriptor(name, filled=filled, data_group_name=self.group_name)


@dataclass
class FitIdentifier:
    initial_params: lm.Parameters
    func: Callable  # Or should I just use func name here? Or func code?
    data: InitVar[np.ndarray]
    data_hash: str = field(init=False)

    def __post_init__(self, data: np.ndarray):
        assert isinstance(self.initial_params, lm.Parameters)
        self.data_hash = self._hash_data(data)


    @staticmethod
    def _hash_data(data: np.ndarray):
        if data.ndim == 1:
            data = data[~np.isnan(data)]  # Because fits omit NaN data so this will make data match the fit data.
        return md5(data.tobytes()).hexdigest()

    def __hash__(self):
        """The default hash of FitIdentifier which will allow comparison between instances
        Using hashlib hashes makes this deterministic rather than runtime specific, so can compare to saved values
        """
        pars_hash = self._hash_params()
        func_hash = self._hash_func()
        data_hash = self.data_hash
        h = md5(pars_hash.encode())
        h.update(func_hash.encode())
        h.update(data_hash.encode())
        return int(h.hexdigest(), 16)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if hash(self) == hash(other):
                return True
        return False

    def _hash_params(self) -> str:
        h = md5(str(sorted(self.initial_params.valuesdict().items())).encode())
        return h.hexdigest()

    def _hash_func(self) -> str:
        # hash(self.func)   # Works pretty well, but if function is executed later even if it does the same thing it
        # will change
        h = md5(str(self.func.__name__).encode())
        return h.hexdigest()

    def generate_name(self):
        """ Will be some thing reproducible and easy to read. Note: not totally guaranteed to be unique."""
        return str(hash(self))[0:5]


# def get_all_fit_paths(group: h5py.Group) -> List[str]:
#     fit_paths = []
#     for k in group.keys():
#         g = group.get(k)
#         if g.attrs.get('description', None) == 'FitInfo':
#             fit_paths.append(g.name)
#         else:
#             fit_paths.extend(get_all_fit_paths(g))  # Recursively search deeper until finding FitInfo then go no further
#     return fit_paths


@dataclass
class FitPaths:
    all_fits_hash: Dict[int, str] = field(default_factory=dict)  # {hash: path}
    avg_fits: Dict[str, str] = field(default_factory=dict)  # {name: path}
    row_fits: Dict[str, str] = field(default_factory=dict)  # {name: path}
    all_fits: Dict[str, str] = field(default_factory=dict)  # {name: path}

    @classmethod
    def from_groups(cls, avg_fit_group: h5py.Group, row_fit_group: h5py.Group):
        hdf = avg_fit_group.file

        def get_paths_in_group(group: h5py.Group) -> List[str]:
            paths = HDU.find_all_groups_names_with_attr(group,
                                                        attr_name='description',
                                                        attr_value='FitInfo')
            # paths = get_all_fit_paths(group)
            return paths

        def get_hash_dict_from_paths(paths: List[str]) -> Dict[int, str]:
            hash_dict = {}
            for path in paths:
                g = hdf.get(path)
                hash = HDU.get_attr(g, 'hash', None)
                if hash is None:
                    raise RuntimeError(f'No hash found for fit')
                hash_dict[hash] = path
            return hash_dict

        avg_fit_paths = get_paths_in_group(avg_fit_group)
        row_fit_paths = get_paths_in_group(row_fit_group)

        avg_fits = {os.path.split(p)[-1]: p for p in avg_fit_paths}
        row_fits = {os.path.split(p)[-1]: p for p in row_fit_paths}

        all_fits = {**avg_fits, **row_fits}

        all_fits_hash = {
            **get_hash_dict_from_paths(avg_fit_paths),
            **get_hash_dict_from_paths(row_fit_paths)
        }
        return cls(all_fits_hash, avg_fits, row_fits, all_fits)

    def update(self, fit: FitInfo, name: str, which: str, group_name: str):
        self.all_fits_hash.update({fit.hash: group_name + f'/{name}'})
        self.all_fits.update({name: group_name + f'/{name}'})
        if which == 'avg':
            self.avg_fits.update({name: group_name + f'/{name}'})
        elif which == 'row':
            self.row_fits.update({name: group_name + f'/{name}'})
        else:
            raise ValueError(f'{which} not in ["avg", "row"]')


class FittingAttribute(DatAttributeWithData, DatAttribute, abc.ABC):
    AUTO_BIN_SIZE = 1000  # TODO: Think about how to handle this better

    @property
    @abc.abstractmethod
    def DEFAULT_DATA_NAME(self) -> str:
        """Override to return the name to use by default for data (i.e. so self.data property points to correct data)
        Note: Only needs to be a class variable
        Examples:
            class SubClass(FittingAttribute):
                DEFAULT_DATA_NAME = 'i_sense'
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_default_params(self, x: Optional[np.ndarray] = None,
                           data: Optional[np.ndarray] = None) -> Union[List[lm.Parameters], lm.Parameters]:
        """Should return lm.Parameters with default values, or estimates based on data passed in"""
        pass

    @abc.abstractmethod
    def get_default_func(self) -> Callable[[Any], float]:
        """Should return function to use for fitting model"""
        pass

    @abc.abstractmethod
    def default_data_names(self) -> List[str]:
        """
        Override this to return a list of data names (will look through descriptors then data names directly)
         which will be used in fitting. More specific data initialization (i.e. something which requires processing and
         saving as a new dataset) should be implemented in a property so that minimum initialization is not slowed down
         but data can still be retrieved when asked for.
        """
        return ['x', 'i_sense']  # Note: if something like 'y' is specified, then 1D data will fail.

    @abc.abstractmethod
    def clear_caches(self):
        """Override this so that any cached data can be cleared. Remember to call super().clear_caches()

        Examples:
            def clear_caches(self):
                super().clear_caches()
                self.some_data_cache = []
        """
        self.fit_paths = self._get_FitPaths()  # Note: This is only sort of like a cache in that it may need updating,
        # but not expensive enough to make a property
        self._avg_x = None
        self._avg_data = None
        self._avg_data_std = None
        self._avg_fit = None
        self._row_fits = None

    @abc.abstractmethod
    def get_centers(self) -> List[float]:
        """
        Override to return a list of centers to use for averaging data
        # TODO: Need to think about how to handle data with missing rows
        Returns:
            (List[float]): list of center positions relative to x to use for averaging data
        """
        raise NotImplementedError

    @abc.abstractmethod
    def initialize_additional_FittingAttribute_minimum(self):
        """
        Override this to do whatever needs to be done to MINIMALLY initialize any additional info required for specific
        FittingAttribute.
        i.e. this should be as fast as possible, leaving any intensive stuff to be done lazily or in a separate call
        Should not be called directly as it will be called as part of DatAttribute init (which also does other group attrs)
        """
        pass

    def __init__(self, dat):
        super().__init__(dat)
        # TODO: Check that getting all FitPaths is not slow!
        self.fit_paths: FitPaths = self._get_FitPaths()  # Container for different ways to look at fit paths
        self._avg_x = None
        self._avg_data = None
        self._avg_data_std = None
        self._avg_fit = None
        self._row_fits = None

    def _get_default_x(self) -> np.ndarray:
        return self.get_data('x')

    def _set_default_x(self, value: np.ndarray):
        self.set_data('x', value)

    def _get_default_data(self) -> np.ndarray:
        return self.get_data(self.DEFAULT_DATA_NAME)

    def _set_default_data(self, value: np.ndarray):
        self.set_data(self.DEFAULT_DATA_NAME, value)

    x: np.ndarray = LateBindingProperty(_get_default_x, _set_default_x)
    data: np.ndarray = LateBindingProperty(_get_default_data, _set_default_data)

    @property
    def fit_names(self):
        avg_fits = self.fit_paths.avg_fits
        avg_fit_names = [k[:-4] for k in avg_fits.keys()]
        return avg_fit_names

    @property
    def avg_data(self):
        """Quick access for DEFAULT avg_data ONLY"""
        if self._avg_data is None:
             self._avg_data, self._avg_data_std, self._avg_x = self.get_avg_data(x=self.x, data=self.data,
                                                                                centers=None, return_x=True,
                                                                                return_std=True)
        return self._avg_data

    @property
    def avg_x(self):
        """Quick access for DEFAULT avg_x ONLY (although this likely be the same all the time)"""
        if self._avg_x is None:
             self._avg_data, self._avg_data_std, self._avg_x = self.get_avg_data(x=self.x, data=self.data,
                                                                                centers=None, return_x=True,
                                                                                return_std=True)
        return self._avg_x

    @property
    def avg_data_std(self):
        """Quick access for DEFAULT avg_data_std ONLY"""
        if self._avg_data_std is None:
             self._avg_data, self._avg_data_std, self._avg_x = self.get_avg_data(x=self.x, data=self.data,
                                                                            centers=None, return_x=True,
                                                                            return_std=True)
        return self._avg_data_std

    @property
    def avg_fit(self) -> FitInfo:
        """Quick access to DEFAULT avg_fit ONLY"""
        if not self._avg_fit:
            self._avg_fit = self.get_fit('avg', check_exists=False)
        return self._avg_fit

    @property
    def row_fits(self) -> List[FitInfo]:
        """Quick acess to DEFAULT row_fits ONLY"""
        if not self._row_fits:
            self._row_fits = self._run_all_row_fits()  # Note: will fail for 1D
        return self._row_fits

    @with_hdf_write
    def _run_all_row_fits(self):
        row_fits = [self.get_fit('row', i, check_exists=False) for i in range(self.data.shape[0])]
        return row_fits

    def get_avg_data(self, x: Optional[np.ndarray] = None,
                     data: Optional[np.ndarray] = None,
                     centers: Optional[Union[List[float], np.ndarray]] = None,
                     return_x: bool = False, return_std: bool = False,
                     name: Optional[str] = None) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Looks for previously calculated avg_data, and if not found, calculates it and saves it for next time.
        # TODO: Improve this function... Need to think about how to handle data with missing rows, also how to
        Args:
            x ():
            data ():
            centers ():
            return_x ():

        Returns:
            Data: Avg_data, [avg_data_std], [avg_x]
        """
        # Try to get saved avg_data and avg_x
        if not name:
            avg_data_name = self.DEFAULT_DATA_NAME+'_avg'
            avg_x_name = 'x_avg'
        else:
            avg_data_name = name+'_avg'
            avg_x_name = f'x_avg_for_{name}'

        if all([v in self.specific_data_descriptors_keys.keys() for v in [avg_data_name, avg_x_name, avg_data_name + '_std']]):
            avg_data = self.get_data(avg_data_name)
            avg_x = self.get_data(avg_x_name)
            avg_data_std = self.get_data(avg_data_name+'_std')
        else:
            # Otherwise create avg_data and avg_x
            if x is None:
                x = self.x
            if data is None:
                data = self.data
            if centers is None:
                centers = self.get_centers()
            avg_x, avg_data, avg_data_std = self._make_avg_data(x, data, centers)
            self.set_data(avg_x_name, avg_x)
            self.set_data(avg_data_name, avg_data)
            self.set_data(avg_data_name+'_std', avg_data_std)

        ret = [avg_data]
        if return_std:
            ret.append(avg_data_std)
        if return_x:
            ret.append(avg_x)
        if len(ret) == 1:
            ret = ret[0]
        return ret

    def _make_avg_data(self, x: np.ndarray, data: np.ndarray, centers: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates average data using inputs and returns avg_x, avg_data, avg_data_std
        Args:
            x ():
            data ():
            centers ():

        Returns:

        """
        if centers is None:
            centers = self.get_centers()  # TODO: Need to think about this for data with rows removed
        assert all([v is not None for v in [x, data, centers]])
        avg_data, avg_x, avg_data_std = CU.mean_data(x=x, data=data, centers=centers,
                                                     return_x=True, return_std=True, nan_policy='omit')
        return avg_x, avg_data, avg_data_std

    def get_fit(self, which: str = 'avg',
                row: int = 0,
                name: Optional[str] = None,
                initial_params: Optional[lm.Parameters] = None,
                fit_func: Optional[Callable] = None,
                data: Optional[np.ndarray] = None,
                x: Optional[np.ndarray] = None,
                check_exists=True,
                overwrite=False) -> FitInfo:
        """
        Get's fit either from saved file or by running fit (if check_exists is False otherwise will raise error).
        If name is provided, the named fit will be looked for and params, func, data are not required.
        If name AND params, func, data are provided, then if a fit is found, it will be checked against the further
        information, and if not equal, will run the fit again and save with name (unless check exists is True)
        Args:
            which (): 'avg' or 'row'
            row (): Which row num (ignored for avg)
            initial_params (): initial params of fit (must be INITIAL so that I can check if it exists without running
            the fit again).
            fit_func (): Function to fit with
            data (): Data to fit
            check_exists (): If True, will only check if already exists, if False will run fit if not existing
            overwrite (): Force to rerun fits even if it looks like the same fit already exists somewhere

        Returns:
            (FitInfo): Returns requested fit as an instance of FitInfo
        """
        # TODO: This function should be refactored to make things more clear!
        fit, fit_path = None, None
        if name and overwrite is False:  # Look for named fit
            fit_path = self._get_fit_path_from_name(name, which, row)
            if fit_path:  # If found get fit
                fit = self._get_fit_from_path(fit_path)
                if not any((initial_params, fit_func, data)):  # If nothing to compare to
                    return fit

        # Special name default if nothing else specified
        if not name and not any((initial_params, fit_func, data)):
            name = 'default'

        # Get defaults if necessary
        if fit_func is None:
            fit_func = self.get_default_func()
        if data is None:
            if which == 'row':
                data = self.data[row]
            elif which == 'avg':
                data = self.avg_data
            else:
                raise ValueError(f'{which} not in ["avg", "row"]')
        if x is None:  # TODO: Need to think about if this is always right
            if which == 'row':
                x = self.x
            elif which == 'avg':
                x = self.avg_x
            else:
                raise ValueError(f'{which} not in ["avg", "row"]')

        if not initial_params:
            initial_params = self.get_default_params(x=x, data=data)

        # Make a fit_id from fitting arguments
        fit_id = FitIdentifier(initial_params, fit_func, data)

        if not fit and overwrite is False:  # If no named fit, then try find a matching fit from arguments
            fit_path = self._get_fit_path_from_fit_id(fit_id)
            if fit_path:
                fit = self._get_fit_from_path(fit_path)

            elif check_exists:
                raise NotFoundInHdfError(f'{name} fit not found with fit_id = {fit_id}')

        # If fit found check it still matches hash and return if so
        if fit and overwrite is False:
            if hash(fit) == hash(fit_id):
                if name and fit_path and name not in fit_path:
                    if check_exists:
                        raise NotFoundInHdfError(f'{name} was not found, although a fit with the same parameters WAS found at {fit_path}')
                    logger.warning(f'Asked for {name} but fit already exists at {fit_path}. A duplicate will be saved')
                    self._save_fit(fit, which, name, row=row)
                return fit
            else:
                logger.warning(f'Fit found with same initial arguments, but hash does not match. Recalculating fit now')

        # Otherwise start generating new fit
        if not name:  # Generate anything other than default name
            name = fit_id.generate_name()
        fit = self._calculate_fit(x, data, params=initial_params, func=fit_func, auto_bin=True)
        if fit:
            self._save_fit(fit, which, name, row=row)
        return fit

    @with_hdf_read
    def _get_fit_from_path(self, path: str) -> FitInfo:
        """Returns Fit from full path to fit (i.e. path includes the FitInfo group rather than the parent group with
        a name)"""
        path, name = os.path.split(path)
        return self.get_group_attr(name, check_exists=True, group_name=path, DataClass=FitInfo)

    @with_hdf_read
    def _get_FitPaths(self) -> FitPaths:
        avg_fit_group = self.hdf.get(self._get_fit_parent_group_name('avg'))
        row_fit_group = self.hdf.get(os.path.split(self._get_fit_parent_group_name('row', 0))[0])
        if not avg_fit_group or not row_fit_group:
            raise NotFoundInHdfError
        return FitPaths.from_groups(avg_fit_group=avg_fit_group, row_fit_group=row_fit_group)

    def _get_fit_path_from_fit_id(self, fit_id: FitIdentifier) -> Optional[str]:
        """Looks for existing path to fit using hash ID"""
        hash_id = hash(fit_id)  # Note: Returns deterministic hash
        all_fits = self.fit_paths.all_fits_hash
        if hash_id in all_fits:
            path = all_fits[hash_id]
            return path
        return None

    def _get_fit_path_from_name(self, name: str, which: str, row: Optional[int] = 0) -> Optional[str]:
        """Looks for existing path to fit using fit name"""
        fit_path = self._generate_fit_path(which, row, name)
        if fit_path in self.fit_paths.all_fits.values():
            return fit_path
        return None

    def _generate_fit_path(self, which: str, row: Optional[int] = 0, name: Optional[str] = 'default') -> str:
        """Generates path for fit (does not check whether exists or not)"""
        fit_group = self._get_fit_parent_group_name(which, row)
        name = self._generate_fit_saved_name(name, which, row)
        return '/'.join((fit_group, name))

    @with_hdf_write
    def _save_fit(self, fit: FitInfo, which: str, name: str, row: int = 0):
        """Saves fit in correct group with base 'name' plus identifier for avg/row which should make it unique even
        if not a hash generated name"""
        group_name = self._get_fit_parent_group_name(which, row)
        group = self.hdf.hdf.require_group(group_name)
        name = self._generate_fit_saved_name(name, which, row)
        fit.save_to_hdf(group, name)

        # Update self.fit_paths
        self.fit_paths.update(fit, name, which, group.name)

    def _generate_fit_saved_name(self, name: str, which: str, row: Optional[int] = 0):
        """
        Generates a fit name based on the general name and whether it is avg or a particular row (i.e. so each fit name
        is unique. Potentially useful for finding named fits later)

        Args:
            name (): General name of fit (i.e. 'default' or 'const_fixed' or something like that
            which (): Whether 'avg' or 'row' fit
            row (): if a 'row' fit then which row

        Returns:
            (str): Name to save fit as (or to find fit as)
        """
        if which == 'avg':
            return name+'_avg'
        elif which == 'row':
            return name+f'_row[{row}]'
        else:
            raise ValueError(f'{which} not in ["avg", "row"]')

    def _get_fit_parent_group_name(self, which: str, row: int = 0) -> str:
        """Get path to parent group of avg or row fit"""
        if which == 'avg':
            group_name = '/'+'/'.join((self.group_name, 'Avg Fits'))
        elif which == 'row':
            group_name = '/'+'/'.join((self.group_name, 'Row Fits', str(row)))
        else:
            raise ValueError(f'{which} not in ["avg", "row"]')
        return group_name

    def _calculate_fit(self, x: np.ndarray, data: np.ndarray, params: lm.Parameters, func: Callable[[Any], float],
                       auto_bin=True) -> FitInfo:
        """
        Calculates fit on data (Note: assumes that 'x' is the independent variable in fit_func)
        Args:
            x (np.ndarray): x_array (Note: fit_func should have variable with name 'x')
            data (np.ndarray): Data to fit
            params (lm.Parameters): Initial parameters for fit
            func (Callable): Function to fit to
            auto_bin (bool): if True will bin data into self.AUTO_BIN_SIZE if data has more data points (can massively
            increase computation speed without noticeable change to fit values for ~1000)

        Returns:
            (FitInfo): FitInfo instance (with FitInfo.fit_result filled)
        """
        model = lm.model.Model(func)
        hash_ = hash(FitIdentifier(params, func, data))  # Needs to be done BEFORE binning data.
        if auto_bin and data.shape[-1] > self.AUTO_BIN_SIZE:
            bin_size = int(np.floor(data.shape[-1] / self.AUTO_BIN_SIZE))
            x, data = [CU.bin_data_new(arr, bin_x=bin_size) for arr in [x, data]]
        try:
            fit = FitInfo.from_fit(model.fit(data, params, x=x, nan_policy='omit'), hash_)
        except TypeError as e:
            logger.warning(f'{e} while fitting in {self.group_name} for {self.dat.dat_id}')
            fit = None
        return fit

    @with_hdf_write
    def initialize_minimum(self):
        self._set_default_fit_groups()
        self.set_default_data_descriptors()
        self.initialize_additional_FittingAttribute_minimum()
        self.initialized = True

    @with_hdf_write
    def _set_default_fit_groups(self):
        group = self.hdf.group
        group.require_group('Avg Fits')
        group.require_group('Row Fits')

    def set_default_data_descriptors(self):
        """
        Set the data descriptors required for fitting (e.g. x, and i_sense)
        Returns:

        """
        for name in self.default_data_names():
            descriptor = self.get_descriptor(name)
            # Note: Can override to change things here (e.g. descriptor.multiple = 10.0) but likely this should be done
            # in Experiment Config instead!
            self.set_data_descriptor(descriptor, name)  # Will put this in DatAttribute specific DataDescriptors


@deprecated
def ensure_fit(fit: Union[FitInfo, lm.model.ModelResult]):
    if isinstance(fit, FitInfo):
        pass
    elif isinstance(fit, lm.model.ModelResult):
        fit = FitInfo.from_fit(fit)
    else:
        raise ValueError(f'trying to set avg_fit to something which is not a fit')
    return fit


def row_fits_to_group(group, fits, y_array=None):
    """For saving all row fits in a dat in a group. To get back to original, use rows_group_to_all_FitInfos"""
    if y_array is None:
        y_array = [None] * len(fits)
    else:
        assert len(y_array) == len(fits)
    for i, (fit_info, y_val) in enumerate(zip(fits, y_array)):
        name = f'Row{i}:{y_val:.5g}' if y_val is not None else f'Row{i}'
        row_group = group.require_group(name)
        row_group.attrs['row'] = i  # Used when rebuilding to make sure things are in order
        row_group.attrs['y_val'] = y_val if y_val is not None else np.nan
        fit_info.save_to_hdf(row_group)


@deprecated(details="Use what is in FittingAttributue class")
def rows_group_to_all_FitInfos(group: h5py.Group):
    """For loading row fits saved with row_fits_to_group"""
    row_group_dict = {}
    for key in group.keys():
        row_id = group[key].attrs.get('row', None)
        if row_id is not None:
            if group[key].attrs.get('description',
                                    None) == "FitInfo":  # Old as of 18/9/2020 (But here for backwards compatability)
                row_group_dict[row_id] = group[key]
            elif 'FitInfo' in group[key].keys():  # New way data is stored as of 18/9/2020
                row_group_dict[row_id] = group[key].get('FitInfo')
            else:
                raise NotImplementedError(f'Something has gone wrong... fit seems to exist in HDF, but cant find group')
    fit_infos = [FitInfo() for _ in row_group_dict]  # Makes a new FitInfo() [FI()]*10 just gives 10 pointers to 1 obj
    for key in sorted(
            row_group_dict.keys()):  # TODO: Old way of loading FitInfo, but need to not break backwards compatability if possible. This works but is not ideal
        fit_infos[key].init_from_hdf(row_group_dict[key])
    return fit_infos


@deprecated
def fit_group_to_FitInfo(group: h5py.Group):
    """For loading a single Fit group from HDF (i.e. if saved using FitInfo.save_to_hdf()"""
    assert group.attrs.get('description', None) in ["FitInfo", 'Single Parameters of fit']
    fit_info = FitInfo()
    fit_info.init_from_hdf(group)
    return fit_info


@dataclass
class DataDescriptor(DatDataclassTemplate):
    """
    Place to group together information required to get Data from Experiment_Copy (or somewhere else) in correct form
    i.e. accounting for offset/multiplying/bad rows of data etc
    """
    data_path: Optional[str] = None  # Path to data in HDF (as a str)
    offset: float = 0.0  # How much to offset all the data (i.e. systematic error)
    multiply: float = 1.0  # How much to multiply all the data (e.g. convert to nA or some other standard)
    bad_rows: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    bad_columns: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    # May want to add more optional things here later (e.g. replace clipped values with NaN etc)

    data: np.ndarray = field(default=None, repr=False, compare=False)  # For temp data storage (not for storing in HDF)
    data_link: h5py.SoftLink = field(default=None, repr=False)  # To make data show up in HDF only

    def __post_init__(self):
        if self.data_path and not self.data_link:
            self.data_link = h5py.SoftLink(self.data_path)  # This will show up as a dataset in the HDF when saved
        elif self.data_path and self.data_path != self.data_link.path:
            logger.error(f'data_path = {self.data_path} != data_link = {self.data_link.path}. Something wrong - change'
                         f'data_path or data_link accordingly')

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return np.all([
                self.data_path == other.data_path,
                self.offset == other.offset,
                self.multiply == other.multiply,
                (self.bad_rows == other.bad_rows).all(),
                (self.bad_columns == other.bad_columns).all(),
            ])
        else:
            return False

    def __hash__(self):
        return hash((self.data_path, self.offset, self.multiply, self.bad_rows.tobytes(), self.bad_columns.tobytes()))

    @staticmethod
    def ignore_keys_for_hdf():
        """Don't want to save 'data' to HDF here because it will be duplicating data saved at 'data_path'"""
        return 'data'

    def get_array_from_hdf(self, hdf: h5py.File):
        """
        NOT EASY TO CACHE READ (See self.calculate_from_raw_data)
        Same as self.calculate_array but handles getting data from HDF (Note: reads are uncached!)
        Opens the dataset, applies multiply/offset/bad_rows etc and returns the result
        Args:
            hdf (): HDF file the data exists in

        Returns:
            (np.ndarray): Array of data after necessary modification
        """
        dataset = hdf.get(self.data_path)
        assert isinstance(dataset, h5py.Dataset)
        good_slice = self._good_slice(dataset.shape)
        data = dataset[good_slice]
        data = self._calculate_offset_multiple(data)
        return data

    def calculate_from_raw_data(self, data: np.ndarray):
        """
        USE FOR CACHING
        Same as self.get_array but with the data passed in (which should be the same as from data_path in the hdf!)
        Reason is that the read from HDF can be cached somewhere else (i.e. in Dat.Data class)
        Args:
            data (np.ndarray): Data which should be exactly what is found at self.data_path in HDF

        Returns:
            (np.ndarray): Array of data after necessary modification

        Examples:
            # To cache the read part something like this is good
            @lru_cache
            def get_filled_DataDescriptor(descriptor: DataDescriptor):
                raw_data = hdf.get(descriptor.data_path)
                descriptor.data = descriptor.calculate_from_raw_data(raw_data)
                return descriptor
        """
        good_slice = self._good_slice(data.shape)
        data = data[good_slice]
        data = self._calculate_offset_multiple(data)
        return data

    def _calculate_offset_multiple(self, data: np.ndarray):
        """Does calculation on data if necessary"""
        if self.multiply == 1.0 and self.offset == 0.0:
            return data
        else:
            data = (data + self.offset) * self.multiply
            return data

    def get_orig_array(self, hdf: h5py.File):
        return hdf.get(self.data_path)[:]

    def _good_slice(self, shape: tuple):
        if self.bad_rows.size == 0 and self.bad_columns.size == 0:
            return np.s_[:]
        else:
            raise NotImplementedError(f'Still need to write how to get slice of only good rows! ')  # TODO: Do this

