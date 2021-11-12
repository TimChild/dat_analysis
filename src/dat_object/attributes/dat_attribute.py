from __future__ import annotations
import datetime
import os
from typing import Union, List, Optional, Type, Callable, Any, Dict, Tuple, TYPE_CHECKING
import abc
import h5py
import logging
import numpy as np
import lmfit as lm
from dataclasses import dataclass, field

from src.hdf_util import NotFoundInHdfError, DatDataclassTemplate
from src import core_util as CU
from src.hdf_util import with_hdf_read, with_hdf_write
import src.hdf_util as HDU

if TYPE_CHECKING:
    from src.dat_object.dat_hdf import DatHDF
    from src.dat_object.attributes.Data import Data
    from src.analysis_tools.general_fitting import FitInfo, FitIdentifier

logger = logging.getLogger(__name__)

FIT_NUM_BINS = 1000  # TODO: This should be somewhere else (use in FitInfo)


def update_meta(self, other):
    """https://code.activestate.com/recipes/408713-late-binding-properties-allowing-subclasses-to-ove/"""
    self.__name__ = other.__name__
    self.__doc__ = other.__doc__
    self.__dict__.update(other.__dict__)
    return self


class LateBindingProperty(property):
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
        self.set_group_attr('date_initialized', str(datetime.datetime.now()))

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

    def clear_caches(self):
        """Should clear out any caches (or equivalent of caches)
        e.g. self.cached_method.clear_cache() if using @functools.LRU_cache, or del self._<manual_cache>"""
        logger.warning(f'Clear cache has not been overwritten for {self.__class__} so has no effect')
        pass


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
class FitPaths:
    all_fits_hash: Dict[int, str] = field(default_factory=dict)  # {hash: path}
    avg_fits: Dict[str, str] = field(default_factory=dict)  # {name: path}
    row_fits: Dict[str, str] = field(default_factory=dict)  # {name: path}
    all_fits: Dict[str, str] = field(default_factory=dict)  # {name: path}

    @classmethod
    def from_groups(cls, avg_fit_group: h5py.Group, row_fit_group: h5py.Group):
        hdf = avg_fit_group.file

        def get_paths_in_group(group: h5py.Group) -> List[str]:
            if (paths := group.attrs.get('all_paths', None)) is None:
                paths = HDU.find_all_groups_names_with_attr(group,
                                                            attr_name='description',
                                                            attr_value='FitInfo')
            return paths

        avg_fit_paths = get_paths_in_group(avg_fit_group)
        row_fit_paths = get_paths_in_group(row_fit_group)

        return cls.from_paths(hdf, avg_fit_paths, row_fit_paths)

    @classmethod
    def from_paths(cls, hdf: h5py.File, avg_fit_paths: List[str], row_fit_paths: List[str]):
        """Assumes list of paths provided is accurate instead of searching through all of them. Still needs hdf in
        order to get hashes from fits"""
        avg_fits = {os.path.split(p)[-1]: p for p in avg_fit_paths}
        row_fits = {os.path.split(p)[-1]: p for p in row_fit_paths}

        all_fits = {**avg_fits, **row_fits}
        all_fits_hash = {
            **cls._get_hash_dict_from_paths(hdf, avg_fit_paths),
            **cls._get_hash_dict_from_paths(hdf, row_fit_paths)
        }
        return cls(all_fits_hash, avg_fits, row_fits, all_fits)

    @staticmethod
    def _get_hash_dict_from_paths(hdf: h5py.File, paths: List[str]) -> Dict[int, str]:
        hash_dict = {}
        for path in paths:
            g = hdf.get(path)
            hash = HDU.get_attr(g, 'hash', None)
            if hash is None:
                logger.warning(f'Fit at {path} had no hash')
            else:
                hash_dict[hash] = path
        return hash_dict

    def update(self, fit: FitInfo, name: str, which: str, group: h5py.Group):
        """Update the FitPaths instance and save new lists of paths to HDF in either Avg Fits or Row Fits"""
        self.all_fits_hash.update({fit.hash: group.name + f'/{name}'})
        self.all_fits.update({name: group.name + f'/{name}'})
        if which == 'avg':
            self.avg_fits.update({name: group.name + f'/{name}'})
            group.attrs['all_paths'] = list(self.avg_fits.values())
        elif which == 'row':
            self.row_fits.update({name: group.name + f'/{name}'})
            # ps = group.require_dataset('all_paths', shape=(100000,), dtype='S100')  # TODO: Could make this better...
            # # TODO: Using a fixed array of 100K strings of max 100 length each to store fit paths. This is WAY overkill
            # # TODO: for small datasets, but necessary for large ones (think 5000 rows 20 different saved fits)
            # # TODO: S100 means 100 length string, path to fit cannot exceed 100 characters with this limit...
            # # Note: Takes ~15ms to read/write/update etc... pretty slow...
            # paths = [str(v).encode("ascii") for v in self.row_fits.values()]
            # if any([len(v) > 100 for v in paths]):
            #     raise RuntimeError(f'Some fit paths are too long to be stored in HDF (max len is 100)')
            # ps[:] = b''  # Reset if previously existing
            # ps[:len(paths)] = paths  # Update with new values
            group.parent.attrs['all_paths'] = list(self.row_fits.values())  # parent otherwise in Row specific group
        else:
            raise ValueError(f'{which} not in ["avg", "row"]')


class FittingAttribute(DatAttributeWithData, DatAttribute, abc.ABC):
    AUTO_BIN_SIZE = 1000  # TODO: Think about how to handle this better
    FIT_METHOD = 'leastsq'

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
        # TODO: Check that getting all FitPaths is not slow! ... it is slow ...
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
            if self.data.shape[0] > 100 and 'default' not in self.fit_names:
                raise RuntimeError(f'Running default fits for dat{self.dat.dat_id} might take a long time. '
                                   f'Use .get_fit(check_exists=False) if you mean to do this')
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
                     name: Optional[str] = None,
                     check_exists: bool = False,
                     overwrite: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Looks for previously calculated avg_data, and if not found, calculates it and saves it for next time.
        Args:
            x ():
            data ():
            centers ():
            return_x ():
            return_std:
            name: Name to save under
            check_exists: If True, error will be raised if requested data is not already in HDF
            overwrite: whether to overwrite previously saved data

        Returns:
            Data: Avg_data, [avg_data_std], [avg_x]
        """
        # Try to get saved avg_data and avg_x
        if not name:
            avg_data_name = self.DEFAULT_DATA_NAME + '_avg'
            avg_x_name = 'x_avg'
        else:
            avg_data_name = name + '_avg'
            avg_x_name = f'x_avg_for_{name}'

        if not overwrite and all([v in self.specific_data_descriptors_keys.keys() for v in
                [avg_data_name, avg_x_name, avg_data_name + '_std']]):
            avg_data = self.get_data(avg_data_name)
            avg_x = self.get_data(avg_x_name)
            avg_data_std = self.get_data(avg_data_name + '_std')
        elif check_exists is False or overwrite:
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
            self.set_data(avg_data_name + '_std', avg_data_std)
        else:
            raise NotFoundInHdfError(f'{avg_data_name} or {avg_x_name} or {avg_data_name}_std not in HDF for '
                                     f'dat{self.dat.datnum}')

        ret = [avg_data]
        if return_std:
            ret.append(avg_data_std)
        if return_x:
            ret.append(avg_x)
        if len(ret) == 1:
            ret = ret[0]
        return ret

    def _make_avg_data(self, x: np.ndarray, data: np.ndarray, centers: Optional[List[float]] = None) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
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

    def get_row_fits(self,
                     name: Optional[str] = None,
                     initial_params: Optional[lm.Parameters] = None,
                     fit_func: Optional[Callable] = None,
                     data: Optional[np.ndarray] = None,
                     x: Optional[np.ndarray] = None,
                     check_exists=True,
                     overwrite=False) -> List[FitInfo]:
        """Convenience function for calling get_fit for each row"""
        data = data if data is not None else self.data
        assert data.ndim >= 2
        return [self.get_fit(which='row', row=i, name=name,
                             initial_params=initial_params, fit_func=fit_func,
                             data=d, x=x,
                             check_exists=check_exists,
                             overwrite=overwrite) for i, d in enumerate(data)]

    def get_fit(self, which: str = 'avg',
                row: int = 0,
                name: Optional[str] = None,
                initial_params: Optional[lm.Parameters] = None,
                fit_func: Optional[Callable] = None,
                data: Optional[np.ndarray] = None,
                x: Optional[np.ndarray] = None,
                calculate_only=False,
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
            calculate_only (): Do not try to load or save, just calculate and return fit
            check_exists (): If True, will only check if already exists, if False will run fit if not existing
            overwrite (): Force to rerun fits even if it looks like the same fit already exists somewhere

        Returns:
            (FitInfo): Returns requested fit as an instance of FitInfo
        """
        # TODO: This function should be refactored to make things more clear!
        from src.analysis_tools.general_fitting import FitIdentifier
        fit, fit_path = None, None
        if not calculate_only:
            if name and overwrite is False:  # Look for named fit
                fit_path = self._get_fit_path_from_name(name, which, row)
                if fit_path:  # If found get fit
                    fit = self._get_fit_from_path(fit_path)
                    if not any((initial_params, fit_func,
                                data is not None)) or check_exists:  # If nothing to compare to or ONLY looking for existing
                        return fit
                elif check_exists:
                    raise NotFoundInHdfError(f'{name} not found for dat{self.dat.datnum} in {self.group_name}')

            # Should ONLY get past here with check_exists == True if name is None
            if check_exists:
                if name is not None:
                    raise RuntimeError(
                        f'Dat{self.dat.datnum}: should not have got here with name={name} and check_exists={check_exists}')

            # Special name default if nothing else specified
            if not name and not any((initial_params, fit_func, data is not None)):
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

        if not calculate_only:
            # Make a fit_id from fitting arguments
            fit_id = FitIdentifier(initial_params, fit_func, data)

            if not fit and overwrite is False:  # If no named fit, then try find a matching fit from arguments
                fit_path = self._get_fit_path_from_fit_id(fit_id)
                if fit_path:
                    fit = self._get_fit_from_path(fit_path)

                elif check_exists:
                    raise NotFoundInHdfError(f'Dat{self.dat.datnum}: {name} fit not found with fit_id = {fit_id}')

            # If fit found check it still matches hash and return if so
            if fit and overwrite is False:
                if hash(fit) == hash(fit_id):
                    if name and fit_path and name not in fit_path:
                        if check_exists:
                            raise NotFoundInHdfError(
                                f'{name} was not found, although a fit with the same parameters WAS found at {fit_path}')
                        logger.warning(
                            f'Asked for {name} but fit already exists at {fit_path}. A duplicate will be saved')
                        self._save_fit(fit, which, name, row=row)
                    return fit
                else:
                    logger.warning(
                        f'Fit found with same initial arguments, but hash does not match. Recalculating fit now')

            # Otherwise start generating new fit
            if not name:  # Generate anything other than default name
                name = fit_id.generate_name()

        fit = self._calculate_fit(x, data, params=initial_params, func=fit_func, auto_bin=True,
                                  generate_hash=True if not calculate_only else False)
        if fit and not calculate_only:
            self._save_fit(fit, which, name, row=row)
        return fit

    @with_hdf_read
    def _get_fit_from_path(self, path: str) -> FitInfo:
        """Returns Fit from full path to fit (i.e. path includes the FitInfo group rather than the parent group with
        a name)"""
        from src.analysis_tools.general_fitting import FitInfo
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
        self.fit_paths.update(fit, name, which, self.hdf.get(group_name))

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
            return name + '_avg'
        elif which == 'row':
            return name + f'_row[{row}]'
        else:
            raise ValueError(f'{which} not in ["avg", "row"]')

    def _get_fit_parent_group_name(self, which: str, row: int = 0) -> str:
        """Get path to parent group of avg or row fit"""
        if which == 'avg':
            group_name = '/' + '/'.join((self.group_name, 'Avg Fits'))
        elif which == 'row':
            group_name = '/' + '/'.join((self.group_name, 'Row Fits', str(row)))
        else:
            raise ValueError(f'{which} not in ["avg", "row"]')
        return group_name

    def _calculate_fit(self, x: np.ndarray, data: np.ndarray, params: lm.Parameters, func: Callable[[Any], float],
                       auto_bin=True, generate_hash: bool = True) -> FitInfo:
        """
        Calculates fit on data (Note: assumes that 'x' is the independent variable in fit_func)
        Args:
            x (np.ndarray): x_array (Note: fit_func should have variable with name 'x')
            data (np.ndarray): Data to fit
            params (lm.Parameters): Initial parameters for fit
            func (Callable): Function to fit to
            auto_bin (bool): if True will bin data into self.AUTO_BIN_SIZE if data has more data points (can massively
            increase computation speed without noticeable change to fit values for ~1000)
            generate_hash: Generate hash before binning data so that a future call can be compared before calculation

        Returns:
            (FitInfo): FitInfo instance (with FitInfo.fit_result filled)
        """
        from src.analysis_tools.general_fitting import calculate_fit
        return calculate_fit(x=x, data=data, params=params, func=func, auto_bin=auto_bin, min_bins=self.AUTO_BIN_SIZE,
                             generate_hash=generate_hash, warning_id=f'Dat{self.dat.datnum}', method=self.FIT_METHOD)

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


