from __future__ import annotations
import numpy as np
import os

from src.HDF_Util import is_DataDescriptor, find_all_groups_names_with_attr, find_data_paths, NotFoundInHdfError
from src.CoreUtil import MyLRU
from src.DatObject.Attributes.DatAttribute import DataDescriptor
from src.DatObject.Attributes.DatAttribute import DatAttribute as DatAttr
import h5py
import logging
from src.DataStandardize.ExpConfig import ExpConfigGroupDatAttribute
import src.HDF_Util as HDU
from src.HDF_Util import with_hdf_write, with_hdf_read
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional

if TYPE_CHECKING:
    from src.DatObject.DatHDF import DatHDF
logger = logging.getLogger(__name__)

POSSIBLE_DATA_GROUPS = ['Transition', 'Entropy', 'Square Entropy', 'Awg', 'Other']


class Data(DatAttr):
    version = '2.0.0'
    group_name = 'Data'
    description = 'Container of data or information on how to get relevant data from Experiment Copy (i.e. if there ' \
                  'are bad rows to avoid, or the whole dataset needs to be multiplied by 10, or offset by something' \
                  'etc.'

    def __getattr__(self, item):
        """
        Called when an attribute doesn't already exist on Data. Will check if it is in self.keys, and if so will return
        the array for that data
        Args:
            item (str): Attribute to look for

        Returns:

        """
        if item.startswith('__') or item.startswith(
                '_') or item == 'data_descriptors':  # To avoid infinite recursion (CRUCIAL)
            return super().__getattribute__(self, item)
        elif item in self.keys or item in [k.split('/')[-1] for k in self.keys]:
            val = self.get_data(item)
            setattr(self, item, val)  # After this it will be in Data attrs
            self._runtime_keys.append(item)  # Keep track of which were loaded like this so I can clear in clear_cache()
            return val
        else:
            return super().__getattribute__(self, item)

    def __init__(self, dat: DatHDF):
        self._keys: List[str] = list()
        self._data_keys: List[str] = list()
        self._runtime_keys: List[str] = list()  # Attributes added to Data during runtime
        self._data_descriptors: Dict[str, DataDescriptor] = dict()
        super().__init__(dat)

    def initialize_minimum(self):
        self._set_exp_config_DataDescriptors()
        self.initialized = True

    @property
    def keys(self) -> Tuple[str, ...]:
        """
        All keys that can be asked for (i.e. any name of data, or any name of DataDescriptor (there may be several
        pointing to one dataset)

        Returns:
            List[str]: list of available keys to ask for (All DataDescriptors plus any data that doesn't have a
            descriptor yet).
        """
        if not self._keys:
            descriptor_keys_paths = self.data_descriptors.keys()
            descriptor_keys = []
            for path in descriptor_keys_paths:
                if path.split('/')[0] == 'Data':  # if 'Data/<more path>
                    descriptor_keys.append('/'.join(path.split('/')[1:]))  # TODO: is this ever reached?
                elif path.split('/')[1] == 'Data':  # If '/Data/<more path>
                    descriptor_keys.append('/'.join(path.split('/')[2:]))
                else:
                    descriptor_keys.append(path)
            data_keys = self.data_keys
            self._keys = list(set(descriptor_keys).union(set(data_keys)))
        return tuple(self._keys)

    @property
    def data_keys(self) -> Tuple[str, ...]:
        """Keys of datasets in Data group (and subgroups) and Experiment Copy group"""
        if not self._data_keys:
            self._data_keys = self._get_data_keys()
        return tuple(self._data_keys)

    @property
    def data_descriptors(self) -> Dict[str, DataDescriptor]:
        if not self._data_descriptors:
            self._data_descriptors = self._get_all_descriptors()
        return self._data_descriptors

    def get_data_descriptor(self, key, filled=True, data_group_name: Optional[str] = None) -> DataDescriptor:
        """
        Returns a filled DataDescriptor (i.e. DataDescriptor instance where inst.data is filled if True (or if already
        filled previously since that is in memory now anyway))

        Args:
            key (str): Name of DataDescriptor to get
            filled (bool): Whether to return the DataDescriptor with data loaded into it (Note: if data already exists, False will still return cached data)
            data_group_name (Optional[str]): Optional sub group to save data descriptor in (i.e. for saving data
            specific to a particular DatAttribute
        Returns:
            (DataDescriptor): Info about data along with data
        """
        full_key = self._get_descriptor_name(key, data_group_name)
        key_data = self._get_descriptor_name(key, None)  # the key for 'key' in Data/Descriptors instead
        if full_key in self.data_descriptors:  # Check if full path exists first (i.e. including sub_group)
            descriptor = self.data_descriptors[full_key]
        elif key_data in self.data_descriptors:  # Check if in Data group
            logger.debug(f"Didn't find {full_key}, looking for {key_data} instead in Data.data_descriptors")
            descriptor = self.data_descriptors[key_data]
        elif key in self.data_keys or key in [k.split('/')[-1] for k in self.data_keys]:  # Check if exists as data somewhere
            descriptor = self._get_default_descriptor_for_data(key, data_group_name=data_group_name)
            self._data_descriptors[full_key] = descriptor
        else:
            raise NotFoundInHdfError(f'No DataDescriptors found for {full_key} or {key_data} and no data found with '
                                     f'name {key}... (originally looked in data_group_name = {data_group_name})')
        if filled and descriptor.data is None:
            self._fill_descriptor(descriptor)  # Note: this acts as caching because I keep
            # hold of the descriptors (and cache reads from disk in this process too)
        return descriptor

    @with_hdf_write
    def set_data_descriptor(self, descriptor: DataDescriptor, name: Optional[str] = None,
                            data_group_name: Optional[str] = None):
        """
        For saving new DataDescriptors in Data (i.e. new way to open existing Data, descriptor.data_path must point to
        an existing dataset.

        Args:
            descriptor (DataDescriptor): New DataDescriptor to save in HDF
            name (Optional[str]): Optional name to save descriptor with (otherwise uses descriptor.name)
            data_group_name (Optional[str]): Optional sub group to save data descriptor in (i.e. for saving data
            specific to a particular DatAttribute
        """
        assert isinstance(descriptor, DataDescriptor)
        name = name if name else descriptor.data_path.split('/')[-1]
        class_ = self.hdf.hdf.get(descriptor.data_path, None, getclass=True)  # Check points to Dataset
        if not class_ is h5py.Dataset:
            raise FileNotFoundError(f'No dataset found at {descriptor.data_path}, found {class_} instead')
        dg_name = self._get_data_group_name(data_group_name)
        self.set_group_attr(name, descriptor,
                            group_name=dg_name + '/Descriptors',
                            DataClass=DataDescriptor)
        descriptor.data = None  # Ensure it is not holding onto old data
        full_name = self._get_descriptor_name(name, data_group_name=data_group_name)
        self._data_descriptors[full_name] = descriptor

    def _get_descriptor_name(self, name: str, data_group_name: Optional[str] = None):
        """Gets consistent and unique descriptor names to be used for self._data_descriptor cache keys"""
        dg_name = self._get_data_group_name(data_group_name)
        full_name = '/'.join((dg_name, name))
        return full_name

    def get_data(self, key, data_group_name: Optional[str] = None) -> np.ndarray:
        """
        Gets array of data given key from descriptor. Alternatively can just call Data.<name> to get array
        Args:
            key (str): Name of Data to get array of
            data_group_name (): Optional sub group to save data descriptor in (i.e. for saving data
            specific to a particular DatAttribute

        Returns:
            (np.ndarray): Array of data
        """
        descriptor = self.get_data_descriptor(key, data_group_name=data_group_name)
        return descriptor.data

    @lru_cache
    @with_hdf_read
    def get_orig_data(self, key: str, data_group_name: Optional[str] = None) -> np.ndarray:
        """
        Gets full array of data without any modifications
        Args:
            key ():

        Returns:

        """
        descriptor = self.get_data_descriptor(key, filled=False, data_group_name=data_group_name)
        return descriptor.get_orig_array(self.hdf.hdf)

    @with_hdf_write
    def set_data(self, data: np.ndarray, name: str, descriptor: Optional[DataDescriptor] = None,
                 data_group_name: str = None):
        """
        Set Data in Data group with 'name'. Optionally pass in a DataDescriptor to be saved at the same time.
        Note: descriptor.data_path will be overwritten to point to Data group
        Args:
            data (): Data array to save
            name (): Name to save data under in Data group
            descriptor (): Optional descriptor to save at same time (descriptor.data_path will be overwritten, but everything else will be kept as is)
        """
        group = self.hdf.get(self._get_data_group_name(data_group_name))
        assert isinstance(data, np.ndarray)
        HDU.set_data(group, name, data)
        if not descriptor:
            descriptor = DataDescriptor()
        data_path = f'{group.name}/{name}'
        descriptor.data_path = data_path
        descriptor.data_link = h5py.SoftLink(data_path)
        self.set_data_descriptor(descriptor, name=name, data_group_name=data_group_name)

    @with_hdf_read
    def _get_data_group_name(self, data_group_name: Optional[str] = None):
        """Get name of possible sub group of Data otherwise just path to Data"""
        if data_group_name:
            data_group_name = data_group_name.split('/')[-2] if data_group_name.split('/')[
                                                                    -1] == 'Descriptors' else data_group_name  # get only the part before '/Descriptors'
            data_group_name = data_group_name.title()
            if data_group_name == 'Data':
                return self.hdf.group.name
            if data_group_name in POSSIBLE_DATA_GROUPS:
                if data_group_name not in self.hdf.group.keys():
                    self._set_new_data_group(data_group_name=data_group_name)  
                group = self.hdf.group.get(data_group_name)
            else:
                raise ValueError(f'{data_group_name} is not an allowed data group name for Data, did you mean '
                                 f'one of {POSSIBLE_DATA_GROUPS}? Otherwise add value to Data.POSSIBLE_DATA_GROUPS '
                                 f'first')
            return group.name
        else:
            return self.hdf.group.name

    @with_hdf_write
    def _set_new_data_group(self, data_group_name):
        """Create a new sub group in Data group for storing data and DataDescriptors specific to another
        DatAttribute"""
        base_group = self.hdf.group
        group = base_group.create_group(data_group_name)
        HDU.set_attr(group, 'description', f'Data specific to {data_group_name}')
        HDU.set_attr(group, 'contains data', True)

        descriptor_group = group.create_group('Descriptors')
        HDU.set_attr(descriptor_group, 'contains DataDescriptors', True)

    def _get_data_keys(self):
        """Get names of all data in Experiment Copy AND Data group
        Note: if any duplicates in data_keys, then the same data is saved in Data and Experiment Copy
        """
        paths = self.get_data_paths()
        keys = []
        for path in [p[1:] for p in paths]:  # Chop off the leading '/'
            if path.split('/')[0] == 'Data':
                keys.append('/'.join(path.split('/')[1:]))
            else:
                keys.append(path)
        return keys

    @with_hdf_read
    def get_data_paths(self):
        """Get the full data paths to all data in Data (and sub groups) and Experiment Copy"""
        def get_dataset_paths_in_group(group: h5py.Group) -> List[str]:
            paths = []
            for k in group.keys():
                class_ = group.get(k, getclass=True)
                if class_ is h5py.Dataset:
                    paths.append(group.get(k).name)
            return paths

        dg = self.hdf.group
        exp_dg = self.hdf.get('Experiment Copy')
        other_data_groups = [self.hdf.get(name) for name in find_all_groups_names_with_attr(dg, 'contains data', True)]
        data_paths = []
        data_paths.extend(get_dataset_paths_in_group(dg))
        for group in other_data_groups:
            data_paths.extend(get_dataset_paths_in_group(group))
        data_paths.extend(get_dataset_paths_in_group(exp_dg))
        return data_paths

    @with_hdf_read
    def _get_all_descriptors(self):
        """Gets all existing DataDescriptors found in HDF.Data.Descriptors and sub groups of Data"""
        group_names = find_all_groups_names_with_attr(self.hdf.group, 'contains DataDescriptors', True)
        descriptors = {}
        for g_name in group_names:
            group = self.hdf.get(g_name)
            for k in group.keys():
                g = group.get(k)
                if is_DataDescriptor(g):
                    full_name = self._get_descriptor_name(k, g_name)
                    descriptors[full_name] = self.get_group_attr(k, group_name=group.name,
                                                                 DataClass=DataDescriptor)  # Avoiding infinite loop by loading directly here
        return descriptors

    @with_hdf_write
    def _set_exp_config_DataDescriptors(self):
        ExpConfig: ExpConfigGroupDatAttribute = self.dat.ExpConfig
        group = self.hdf.group.require_group('Descriptors')
        HDU.set_attr(group, 'contains DataDescriptors', True)
        data_infos = ExpConfig.get_default_data_infos()
        for name, info in data_infos.items():
            if name in [k.split('/')[-1] for k in self.data_keys]:
                path = self._get_data_path_from_hdf(name, prioritize='experiment_only')
                if path:
                    descriptor = DataDescriptor(data_path=path, offset=info.offset, multiply=info.multiply)
                    self.set_data_descriptor(descriptor, info.standard_name)
                else:
                    logger.warning(f'{name} in data_keys but could not find path to data in Experiment Copy')

    @with_hdf_read
    def _fill_descriptor(self, descriptor: DataDescriptor):
        # descriptor.data = descriptor.get_array(self.hdf.hdf) # Works but does not cache read from disk
        raw_data = self._get_cached_data(descriptor.data_path)  # So that the read from disk can be cached
        descriptor.data = descriptor.calculate_from_raw_data(raw_data)

    # @MyLRU
    @with_hdf_read
    def _get_cached_data(self, data_path: str) -> np.ndarray:
        """For caching the read from disk which may be done for descriptors with the same name """
        return self.hdf.get(data_path)[:]

    def _get_default_descriptor_for_data(self, name, data_group_name: Optional[str] = None) -> DataDescriptor:
        if len(name.split('/')) > 1:
            split = name.split('/')
            if not data_group_name:
                data_group_name = split[0]
            name = split[-1]
            if len(split) > 2:
                logger.warning(f'{split} has more than 2 parts, only first and last being used')
        if not data_group_name:
            data_group_name = 'Data'
        path = self._get_data_path_from_hdf(name, prioritize=data_group_name)
        descriptor = DataDescriptor(path)
        return descriptor

    @with_hdf_read
    def _get_data_path_from_hdf(self, name, prioritize='Data'):
        """
        Get's the path to Data in HDF prioritizing Data group over Experiment Copy
        Args:
            prioritize(str): Whether to prioritize 'Data' group or 'Experiment' or group (Experiment Copy) (or 'Experiment_only')

        Returns:
            (str): path to data
        """
        def find_paths_in(group):
            paths = find_data_paths(group, name)
            if len(paths) == 1:
                return paths[0]
            elif len(paths) > 1:
                logger.warning(f'Multiple paths found to {name}, return first of this list only: {paths}')
                return paths[0]
            else:
                return []

        prioritize = prioritize.lower()
        if prioritize in ['experiment_only', 'experiment']:
            path = find_paths_in(self.hdf.get('Experiment Copy'))
            if not path:
                path = None
            if prioritize == 'experiment_only' or path:
                return path

        elif prioritize != 'data' and prioritize.title() in POSSIBLE_DATA_GROUPS:
            path = find_paths_in(self.hdf.group.get(prioritize.title()))
            if path:
                return path

        path = find_paths_in(self.hdf.group)
        if path:
            return path

        path = find_paths_in(self.hdf.get('Experiment Copy'))
        if path:
            return path
        raise FileNotFoundError(f'No path found for {name} in Data or Experiment Copy')

    def clear_caches(self):
        self._keys = list()
        self._data_descriptors = {}
        self._data_keys = list()
        self.get_orig_data.cache_clear()
        # self._get_cached_data.cache_clear()
        for key in self._runtime_keys:
            delattr(self, key)

# class Data(DA.DatAttribute):
#     version = '2.0'
#     group_name = 'Data'
#
#     def __getattr__(self, item):
#         """Gets any other data from HDF if it exists, can use self.data_keys to see what there is"""
#         if item.startswith('__') or item.startswith('_'):  # So don't complain about things like __len__
#             return super().__getattribute__(self, item)
#         else:
#             if item in self.data_keys:
#                 return self.get_data(item)
#             else:
#                 logger.warning(f'Dataset "{item}" does not exist for this Dat')
#                 return None
#
#     def __setattr__(self, key, value):
#         """Overriding so that making changes to data updates the HDF"""
#         if not key in self.data_keys or not hasattr(self, '_data_dict'):
#             super().__setattr__(key, value)
#         else:
#             if key in self._data_dict:
#                 self.set_data(key, value)
#
#     def get_data(self, name):
#         """Returns Data (caches so second access is fast)
#         Can use this directly, or just call Data.<data_key> which calls this anyway"""
#         if name in self.data_keys:
#             if name not in self._data_dict:
#                 logger.debug(f'Loading {name} from HDF')
#                 self._data_dict[name] = self._get_data(name)
#             return self._data_dict[name]
#         else:
#             raise KeyError(f'{name} not in data_keys: {self._data_keys}')
#
#     @with_hdf_write
#     def set_data(self, name, data, dtype=np.float32):
#         """Sets data in HDF"""
#         group = self.hdf.group
#         self._data_dict[name] = data.astype(dtype)
#         HDU.set_data(group, name, data, dtype=dtype)
#
#     def __init__(self, dat):
#         super().__init__(dat)
#         # Don't load data here, only when it is actually requested (to make opening HDF faster)
#         self._data_dict = dict()  # For storing loaded data (effectively a cache)
#
#     # Some standard datas that exist, made as properties
#     x_array: np.ndarray = property(my_partial(get_data, 'x_array', arg_start=1),
#                                    my_partial(set_data, 'x_array', arg_start=1))
#     y_array: np.ndarray = property(my_partial(get_data, 'y_array', arg_start=1),
#                                    my_partial(set_data, 'y_array', arg_start=1))
#     i_sense: np.ndarray = property(my_partial(get_data, 'i_sense', arg_start=1),
#                                    my_partial(set_data, 'i_sense', arg_start=1))
#
#     def _initialize_minimum(self):
#         self._copy_from_experiment()
#         self.initialized = True
#
#     def _copy_from_experiment(self):
#         Exp_config: ExpConfigGroupDatAttribute = self.dat.ExpConfig
#         keys_to_copy = Exp_config.get_data_
#
#
#     def clear_caches(self):
#         del self._data_dict
#         self._data_dict = dict()
#
#     @property
#     def data_keys(self):
#         """Returns list of Data keys in HDF"""
#         return self._get_data_keys()
#
#     @with_hdf_read
#     def _get_data(self, name):
#         """Gets the data from the HDF"""
#         if name in self.data_keys:
#             group = self.hdf[self.group_name]
#             return HDU.get_dataset(group, name)[:]  # Loads data from file here and returns all data
#         else:
#             raise KeyError(f'{name} not in data_keys: {self._data_keys}')
#
#     @with_hdf_write
#     def _link_data(self, new_name: str, old_name: str, from_group: str = None):
#         """
#         Create link from 'old_name' data in 'from_group' to 'new_name' data in 'to_group'
#
#         Args:
#             new_name (): New name of data (link)
#             old_name (): Old name of data to link to
#             from_group (): Group to link data from
#
#         Returns:
#
#         """
#         group = self.hdf.group
#         from_group = from_group if from_group else group
#         try:
#             HDU.link_data(from_group, group, old_name, new_name)
#         except FileExistsError as e:
#             logger.debug(f'Data [{new_name}] already exists in Data. Nothing changed')
#
#     @with_hdf_read
#     def _get_data_keys(self):  # Takes ~1ms, just opening the file is ~250us
#         """Gets the name of all datasets in Data group of HDF"""
#         group = self.hdf.group
#         keylist = group.keys()
#         data_keys = set()
#         for key in keylist:
#             if isinstance(group[key], h5py.Dataset):  # Make sure it's a dataset not metadata
#                 data_keys.add(key)
#         return data_keys
#
#     @with_hdf_write
#     def _set_links_to_measured_data(self):
#         """Creates links in Data group to data stored in Exp_measured_data group (not Exp HDF file directly,
#         that needs to be built in Builders """
#         group = self.hdf.group
#         if 'Exp_measured_data' not in self.hdf.keys():
#             self.hdf.create_group('Exp_measured_data')
#         exp_data_group = self.hdf['Exp_measured_data']
#         data_keys = set()
#         for key in exp_data_group.keys():
#             if isinstance(exp_data_group[key], h5py.Dataset):
#                 data_keys.add(key)
#         for key in data_keys:
#             new_key = f'Exp_{key}'  # Store links to Exp_data with prefix so it's obvious
#             if new_key not in group.keys():
#                 self._link_data(new_key, key, from_group=exp_data_group)
#
#
# def init_Data(data_attribute: Data, setup_dict):
#     dg = data_attribute.group
#     for item in setup_dict.items():  # Use Data.get_setup_dict to create
#         standard_name = item[0]  # The standard name used in rest of this analysis
#         info = item[1]  # The possible names, multipliers, offsets to look for in exp data  (from setupDF)
#         exp_names = CU.ensure_list(info[0])  # All possible names in exp
#         exp_names = [f'Exp_{name}' for name in exp_names]  # stored with prefix in my Data folder
#         exp_name, index = HDU.match_name_in_group(exp_names, dg)  # First name which matches a dataset in exp
#         if None not in [exp_name, index]:
#             multiplier = info[1][index]  # Get the correction multiplier
#             offset = info[2][index] if len(info) == 3 else 0  # Get the correction offset or default to zero
#             if multiplier == 1 and offset == 0:  # Just link to exp data
#                 data_attribute._link_data(standard_name, exp_name,
#                                           dg)  # Hard link to data (so not duplicated in HDF file)
#             else:  # duplicate and alter dataset before saving in HDF
#                 data = dg.get(exp_name)[:]  # Get copy of exp Data
#                 data = data * multiplier + offset  # Adjust as necessary
#                 data_attribute.set_data(standard_name, data)  # Store as new data in HDF
