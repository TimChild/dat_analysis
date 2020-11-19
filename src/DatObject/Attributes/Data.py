from __future__ import annotations
import numpy as np
from DatObject.Attributes.DatAttribute import DataDescriptor
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
        if item.startswith('__') or item.startswith('_'):  # To avoid infinite recursion (CRUCIAL)
            return super().__getattribute__(self, item)
        elif item in self.keys:
            val = self.get_data(item)
            setattr(self, item, val)  # After this it will be in Data attrs
            self._runtime_keys.append(item)  # Keep track of which were loaded like this so I can clear in clear_cache()
            return val
        else:
            return super().__getattribute__(self, item)

    # def __setattr__(self, key, value):
    #     """
    #     Overrides setting attributes on Data class. Using this to make easy way to save DataDescriptor
    #     Args:
    #         key ():
    #         value ():
    #
    #     Returns:
    #
    #     """

    def __init__(self, dat: DatHDF):
        self._keys: List[str] = list()
        self._data_keys: List[str] = list()
        self._runtime_keys: List[str] = list()  # Attributes added to Data during runtime
        self._data_descriptors: Dict[str, DataDescriptor] = dict()
        super().__init__(dat)

    def _initialize_minimum(self):
        self._set_exp_config_DataDescriptors()
        self.initialized = True

    def _set_exp_config_DataDescriptors(self):
        ExpConfig: ExpConfigGroupDatAttribute = self.dat.ExpConfig
        data_infos = ExpConfig.get_default_data_infos()
        for name, info in data_infos.items():
            if name in self.data_keys:
                path = self._get_data_path_from_hdf(name, prioritize='experiment_only')
                descriptor = DataDescriptor(data_path=path, offset=info.offset, multiply=info.multiply)
                self.set_data_descriptor(descriptor, info.standard_name)

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
            descriptor_keys = self.data_descriptors.keys()
            data_keys = self.data_keys
            self._keys = list(set(descriptor_keys).union(set(data_keys)))
        return tuple(self._keys)

    @property
    def data_keys(self) -> Tuple[str, ...]:
        if not self._data_keys:
            self._data_keys = self._get_data_keys()
        return tuple(self._data_keys)

    @with_hdf_read
    def _get_data_keys(self):
        """Get names of all data in Experiment Copy AND Data group
        Note: if any duplicates in data_keys, then the same data is saved in Data and Experiment Copy
        """
        dg = self.hdf.group
        exp_dg = self.hdf.get('Experiment Copy')
        data_keys = []
        for key in dg.keys():
            if is_Dataset(dg, key):
                data_keys.append(key)
        for key in exp_dg.keys():
            if is_Dataset(exp_dg, key):
                data_keys.append(key)
        return data_keys

    @property
    def data_descriptors(self) -> Dict[str, DataDescriptor]:
        if not self._data_descriptors:
            self._data_descriptors = self._get_all_descriptors()
        return self._data_descriptors

    @with_hdf_read
    def _get_all_descriptors(self):
        """Gets all existing DataDescriptor found in HDF.Data.Descriptors"""
        group = self.hdf.group.get('Descriptors')
        descriptors = {}
        for k in group.keys():
            g = group.get(k)
            if is_DataDescriptor(g):
                descriptors[k] = self.get_group_attr(k, DataClass=DataDescriptor)  # Avoiding infinite loop by loading directly here
        return descriptors

    def get_data_descriptor(self, key, filled=True) -> DataDescriptor:
        """
        Returns a filled DataDescriptor (i.e. DataDescriptor instance where inst.data is filled if True (or if already
        filled previously since that is in memory now anyway))

        Args:
            key (str): Name of DataDescriptor to get
            filled (bool): Whether to return the DataDescriptor with data loaded into it

        Returns:
            (DataDescriptor): Info about data along with data
        """
        if key in self.data_descriptors:
            descriptor = self.data_descriptors[key]
        elif key in self.data_keys:
            descriptor = self._get_default_descriptor_for_data(key)
        else:
            raise KeyError(f'No DataDescriptors found for {key}')
        if filled and not descriptor.data:
            descriptor.data = descriptor.get_array(self.hdf.hdf)  # Note: this acts as caching because I keep
            # hold of the descriptors
        return descriptor

    def _get_default_descriptor_for_data(self, name):
        path = self._get_data_path_from_hdf(name, prioritize='Data')
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
        dg = self.hdf.group
        exp_dg = self.hdf.get('Experiment Copy')
        if prioritize.lower() != 'experiment_only':
            ds = dg.get(name, None)
        else:
            ds = None
        exp_ds = exp_dg.get(name, None)
        if ds and exp_ds:
            if prioritize.lower() == 'data':
                return ds.path
            elif prioritize.lower() in ['experiment', 'experiment_only']:
                return exp_ds.path
            else:
                raise ValueError(f'{prioritize} not a valid argument')
        elif ds and not exp_ds:
            return ds.path
        elif exp_ds and not ds:
            return exp_ds.path
        else:
            raise FileNotFoundError(f'{name} not found in Data or Experiment Copy groups.')

    @with_hdf_write
    def set_data_descriptor(self, descriptor: DataDescriptor, name: Optional[str] = None):
        """
        For saving new DataDescriptors in Data (i.e. new way to open existing Data, descriptor.data_path must point to
        an existing dataset.

        Args:
            descriptor (DataDescriptor): New DataDescriptor to save in HDF
            name (Optional[str]): Optional name to save descriptor with (otherwise uses descriptor.name)
        """
        assert isinstance(descriptor, DataDescriptor)
        name = name if name else descriptor.name
        class_ = self.hdf.hdf.get(descriptor.data_path, None, getclass=True)  # Check points to Dataset
        if not class_ is h5py.Dataset:
            raise FileNotFoundError(f'No dataset found at {descriptor.data_path}, found {class_} instead')
        self.set_group_attr(name, descriptor, group_name=self.group_name + '/Descriptors', DataClass=DataDescriptor)
        self._data_descriptors[name] = descriptor

    def get_data(self, key) -> np.ndarray:
        """
        Gets array of data given key from descriptor. Alternatively can just call Data.<name> to get array
        Args:
            key (str): Name of Data to get array of

        Returns:
            (np.ndarray): Array of data
        """
        descriptor = self.get_data_descriptor(key)
        return descriptor.data

    @lru_cache
    @with_hdf_read
    def get_orig_data(self, key) -> np.ndarray:
        """
        Gets full array of data without any modifications
        Args:
            key ():

        Returns:

        """
        descriptor = self.get_data_descriptor(key)
        return descriptor.get_orig_array(self.hdf.hdf)

    @with_hdf_write
    def set_data(self, data: np.ndarray, name: str, descriptor: Optional[DataDescriptor] = None):
        """
        Set Data in Data group with 'name'. Optionally pass in a DataDescriptor to be saved at the same time.
        Note: descriptor.data_path will be overwritten to point to Data group
        Args:
            data (): Data array to save
            name (): Name to save data under in Data group
            descriptor (): Optional descriptor to save at same time (descriptor.data_path will be overwritten, but everything else will be kept as is)
        """
        assert isinstance(data, np.ndarray)
        HDU.set_data(self.hdf.group, name, data)
        if descriptor:
            data_path = f'/Data/{name}'
            descriptor.data_path = f'/Data/{name}'
            descriptor.data_link = h5py.SoftLink(data_path)
            self.set_data_descriptor(descriptor, name=name)


    def clear_caches(self):
        self._keys = list()
        self._data_descriptors = {}
        self._data_keys = list()
        self.get_orig_data.cache_clear()
        for key in self._runtime_keys:
            delattr(self, key)


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
    if hasattr(group.attrs, 'data_link'):  # Check the group is a DataDescriptor
        return True
    else:
        return False


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





