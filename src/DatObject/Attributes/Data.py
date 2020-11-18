from __future__ import annotations
import numpy as np
import src.DatObject.Attributes.DatAttribute as DA
from src.DatObject.Attributes.DatAttribute import DatAttribute as DatAttr
import h5py
import src.CoreUtil as CU
from src.CoreUtil import my_partial
import logging
from src.DataStandardize.ExpConfig import ExpConfigGroupDatAttribute
from dataclasses import dataclass, field
import src.HDF_Util as HDU
from src.HDF_Util import with_hdf_write, with_hdf_read
from functools import lru_cache
from typing import TYPE_CHECKING, Dict
if TYPE_CHECKING:
    from src.DatObject.DatHDF import DatHDF
logger = logging.getLogger(__name__)


@dataclass
class DataDescriptor(DA.DatDataclassTemplate):
    """
    Place to group together information required to get Data from Experiment_Copy (or somewhere else) in correct form
    i.e. accounting for offset/multiplying/bad rows of data etc
    """
    data_path: str
    name: str  # Name of data in Experiment_Copy (just so it is clear to user looking in HDF file directly)
    offset: float = 0.0  # How much to offset all the data (i.e. systematic error)
    multiply: float = 1.0  # How much to multiply all the data (e.g. convert to nA or some other standard)
    bad_rows: list = field(default_factory=list)
    bad_columns: list = field(default_factory=list)
    # May want to add more optional things here later (e.g. replace clipped values with NaN etc)

    data: np.ndarray = field(default=None, repr=False, compare=False)  # For temp data storage (not for storing in HDF)

    data_link: h5py.SoftLink = field(default=None, repr=False)

    def __post_init__(self):
        if self.data_path and not self.data_link:
            self.data_link = h5py.SoftLink(self.data_path)  # This will show up as a dataset in the HDF
        elif self.data_path != self.data_link.path:
            logger.error(f'data_path = {self.data_path} != data_link = {self.data_link.path}. Something wrong - change'
                         f'data_path or data_link accordingly')

    def ignore_keys_for_saving(self):
        """Don't want to save 'data' to HDF here because it will be duplicating data saved at 'data_path'"""
        return 'data'

    def get_array(self, hdf: h5py.File):
        """
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
        if self.multiply == 1.0 and self.offset == 0.0:
            return data
        else:
            data = data*self.multiply+self.offset
            return data

    def get_orig_array(self, hdf: h5py.File):
        return hdf.get(self.data_path)[:]

    def _good_slice(self, shape: tuple):
        if not self.bad_rows and not self.bad_columns:
            return np.s_[:]
        else:
            raise NotImplementedError(f'Still need to write how to get slice of only good rows! ')  # TODO: Do this



class Data(DatAttr):
    version = '2.0.0'
    group_name = 'Data'
    description = 'Container of data or information on how to get relevant data from Experiment Copy (i.e. if there ' \
                  'are bad rows to avoid, or the whole dataset needs to be multiplied by 10, or offset by something' \
                  'etc.'

    def __init__(self, dat: DatHDF):
        super().__init__(dat)
        self._data_descriptors: Dict[str: DataDescriptor] = self._get_all_descriptors()

    def get_data(self, key) -> np.ndarray:
        """
        Gets array of data given key from descriptor
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

    @with_hdf_read
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
        if key in self._data_descriptors:
            descriptor = self._data_descriptors[key]
            if filled and not descriptor.data:
                descriptor.data = descriptor.get_array(self.hdf.hdf)    # Note: this acts as caching because I keep
                                                                        # hold of the descriptors
            return descriptor

    def _initialize_minimum(self):
        self._get_exp_config_data_corrections()
        self.initialized = True

    def _get_exp_config_data_corrections(self):
        ExpConfig: ExpConfigGroupDatAttribute = self.dat.ExpConfig
        raise NotImplemented

    @with_hdf_read
    def _get_all_descriptors(self):
        """Gets all existing DataDescriptor found in HDF.Data"""
        group = self.hdf.get(self.group_name)
        descriptors = {}
        for k in group.keys():
            if is_Group(group, k):
                g = group.get(k)
                if is_DataDescriptor(g):
                    descriptors[k] = self.get_group_attr(k, dataclass=DataDescriptor)
        return descriptors

    def clear_caches(self):
        self._data_descriptors = {}
        self._data_descriptors = self._get_all_descriptors()
        self.get_orig_data.cache_clear()


def is_Group(parent_group, key):
    class_ = parent_group.get(key, getclass=True)
    if class_ is h5py.Group:
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
#         group = self.hdf.get(self.group_name)
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
#         group = self.hdf.get(self.group_name)
#         from_group = from_group if from_group else group
#         try:
#             HDU.link_data(from_group, group, old_name, new_name)
#         except FileExistsError as e:
#             logger.debug(f'Data [{new_name}] already exists in Data. Nothing changed')
#
#     @with_hdf_read
#     def _get_data_keys(self):  # Takes ~1ms, just opening the file is ~250us
#         """Gets the name of all datasets in Data group of HDF"""
#         group = self.hdf.get(self.group_name)
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
#         group = self.hdf.get(self.group_name)
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





