from __future__ import annotations
import numpy as np
import src.DatObject.Attributes.DatAttribute as DA
import h5py
import src.CoreUtil as CU
from functools import partial
import logging

import src.HDF_Util
from src.HDF_Util import with_hdf_write, with_hdf_read

logger = logging.getLogger(__name__)


# def _data_property_maker(data_key):
#     """Makes a property getter for 'data_key' which will load only when called (to be used for common datasets)"""
#
#     def _prop(self: Data, ):
#         return self.get_data(data_key)
#
#     def _set(self: Data, value: np.ndarray):
#         self.set_data(data_key, value, value.dtype)
#
#     def _del(self: Data, ):
#         if data_key in self._data_dict:
#             del self._data_dict[data_key]
#
#     return property(_prop, _set, _del)


class Data(DA.DatAttribute):
    version = '2.0'
    group_name = 'Data'

    def __getattr__(self, item):
        """Gets any other data from HDF if it exists, can use self.data_keys to see what there is"""
        if item.startswith('__') or item.startswith('_'):  # So don't complain about things like __len__
            return super().__getattribute__(self, item)
        else:
            if item in self.data_keys:
                return self.get_data(item)
            else:
                logger.warning(f'Dataset "{item}" does not exist for this Dat')
                return None

    def __setattr__(self, key, value):
        """Overriding so that making changes to data updates the HDF"""
        if not key in self.data_keys or not hasattr(self, '_data_dict'):
            super().__setattr__(key, value)
        else:
            if key in self._data_dict:
                self.set_data(key, value)

    def get_data(self, name):
        """Returns Data (caches so second access is fast)
        Can use this directly, or just call Data.<data_key> which calls this anyway"""
        if name in self.data_keys:
            if name not in self._data_dict:
                logger.debug(f'Loading {name} from HDF')
                self._data_dict[name] = self._get_data(name)
            return self._data_dict[name]
        else:
            raise KeyError(f'{name} not in data_keys: {self._data_keys}')

    @with_hdf_write
    def set_data(self, name, data, dtype=np.float32):
        """Sets data in HDF"""
        group = self.hdf.get(self.group_name)
        self._data_dict[name] = data.astype(dtype)
        if name not in self.data_keys:
            group.create_dataset(name, data.shape, dtype, data)
        else:
            logger.warning(
                f'Data with name [{name}] already exists. Overwriting now')  # TODO: Does this alter original data if it was linked?
            group[name] = data  # TODO: Check this works when resizing or changing dtype

    def __init__(self, dat):
        super().__init__(dat)
        # Don't load data here, only when it is actually requested (to make opening HDF faster)

        self._data_dict = dict()  # For storing loaded data (effectively a cache)
        # self.get_from_HDF()  # Don't do self.get_from_HDF() here by default because it is slow

    # Some standard datas that exist, made as properties
    x_array: np.ndarray = property(partial(get_data, 'x_array'), partial(set_data, 'x_array'))
    y_array: np.ndarray = property(partial(get_data, 'y_array'), partial(set_data, 'y_array'))
    i_sense: np.ndarray = property(partial(get_data, 'i_sense'), partial(set_data, 'i_sense'))
    # x_array: np.ndarray = _data_property_maker('x_array')
    # y_array: np.ndarray = _data_property_maker('y_array')
    # i_sense: np.ndarray = _data_property_maker('i_sense')

    def update_HDF(self):
        logger.warning('Calling update_HDF on Data attribute has no effect')
        pass

    def _initialize_minimum(self):
        raise NotImplementedError

    def clear_caches(self):
        del self._data_dict
        self._data_dict = dict()

    @property
    def data_keys(self):
        """Returns list of Data keys in HDF"""
        return self._get_data_keys()

    @with_hdf_read
    def _get_data(self, name):
        """Gets the data from the HDF"""
        if name in self.data_keys:
            group = self.hdf[self.group_name]
            return group.get(name)[:]  # Loads data from file here and returns all data
        else:
            raise KeyError(f'{name} not in data_keys: {self._data_keys}')

    @with_hdf_write
    def link_data(self, new_name, old_name, from_group=None):
        """
        Create link to data from group in Data group

        @param new_name: Name of dataset in Data group
        @type new_name: str
        @param old_name: Name of original dataset in given group (or in Data by default)
        @type old_name: str
        @param from_group: Group that data lies in, by default Data group
        @type from_group: Union[None, h5py.Group]
        @return: None, just sets new link in Data
        @rtype: None
        """
        group = self.hdf.get(self.group_name)
        from_group = from_group if from_group else group
        if new_name not in group.keys():
            ds = from_group[old_name]
            assert isinstance(ds, h5py.Dataset)
            self.group[new_name] = ds  # make link to dataset with new name
        else:
            logger.debug('Data [{new_name}] already exists in Data. Nothing changed')
        return



    @with_hdf_read
    def get_from_HDF(self):
        """Only call this if trying to pre load Data!
        Data should load any other data from HDF lazily
        Data is already accessible through getattr override (names available from self.data_keys)"""
        # Only run this when trying to pre load data!
        group = self.hdf.get(self.group_name)
        self.x_array = group.get('x_array', None)
        self.y_array = group.get('y_array', None)
        self.i_sense = group.get('i_sense', None)  # TODO: Make subclass which has these exp specific datas

    def _check_default_group_attrs(self):
        super()._check_default_group_attrs()
        # add other attrs here
        pass

    @with_hdf_read
    def _get_data_keys(self):  # Takes ~1ms, just opening the file is ~250us
        """Gets the name of all datasets in Data group of HDF"""
        group = self.hdf.get(self.group_name)
        keylist = group.keys()
        data_keys = set()
        for key in keylist:
            if isinstance(group[key], h5py.Dataset):  # Make sure it's a dataset not metadata
                data_keys.add(key)
        return data_keys

    @with_hdf_write
    def set_links_to_measured_data(self):
        """Creates links in Data group to data stored in Exp_measured_data group (not Exp HDF file directly,
        that needs to be built in Builders """
        group = self.hdf.get(self.group_name)
        if 'Exp_measured_data' not in self.hdf.keys():
            self.hdf.create_group('Exp_measured_data')
        exp_data_group = self.hdf['Exp_measured_data']
        data_keys = set()
        for key in exp_data_group.keys():
            if isinstance(exp_data_group[key], h5py.Dataset):
                data_keys.add(key)
        for key in data_keys:
            new_key = f'Exp_{key}'  # Store links to Exp_data with prefix so it's obvious
            if new_key not in group.keys():
                self.link_data(new_key, key, from_group=exp_data_group)


def init_Data(data_attribute: Data, setup_dict):
    dg = data_attribute.group
    for item in setup_dict.items():  # Use Data.get_setup_dict to create
        standard_name = item[0]  # The standard name used in rest of this analysis
        info = item[1]  # The possible names, multipliers, offsets to look for in exp data  (from setupDF)
        exp_names = CU.ensure_list(info[0])  # All possible names in exp
        exp_names = [f'Exp_{name}' for name in exp_names]  # stored with prefix in my Data folder
        exp_name, index = src.HDF_Util.match_name_in_group(exp_names, dg)  # First name which matches a dataset in exp
        if None not in [exp_name, index]:
            multiplier = info[1][index]  # Get the correction multiplier
            offset = info[2][index] if len(info) == 3 else 0  # Get the correction offset or default to zero
            if multiplier == 1 and offset == 0:  # Just link to exp data
                data_attribute.link_data(standard_name, exp_name,
                                         dg)  # Hard link to data (so not duplicated in HDF file)
            else:  # duplicate and alter dataset before saving in HDF
                data = dg.get(exp_name)[:]  # Get copy of exp Data
                data = data * multiplier + offset  # Adjust as necessary
                data_attribute.set_data(standard_name, data)  # Store as new data in HDF





