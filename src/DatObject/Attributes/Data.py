import numpy as np
import src.DatObject.Attributes.DatAttribute as DA
import h5py
import src.CoreUtil as CU

import logging

import src.HDF_Util

logger = logging.getLogger(__name__)

class NewData(DA.DatAttribute):
    version = '1.0'
    group_name = 'Data'

    def __getattr__(self, item):
        """Gets any other data from HDF if it exists, can use self.data_keys to see what there is"""
        if item.startswith('__') or item.startswith('_'):  # So don't complain about things like __len__
            return super().__getattribute__(self, item)
        else:
            if item in self.data_keys:
                return self.group[item][:]  # loads into memory here. Use get_dataset if trying to avoid this
            else:
                print(f'Dataset "{item}" does not exist for this Dat')
                return None

    def __init__(self, hdf):
        assert isinstance(hdf, h5py.File)
        super().__init__(hdf)
        # By default just pointers to Datasets if they exist, not loaded into memory
        self.x_array = None
        self.y_array = None
        self.i_sense = None
        self.entx = None
        self.enty = None
        self.get_from_HDF()

    def update_HDF(self):
        logger.warning('Calling update_HDF on Data attribute has no effect')
        pass

    @property
    def data_keys(self):
        return self._get_data_keys()

    def get_dataset(self, name):
        if name in self.data_keys:
            return self.group[name]  # Returns pointer to Dataset

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

        from_group = from_group if from_group else self.group
        if new_name not in self.group.keys():
            ds = from_group[old_name]
            assert isinstance(ds, h5py.Dataset)
            self.group[new_name] = ds  # make link to dataset with new name
        else:
            logger.info(f'Data [{new_name}] already exists in Data. Nothing changed')
        return

    def set_data(self, name, data, dtype=np.float32):
        if name not in self.data_keys:
            self.group.create_dataset(name, data.shape, dtype, data)
        else:
            logger.warning(f'Data with name [{name}] already exists. Overwriting now')  # TODO: Does this alter original data if it was linked?
            self.group[name] = data  # TODO: Check this works when resizing or changing dtype

        if name in ['x_array', 'y_array', 'i_sense', 'entx', 'enty']:
            setattr(self, name, self.group.get(name))  # Update e.g. self.i_sense

    def get_from_HDF(self):
        """Data should load any other data from HDF lazily
        Datasets are already accessible through getattr override (names available from self.data_keys)"""
        self.x_array = self.group.get('x_array', None)
        self.y_array = self.group.get('y_array', None)
        self.i_sense = self.group.get('i_sense', None)  # TODO: Make subclass which has these exp specific datas
        self.entx = self.group.get('entx', None)
        self.enty = self.group.get('enty', None)

    def _set_default_group_attrs(self):
        super()._set_default_group_attrs()
        # add other attrs here
        pass

    def _get_data_keys(self):
        keylist = self.group.keys()
        data_keys = set()
        for key in keylist:
            if isinstance(self.group[key], h5py.Dataset):  # Make sure it's a dataset not metadata
                data_keys.add(key)
        return data_keys

    def set_links_to_measured_data(self):  # TODO: This should be part of builder class
        """Creates links in Data group to data stored in Exp_measured_data group (not Exp HDF file directly,
        that needs to be built in Builders """
        if 'Exp_measured_data' not in self.hdf.keys():
            logger.warning(f'"Exp_measured_data" not currently in DatHDF. Has been added')
            self.hdf.create_group('Exp_measured_data')
        exp_data_group = self.hdf['Exp_measured_data']
        data_keys = set()
        for key in exp_data_group.keys():
            if isinstance(exp_data_group[key], h5py.Dataset):
                data_keys.add(key)
        for key in data_keys:
            new_key = f'Exp_{key}'  # Store links to Exp_data with prefix so it's obvious
            if new_key not in self.group.keys():
                self.link_data(new_key, key, from_group=exp_data_group)


def init_Data(data_attribute: NewData, setup_dict):
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
                data_attribute.link_data(standard_name, exp_name, dg)  # Hard link to data (so not duplicated in HDF file)
            else:  # duplicate and alter dataset before saving in HDF
                data = dg.get(exp_name)[:]  # Get copy of exp Data
                data = data * multiplier + offset  # Adjust as necessary
                data_attribute.set_data(standard_name, data)  # Store as new data in HDF

#
# class Data(object):
#     """Stores all raw data of Dat"""
#
#     def __getattr__(self, item):
#         """Overrides behaviour when attribute is not found for Data"""
#         if item.startswith('__'):  # So don't complain about things like __len__
#             return super().__getattr__(self, item)
#         else:
#             if item in self.data_keys:
#                 hdf = h5py.File(CU.get_full_path(self.hdfpath), 'r')
#                 return hdf[item]
#             else:
#                 print(f'Dataset "{item}" does not exist for this Dat')
#                 return None
#
#     def __getstate__(self):
#         """Required for pickling because of __getattr__ override"""
#         return self.__dict__
#
#     def __setstate__(self, state):
#         """Required for unpickling because of __getattr__ override"""
#         self.__dict__.update(state)
#
#     def __init__(self, infodict=None, hdfpath=None):
#         """Creates Data Attribue for Dat. Can specify path to hdf if desired"""
#         self.data_keys = None
#         if infodict is not None:
#             self.x_array = infodict['Logs'].get('x_array')
#             self.y_array = infodict['Logs'].get('y_array')
#             self.i_sense = infodict.get('i_sense')
#             self.entx = infodict.get('entx')
#             self.enty = infodict.get('enty')
#             self.conductance = infodict.get('conductance')
#             self.current = infodict.get('current')
#             if hdfpath is None:
#                 hdfpath = infodict.get('hdfpath')
#             if 'li_theta_keys' in infodict.keys():
#                 self.li_theta_keys = infodict.get('li_theta_keys', None)
#                 self.li_multiplier = infodict.get('li_multiplier', 1)
#
#         self.hdfpath = hdfpath
#         if self.hdfpath is not None:
#             hdf = h5py.File(CU.get_full_path(self.hdfpath), 'r')
#             keylist = hdf.keys()
#             data_keys = []
#             for key in keylist:
#                 if isinstance(hdf[key], h5py.Dataset):  # Make sure it's a dataset not metadata
#                     data_keys.append(key)
#             self.data_keys = data_keys
#
#     def get_names(self):
#         """Returns list of data names that is not None"""
#         datanames = [attrname for attrname in self.__dict__.keys() if getattr(self, attrname, None) is not None]
#         datanames += self.data_keys
#         datanames.remove('data_keys')
#         datanames.remove('hdfpath')
#         return datanames
#

