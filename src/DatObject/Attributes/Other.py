"""General DatAttribute for storing work on a dat which does not fall into a well defined
DatAttribute. e.g. pinch off plots etc. Provides functionality for storing data, plots, code, notes, etc"""

from src.DatObject.Attributes import DatAttribute as DA
from src import HDF_Util as HDU
import numpy as np
import logging
import h5py

logger = logging.getLogger(__name__)

HDF_ONLY_KEYS = ['version', 'description', 'Data']


class Other(DA.DatAttribute):
    group_name = 'Other'
    version = '1.0'

    def __setattr__(self, key: str, value):
        if key.startswith('__') or key.startswith('_') or not hasattr(self, '_attr_keys') or key in HDF_ONLY_KEYS+['Code']:  # I save 'Code' separately
            super().__setattr__(key, value)
        else:
            if HDU.allowed(value):
                if isinstance(value, np.ndarray) and value.size > 500:
                    logger.warning(f'Attr {key} has size {value.size}. To store in HDF use Other.set_data() instead')
                    super().__setattr__(key, value)
                logger.info(f'{key} added to self.attr_keys and will be stored in HDF upon update')
                self._attr_keys.add(key)
            else:
                logger.info(f'value of {key} has type {type(value)} which is not allowed in HDF, will not be stored!')
                if key in self._attr_keys:
                    self._attr_keys.remove(key)
            super().__setattr__(key, value)

    def __delattr__(self, item):
        if hasattr(self, '_attr_keys'):
            if item in self._attr_keys:
                self._attr_keys.remove(item)
        super().__delattr__(item)

    def __init__(self, hdf):
        super().__init__(hdf)
        self._attr_keys = set()
        self.Data = self.group.require_group('Data')
        self.Code: dict = {}  # Will be loaded in get_from_HDF() if anything saved there
        self.get_from_HDF()

    def _check_default_group_attrs(self):
        super()._check_default_group_attrs()
        self.group.attrs['description'] = 'General DatAttribute for storing work on a dat which does not fall into a ' \
                                          'well defined DatAttribute. e.g. pinch off plots etc. Provides ' \
                                          'functionality for storing data, plots, code, notes, etc '

    def get_from_HDF(self):
        # Load things set in HDF with HDU.set_attr()
        self._load_attrs()  # This should also lode 'Code' part OK.

    def update_HDF(self):
        super().update_HDF()
        for key in self._attr_keys - {'Code'}:  # I save 'Code' separately to force saving in dict group
            val = getattr(self, key)
            if isinstance(val, np.ndarray) and val.size > 1000:
                logger.warning(f'Ignoring adding {key} with size {val.size} as an attr, should be saved as dataset instead')
            else:
                HDU.set_attr(self.group, key, getattr(self, key))

        if self.Code:
            self._save_code_to_hdf(flush=False)
        self.group.file.flush()

    def _load_attrs(self):
        # Make keys of group and group attrs unique if there are overlaps
        HDU.check_group_attr_overlap(self.group, make_unique=True, exceptions=HDF_ONLY_KEYS)

        for key in list(self.group.keys()) + list(self.group.attrs.keys()):
            if key not in HDF_ONLY_KEYS:  # If not in HDF only attrs
                value = HDU.get_attr(self.group, key, None, check_exists=False)
                # If data was stored by HDU.set_attr() then retrieve it
                if value is not None:
                    setattr(self, key, value)

    def set_data(self, name, data):
        assert isinstance(data, (np.ndarray, h5py.Dataset))
        if name in self.Data.keys():
            logger.info(f'Overwriting data [{name}] in [{self.Data.name}]')
            del self.Data[name]
        self.Data[name] = data

    def save_code(self, code, name):
        """Saves string of code to HDF.Other.Code dict group
        Note: Can be done by working with Other.Code[name]=code directly,
        but this will save to HDF as well"""
        assert type(code) == str
        assert type(name) == str
        self.Code = self.Code if self.Code else {}
        if name in self.Code:
            logger.debug(f'Overwriting code for {name}')
        self.Code[name] = code
        self._save_code_to_hdf()

    def _save_code_to_hdf(self, flush=True):
        code_group = self.group.require_group('Code')
        HDU.save_dict_to_hdf_group(code_group, self.Code)
        if flush:
            self.group.file.flush()
