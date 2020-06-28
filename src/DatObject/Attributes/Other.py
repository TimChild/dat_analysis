"""General DatAttribute for storing work on a dat which does not fall into a well defined
DatAttribute. e.g. pinch off plots etc. Provides functionality for storing data, plots, code, notes, etc"""

from src.DatObject.Attributes import DatAttribute as DA
from src import HDF_Util as HDU
import numpy as np
import logging

logger = logging.getLogger(__name__)

HDF_ONLY_KEYS = ['version', 'description', 'Data']


class Other(DA.DatAttribute):
    group_name = 'Other'
    version = '1.0'

    def __setattr__(self, key: str, value):
        if key.startswith('__') or key.startswith('_') or not hasattr(self, '_attr_keys') or key in HDF_ONLY_KEYS:
            super().__setattr__(key, value)
        else:
            if value in HDU.ALLOWED_TYPES:
                if isinstance(value, np.ndarray) and value.size > 30:
                    raise ValueError(f'Trying to add array with size {value.size} as an attr. Use set_data() instead')
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

        self.get_from_HDF()

    def _set_default_group_attrs(self):
        super()._set_default_group_attrs()
        self.group.attrs['description'] = 'General DatAttribute for storing work on a dat which does not fall into a ' \
                                          'well defined DatAttribute. e.g. pinch off plots etc. Provides ' \
                                          'functionality for storing data, plots, code, notes, etc '

    def get_from_HDF(self):
        # Load things set in HDF with HDU.set_attr()
        self._load_attrs()

    def update_HDF(self):
        for key in self._attr_keys:
            HDU.set_attr(self.group, key, getattr(self, key))

    def _load_attrs(self):
        # Make keys of group and group attrs unique if there are overlaps
        HDU.check_group_attr_overlap(self.group, make_unique=True, exceptions=HDF_ONLY_KEYS)

        for key in list(self.group.keys()) + list(self.group.attrs.keys()):
            if key not in HDF_ONLY_KEYS:  # If not in HDF only attrs
                value = HDU.get_attr(self.group, key, None, check_exists=False)
                # If data was stored by HDU.set_attr() then retrieve it
                if value is not None:
                    setattr(self, key, value)  # TODO: Does this use my override

    def set_data(self, name, data):
        assert isinstance(data, np.ndarray)
        if name in self.Data.keys():
            logger.info(f'Overwriting data [{name}] in [{self.Data.name}]')
        self.Data[name] = data
