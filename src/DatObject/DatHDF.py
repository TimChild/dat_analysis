from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Union, Optional, Any
from src.DatObject.Attributes import Transition as T, Data as D, Entropy as E, Other as O, \
    Logs as L, AWG as A, SquareEntropy as SE, DatAttribute as DA
from src.DatObject.Attributes.DatAttribute import LateBindingProperty
from src.CoreUtil import my_partial
import src.CoreUtil as CU
from src import HDF_Util as HDU
from src.HDF_Util import with_hdf_read, with_hdf_write
import os
from src.DataStandardize.ExpConfig import ExpConfigGroupDatAttribute
import h5py

if TYPE_CHECKING:
    from src.DataStandardize.BaseClasses import Exp2HDF

logger = logging.getLogger(__name__)

_NOT_SET = object()
BASE_ATTRS = ['datnum', 'datname', 'dat_id', 'dattypes', 'date_initialized']

DAT_ATTR_DICT = {
    'expconfig': ExpConfigGroupDatAttribute,
    'data': D.Data,
    'logs': L.Logs,
    'entropy': E.NewEntropy,
    'transition': T.OldTransitions,
    'awg': A.AWG,
    'other': O.Other,
    'square entropy': SE.SquareEntropy,
}


class DatHDF(object):
    """Overall Dat object which contains general information about dat, more detailed info should be put
    into DatAttribute classes. Everything in this overall class should be useful for 99% of dats
    """
    version = '1.0'
    """
    Version history
        1.0 -- HDF based save files
    """

    def _dat_attr_prop(self, name: str) -> DA.DatAttribute:
        """
        General property method to get any DatAttribute which is not specified in a subclass

        Args:
            name (str): Name of Dat_attr to look for

        Returns:
            DA.DatAttribute: Named DatAttribute
        """
        name = name.lower()
        _check_is_datattr(name)
        private_key = _get_private_key(name)
        if not getattr(self, private_key, None):
            setattr(self, private_key, DAT_ATTR_DICT[name](self))
        return getattr(self, private_key)

    def _dat_attr_set(self, name, value: DA.DatAttribute):
        """
        General property method to set and DatAttribute which is not specified in subclass

        Args:
            name (): Name of DatAttribute
            value (): DatAttribute with correct class type

        Returns:
            None
        """
        name = name.lower()
        _check_is_datattr(name)
        private_key = _get_private_key(name)
        if not isinstance(value, DAT_ATTR_DICT.get(name)):
            raise TypeError(f'{value} is not an instance of {DAT_ATTR_DICT[name]}')
        else:
            setattr(self, private_key, value)

    def _dat_attr_del(self, name):
        """
        General property method to delete a DatAttribute which is not specified in subclass

        Args:
            name (): Name of DatAttribute

        Returns:
            None
        """
        name = name.lower()
        _check_is_datattr(name)
        private_key = _get_private_key(name)
        if getattr(self, private_key, None) is not None:
            delattr(self, private_key)

    def __init__(self, hdf_container: Union[HDU.HDFContainer, h5py.File]):
        """Very basic initialization, everything else should be done as properties that are run when accessed"""
        hdf_container = HDU.ensure_hdf_container(hdf_container)
        self.version = DatHDF.version
        self.hdf = hdf_container  # Should be left in CLOSED state! If it is ever left OPEN something is WRONG!
        # self.date_initialized = datetime.now().date()
        self._datnum = None
        self._datname = None
        self._date_initialized = None

    @property
    def datnum(self):
        if not self._datnum:
            self._datnum = self._get_attr('datnum')
        return self._datnum

    @property
    def datname(self):
        if not self._datname:
            self._datname = self._get_attr('datname')
        return self._datname

    @property
    def dat_id(self):
        return CU.get_dat_id(self.datnum, self.datname)

    @property
    def date_initialized(self):
        """This should be written when HDF is first made"""
        if not self._date_initialized:
            self._date_initialized = self._get_attr('date_initialized')
        return self._date_initialized

    @with_hdf_read
    def _get_attr(self, name: str, default: Optional[Any] = _NOT_SET, group_name: Optional[str] = None) -> Any:
        """
        For getting attributes from HDF group for DatHDF
        Args:
            name (str): Name of attribute to look for
            default (Any): Optional default value if not found
            group_name (str): Optional group name to look for attr in

        Returns:
            Any: value of attribute
        """
        check = True if default == _NOT_SET else False  # only check if no default passed
        if group_name:
            group = self.hdf.get(group_name)
        else:
            group = self.hdf.hdf
        attr = HDU.get_attr(group, name, default, check_exists=check)
        return attr

    ExpConfig = LateBindingProperty(my_partial(_dat_attr_prop, 'ExpConfig', arg_start=1),
                         my_partial(_dat_attr_set, 'ExpConfig', arg_start=1),
                         my_partial(_dat_attr_del, 'ExpConfig', arg_start=1))

    Data: D.Data = LateBindingProperty(my_partial(_dat_attr_prop, 'Data', arg_start=1),
                    my_partial(_dat_attr_set, 'Data', arg_start=1),
                    my_partial(_dat_attr_del, 'Data', arg_start=1))

    Logs = LateBindingProperty(my_partial(_dat_attr_prop, 'Logs', arg_start=1),
                    my_partial(_dat_attr_set, 'Logs', arg_start=1),
                    my_partial(_dat_attr_del, 'Logs', arg_start=1))

    # TODO: add more of above properties...


def _check_is_datattr(name):
    if name not in DAT_ATTR_DICT:
        raise ValueError(f'{name} not in DAT_ATTR_DICT.keys(): {DAT_ATTR_DICT.keys()}')


def _get_private_key(name):
    return '_' + name.lower()


class DatHDFBuilder:
    """Class to build basic DatHDF and then do some amount of initialization
    Subclass this class to do more initialization on creation"""

    def __init__(self, exp2hdf: Exp2HDF, init_level: str):
        self.init_level = init_level
        self.exp2hdf = exp2hdf

        # Initialized in build_dat
        self.hdf_container: HDU.HDFContainer = None
        self.dat: DatHDF = None

    def build_dat(self) -> DatHDF:
        """Build a dat from scratch (Should have already checked that data exists,
        and will fail if DatHDF already exists)"""
        self.create_hdf()
        self.copy_exp_data()
        self.init_DatHDF()
        self.init_ExpConfig()
        self.other_inits()
        self.init_base_attrs()
        assert self.dat is not None
        return self.dat

    def create_hdf(self):
        """Create the empty DatHDF (fail if already exists)"""
        path = self.exp2hdf.get_datHDF_path()
        if os.path.isfile(path):
            raise FileExistsError(f'Existing DatHDF at {os.path.abspath(path)} needs to be deleted before building a new HDF')
        elif not os.path.exists(os.path.dirname(path)):
            raise NotADirectoryError(f'No directory for {os.path.abspath(path)} to be written into')
        hdf = h5py.File(path, mode='w-')
        self.hdf_container = HDU.HDFContainer.from_hdf(hdf)  # Note: hdf if closed here

    def copy_exp_data(self):
        """Copy everything from original Exp Dat into Experiment Copy group of DatHDF"""
        data_path = self.exp2hdf.get_exp_dat_path()
        with h5py.File(data_path, 'r') as df, h5py.File(self.hdf_container.hdf_path, 'r+') as hdf:
            exp_data_group = hdf.require_group('Experiment Copy')
            exp_data_group.attrs['description'] = 'This group contains an exact copy of the Experiment Dat file'
            for key in df.keys():
                df.copy(key, exp_data_group)

    def init_DatHDF(self):
        """Init DatHDF"""
        self.dat = DatHDF(self.hdf_container)

    def init_ExpConfig(self):
        """Initialized the ExpConfig group of Dat"""
        exp_config = ExpConfigGroupDatAttribute(self.dat,
                                                self.exp2hdf.ExpConfig)  # Explicit initialization to pass in ExpConfig
        self.dat.ExpConfig = exp_config  # Should be able to set this because it has the right Class type
        assert exp_config.initialized == True

    def other_inits(self):
        pass

    def init_base_attrs(self):
        """
        Make the base attrs that should show up in the datHDF (i.e. datnum, datname, date_init etc)
        Returns:
            None
        """
        attr_dict = self._get_base_attrs()
        with h5py.File(self.dat.hdf.hdf_path, 'r+') as f:  # Can't use the usual wrapper here because no self.hdf attr
            for key, val in attr_dict.items():
                HDU.set_attr(f, key, val)

    def _get_base_attrs(self) -> dict:
        """
        Override this to add/change the default attrs
        Returns:
            dict: Dictionary of name: value pairs for base HDF attrs
        """
        attrs = dict(
            datnum=self.exp2hdf.datnum,
            datname=self.exp2hdf.datname,
            date_initialized=CU.time_now(),
        )
        return attrs


if __name__ == '__main__':
    DAT_ATTR_DICT.get('expconfig')
