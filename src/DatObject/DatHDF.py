from __future__ import annotations
import logging
from threading import Lock, RLock
from typing import TYPE_CHECKING, Union, Optional, Any, List
from src.DatObject.Attributes import Transition as T, Data as D, Entropy as E, Other as O, \
    Logs as L, AWG as A, SquareEntropy as SE, DatAttribute as DA, Figures
from src.DatObject.Attributes.DatAttribute import LateBindingProperty
from src.CoreUtil import my_partial
import datetime
from src import HDF_Util as HDU
from src.HDF_Util import with_hdf_read, with_hdf_write
import os
from src.DataStandardize.ExpConfig import ExpConfigGroupDatAttribute
import h5py

if TYPE_CHECKING:
    from src.DataStandardize.BaseClasses import Exp2HDF

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_NOT_SET = object()
BASE_ATTRS = ['datnum', 'datname', 'dat_id', 'dattypes', 'date_initialized']

DAT_ATTR_DICT = {
    'expconfig': ExpConfigGroupDatAttribute,
    'data': D.Data,
    'logs': L.Logs,
    'entropy': E.Entropy,
    'transition': T.Transition,
    'awg': A.AWG,
    # 'other': O.Other,
    'squareentropy': SE.SquareEntropy,
    'figures': Figures.Figures
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

    def _dat_attr_prop(self, name: str) -> Any:
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
            name_lower (): Name of DatAttribute

        Returns:
            None
        """
        name_lower = name.lower()
        _check_is_datattr(name_lower)
        private_key = _get_private_key(name_lower)
        if getattr(self, private_key, None) is not None:
            hdf_key = getattr(self, private_key).group_name
            delattr(self, private_key)
            self.del_dat_attr(hdf_key)
        else:
            logger.warning(f'{name} not currently loaded, trying to delete {name} from HDF, but may not be saved under this name')
            self.del_dat_attr(name)

    @with_hdf_write
    def del_dat_attr(self, hdf_key: str):
        """Deletes top level group in HDF (i.e. whole DatAttribute)"""
        if hdf_key in self.hdf.hdf.keys():
            del self.hdf.hdf[hdf_key]
        else:
            logger.warning(f'{hdf_key} not found in dat{self.datnum} that has keys:\n{self.list_contents_of_hdf("")}')

    @with_hdf_write
    def del_hdf_item(self, hdf_path: str):
        """
        Deletes item from HDF at specified path
        Args:
            hdf_path (): '/' separated path to item which should be deleted

        Returns:

        """
        item = self.hdf.hdf.get(hdf_path)
        if item is not None:
            logger.info(f'Deleting {item} from Dat{self.datnum}')
            del self.hdf.hdf[hdf_path]
        else:
            logger.warning(f'No item found at {hdf_path} for dat{self.datnum}')
            self.list_contents_of_hdf(hdf_path)

    def __init__(self, hdf_container: Union[HDU.HDFContainer, h5py.File]):
        """Very basic initialization, everything else should be done as properties that are run when accessed"""
        hdf_container = HDU.ensure_hdf_container(hdf_container)
        self.version = DatHDF.version
        self.hdf = hdf_container  # Should be left in CLOSED state! If it is ever left OPEN something is WRONG!
        # self.date_initialized = datetime.now().date()
        self._datnum = None
        self._datname = None
        self._date_initialized = None

        self.lock = Lock()
        self.rlock = RLock()
        self._threaded_test_var = None

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
        return get_dat_id(self.datnum, self.datname)

    @property
    def date_initialized(self):
        """This should be written when HDF is first made"""
        if not self._date_initialized:
            self._date_initialized = self._get_attr('date_initialized')
        return self._date_initialized

    @with_hdf_read
    def list_contents_of_hdf(self, path: str, get_attrs: bool = False) -> Optional[List[str]]:
        """
        Lists the contents of the HDF at given path (i.e. lists hdf.get(path).keys()
        Args:
            path (): '/' separated path into HDF file (e.g. 'Entropy/Avg Fits')
            get_attrs (): Whether to look for keys of attrs at path instead of just keys

        Returns:
            (List[str]): List of keys at given path
        """
        if path == '':
            path = '/'
        obj = self.hdf.hdf.get(path)
        if obj is not None:
            if isinstance(obj, h5py.Group):
                if get_attrs is False:
                    return list(obj.keys())
                else:
                    return list(obj.attrs.keys())
            elif isinstance(obj, h5py.Dataset):
                logger.warning(f'{path} points to a Dataset in dat{self.datnum}, not a group')
                return None
        else:  # Path didn't exists, find the contents of the furthest part along path which exists
            prev_obj = self.hdf.hdf
            valid_path = ''
            for p in path.split('/'):
                if p == '':
                    continue
                new_obj = prev_obj.get(p)
                if not isinstance(new_obj, h5py.Group):
                    logger.warning(f'{path} is only valid up to {valid_path} which contains:\n'
                                   f'{prev_obj.keys()}')
                    return None
                valid_path += '/' + p
                prev_obj = new_obj
        raise RuntimeError(f"{path} was not a Group or Dataset, but seemed to work all the way. Shouldn't get here")

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

    # ExpConfig = property(my_partial(_dat_attr_prop, 'ExpConfig'),
    #                      my_partial(_dat_attr_set, 'ExpConfig'),
    #                      my_partial(_dat_attr_del, 'ExpConfig'))
    #
    # Data: D.Data = property(my_partial(_dat_attr_prop, 'Data'),
    #                         my_partial(_dat_attr_set, 'Data'),
    #                         my_partial(_dat_attr_del, 'Data'))
    #
    # Logs: L.Logs = property(my_partial(_dat_attr_prop, 'Logs'),
    #                         my_partial(_dat_attr_set, 'Logs'),
    #                         my_partial(_dat_attr_del, 'Logs'))
    #
    # Figures: Figures.Figures = property(my_partial(_dat_attr_prop, 'Figures'),
    #                                     my_partial(_dat_attr_set, 'Figures'),
    #                                     my_partial(_dat_attr_del, 'Figures'))
    #
    # Transition: T.Transition = property(my_partial(_dat_attr_prop, 'Transition'),
    #                                     my_partial(_dat_attr_set, 'Transition'),
    #                                     my_partial(_dat_attr_del, 'Transition'))
    # Entropy: E.Entropy = property(my_partial(_dat_attr_prop, 'Entropy'),
    #                               my_partial(_dat_attr_set, 'Entropy'),
    #                               my_partial(_dat_attr_del, 'Entropy'))
    # SquareEntropy: SE.SquareEntropy = property(my_partial(_dat_attr_prop, 'SquareEntropy'),
    #                                            my_partial(_dat_attr_set, 'SquareEntropy'),
    #                                            my_partial(_dat_attr_del, 'SquareEntropy'))
    # AWG: A.AWG = property(my_partial(_dat_attr_prop, 'AWG'),
    #                       my_partial(_dat_attr_set, 'AWG'),
    #                       my_partial(_dat_attr_del, 'AWG'))

    @property
    def Transition(self) -> T.Transition:
        return self._dat_attr_prop('Transition')

    @Transition.setter
    def Transition(self, value):
        self._dat_attr_set('Transition', value)

    @Transition.deleter
    def Transition(self):
        self._dat_attr_del('Transition')

    @property
    def Entropy(self) -> E.Entropy:
        return self._dat_attr_prop('Entropy')

    @Entropy.setter
    def Entropy(self, value):
        self._dat_attr_set('Entropy', value)

    @Entropy.deleter
    def Entropy(self):
        self._dat_attr_del('Entropy')

    @property
    def SquareEntropy(self) -> SE.SquareEntropy:
        return self._dat_attr_prop('SquareEntropy')

    @SquareEntropy.setter
    def SquareEntropy(self, value):
        self._dat_attr_set('SquareEntropy', value)

    @SquareEntropy.deleter
    def SquareEntropy(self):
        self._dat_attr_del('SquareEntropy')

    @property
    def AWG(self) -> A.AWG:
        return self._dat_attr_prop('AWG')

    @AWG.setter
    def AWG(self, value):
        self._dat_attr_set('AWG', value)

    @AWG.deleter
    def AWG(self):
        self._dat_attr_del('AWG')

    @property
    def ExpConfig(self):
        return self._dat_attr_prop('ExpConfig')

    @ExpConfig.setter
    def ExpConfig(self, value):
        self._dat_attr_set('ExpConfig', value)

    @ExpConfig.deleter
    def ExpConfig(self):
        self._dat_attr_del('ExpConfig')

    @property
    def Data(self) -> D.Data:
        return self._dat_attr_prop('Data')

    @Data.setter
    def Data(self, value):
        self._dat_attr_set('Data', value)

    @Data.deleter
    def Data(self):
        self._dat_attr_del('Data')

    @property
    def Logs(self) -> L.Logs:
        return self._dat_attr_prop('Logs')

    @Logs.setter
    def Logs(self, value):
        self._dat_attr_set('Logs', value)

    @Logs.deleter
    def Logs(self):
        self._dat_attr_del('Logs')

    @property
    def Figures(self) -> Figures.Figures:
        return self._dat_attr_prop('Figures')

    @Figures.setter
    def Figures(self, value):
        self._dat_attr_set('Figures', value)

    @Figures.deleter
    def Figures(self):
        self._dat_attr_del('Figures')

    def _threaded_manipulate_test(self):
        """Testing how multiple threads interact with object attributes"""
        import time
        import random
        with self.rlock:
            new_val = random.random()
            logger.debug(f'Old: {self._threaded_test_var}, Replace: {new_val}')
            self._threaded_test_var = new_val
            time.sleep(0.2)
            eq = new_val == self._threaded_test_var
            logger.debug(f'After sleeping: {self._threaded_test_var}.'
                         f' Current == Replaced? {eq}')
            return eq

    def _threaded_reentrant_test(self, i=0):
        """Testing how multiple threads interact with reentrant functions which manipulate attrs"""
        import time
        with self.rlock:
            self._threaded_test_var = i
            if i < 3:
                self._threaded_test_var = self._threaded_reentrant_test(i + 1)
            time.sleep(0.2)
            return self._threaded_test_var

    @with_hdf_read
    def _threaded_read_test(self):
        """Testing how multiple threads interact with reading from HDFs"""
        import time
        hdf = self.hdf.hdf
        time.sleep(0.2)
        stored_test_var = hdf.attrs.get('threading_test_var', None)
        logger.debug(f'Read: {stored_test_var}')
        return stored_test_var

    @with_hdf_write
    def _threaded_write_test(self, value=0):
        """Testing how multiple threads interact with writing to HDF"""
        import time
        hdf = self.hdf.hdf
        time.sleep(0.2)
        logger.debug(f'Storing: {value}')
        hdf.attrs['threading_test_var'] = value
        return value

    @with_hdf_write
    def _write_test(self):
        """Test write to HDF"""
        x = self.hdf.hdf.attrs.get('test_var', -1)
        self.hdf.hdf.attrs['test_var'] = x+1
        return x+1

    @with_hdf_read
    def _read_test(self):
        """Test read to HDF"""
        return self.hdf.hdf.attrs.get('test_var', -1)

    @with_hdf_read
    def _write_inside_read_test(self):
        """Testing that switching to write mode from read mode isn't a problem"""
        before = self.hdf.hdf.attrs.get('test_var', -1)
        self._write_test()
        after = self.hdf.hdf.attrs.get('test_var', -1)
        return before, after

    @with_hdf_write
    def _read_inside_write_test(self):
        """Testing that switching to read mode from write mode isn't a problem"""
        before = self._read_test()
        self.hdf.hdf.attrs['test_var'] = before+1
        after = self._read_test()
        return before, after

    def __repr__(self):
        return f'{self.dat_id}: Initialized at {self.date_initialized}'


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
            raise FileExistsError(
                f'Existing DatHDF at {os.path.abspath(path)} needs to be deleted before building a new HDF')
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
            date_initialized=datetime.datetime.now(),
        )
        return attrs



if __name__ == '__main__':
    DAT_ATTR_DICT.get('expconfig')


def get_dat_id(datnum, datname):
    """Returns unique dat_id within one experiment."""
    name = f'Dat{datnum}'
    if datname != 'base':
        name += f'[{datname}]'
    return name