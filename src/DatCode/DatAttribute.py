from typing import Union, NamedTuple
import src.Configs.Main_Config as cfg
import src.CoreUtil as CU
from src.CoreUtil import data_to_NamedTuple
import abc
import datetime
import h5py
import logging

logger = logging.getLogger(__name__)


class HDFGroupMeta(object):
    """Info that will be stored in HDF_meta for a DatAttribute group"""
    name = None
    date = datetime.datetime.now()
    version = None


class HDFDatasetMeta(object):
    """Info that will be stored in HDF_meta for a Dataset in a DatAttribute"""
    name = None
    date = datetime.datetime.now()
    version = None


class DatAttribute(abc.ABC):
    version = 'NEED TO OVERRIDE'
    group_name = 'NEED TO OVERRIDE'

    def __init__(self, hdf):
        self.hdf = hdf
        self.group = self.get_group()
        self._set_default_group_attrs()

    def get_group(self):
        group_name = self.__class__.group_name
        if group_name not in self.hdf.keys():
            self.hdf.create_group(group_name)
        group = self.hdf[group_name]  # type: h5py.Group
        return group

    @abc.abstractmethod
    def _set_default_group_attrs(self):
        """Set default attributes of group if not already existing
        e.g. upon creation of new dat, add description of Entropy group in attrs"""
        if not hasattr(self.group_name, 'version'):
            self.group.attrs['version'] = self.__class__.version

    @abc.abstractmethod
    def get_DF_dict(self):
        """Should be able to return a nice dictionary/df? of summary info to put in DF"""

    # @abc.abstractmethod
    def get_HDF_attrs(self):
        """HDF meta data to store with DatAttribute HDF group"""
        # TODO: Need to look at HDF group class etc to see if I can just use that directly. Would make more sense...
        meta = HDFGroupMeta()  # standard format
        meta.name = None
        meta.version = None  # set attrs
        return meta

    # @abc.abstractmethod
    def get_HDF_groups(self):
        """Get Groups which should be in top level of DatAttribute"""
        return None

    # @abc.abstractmethod
    def get_HDF_datasets(self):
        """Get Datasets which should be stored in top level of DatAttribute"""
        meta = HDFDatasetMeta()
        meta.name = None
        meta.version = None






##################################

def get_instr_vals(instr: str, instrid: Union[int, str, None], infodict) -> NamedTuple:
    instrname, instr_tuple = get_key_ntuple(instr, instrid)
    logs = infodict.get('Logs', None)
    if logs is not None:
        try:
            if instrname in logs.keys():
                instrinfo = logs[instrname]
            elif instr+'s' in logs.keys() and logs[instr+'s'] is not None and instrname in logs[instr+'s'].keys():
                instrinfo = logs[instr+'s'][instrname]
            else:
                return None
            if instrinfo is not None:
                ntuple = data_to_NamedTuple(instrinfo, instr_tuple)  # Will leave warning in cfg.warning if necessary
            else:
                return None
            if cfg.warning is not None:
                logger.warning(f'For {instrname} - {cfg.warning}')
        except (TypeError, KeyError):
            logger.info(f'No {instr} found')
            return None
        return ntuple
    return None


def get_key_ntuple(instrname: str, instrid: Union[str, int] = None) -> [str, NamedTuple]:
    """Returns instrument key and namedtuple for that instrument"""
    instrtupledict = {'srs': SRStuple, 'mag': MAGtuple, 'temperatures': TEMPtuple}
    if instrname not in instrtupledict.keys():
        raise KeyError(f'No {instrname} in instruments.py')
    else:
        if instrid is None:
            instrid = ''
        instrkey = instrname + str(instrid)
    return instrkey, instrtupledict[instrname]


#  name in Logs dict has to be exactly the same as NamedTuple attr names
class SRStuple(NamedTuple):
    gpib: int
    out: int
    tc: float
    freq: float
    phase: float
    sens: float
    harm: int
    CH1readout: int


class MAGtuple(NamedTuple):
    field: float
    rate: float


class TEMPtuple(NamedTuple):
    mc: float
    still: float
    mag: float
    fourk: float
    fiftyk: float

