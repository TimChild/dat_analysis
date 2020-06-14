from typing import Union, NamedTuple
import src.Configs.Main_Config as cfg
import src.CoreUtil as CU
from src.CoreUtil import data_to_NamedTuple
import abc
import datetime
import h5py
import logging

logger = logging.getLogger(__name__)


class DatAttribute(abc.ABC):
    version = 'NEED TO OVERRIDE'
    group_name = 'NEED TO OVERRIDE'

    def __init__(self, hdf):
        self.version = self.__class__.version
        self.hdf = hdf
        self.group = self.get_group()
        self._set_default_group_attrs()


    def get_group(self):
        """Sets self.group to be the appropriate group in HDF for given DatAttr
        based on the class.group_name which should be overridden.
        Will create group in HDF if necessary"""
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
    def get_from_HDF(self):
        """Should be able to run this to get all data from HDF into expected attrs of DatAttr"""
        pass
    # @abc.abstractmethod
    # def get_DF_dict(self):
    #     """Should be able to return a nice dictionary/df? of summary info to put in DF"""




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
        raise KeyError(f'No {instrname} found')
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

