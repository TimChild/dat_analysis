from typing import Union, NamedTuple
import src.Configs.Main_Config as cfg
import src.CoreUtil as CU
from src.CoreUtil import data_to_NamedTuple


class DatAttribute(object):
    pass


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
            ntuple = data_to_NamedTuple(instrinfo, instr_tuple)
            if cfg.warning is not None:
                print(f'WARNING: For {instrname} - {cfg.warning}')
        except (TypeError, KeyError):
            # region Verbose  get_instr_vals
            if cfg.verbose is True:
                CU.verbose_message(f'No {instr} found')
            # endregion
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
    gpib: int
    field: float
    ramprate: float


class TEMPtuple(NamedTuple):
    mc: float
    still: float
    mag: float
    fourk: float
    fiftyk: float


def DF_do_no_search():
    pass