from typing import Union, NamedTuple
import src.Configs.Main_Config as cfg
import src.CoreUtil as CU


class DatAttribute(object):
    pass


def get_instr_vals(instr: str, instrid: Union[int, str, None], infodict) -> NamedTuple:
    instrname, instr_tuple = get_key_ntuple(instr, instrid)
    try:
        if instrid is None:
            instrinfo = infodict[instr]
        else:
            instrinfo = infodict[instr][instrname]
        tupledict = instr_tuple.__annotations__  # Get ordered dict of keys of namedtuple
        for key in tupledict.keys():  # Set all values to None so they will default to that if not entered
            tupledict[key] = None
        for key in set(instrinfo.keys()) & set(tupledict.keys()):  # Enter valid keys values
            tupledict[key] = instrinfo[key]
        if set(instrinfo.keys())-set(tupledict.keys()) is not None:
            print(f'WARNING: This data is not being stored for {instr}{instrid}: {set(instrinfo.keys())-set(tupledict.keys())}')
        ntuple = instr_tuple(tupledict.values())
    except (TypeError, KeyError):
        # region Verbose  get_instr_vals
        if cfg.verbose is True:
            CU.verbose_message(f'No {instr} found')
        # endregion
        return None
    return ntuple


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
