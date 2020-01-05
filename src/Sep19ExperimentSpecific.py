from src.Core import *
import src.config as cfg

################# Connected Instruments #######################
from src.CoreUtil import verbose_message, make_basicinfodict, open_hdf5
from src.Dat.Dat import Dat
from src.DFcode.DatDF import DatDF
from typing import List, Union, NamedTuple

instruments = ['SRS']


def SRSmeta(instrid: int):
    class SRStuple(NamedTuple):
        gpib: int
        out: int
        tc: float
        freq: float
        phase: float
        sens: float
        harm: int
        CH1readout: int

    instrname = f'SRS_{instrid:d}'
    return instrname, SRStuple


def TEMPERATUREmeta(*args):  # just so it can be passed info like others
    class TEMPtuple(NamedTuple):
        mc: float
        still: float
        mag: float
        fourk: float

    return 'ls370', TEMPtuple


###############################################################


def make_dat_standard(datnum, datname:str = 'base', dfoption: str = 'sync', dattypes: Union[str, List[str]] = None, dfname: str = None) -> Dat:
    """Loads or creates dat object. Ideally this is the part that changes between experiments"""

    # TODO: Check dict or something for whether datnum needs scaling differently (i.e. 1e8 on current amp)

    def get_instr_vals(instr: str, instrid: Union[int, str, None]) -> NamedTuple:
        instrname, instr_tuple = globals()[f'{instr.upper()}meta'](
            instrid)  # e.g. SRSmeta(1) to get NamedTuple for SRS_1
        try:
            keys = list(sweeplogs[instrname].keys())  # first is gbip
            ntuple = instr_tuple([sweeplogs[instrname][key] for key in keys])
        except (TypeError, KeyError):
            # region Verbose  get_instr_vals
            if cfg.verbose is True:
                verbose_message(f'No {instr} found')
            # endregion
            return None
        return ntuple

    if dfoption == 'load':  # If only trying to load dat then skip everything else and send instruction to load from DF
        return datfactory(datnum, datname, dfname, 'load')

    # region Pulling basic info from hdf5
    hdf = open_hdf5(datnum, path=cfg.ddir)

    sweeplogs = hdf['metadata'].attrs['sweep_logs']  # Not JSON data yet
    sweeplogs = metadata_to_JSON(sweeplogs)

    sc_config = hdf['metadata'].attrs['sc_config']  # Not JSON data yet
    # sc_config = metadata_to_JSON(sc_config) # FIXME: Something is wrong with sc_config metadata...
    sc_config = {'Need to fix sc_config metadata': 'Need to fix sc_config metadata'}

    xarray = hdf['x_array'][:]
    try:
        yarray = hdf['y_array'][:]
        dim = 2
    except KeyError:
        yarray = None
        dim = 1

    temperatures = get_instr_vals('temperature', None)
    srss = [get_instr_vals('SRS', num) for num in [1, 2, 3, 4]]
    # mags = [get_instr_vals('MAG', direction) for direction in ['x', 'y', 'z']]
    mags = None  # TODO: Need to fix how the json works with Magnets first
    # endregion
    infodict = make_basicinfodict(xarray, yarray, dim, sweeplogs, sc_config, srss, mags, temperatures)
    if dattypes is None:  # Will return basic dat only
        infodict = infodict
        dattypes = 'none'

    i_sense_keys = ['FastScanCh0_2D', 'FastScan2D', 'fd_0adc']
    if 'isense' in dattypes:
        i_sense = get_corrected_data(datnum, i_sense_keys, hdf)
        infodict['i_sense'] = i_sense

    entropy_x_keys = ['FastScanCh1_2D', 'fd_1adc']
    entropy_y_keys = ['FastScanCh2_2D', 'fd_2adc']
    if 'entropy' in dattypes:  # FIXME: Need to fill this in
        entx = get_corrected_data(datnum, entropy_x_keys, hdf)
        enty = get_corrected_data(datnum, entropy_y_keys, hdf)
        infodict['entx'] = entx
        infodict['enty'] = enty

    return datfactory(datnum, datname, dfname, dfoption, infodict)


import h5py
from src.DFcode.SetupDF import SetupDF


def get_corrected_data(datnum, wavenames: Union[List, str], hdf:h5py.File):
    if type(wavenames) != list:
        wavenames = [wavenames]
    setupdf = SetupDF()
    setupdata = setupdf.get_valid_row(datnum)
    data = None
    for name in wavenames:
        if name in hdf.keys():
            data = hdf[name][:]
            if name in setupdata.keys() and setupdata[name] is not None:
                data = data*setupdata[name]  # Multiplier stored in setupdata
    return data


if __name__ == '__main__':
    dat = make_dat_standard(2700, dattypes='isense', dfname='testing')
    sf = SetupDF()
    df = DatDF(dfname='testing')
    df.add_dat(dat)

    