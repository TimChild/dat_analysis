from src.Core import *
import src.config as cfg

################# Connected Instruments #######################
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


def TEMPERATUREmeta(instrid: int):  # instrid just so it is same as others
    class TEMPtuple(NamedTuple):
        mc: float
        still: float
        mag: float
        fourk: float

    return 'ls370', TEMPtuple


###############################################################


def make_dat_standard(datnum, datname:str = 'base', dfoption: str = 'sync', type: Union[str, List[str]] = None, dfname: str = None) -> Dat:
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
        return Dat(datnum, None, 'load', dfname=dfname)

    # region Pulling basic info from hdf5
    hdf = open_hdf5(datnum, path=cfg.ddir)
    keylist = [key for key in hdf.keys()]

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
    if type is None:  # Will return basic dat only
        infodict = infodict

    # Pull type specific information from hdf5
    if type is None:
        type = 'none'  # So if statements below will work
    if 'isense' in type:
        if 'FastScanCh0_2D' in keylist:
            i_sense = hdf['FastScanCh0_2D'][:]
        elif 'FastScan2D' in keylist:
            i_sense = hdf['FastScan2D'][:]
        elif 'fd_0adc' in keylist:
            i_sense = hdf['fd_0adc'][:]
        else:
            i_sense = None
        infodict += {'i_sense': i_sense}

    if 'entropy' in type:  # FIXME: Need to fill this in
        entx = hdf['FastScan...']
        enty = hdf['FastScan...']
        infodict += {'entx': entx, 'enty': enty}
    return datfactory(datnum, datname, dfname, dfoption, infodict)

  


if __name__ == '__main__':
    dat = make_dat_standard(2700)
    df = DatDF()
    # Dat(2700, None, {'test': 1})
    