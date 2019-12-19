from src.Core import *
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


def TEMPERATUREmeta():
    class TEMPtuple(NamedTuple):
        mc: float
        still: float
        mag: float
        fourk: float

    return 'ls370', TEMPtuple


###############################################################


def make_dat_standard(datnum) -> Dat:
    """Loads or creates dat object. Ideally this is the part that changes between experiments"""

    # TODO: Put in load check here to load from pickle
    # TODO: Check dict or something for whether datnum needs scaling differently (i.e. 1e8 on current amp)

    def get_instr_vals(instr: str, instrid: Union[int, str, None]) -> NamedTuple:
        instrname, instr_tuple = globals()[f'{instr.upper()}meta'](
            instrid)  # e.g. SRSmeta(1) to get NamedTuple for SRS_1
        try:
            keys = list(sweeplogs[instrname].keys())  # first is gbip
            ntuple = instr_tuple([sweeplogs[instr][key] for key in keys])
        except:
            if verbose is True: print(f'No {instr} found')
            return None
        return ntuple

    hdf = open_hdf5(datnum, path=ddir)
    keylist = [key for key in hdf.keys()]

    sweeplogs = hdf['metadata'].attrs['sweep_logs']  # Not JSON data yet
    sweeplogs = metadata_to_JSON(sweeplogs)

    sc_config = hdf['metadata'].attrs['sc_config']  # Not JSON data yet
    sc_config = metadata_to_JSON(sc_config)
    ######################## Might want to remove this from standard Dat
    if 'FastScanCh0_2D' in keylist:
        i_sense = hdf['FastScanCh0_2D'][:]
    elif 'FastScan2D' in keylist:
        i_sense = hdf['FastScan2D'][:]
    elif 'fd_0adc' in keylist:
        i_sense = hdf['fd_0adc'][:]
    else:
        i_sense = None
    ##################################################
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

    return Dat(datnum, xarray, yarray, dim, sweeplogs, sc_config, i_sense, srss, mags, temperatures)