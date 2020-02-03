from src.Core import *
import src.Configs.Main_Config as cfg
from src.CoreUtil import verbose_message, add_infodict_Logs
from src.DatCode.Dat import Dat
from typing import List, Union
import src.Configs.Jan20Config as ES

###############################################################


def make_dat_standard(datnum, datname: str = 'base', dfoption: str = 'sync', dattypes: Union[str, List[str]] = None,
                      dfname: str = None) -> Dat:
    """Loads or creates dat object. Ideally this is the part that changes between experiments"""

    if dfoption == 'load':  # If only trying to load dat then skip everything else and send instruction to load from DF
        return datfactory(datnum, datname, dfname, 'load')

    # region Pulling basic info from hdf5
    hdfpath = os.path.join(cfg.ddir, f'dat{datnum:d}.h5')
    hdf = h5py.File(hdfpath, 'r')

    sweeplogs = hdf['metadata'].attrs['sweep_logs']  # Not JSON data yet
    sweeplogs = metadata_to_JSON(sweeplogs)

    sc_config = hdf['metadata'].attrs['sc_config']  # Not JSON data yet
    # sc_config = metadata_to_JSON(sc_config) # FIXME: Something is wrong with sc_config metadata...
    sc_config = {'Need to fix sc_config metadata': 'Need to fix sc_config metadata'}

    xarray = hdf['x_array'][:]
    xlabel = sweeplogs['axis_labels']['x']
    ylabel = sweeplogs['axis_labels']['y']
    try:
        yarray = hdf['y_array'][:]
        dim = 2
    except KeyError:
        yarray = None
        dim = 1

    temperatures = temp_from_json(sweeplogs, fridge=ES.instruments['fridge'])  # fridge is just a placeholder for now
    srss = {'srs' + str(i): srs_from_json(sweeplogs, i, srs_type=ES.instruments['srs']) for i in range(1, ES.instrument_num['srs'] + 1)}
    # mags = [get_instr_vals('MAG', direction) for direction in ['x', 'y', 'z']]
    mags = None  # TODO: Need to fix how the json works with Magnets first
    # endregion

    dacs = {int(key[2:]): sweeplogs['BabyDAC'][key] for key in sweeplogs['BabyDAC'] if key[-4:] not in ['name', 'port']}
    # make dict of dac values from sweeplogs for keys that are for dac values and not dacnames. Dict is {0: val,
    # 1: val...}
    dacnames = {int(key[2:-4]): sweeplogs['BabyDAC'][key] for key in sweeplogs['BabyDAC'] if key[-4:] == 'name'}
    # Make dict of dacnames from sweeplogs for keys that end in 'name'. Dict is {0: name, 1: name... }
    time_elapsed = sweeplogs['time_elapsed']
    time_completed = sweeplogs['time_completed']
    infodict = add_infodict_Logs(None, xarray, yarray, xlabel, ylabel, dim, srss, mags, temperatures, time_elapsed,
                                 time_completed, dacs, dacnames)
    infodict['hdfpath'] = hdfpath
    if dattypes is None:  # Will return basic dat only
        dattypes = ['none']
        infodict = infodict

    if type(dattypes) != list:
        dattypes = [dattypes]

    if 'comment' in sweeplogs.keys():  # Adds dattypes from comment stored in hdf
        for key in ES.dat_types_list:
            if key in [val.strip() for val in sweeplogs['comment'].split(',')]:
                dattypes += key

    infodict['dattypes'] = dattypes

    if ('i_sense' or 'transition') in dattypes:
        i_sense = get_corrected_data(datnum, ES.i_sense_keys, hdf)
        infodict['i_sense'] = i_sense

    if 'entropy' in dattypes:
        entx = get_corrected_data(datnum, ES.entropy_x_keys, hdf)
        enty = get_corrected_data(datnum, ES.entropy_y_keys, hdf)
        infodict['entx'] = entx
        infodict['enty'] = enty

    current_keys = ['']
    conductance_keys = ['']
    if 'pinch' in dattypes:
        # TODO: Finish this, had to give up due to lack of time 27/01/2020
        current = get_corrected_data(datnum, current_keys, hdf)
        bias = 10e-6  # voltage bias of 10uV # TODO: This shouldn't be fixed
        # conductance = current / bias
        infodict['current'] = current
        # infodict['conductance'] = conductance
        # TODO: Think about how to make this conductance in general

    if ('current' or 'pinch') in dattypes:
        # TODO: Think about how to collect all the right data... Maybe should look at wave names in hdf by default?
        pass

    if 'dot_tuning' in dattypes:
        # TODO: Do this
        pass

    return datfactory(datnum, datname, dfname, dfoption, infodict)


import h5py
from src.DFcode.SetupDF import SetupDF


def get_corrected_data(datnum, wavenames: Union[List, str], hdf: h5py.File):
    if type(wavenames) != list:
        wavenames = [wavenames]
    setupdf = SetupDF()
    setupdata = setupdf.get_valid_row(datnum)
    data = None
    for name in wavenames:
        if name in hdf.keys():
            data = hdf[name][:]
            if name in setupdata.keys() and setupdata[name] is not None:
                data = data * setupdata[name]  # Multiplier stored in setupdata
    return data


def temp_from_json(jsondict, fridge='ls370'):
    if 'BF Small' in jsondict.keys():
        return temp_from_bfsmall(jsondict['BF Small'])
    else:
        # region Verbose  temp_from_json
        if cfg.verbose is True:
            verbose_message(f'Verbose[][temp_from_json] - Did not find "BF Small" in json')
        # endregion
    return None


def srs_from_json(jsondict, id, srs_type='srs830'):
    if 'SRS_' + str(id) in jsondict.keys():
        srsdict = jsondict['SRS_' + str(id)]
        srsdata = {'gpib': srsdict['gpib_address'],
                   'amp': srsdict['amplitude V'],
                   'tc': srsdict['time_const ms'],
                   'freq': srsdict['frequency Hz'],
                   'phase': srsdict['phase deg'],
                   'sens': srsdict['sensitivity V'],
                   'harm': srsdict['harmonic'],
                   # 'CH1readout': srsdict['CH1readout'],
                   }
    else:
        srsdata = None
    return srsdata


def temp_from_bfsmall(tempdict):
    tempdata = {'mc': tempdict['MC K'],
                'still': tempdict['Still K'],
                'fourk': tempdict['4K Plate K'],
                'mag': tempdict['Magnet K'],
                'fiftyk': tempdict['50K Plate K']}
    return tempdata



