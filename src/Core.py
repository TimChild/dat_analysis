"""Core of PyDatAnalysis. This should remain unchanged between experiments in general, or be backwards compatible"""

import json
import os
import pickle
import re
from typing import Union, List

import h5py
import pandas as pd
import src.Configs.Main_Config as cfg
from src import CoreUtil as CU
from src.Configs import Main_Config as cfg, Jan20Config as ES
from src.CoreUtil import verbose_message
from src.DFcode.SetupDF import SetupDF
from src.DatCode.Dat import Dat
from src.DFcode.DatDF import DatDF, _dat_exists_in_df
import src.DFcode.DFutil as DU
import src.CoreUtil as CU
import src.DFcode.DatDF as DF


################# Sweeplog fixes ##############################


def metadata_to_JSON(data: str) -> dict:
    jsonsubs = cfg.jsonsubs  # Get experiment specific json subs from config
    if jsonsubs is not None:
        for pattern_repl in jsonsubs:
            data = re.sub(pattern_repl[0], pattern_repl[1], data)
    try:
        jsondata = json.loads(data)
    except json.decoder.JSONDecodeError as e:
        print(data)
        raise e
    return jsondata


def datfactory(datnum, datname, dfname, dfoption, infodict=None):
    datdf = DatDF(dfname=dfname)  # Load DF
    datcreator = _creator(dfoption)  # get creator of Dat instance based on df option
    datinst = datcreator(datnum, datname, datdf, infodict)  # get an instance of dat using the creator
    return datinst  # Return that to caller


def _creator(dfoption):
    if dfoption in ['load', 'load_pickle']:
        return _load_pickle
    if dfoption in ['load_df']:
        return _load_df
    elif dfoption == 'sync':
        return _sync
    elif dfoption == 'overwrite':
        return _overwrite
    else:
        raise ValueError("dfoption must be one of: load, sync, overwrite, load_df")


def _load_pickle(datnum: int, datname, datdf, infodict=None):
    exists = DF._dat_exists_in_df(datnum, datname, datdf)
    if exists is True:
        datpicklepath = datdf.get_path(datnum, datname=datname)

        if os.path.isfile(datpicklepath) is False:
            inp = input(
                f'Pickle for dat{datnum}[{datname}] doesn\'t exist in "{datpicklepath}", would you like to load using DF[{datdf.name}]?')
            if inp in ['y', 'yes']:
                return _load_df(datnum, datname, datdf, infodict)
            else:
                raise FileNotFoundError(f'Pickle file for dat{datnum}[{datname}] doesn\'t exist')
        with open(datpicklepath, 'rb') as f:  # TODO: Check file exists
            inst = pickle.load(f)
        return inst
    else:
        return None


def _load_df(datnum: int, datname: str, datdf, *args):
    # TODO: make infodict from datDF then run overwrite with same info to recreate Datpickle
    DF._dat_exists_in_df(datnum, datname, datdf)
    infodict = datdf.infodict(datnum, datname)
    inst = _overwrite(datnum, datname, datdf, infodict)
    return inst


def _sync(datnum, datname, datdf, infodict):
    if (datnum, datname) in datdf.df.index:
        ans = CU.option_input(f'Dat{datnum}[{datname}] already exists, do you want to \'load\' or \'overwrite\'', {'load': 'load', 'overwrite': 'overwrite'})
        if ans == 'load':
            if pd.isna(DU.get_single_value_pd(datdf.df, (datnum, datname), ('picklepath',))) is not True:
                inst = _load_pickle(datnum, datname, datdf)
            else:
                raise NotImplementedError('Not implemented loading from DF yet, infodict from DF needs work first')
                inst = _load_df(datnum, datname,
                                datdf)  # FIXME: Need a better way to get infodict from datDF for this to work
        elif ans == 'overwrite':
            inst = _overwrite(datnum, datname, datdf, infodict)
        else:
            raise ValueError('Must choose either \'load\' or \'overwrite\'')
    else:
        inst = _overwrite(datnum, datname, datdf, infodict)
    return inst


def _overwrite(datnum, datname, datdf, infodict):
    inst = Dat(datnum, datname, infodict, dfname=datdf.name)
    return inst


def load_dats(autosave = False, dfname: str = 'default', datname: str = 'base', dattypes: Union[str, List[str]] = None):
    """For loading a batch of new dats into DataFrame"""
    datdf = DF.DatDF()
    setupdf = SetupDF()
    datdf.sort_indexes()
    dfdatnums = [x[0] for x in datdf.df.index]
    ddir = cfg.ddir
    h5datnums = [int(n[3:-3]) for n in os.listdir(ddir) if n[0:3] == 'dat']
    print(f'Going to try to add datnums: {set(h5datnums) - set(dfdatnums)}')
    for datnum in set(h5datnums) - set(dfdatnums):  # For all datnums that aren't in df yet
        print(f'Adding dat{datnum}')
        setupdata = setupdf.get_valid_row(datnum)
        if setupdata['junk'] == True:  # Looks for whether Junk is marked True in setupdf
            datdf.df.loc[(datnum, 'junk'), ('junk', '.')] = True
            continue
        try:
            dat = make_dat_standard(datnum, datname=datname, dfoption='sync', dattypes=dattypes, dfname=dfname)
            datdf.update_dat(dat)
        except OSError as e:
            datdf.df.loc[(datnum, 'FileError'), ('junk', '.')] = True
    if autosave is True:
        datdf.save()
    return datdf


def make_dat_standard(datnum, datname: str = 'base', dfoption: str = 'sync', dattypes: Union[str, List[str]] = None,
                      dfname: str = None) -> Dat:
    """Loads or creates dat object. Ideally this is the part that changes between experiments"""

    if dfoption == 'load':  # If only trying to load dat then skip everything else and send instruction to load from DF
        return datfactory(datnum, datname, dfname, 'load')

    # region Pulling basic info from hdf5
    hdfpath = os.path.join(cfg.ddir, f'dat{datnum:d}.h5')
    if os.path.isfile(hdfpath):
        hdf = h5py.File(hdfpath, 'r')
    else:
        print(f'No hdf5 file found for dat{datnum:d} in directory {cfg.ddir}')
        return None  # Return None if no data exists
    sweeplogs = hdf['metadata'].attrs['sweep_logs']  # Not JSON data yet
    sweeplogs = metadata_to_JSON(sweeplogs)

    sc_config = hdf['metadata'].attrs['sc_config']  # Not JSON data yet
    # sc_config = metadata_to_JSON(sc_config) # FIXME: Something is wrong with sc_config metadata...
    sc_config = {'Need to fix sc_config metadata': 'Need to fix sc_config metadata'}

    xarray = hdf['x_array'][:]
    x_label = sweeplogs['axis_labels']['x']
    y_label = sweeplogs['axis_labels']['y']
    try:
        yarray = hdf['y_array'][:]
        dim = 2
    except KeyError:
        yarray = None
        dim = 1

    temperatures = _temp_from_json(sweeplogs, fridge=ES.instruments['fridge'])  # fridge is just a placeholder for now
    srss = {'srs' + str(i): _srs_from_json(sweeplogs, i, srs_type=ES.instruments['srs']) for i in range(1, ES.instrument_num['srs'] + 1)}
    # mags = [get_instr_vals('MAG', direction) for direction in ['x', 'y', 'z']]
    mags = None  # TODO: Need to fix how the json works with Magnets first
    # endregion

    dacs = {int(key[2:]): sweeplogs['BabyDAC'][key] for key in sweeplogs['BabyDAC'] if key[-4:] not in ['name', 'port']}

    # make dict of dac values from sweeplogs for keys that are for dac values and not dacnames. Dict is {0: val,
    # 1: val...}
    dacnames = {int(key[2:-4]): sweeplogs['BabyDAC'][key] for key in sweeplogs['BabyDAC'] if key[-4:] == 'name'}


    try:
        fdacs = {int(key[2:]): sweeplogs['FastDAC'][key] for key in sweeplogs['FastDAC'] if key[-4:] not in ['name', 'Keys', 'Freq']}
        fdacnames = {int(key[2:-4]): sweeplogs['FastDAC'][key] for key in sweeplogs['FastDAC'] if key[-4:] == 'name'}
        fdacfreq = sweeplogs['FastDAC']['SamplingFreq']
    except KeyError as e:  # No fastdacs connected
        print(e)
        fdacs = None
        fdacnames = None
        fdacfreq = None
        pass


    # Make dict of dacnames from sweeplogs for keys that end in 'name'. Dict is {0: name, 1: name... }
    time_elapsed = sweeplogs['time_elapsed']
    time_completed = sweeplogs['time_completed']  # TODO: make into datetime format here

    if type(dattypes) != list and dattypes is not None:
        dattypes = {dattypes}
    elif dattypes is None:
        dattypes = {'none_given'}

    if 'comment' in sweeplogs.keys():  # Adds dattypes from comment stored in hdf
        sk = [item.strip() for item in sweeplogs['comment'].split(',')]
        dt = ES.dat_types_list
        for key in list(set(sk) & set(dt)):
            dattypes.add(key)
            sk.remove(key)
        comments = ','.join(sk)  # Save the remaining info in self.Logs.comments
    else:
        comments = None

    infodict = CU.add_infodict_Logs(None, xarray, yarray, x_label, y_label, dim, srss, mags, temperatures, time_elapsed,
                                 time_completed, dacs, dacnames, fdacs, fdacnames, fdacfreq, comments)
    infodict['hdfpath'] = hdfpath
    if dattypes is None:  # Will return basic dat only
        dattypes = {'none_given'}
        infodict = infodict

        for key in ES.dat_types_list:
            if key in [val.strip() for val in sweeplogs['comment'].split(',')]:
                dattypes.add(key)

    infodict['dattypes'] = dattypes

    if {'i_sense', 'transition', 'entropy', 'dcbias'} & set(dattypes):  # If there is overlap between lists then...
        i_sense = _get_corrected_data(datnum, ES.i_sense_keys, hdf)
        infodict['i_sense'] = i_sense

    if 'entropy' in dattypes:
        entx = _get_corrected_data(datnum, ES.entropy_x_keys, hdf)*infodict['Logs']['srss']['srs3']['sens']/10*1e-3*1e-8*1e9  # /10 because 10V range of output, 1e-3 to go to V, 1e-8 because of Current amp, 1e9 to go to nA
        enty = _get_corrected_data(datnum, ES.entropy_y_keys, hdf)*infodict['Logs']['srss']['srs3']['sens']/10*1e-3*1e-8*1e9  # /10 because 10V range of output, 1e-3 to go to V, 1e-8 because of Current amp, 1e9 to go to nA
        infodict['entx'] = entx
        infodict['enty'] = enty
        dattypes.add('transition')

    if 'dcbias' in dattypes:
        dattypes.add('transition')

    current_keys = ['current']
    conductance_keys = ['']
    if 'pinch' in dattypes:
        # TODO: Finish this, had to give up due to lack of time 27/01/2020
        current = _get_corrected_data(datnum, current_keys, hdf)
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


def _get_corrected_data(datnum, wavenames: Union[List, str], hdf: h5py.File):
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


def _temp_from_json(jsondict, fridge='ls370'):
    if 'BF Small' in jsondict.keys():
        return _temp_from_bfsmall(jsondict['BF Small'])
    else:
        # region Verbose  temp_from_json
        if cfg.verbose is True:
            verbose_message(f'Verbose[][temp_from_json] - Did not find "BF Small" in json')
        # endregion
    return None


def _srs_from_json(jsondict, id, srs_type='srs830'):
    if 'SRS_' + str(id) in jsondict.keys():
        srsdict = jsondict['SRS_' + str(id)]
        srsdata = {'gpib': srsdict['gpib_address'],
                   'out': srsdict['amplitude V'],
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


def _temp_from_bfsmall(tempdict):
    tempdata = {'mc': tempdict['MC K'],
                'still': tempdict['Still K'],
                'fourk': tempdict['4K Plate K'],
                'mag': tempdict['Magnet K'],
                'fiftyk': tempdict['50K Plate K']}
    return tempdata
