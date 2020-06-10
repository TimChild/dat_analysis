"""Core of PyDatAnalysis. This should remain unchanged between experiments in general, or be backwards compatible"""
from __future__ import annotations
import json
import os
import pickle
import re
from typing import Union, List, Set
import logging

import h5py
import pandas as pd
import src.Configs.Main_Config as cfg
from src.Configs import Main_Config as cfg
from src.CoreUtil import verbose_message, print_verbose
from src.DFcode.SetupDF import SetupDF
from src.DatCode.Dat import Dat
from src.DFcode.DatDF import DatDF, dat_exists_in_df
import src.DFcode.DFutil as DU
import src.CoreUtil as CU
import src.DFcode.DatDF as DF
import src.DFcode.SetupDF as SF

logger = logging.getLogger(__name__)

################# Sweeplog fixes ##############################


def metadata_to_JSON(data: str, config=None, datnum=None) -> dict:
    if config is None:
        config = cfg.current_config
    jsonsubs = config.json_subs  # Get experiment specific json subs from config
    if callable(jsonsubs):  # Cheeky way to be able to get datnum specific jsonsubs by returning a function in the first place
        jsonsubs = jsonsubs(datnum)
    if jsonsubs is not None:
        for pattern_repl in jsonsubs:
            data = re.sub(pattern_repl[0], pattern_repl[1], data)
    try:
        jsondata = json.loads(data)
    except json.decoder.JSONDecodeError as e:
        print(data)
        raise e
    return jsondata


def datfactory(datnum, datname, dat_df: DatDF, dfoption, infodict=None):
    if type(dat_df) == str:
        logger.warning(f'DEPRICATION WARNING: Should really pass in whole datdf here to be safe with different'
              f'configs etc. This is still being called: datdf = DatDF(dfname={dat_df}) for now.')
        datdf = DatDF(dfname=dat_df)  # Load DF
    elif isinstance(dat_df, DF.DatDF):
        datdf = dat_df
    else:
        raise ValueError(f'ERROR[C.datfactory]: [{dat_df}] is not a valid DF.DatDF')
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
    exists = DF.dat_exists_in_df(datnum, datname, datdf)
    if exists is True:
        datpicklepath = CU.get_full_path(datdf.get_path(datnum, datname=datname))
        if os.path.isfile(datpicklepath) is False:
            inp = input(
                f'Pickle for dat{datnum}[{datname}] doesn\'t exist in "{datpicklepath}", would you like to load using DF[{datdf.name}]?')
            if inp in ['y', 'yes']:
                return _load_df(datnum, datname, datdf, infodict)
            else:
                raise FileNotFoundError(f'Pickle file for dat{datnum}[{datname}] doesn\'t exist')
        with open(datpicklepath, 'rb') as f:
            inst = pickle.load(f)
        return inst
    else:
        return None


def _load_df(datnum: int, datname: str, datdf, *args):
    # TODO: make infodict from datDF then run overwrite with same info to recreate Datpickle
    DF.dat_exists_in_df(datnum, datname, datdf)
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
                # inst = _load_df(datnum, datname,
                #                 datdf)  # FIXME: Need a better way to get infodict from datDF for this to work
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


@CU.plan_to_remove  # 9/6
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
            datdf.update_dat(dat, yes_to_all=True)
        except OSError as e:
            datdf.df.loc[(datnum, 'FileError'), ('junk', '.')] = True
        except Exception as e:
            datdf.df.loc[(datnum, 'UnknownError'), ('junk', '.')] = True
            print(f'Could not add dat{datnum} due to error:\n{e}')
    if autosave is True:
        datdf.save()
    return datdf


def make_dat_standard(datnum, datname: str = 'base', dfoption: str = 'sync', dattypes: Union[str, List[str], Set[str]] = None,
                      dfname: str = None, datdf: DF.DatDF = None, setupdf: SetupDF = None, config = None) -> Dat:
    """
    Loads or creates dat object and interacts with Main_Config (and through that the Experiment specific configs)

    @param datnum: dat[datnum].h5
    @type datnum: int
    @param datname: name for storing in datdf and files
    @type datname: str
    @param dfoption: whether to 'load', or 'overwrite' files in datdf (or 'sync' will ask for input if necessary)
    @type dfoption: str
    @param dattypes: what types of info dat contains, e.g. 'transition', 'entropy', 'dcbias'
    @type dattypes: Union[str, List[str], Set[str]]
    @param dfname: DEPRICATED name of datdf to use for loading data
    @type dfname: str
    @param datdf: datdf to load dat from or overwrite to.
    @type datdf: DF.DatDF
    @param setupdf: setup df to use to get corrected data when loading dat in
    @type setupdf: SetupDF
    @param config: config file to use when loading dat in, will default to cfg.current_config. Otherwise pass the whole module in
    @type config: module
    @return: dat object
    @rtype: Dat
    """

    old_config = cfg.current_config
    if config is None:
        config = cfg.current_config
    else:
        cfg.set_all_for_config(config, folder_containing_experiment=None)

    if dfname is not None and datdf is None:
        print(f'DEPRICATION WARNING[C.make_dat_standard]: Should pass in full datdf in datdf=... instead of just dfname.'
              f'to avoid config issues. For now this will load datdf from DF.DatDF(dfname=dfname)')
        datdf = DF.DatDF(dfname=dfname)
    elif datdf is None:
        datdf = DF.DatDF()
    else:
        pass

    if setupdf is None:
        setupdf = SetupDF()


    if setupdf.config_name != datdf.config_name or setupdf.config_name != config.__name__.split('.')[-1]:
        raise AssertionError(f'C.make_dat_standard: setupdf, datdf and config have different config names'
                             f'[{setupdf.config_name, datdf.config_name, config.__name__.split(".")[-1]},'
                             f' probably you dont want to continue!')

    if dfoption == 'load':  # If only trying to load dat then skip everything else and send instruction to load from DF
        dat = datfactory(datnum, datname, datdf, 'load')
        cfg.set_all_for_config(old_config, folder_containing_experiment=None)
        return dat

    # region Pulling basic info from hdf5
    hdfpath = os.path.join(cfg.ddir, f'dat{datnum:d}.h5')
    if os.path.isfile(CU.get_full_path(hdfpath)):
        hdf = h5py.File(CU.get_full_path(hdfpath), 'r')
    else:
        print(f'No hdf5 file found for dat{datnum:d} in directory {cfg.ddir}')
        return None  # Return None if no data exists
    sweeplogs = hdf['metadata'].attrs['sweep_logs']  # Not JSON data yet
    sweeplogs = metadata_to_JSON(sweeplogs, config=config, datnum=datnum)



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

    temperatures = _temp_from_json(sweeplogs, fridge=config.instruments['fridge'])  # fridge is just a placeholder for now
    srss = {'srs' + str(i): _srs_from_json(sweeplogs, i, srs_type=config.instruments['srs']) for i in range(1, config.instrument_num['srs'] + 1)}
    # mags = [get_instr_vals('MAG', direction) for direction in ['x', 'y', 'z']]
    mags = {'mag' + id: _mag_from_json(sweeplogs, id, mag_type=config.instruments['magnet']) for id in ['x', 'y', 'z']}
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
        print(f'Missing key [{e}] in sweep_logs')
        fdacs = None
        fdacnames = None
        fdacfreq = None
        pass


    # Make dict of dacnames from sweeplogs for keys that end in 'name'. Dict is {0: name, 1: name... }
    time_elapsed = sweeplogs['time_elapsed']
    time_completed = sweeplogs['time_completed']  # TODO: make into datetime format here
    if dattypes is not None:
        dattypes = CU.ensure_set(dattypes)
    elif dattypes is None:
        dattypes = {'none_given'}

    if 'comment' in sweeplogs.keys():  # Adds dattypes from comment stored in hdf
        sk = [item.strip() for item in sweeplogs['comment'].split(',')]
        dt = config.dat_types_list
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

        for key in config.dat_types_list:
            if key in [val.strip() for val in sweeplogs['comment'].split(',')]:
                dattypes.add(key)

    infodict['dattypes'] = dattypes

    if {'i_sense', 'transition', 'entropy', 'dcbias'} & set(dattypes):  # If there is overlap between lists then...
        i_sense = _get_corrected_data(datnum, config.i_sense_keys, hdf, setupdf)
        infodict['i_sense'] = i_sense

    if 'entropy' in dattypes:
        entx = _get_corrected_data(datnum, config.entropy_x_keys, hdf, setupdf)
        enty = _get_corrected_data(datnum, config.entropy_y_keys, hdf, setupdf)

        current_amplification = _get_value_from_setupdf(datnum, 'ca0amp', setupdf)
        srs = _get_value_from_setupdf(datnum, 'entropy_srs', setupdf)
        if srs[:3] == 'srs':
            multiplier = infodict['Logs']['srss'][srs][
             'sens'] / 10 * 1e-3 / current_amplification * 1e9  # /10 because 10V range of output, 1e-3 to go to V, 1e9 to go to nA
        else:
            multiplier = 1e9/current_amplification  # 1e9 to nA, /current_amp to current in A.
            print(f'Not using "srs_sens" for entropy signal for dat{datnum} with setupdf config=[{setupdf.config_name}]')
        # if datnum < 1400:
        #     multiplier = infodict['Logs']['srss']['srs3'][
        #      'sens'] / 10 * 1e-3 / current_amplification * 1e9  # /10 because 10V range of output, 1e-3 to go to V, 1e9 to go to nA
        # elif datnum >= 1400:
        #     multiplier = infodict['Logs']['srss']['srs1'][
        #      'sens'] / 10 * 1e-3 / current_amplification * 1e9  # /10 because 10V range of output, 1e-3 to go to V, 1e9 to go to nA
        # else:
        #     raise ValueError('Invalid datnum')
        if entx is not None:
            entx = entx*multiplier
        if enty is not None:
            enty = enty*multiplier
        if entx is not None or enty is not None:
            infodict['entx'] = entx
            infodict['enty'] = enty
            dattypes.add('transition')
        else:
            dattypes.remove('entropy')
            print(f'No entropy data found for dat {datnum} even though "entropy" in dattype')

    if {'lockin theta', 'li_theta'} & set(dattypes):
        dattypes.add('li_theta')
        multiplier = infodict['Logs']['srss']['srs3']['sens'] / 10 * 1e-3 / 1e9 * 1e9  # div by current amp, then to nA
        li_x_key = set(config.li_theta_x_keys) & set(hdf.keys())
        li_y_key = set(config.li_theta_y_keys) & set(hdf.keys())
        assert len(li_x_key) == 1
        assert len(li_y_key) == 1
        infodict['li_theta_keys'] = [list(li_x_key)[0], list(li_y_key)[0]]
        infodict['li_multiplier'] = multiplier

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
    dat = datfactory(datnum, datname, datdf, dfoption, infodict)
    cfg.set_all_for_config(old_config, folder_containing_experiment=None)
    return dat


def make_dat(datnum, datname, dfoption='sync', dattypes = None, datdf=None, setupdf=None, config=None):
    datdf = datdf if datdf else DF.DatDF()
    setupdf = setupdf if setupdf else SF.SetupDF()
    config = config if config else cfg.current_config

    if dattypes is not None:
        dattypes = CU.ensure_set(dattypes)
    elif dattypes is None:
        dattypes = {'none_given'}

    old_config = cfg.current_config
    cfg.set_all_for_config(config, folder_containing_experiment=None)

    if setupdf.config_name != datdf.config_name or setupdf.config_name != config.__name__.split('.')[-1]:
        raise AssertionError(f'C.make_dat_standard: setupdf, datdf and config have different config names'
                             f'[{setupdf.config_name, datdf.config_name, config.__name__.split(".")[-1]},'
                             f' probably you dont want to continue!')

    if dfoption == 'load':  # If only trying to load dat then skip everything else and send instruction to load from DF
        dat = datfactory(datnum, datname, datdf, 'load')
        cfg.set_all_for_config(old_config, folder_containing_experiment=None)
        return dat

    hdfpath = os.path.join(cfg.ddir, f'dat{datnum:d}.h5')
    if os.path.isfile(CU.get_full_path(hdfpath)):
        hdf = h5py.File(CU.get_full_path(hdfpath), 'r')
    else:
        print(f'No hdf5 file found for dat{datnum:d} in directory {cfg.ddir}')
        return None  # Return None if no data exists

    try:
        sweeplogs = hdf['metadata'].attrs['sweep_logs']  # Not JSON data yet
        sweeplogs = metadata_to_JSON(sweeplogs, config=config, datnum=datnum)
    except Exception as e:
        logger.warning(f'Exception when getting sweeplogs: {e}')
        sweeplogs = None

    xarray = hdf['x_array'][:]
    x_label = sweeplogs['axis_labels']['x']
    y_label = sweeplogs['axis_labels']['y']
    try:
        yarray = hdf['y_array'][:]
        dim = 2
    except KeyError:
        yarray = None
        dim = 1

    if sweeplogs is not None:
        temperatures = _temp_from_json(sweeplogs,
                                       fridge=config.instruments['fridge'])  # fridge is just a placeholder for now
        srss = {'srs' + str(i): _srs_from_json(sweeplogs, i, srs_type=config.instruments['srs']) for i in
                range(1, config.instrument_num['srs'] + 1)}
        # mags = [get_instr_vals('MAG', direction) for direction in ['x', 'y', 'z']]
        mags = {'mag' + id: _mag_from_json(sweeplogs, id, mag_type=config.instruments['magnet']) for id in ['x', 'y', 'z']}
        if 'BabyDAC' in sweeplogs.keys():
            dacs = {int(key[2:]): sweeplogs['BabyDAC'][key] for key in sweeplogs['BabyDAC'] if
                    key[-4:] not in ['name', 'port']}
            # make dict of dac values from sweeplogs for keys that are for dac values and not dacnames. Dict is {0: val,
            # 1: val...}
            dacnames = {int(key[2:-4]): sweeplogs['BabyDAC'][key] for key in sweeplogs['BabyDAC'] if key[-4:] == 'name'}
        else:
            dacs = None
            dacnames = None

        if 'comment' in sweeplogs.keys():  # Adds dattypes from comment stored in hdf
            sk = [item.strip() for item in sweeplogs['comment'].split(',')]
            dt = config.dat_types_list
            for key in list(set(sk) & set(dt)):
                dattypes.add(key)
                sk.remove(key)
            comments = ','.join(sk)  # Save the remaining info in self.Logs.comments
        else:
            comments = None

        try:
            fdacs = {int(key[2:]): sweeplogs['FastDAC'][key] for key in sweeplogs['FastDAC'] if key[-4:] not in ['name', 'Keys', 'Freq']}
            fdacnames = {int(key[2:-4]): sweeplogs['FastDAC'][key] for key in sweeplogs['FastDAC'] if key[-4:] == 'name'}
            fdacfreq = sweeplogs['FastDAC']['SamplingFreq']
        except KeyError as e:  # No fastdacs connected
            print(f'Missing key [{e}] in sweep_logs')
            fdacs = None
            fdacnames = None
            fdacfreq = None
            pass

        time_elapsed = sweeplogs['time_elapsed']
        time_completed = sweeplogs['time_completed']  # TODO: make into datetime format here
    else:
        srss, mags, temperatures, time_elapsed, time_completed, dacs, dacnames, fdacs, fdacnames, fdacfreq, comments = \
            [None, None, None,      None,           None,           None, None,     None, None,     None,       None]

    infodict = CU.add_infodict_Logs(None, xarray, yarray, x_label, y_label, dim, srss, mags, temperatures, time_elapsed,
                                    time_completed, dacs, dacnames, fdacs, fdacnames, fdacfreq, comments)
    infodict['hdfpath'] = hdfpath
    if {'i_sense', 'transition', 'entropy', 'dcbias'} & set(dattypes):  # If there is overlap between lists then...
        i_sense = _get_corrected_data(datnum, config.i_sense_keys, hdf, setupdf)
        infodict['i_sense'] = i_sense

    if 'entropy' in dattypes:
        entx = _get_corrected_data(datnum, config.entropy_x_keys, hdf, setupdf)
        enty = _get_corrected_data(datnum, config.entropy_y_keys, hdf, setupdf)

        current_amplification = _get_value_from_setupdf(datnum, 'ca0amp', setupdf)
        srs = _get_value_from_setupdf(datnum, 'entropy_srs', setupdf)
        if srs[:3] == 'srs':
            multiplier = infodict['Logs']['srss'][srs][
             'sens'] / 10 * 1e-3 / current_amplification * 1e9  # /10 because 10V range of output, 1e-3 to go to V, 1e9 to go to nA
        else:
            multiplier = 1e9/current_amplification  # 1e9 to nA, /current_amp to current in A.
            print(f'Not using "srs_sens" for entropy signal for dat{datnum} with setupdf config=[{setupdf.config_name}]')
        # if datnum < 1400:
        #     multiplier = infodict['Logs']['srss']['srs3'][
        #      'sens'] / 10 * 1e-3 / current_amplification * 1e9  # /10 because 10V range of output, 1e-3 to go to V, 1e9 to go to nA
        # elif datnum >= 1400:
        #     multiplier = infodict['Logs']['srss']['srs1'][
        #      'sens'] / 10 * 1e-3 / current_amplification * 1e9  # /10 because 10V range of output, 1e-3 to go to V, 1e9 to go to nA
        # else:
        #     raise ValueError('Invalid datnum')
        if entx is not None:
            entx = entx*multiplier
        if enty is not None:
            enty = enty*multiplier
        if entx is not None or enty is not None:
            infodict['entx'] = entx
            infodict['enty'] = enty
            dattypes.add('transition')
        else:
            dattypes.remove('entropy')
            print(f'No entropy data found for dat {datnum} even though "entropy" in dattype')

    if 'dcbias' in dattypes:
        dattypes.add('transition')

    dat = datfactory(datnum, datname, datdf, dfoption, infodict)
    cfg.set_all_for_config(old_config, folder_containing_experiment=None)
    return dat

def make_dats(datnums: List[int], datname='base', dfoption='load', dfname=None, datdf=None, setupdf=None, config=None) -> List[Dat]:
    """
    Quicker way to get a list of dat objects

    @param datnums: list of datnums to load/overwrite
    @type datnums: list
    @param datname: name of dat in datdf (e.g. 'base', 'digamma' etc)
    @type datname: str
    @param dfoption: whether to load or overwrite datdf
    @type dfoption: str
    @return: List of Dat objects
    @rtype: List[Dat]
    """
    return [make_dat_standard(num, datname=datname, dfoption=dfoption, dfname=dfname, datdf=datdf, setupdf=setupdf, config=config) for num in datnums]


def _get_corrected_data(datnum, wavenames: Union[List, str], hdf: h5py.File, setupdf):
    def _correct_multipliers(data, name):
        mult_found = False
        if setupdata.get(name, None) is not None:
            mult_found = True
            data = data * setupdata[name]
        return data, mult_found

    def _correct_offset(data, name):
        off_found = False
        if setupdata.get(name+'_offset', None) is not None:
            off_found = True
            data = data + setupdata[name+'_offset']
        return data, off_found

    if type(wavenames) != list:
        wavenames = [wavenames]
    setupdata = setupdf.get_valid_row(datnum)
    data = None
    correction_found = False
    for name in wavenames:
        if name in hdf.keys():
            data = hdf[name][:]
            data, offset_found = _correct_offset(data, name)  # This should not necessarily exist
            data, correction_found = _correct_multipliers(data, name)  # This should definitely exist (as a 1 if no change)
    if correction_found is False:
        print(f'WARNING[_get_corrected_data]: No correction found for [{wavenames}] for dat[{datnum}] with '
              f'setupdf config = [{setupdf.config_name}]')
    return data





def _get_value_from_setupdf(datnum, name, setupdf):
    setupdata = setupdf.get_valid_row(datnum)
    if name in setupdata.keys() and setupdata[name] is not None:
        value = setupdata[name]
    else:
        print(f'WARNING[_get_value_from_setupdf]: [{name}] not found in setupdf')
        value = 1
    return value


def _temp_from_json(jsondict, fridge='ls370'):
    if 'BF Small' in jsondict.keys():
        try:
            temps = _temp_from_bfsmall(jsondict['BF Small'])
        except KeyError as e:
            print(jsondict)
            raise e
        return temps
    else:
        logger.info(f'Did not find "BF Small" in json')
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
                   'CH1readout': srsdict.get('CH1readout', None)
                   }
    else:
        srsdata = None
    return srsdata


def _mag_from_json(jsondict, id, mag_type='ls625'):
    if 'LS625 Magnet Supply' in jsondict.keys():  # FIXME: This probably only works if there is 1 magnet ONLY!
        mag_dict = jsondict['LS625 Magnet Supply']  #  FIXME: Might just be able to pop entry out then look again
        magname = mag_dict.get('variable name', None)  # Will get 'magy', 'magx' etc
        if magname[-1:] == id:  # compare last letter
            mag_data = {'field': mag_dict['field mT'],
                        'rate': mag_dict['rate mT/min']
                        }
        else:
            mag_data = None
    else:
        mag_data = None
    return mag_data


def _temp_from_bfsmall(tempdict):
    tempdata = {'mc': tempdict.get('MC K', None),
                'still': tempdict.get('Still K', None),
                'fourk': tempdict.get('4K Plate K', None),
                'mag': tempdict.get('Magnet K', None),
                'fiftyk': tempdict.get('50K Plate K', None)}
    return tempdata


class DatHandler(object):
    """
    Make loading dats a bit more efficient, this will only load dats from pickle if it hasn't already been opened
    in current runtime. Otherwise it will pass reference to the same dat.
    Can also see what dats are open, remove individual dats from DatHandler, or clear all dats from DatHandler
    """
    open_dats = {}

    @staticmethod
    def _get_dat_id(datnum, datname, datdf):
        return f'{datdf.config_name}_{datnum}[{datname}]'

    @classmethod
    def get_dat(cls, datnum, datname, datdf=None, config=None):
        datdf = datdf if datdf else DF.DatDF()
        config = config if config else cfg.current_config
        dat_id = cls._get_dat_id(datnum, datname, datdf)

        if dat_id not in cls.open_dats:
            if DF.dat_exists_in_df(datnum, datname, datdf):
                option = 'load'
            else:
                option = 'sync'  # basically overwrite but sync in case there is something there somehow
                logger.info(f'[{datnum}[{datname}]] for [{datdf.config_name}-{datdf.name}] did not exist so being '
                            f'created. NOT SAVED TO DF BY DEFAULT')
            new_dat = make_dat(datnum, datname, dfoption=option, datdf=datdf, config=config)
            cls.open_dats[dat_id] = new_dat
        return cls.open_dats[dat_id]

    @classmethod
    def list_open_dats(cls):
        return cls.open_dats

    @classmethod
    def remove_dat(cls, datnum, datname, datdf, verbose=True):
        dat_id = cls._get_dat_id(datnum, datname, datdf)
        if dat_id in cls.open_dats:
            del cls.open_dats[dat_id]
            print_verbose(f'Removed [{dat_id}] from dat_handler', verbose)
        else:
            print_verbose(f'Nothing to be removed', verbose)

    @classmethod
    def clear_dats(cls):
        cls.open_dats = {}
