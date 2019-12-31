"""Core of PyDatAnalysis. This should remain unchanged between experiments in general, or be backwards compatible"""

import json
import os
import pickle
import types
import h5py
import re
import lmfit as lm
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import digamma
from collections import namedtuple
import sys
from typing import List, Tuple, Union, NamedTuple
from PyQt5 import QtCore, QtGui, QtWidgets
import time
import platform
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import functools
import inspect
import src.config as cfg

################# Settings for Debugging #####################
cfg.verbose = True
cfg.verboselevel = 19  # Max depth of stack to have verbose prints from
timer = False


###############################################################


################# Other configurables #########################
#
# if platform.system() == "Darwin":  # Darwin is Mac... So I can work with Owen
#     os.chdir("/Users/owensheekey/Documents/Research/One_CK_analysis/OneCK-Analysis")
# else:
#     os.chdir('D:/OneDrive/UBC LAB/GitHub/Python/PyDatAnalysis/')

##Just reminder for what exists
# cfg.ddir
# cfg.pickledata
# cfg.plotdir
# cfg.dfdir

###############################################################

################# Sweeplog fixes ##############################


def metadata_to_JSON(data: str) -> dict:  # TODO, FIXME: Move json edits into experiment specific
    data = re.sub(', "range":, "resolution":', "", data)
    data = re.sub(":\+", ':', data)
    try:
        jsondata = json.loads(data)
    except json.JSONDecodeError:
        data = re.sub('"CH0name".*"com_port"', '"com_port"', data)
        jsondata = json.loads(data)
    return jsondata


###############################################################

################# Constants and Characters ####################
pm = '\u00b1'  # plus minus
G = '\u0393'  # Capital Gamma
delta = '\u0394'  # Capital Delta
d = '\u03B4'  # Lower delta


###############################################################


################# Beginning of Classes ########################
class Dats(object):
    def __init__(self):
        pass


def datfactory(datnum, datname, dfname, dfoption, infodict):
    datdf = DatDF(dfname=dfname)  # Load DF
    datcreator = _creator(dfoption)  # get creator of Dat instance based on df option
    datinst = datcreator(datnum, datname, datdf, infodict)  # get an instance of dat using the creator
    return datinst  # Return that to caller


def _creator(dfoption):
    if dfoption == 'load':
        return _load
    elif dfoption == 'sync':
        return _sync
    elif dfoption == 'overwrite':
        return _overwrite
    else:
        raise ValueError("dfoption must be one of: load, sync, overwrite")


def _load(datnum: int, datname, datdf, infodict):
    datpicklepath = datdf.get_path(datnum, datname=datname)
    with open(datpicklepath) as f:  # TODO: Check file exists
        inst = pickle.load(f)
    return inst


def _sync(datnum, datname, datdf, infodict):
    if (datnum, datname) in datdf.df.index:
        inp = input(f'Dat{datnum},{datname} already exists, do you want to \'load\' or \'overwrite\'')
        if inp == 'load':
            inst = _load(datnum, datname, infodict)
        elif inp == 'overwrite':
            inst = _overwrite(datnum, datname, infodict)
        else:
            raise ValueError('Must choose either \'load\' or \'overwrite\'')
    else:
        inst = _overwrite(datnum, datname, datdf, infodict)
    return inst


def _overwrite(datnum, datname, datdf, infodict):
    inst = Dat(datnum, datname, infodict, dfname=datdf.name)
    return inst


class Dat(object):
    """Overall Dat object which contains general information about dat, more detailed info should be put
    into a subclass. Everything in this overall class should be useful for 99% of dats"""

    # def __new__(cls, *args, **kwargs):
    #     return object.__new__(cls)

    def __getattr__(self, name):  # __getattribute__ overrides all, __getattr__ overrides only missing attributes
        # Note: This affects behaviour of hasattr(). Hasattr only checks if getattr returns a value, not whether
        # attribute was defined previously.
        raise AttributeError(f'Attribute {name} does not exist. Maybe want to implement getting attrs from datPD here')

    def __setattr__(self, name, value):
        # region Verbose Dat __setattr__
        if cfg.verbose is True:
            verbose_message(
                f'in override setattr. Being called from {inspect.stack()[1][3]}, hasattr is {hasattr(self, name)}')
        # endregion
        if not hasattr(self, name) and inspect.stack()[1][3] != '__init__':  # Inspect prevents this override
            # affecting init
            # region Verbose Dat __setattr__
            if cfg.verbose is True:
                verbose_message(
                    'testing setattr override')  # TODO: implement writing change to datPD at same time, maybe with a check?
            # endregion

        else:
            super().__setattr__(name, value)

    def __init__(self, datnum: int, datname, infodict: dict, dfname='default'):
        """Constructor for dat"""
        try:
            dattype = infodict['type']
        except:
            dattype = 'none'  # Can't check if str is in None, but can check if in 'none'
        self.datnum = datnum
        self.sweeplogs = infodict['sweeplogs']  # type: dict  # Full JSON formatted sweeplogs
        self.sc_config = infodict['sc_config']  # type: dict  # Full JSON formatted sc_config

        self.x_array = infodict['xarray']  # type:np.ndarray
        self.y_array = infodict['yarray']  # type:np.ndarray
        self.x_label = self.sweeplogs['axis_labels']['x']
        self.y_label = self.sweeplogs['axis_labels']['y']
        self.dim = infodict['dim']  # type: int  # Number of dimensions to data

        self.time_elapsed = self.sweeplogs['time_elapsed']

        self.srs1 = None
        self.srs2 = None
        self.srs3 = None
        self.srs4 = None
        # self.instr_vals('srs', infodict['srss'])  #FIXME

        self.magx = None
        self.magy = None
        self.magz = None
        # self.instr_vals('mag', infodict['mags'])  # FIXME

        self.temps = infodict['temperatures']  # Stores temperatures in tuple e.g. self.temps.mc

        if 'i_sense' in dattype:
            self.i_sense = infodict[
                'i_sense']  # type: np.ndarray  # Charge sensor current in nA  # TODO: Do I want to move this to a subclass?
        if 'entropy' in dattype:
            # TODO: Then init subclass entropy dat here??
            # self.__init_subclass__(Entropy_Dat)
            pass
        self.dfname = dfname
        self.savetodf(dfname=dfname)

    def instr_vals(self, name: str, data: List[NamedTuple]):
        if data is not None:
            for ntuple in data:  # data should be a List of namedtuples for instrument, First field should be ID (e.g. 1 or x)
                evalstr = f'self.{name}{ntuple[0]} = {ntuple}'
                exec(evalstr)
        return None

    def savetodf(self, dfname='default'):
        datDF = DatDF(dfname=dfname)
        datDF.add_dat(self)
        datDF.save(datDF.name)


class Entropy_Dat(Dat):
    """For Dats which contain entropy data"""

    def __init__(self, *args, **kwargs):
        super().__init__(args,
                         kwargs)
        self.entx = kwargs['entx']
        self.enty = kwargs['enty']

        self.entr = None
        self.entrav = None
        self.entangle = None
        self.calc_r(useangle=True)

    def calc_r(self, useangle=True):
        # calculate r data using either constant phase determined at largest value or larger signal
        # create averages - Probably this is still the best way to determine phase angle/which is bigger even if it's not repeat data
        xarray, entxav = average_repeats(self,
                                         returndata="entx")  # FIXME: Currently this requires Charge transition fits to be done first inside average_repeats
        xyarray, entyav = average_repeats(self, returndata="enty")
        sqr_x = np.square(entxav)
        sqr_y = np.square(entyav)
        sqr_xi = np.square(self.entx)  # non averaged data
        sqr_yi = np.square(self.enty)
        if max(np.abs(entxav)) > max(np.abs(entyav)):
            # if x is larger, take sign of x. Individual lines still take sign of averaged data otherwise it goes nuts
            sign = np.sign(entxav)
            signi = np.sign(self.entx)
            if np.abs(np.nanmax(entxav)) > np.abs(np.nanmin(entxav)):
                xmax = np.nanmax(entxav)
                xmaxi = np.nanargmax(entxav)
            else:
                xmax = np.nanmin(entxav)
                xmaxi = np.nanargmin(entxav)
            angle = np.arctan((entyav[xmaxi]) / xmax)
        else:
            # take sign of y data
            sign = np.sign(entyav)
            signi = np.sign(self.enty)
            if np.abs(np.nanmax(entyav)) > np.abs(np.nanmin(entyav)):
                ymax = np.nanmax(entyav)
                ymaxi = np.nanargmax(entyav)
            else:
                ymax = np.nanmin(entyav)
                ymaxi = np.nanargmin(entyav)
            angle = np.arctan(ymax / entxav[ymaxi])
        if useangle is False:
            self.entrav = np.multiply(np.sqrt(np.add(sqr_x, sqr_y)), sign)
            self.entr = np.multiply(np.sqrt(np.add(sqr_xi, sqr_yi)), signi)
            self.entangle = None
        elif useangle is True:
            self.entrav = np.array([x * np.cos(angle) + y * np.sin(angle) for x, y in zip(entxav, entyav)])
            self.entr = np.array([x * np.cos(angle) + y * np.sin(angle) for x, y in zip(self.entx, self.enty)])
            self.entangle = angle


#

class DatDF(object):
    """
    Pandas Dataframe object holding all metadata and parameters of Dat objects. Dat objects should ask DatPD for
    config/save config here
    """
    __instance_dict = {}  # Keeps track of whether DatPD exists or not

    # def __getnewargs_ex__(self):
    # Uses this when dumping pickle.... Not sure if this is useful yet

    def __new__(cls, **kwargs):
        if inspect.stack()[1][3] == '__new__':  # If loading from pickle in this loop, don't start an infinite loop
            return super(DatDF, cls).__new__(cls)
        if 'dfname' in kwargs.keys() and kwargs['dfname'] is not None:
            name = kwargs['dfname']
        else:
            name = 'default'
        datDFpath = os.path.join(cfg.dfdir, f'{name}.pkl')
        datDFexcel = os.path.join(cfg.dfdir, f'{name}.xlsx')
        # TODO: Can add later way to load different versions, or save to a different version etc. Or backup by week or something
        if name not in cls.__instance_dict:  # If named datDF doesn't already exist
            if os.path.isfile(datDFpath):  # check if saved version exists
                with open(datDFpath, 'rb') as f:
                    inst = pickle.load(f)  # FIXME: This loops back to beginning of __new__, not sure why?!
                inst.loaded = True
                if not isinstance(inst, cls):  # Check if loaded version is actually a datPD
                    raise TypeError(f'File saved at {datDFpath} is not of the type {cls}')
                if os.path.isfile(datDFexcel):  # If excel of df only exists
                    tempdf = pd.read_excel(datDFexcel, index_col=0, header=0)
                    if not inst.df.equals(tempdf):
                        inp = input(f'datDF[{name}] has a different pickle and excel version of DF '
                                    f'do you want to use excel version?')
                        if inp.lower() in {'y', 'yes'}:  # Replace pickledf with exceldf
                            inst.df = tempdf
                cls.__instance_dict[name] = inst
            else:
                inst = object.__new__(cls)  # Otherwise create a new instance
                inst.loaded = False
                cls.__instance_dict[name] = inst
        else:
            # region Verbose DatDF __new__
            cls.__instance_dict[name].loaded = True  # loading from existing
            if cfg.verbose is True:
                verbose_message('DatPD already exists, returned same instance')
            # endregion

        return cls.__instance_dict[name]  # Return the instance to __init__

    def __init__(self, **kwargs):
        if self.loaded is False:  # If not loaded from file need to create it
            self.df = pd.DataFrame(columns=['datnum', 'time', 'picklepath'])  # TODO: Add more here
            if 'dfname' in kwargs.keys():
                name = kwargs['dfname']
            else:
                name = 'default'
            self.name = name
            self.save()  # So overwrites without asking (by here you have decided to overwrite anyway)
        else:  # Probably don't need to do much if loaded from file
            pass
        # region Verbose DatDF __init__
        if cfg.verbose is True:
            verbose_message('End of init of DatDF')
        # endregion

    def add_dat(self, dat: Dat):
        """Cycles through all attributes of Dat and adds to Dataframe"""
        for attrname in dat.__dict__.keys():
            self.add_dat_attr(dat.datnum, attrname, getattr(dat, attrname))  # add_dat_attr checks if value can be added

    def add_dat_attr(self, datnum, attrname, attrvalue):
        """Adds single value to dataframe, performs checks on values being entered"""
        if DatDF.allowable_attrvalue(attrname, attrvalue) is True:  # Don't want to fill dataframe with arrays,
            if attrname not in self.df.columns:
                inp = input(f'There is currently no column for {attrname}, would you like to add one?\n')
                if inp.lower() in {'yes', 'y'}:
                    pass
                else:
                    # region Verbose DatDF add_dat_attr
                    if cfg.verbose is True:
                        verbose_message(f'Verbose[DatDF][add_dat_attr] - Not adding {attrname} to df')
                    # endregion
                    return None
            else:
                if self.df.loc[(datnum), attrname] is not np.nan:
                    inp = input(
                        f'{attrname} is currently {self.df.loc[(datnum), attrname]}, do you want to overwrite with {attrvalue}?')
                    if inp in {'yes', 'y'}:
                        pass
                    else:
                        # region Verbose DatDF add_dat_attr
                        if cfg.verbose is True:
                            verbose_message(
                                f'Verbose[DatDF][add_dat_attr] - Not overwriting {attrname} for dat{datnum} in df')
                        # endregion

                        return None
            self.df.at[(datnum), attrname] = attrvalue
        return None

    @staticmethod
    def allowable_attrvalue(attrname, attrvalue) -> bool:
        """Returns true if allowed in df"""
        if type(attrvalue) in {str, int, float, np.float32, np.float64}:
            if attrname not in ['datnum', 'dfname']:
                return True
        # region Verbose DatDF allowable_attrvalue
        if cfg.verbose is True:
            verbose_message(f'Verbose[DatDF][allowable_attrvalue] - Type {type(attrvalue)} not allowed in DF')
        # endregion
        return False

    def save(self, name=None):
        """Defaults to saving over itself"""
        if name is not None and name in DatDF.__instance_dict:
            inp = input(f'datDF with name {name} already exists, do you want to overwrite it?')
            if inp in ['y', 'yes']:
                pass
            else:
                print('DF not saved and name not changed')
                return None
        elif name is None:
            name = self.name  # defaults to save over currently open DatDF
        self.name = name  # Change name of DF if saving under new name
        DatDF.__instance_dict[name] = self  # Copying instance to DatDF dict

        datDFexcel = os.path.join(cfg.dfdir, f'{name}.xlsx')
        datDFpath = os.path.join(cfg.dfdir, f'{name}.pkl')

        self.df.to_excel(datDFexcel)  # Can use pandasExcelWriter if need to do more fancy saving
        with open(datDFpath, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        return None

    def load(self, name=None):  # FIXME: Does this actually reload an older instance?
        del DatDF.__instance_dict[name]
        DatDF.__new__(DatDF, name=name)
        # region Verbose DatDF load
        if cfg.verbose is True:
            verbose_message(f'Loaded {name}')
        # endregion        
        return None

    def sync_dat(self, datnum: int, mode: str = 'sync', **kwargs):
        """
        :param mode: Accepts 'sync', 'overwrite', 'load' to determine behaviour with dataframe
        """
        if mode == 'sync':
            if self.df['datnum'] is not None:  # FIXME: Idea of this is to see if datnum exists in datnum column
                inp = input(f'Do you want to \'overwrite\' or \'load\' for Dat{datnum}')
                if inp == 'overwrite':
                    mode = 'overwrite'
                elif inp == 'load':
                    mode = 'load'
            else:
                mode = 'overwrite'  # Already checked no data there at this point
        if mode == 'load':
            return self.df.loc[self.df['datnum'] == datnum]
        if mode == 'overwrite':
            data = []
            cols = []
            for key, value in kwargs:
                if key in self.df.columns:
                    data.append(value)
                    cols.append(key)
                else:
                    inp = input(f'{key} is not in datPD dataframe, would you like to add it?')
                    if inp.lower() in ['yes', 'y']:
                        data.append(value)
                        cols.append(key)
                    else:
                        print(f'{key} was not added')
            tempdf = pd.DataFrame([data], columns=cols)
            self.df.append(tempdf, ignore_index=True)
        return None

    def get_path(self, datnum, name):
        """Returns path to pickle of dat specified by datnum [, dfname]"""
        if datnum in self.df.index.levels[0]:
            if name is not None and (datnum, name) in self.df.index:
                path = self.df.at[(datnum, name), 'picklepath']
            else:
                path = self.df.at[
                    datnum, 'picklepath']  # FIXME: How does index look for multi index without second entry
        else:
            raise ValueError(f'No dat exists with datnum={datnum}, dfname={name}')
        return path

    @staticmethod
    def print_open_datDFs():
        """Prints and returns __instance_dict.keys() of DatDF"""
        print(DatDF.__instance_dict.keys())
        return DatDF.__instance_dict.keys()

    @staticmethod
    def killinstances():
        DatDF.__instance_dict = {}

################# End of classes ################################

################# Functions #####################################

def verbose_message(printstr: str, forcelevel=None, forceon=False):
    """Prints verbose message if global verbose is True"""
    level = stack_size()  # TODO: set level by how far into stack the function is being called from so that prints can be formatted nicer
    if cfg.verbose is True or forceon is True and level < cfg.verboselevel:
        print(f'{printstr.rjust(level + len(printstr))}')
    return None


def make_basicinfodict(xarray: np.array = None, yarray: np.array = None, dim: int = None, sweeplogs: dict = None,
                       sc_config: dict = None, srss: List[NamedTuple] = None, mags: List[NamedTuple] = None,
                       temperatures: NamedTuple = None) -> dict:
    """Makes dict with all info to pass to Dat. Useful for typehints"""
    infodict = {'xarray': xarray, 'yarray': yarray, 'dim': dim, 'sweeplogs': sweeplogs, 'sc_config': sc_config,
                'srss': srss, 'mags': mags, 'temperatures': temperatures}
    return infodict


def open_hdf5(dat, path=''):
    fullpath = os.path.join(path, 'dat{0:d}.h5'.format(dat))
    return h5py.File(fullpath, 'r')


def average_repeats(dat: Dat, returndata: str = 'i_sense', centerdata: np.array = None, retstd=False) -> Union[
    List[np.array], List[np.array]]:
    """Takes dat object and returns (xarray, averaged_data, centered by charge transition by default)"""
    if centerdata is None:
        if dat.transitionvalues.x0s is not None:  # FIXME: put in try except here if there are other default ways to center
            centerdata = dat.transitionvalues.x0s
        else:
            raise AttributeError('No data available to use for centering')

    xlen = dat.x_array[-1] - dat.x_array[0]
    xarray = np.array(list(np.linspace(-(xlen / 2), (xlen / 2), len(dat.x_array))),
                      dtype=np.float32)  # FIXME: Should think about making this create an array smaller than original with more points to avoid data degredation
    midvals = centerdata
    if returndata in dat.__dict__.keys():
        data = dat.__getattribute__(returndata)
        if data.shape[1] != len(dat.x_array) or data.shape[0] != len(dat.y_array):
            raise ValueError(f'Shape of Data is {data.shape} which is not compatible with x_array={len(dat.x_array)} '
                             f'and y_array={len(dat.y_array)}')
    else:
        raise ValueError(f'Returndata = \'{returndata}\' does not exist in dat... it is case sensitive by the way')
    matrix = np.array([np.interp(xarray, dat.x_array - midvals[i], data[i]) for i in range(len(dat.y_array))],
                      dtype=np.float32)  # interpolated data after shifting data to be centred at 0
    averaged = np.array([np.average(matrix[:, i]) for i in range(len(xarray))],
                        dtype=np.float32)  # average the data over number of rows of repeat
    stderrs = np.array([np.std(matrix[:, i]) for i in range(len(xarray))],
                       dtype=np.float32)  # stderr of each averaged datapoint
    #  All returning np.float32 because lmfit doesn't like float64

    ret = [xarray, averaged]
    if retstd is True:
        ret += [stderrs]
    return ret


def stack_size():
    frame = sys._getframe(1)
    i = 0
    while frame:
        frame = frame.f_back
        i += 1
    return i


def coretest(nums):
    """For testing unit testing"""
    result = 0
    for num in nums:
        result += num
    return result
