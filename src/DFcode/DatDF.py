import inspect
import os
import pickle

import numpy as np
import pandas as pd

from src import config as cfg
from src.CoreUtil import verbose_message
from src.DFcode import DFutil
from src.Dat.Dat import Dat

pd.DataFrame.set_index = DFutil.protect_data_from_reindex(pd.DataFrame.set_index)  # Protect from deleting columns of data
class DatDF(object):
    """
    Pandas Dataframe object holding all metadata and parameters of Dat objects. Dat objects should ask DatPD for
    config/save config here
    """
    __instance_dict = {}  # Keeps track of whether DatPD exists or not

    _default_columns = ['time', 'picklepath', 'x_label', 'y_label', 'dim', 'time_elapsed']
    _default_data = [['Wednesday, January 1, 2020 00:00:00', 'pathtopickle', 'xlabel', 'ylabel', 1, 1]]
    _dtypes = [object, str, str, str, float, float]
    _dtypes = dict(zip(_default_columns, _dtypes))  # puts into form DataFrame can use
    # Can use 'converters' to make custom converter functions if necessary


    def __getnewargs_ex__(self):
        """When loading from pickle, this is passed into __new__"""
        args = None
        kwargs = {'frompickle': True}
        return (args,), kwargs

    def __new__(cls, *args, **kwargs):
        if 'dfname' in kwargs.keys() and kwargs['dfname'] is not None:
            name = kwargs['dfname']
        else:
            name = 'default'
        datDFpath = os.path.join(cfg.dfdir, f'{name}.pkl')
        datDFexcel = os.path.join(cfg.dfdir, f'{name}.xlsx')
        if 'frompickle' in kwargs.keys() and kwargs['frompickle'] is True:
            inst = super(DatDF, cls).__new__(cls)
            return inst

        # TODO: Can add later way to load different versions, or save to a different version etc. Or backup by week or something
        if name not in cls.__instance_dict:  # If named datDF doesn't already exist
            inst = DFutil.load_from_pickle(datDFpath, cls)  # Returns either inst or None
            if inst is not None:
                if os.path.isfile(datDFexcel):  # If excel of df only exists
                    exceldf = pd.read_excel(datDFexcel, index_col=[0, 1], header=0, dtype=DatDF._dtypes)
                    inst.df = DFutil.compare_pickle_excel(inst.df, exceldf, f'DatDF[{inst.name}]')  # Returns either pickle or excel df depending on input
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
            mux = pd.MultiIndex.from_arrays([[0], ['base']], names=['datnum', 'datname'])  # Needs at least one row of data to save
            self.df = pd.DataFrame(DatDF._default_data, index=mux, columns=DatDF._default_columns)
            self.set_dtypes()
            # self.df = pd.DataFrame(columns=['time', 'picklepath'])  # TODO: Add more here
            if 'dfname' in kwargs.keys():
                name = kwargs['dfname']
            else:
                name = 'default'
            self.name = name
            self.filepathpkl = os.path.join(cfg.dfdir, f'{name}.pkl')
            self.filepathexcel = os.path.join(cfg.dfdir, f'{name}.xlsx')
            self.save()  # So overwrites without asking (by here you have decided to overwrite anyway)
        else:  # Probably don't need to do much if loaded from file
            pass
        # region Verbose DatDF __init__
        if cfg.verbose is True:
            verbose_message('End of init of DatDF')
        # endregion

    def set_dtypes(self):
        for key, value in DatDF._dtypes.items():
            if type(value) == type:
                self.df[key] = self.df[key].astype(value)

    def add_dat(self, dat: Dat):
        """Cycles through all attributes of Dat and adds to Dataframe"""
        for attrname in dat.__dict__.keys():
            self.add_dat_attr(dat.datnum, attrname, getattr(dat, attrname), datname=dat.datname)  # add_dat_attr checks if value can be added

    def add_dat_attr(self, datnum, attrname, attrvalue, datname='base'):
        """Adds single value to dataframe, performs checks on values being entered"""
        if DatDF.allowable_attrvalue(self.df, attrname, attrvalue) is True:  # Don't want to fill dataframe with arrays,
            if attrname not in self.df.columns:
                inp = input(f'There is currently no column for "{attrname}", would you like to add one?\n')
                if inp.lower() in {'yes', 'y'}:
                    pass
                else:
                    # region Verbose DatDF add_dat_attr
                    if cfg.verbose is True:
                        verbose_message(f'Verbose[DatDF][add_dat_attr] - Not adding "{attrname}" "to df')
                    # endregion
                    return None
            else:
                if (datnum, datname) in self.df.index and not pd.isna(self.df.loc[(datnum, datname), attrname]):
                    inp = input(
                        f'{attrname} is currently {self.df.loc[(datnum, datname), attrname]}, do you want to overwrite with {attrvalue}?')
                    if inp in {'yes', 'y'}:
                        pass
                    else:
                        # region Verbose DatDF add_dat_attr
                        if cfg.verbose is True:
                            verbose_message(
                                f'Verbose[DatDF][add_dat_attr] - Not overwriting "{attrname}" for dat{datnum}[{datname}] in df')
                        # endregion

                        return None
            self.df.at[(datnum, datname), attrname] = attrvalue
        return None

    def change_in_excel(self):
        """Lets user change df in excel. Does not save changes by default!!!"""
        self.df = DFutil.change_in_excel(self.filepathexcel)
        return None

    @staticmethod
    def allowable_attrvalue(df, attrname, attrvalue) -> bool:
        """Returns true if allowed in df"""
        if type(attrvalue) in {str, int, float, np.float32, np.float64}:  # Right sort of data to add
            if attrname not in ['datnum', 'datname', 'dfname']:  # Not necessary to add these
                if DatDF.check_dtype(df, attrname, attrvalue) is True:
                    return True
        # region Verbose DatDF allowable_attrvalue
        if cfg.verbose is True:
            verbose_message(f'Verbose[DatDF][allowable_attrvalue] - Type "{type(attrvalue)}" not allowed in DF')
        # endregion
        return False

    @staticmethod
    def check_dtype(df, attrname, attrvalue):
        """Checks if dtype matches if column exists"""
        if attrname in df.columns:
            if type(attrvalue) == str:
                t = 'object'
            elif type(attrvalue) == int:
                t = float
            else:
                t = type(attrvalue)
            if t == df[attrname].dtype:
                return True
            else:
                inp = input(f'Trying to add {attrname}="{attrvalue}" with dtype={t} where current '
                            f'dtype is {df[attrname].dtype}, would you like to continue?')
                if inp.lower() in ['y', 'yes']:
                    return True
            return False  # only if column exists and user decided not to add value with new dtype
        else:
            return True

    @DFutil.temp_reset_index
    def save(self, name=None):
        """Defaults to saving over itself"""
        self.df.set_index(['datnum', 'datname'], inplace=True)
        if name is not None and name in DatDF.__instance_dict:
            inp = input(f'datDF with name "{name}" already exists, do you want to overwrite it?')
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

    def load(self, name=None):
        """Loads from named pickle and excel then asks which to keep"""
        if name is None:
            name = self.name
        del DatDF.__instance_dict[name]
        inst = DatDF.__new__(DatDF, dfname=name)
        # region Verbose DatDF load
        if cfg.verbose is True:
            verbose_message(f'Loaded {name}')
        # endregion
        return inst

    def infodict(self, datnum, datname):
        """Returns infodict for named dat so pickle can be made again"""
        _dat_exists_in_df(datnum, datname, self)
        return [val for val in self.df.loc[(datnum, datname)]]

    # def sync_dat(self, datnum: int, mode: str = 'sync', **kwargs):
    #     """
    #     :param mode: Accepts 'sync', 'overwrite', 'load' to determine behaviour with dataframe
    #     """
    #     if mode == 'sync':
    #         if self.df['datnum'] is not None:  # FIXME: Idea of this is to see if datnum exists in datnum column
    #             inp = input(f'Do you want to \'overwrite\' or \'load\' for Dat{datnum}')
    #             if inp == 'overwrite':
    #                 mode = 'overwrite'
    #             elif inp == 'load':
    #                 mode = 'load'
    #         else:
    #             mode = 'overwrite'  # Already checked no data there at this point
    #     if mode == 'load':
    #         return self.df.loc[self.df['datnum'] == datnum]
    #     if mode == 'overwrite':
    #         data = []
    #         cols = []
    #         for key, value in kwargs:
    #             if key in self.df.columns:
    #                 data.append(value)
    #                 cols.append(key)
    #             else:
    #                 inp = input(f'{key} is not in datPD dataframe, would you like to add it?')
    #                 if inp.lower() in ['yes', 'y']:
    #                     data.append(value)
    #                     cols.append(key)
    #                 else:
    #                     print(f'{key} was not added')
    #         tempdf = pd.DataFrame([data], columns=cols)
    #         self.df.append(tempdf, ignore_index=True)
    #     return None

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
        """Clears all instances of dfs from __instance_dict"""
        DatDF.__instance_dict = {}

    @staticmethod
    def killinstance(dfname):
        """Removes instance of named df from __instance_dict if it exists"""
        if dfname in DatDF.__instance_dict:
            del DatDF.__instance_dict[dfname]


##########################
# FUNCTIONS
##########################


def _dat_exists_in_df(datnum, datname, datdf):
    """Checks if datnum, datname in given datdf"""
    if (datnum, datname) in datdf.df.index:
        return True
    else:
        raise NameError(f'Dat{datnum}[{datname}] doesn\'t exist in datdf[{datdf.name}]')


def savetodf(dat: Dat, dfname='default'):
    datDF = DatDF(dfname=dfname)
    datDF.add_dat(dat)
    datDF.save()  # No name so saves without asking. TODO: Think about whether DF should be saved here
