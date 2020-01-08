import inspect
import os
import pickle

import numpy as np
import pandas as pd

from src import config as cfg
from src.CoreUtil import verbose_message
from src.DFcode import DFutil as DU
from src.Dat.Dat import Dat
import src.CoreUtil as CU

pd.DataFrame.set_index = DU.protect_data_from_reindex(pd.DataFrame.set_index)  # Protect from deleting columns of data
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
            inst = DU.load_from_pickle(datDFpath, cls)  # Returns either inst or None
            if inst is not None:
                if os.path.isfile(datDFexcel):  # If excel of df only exists
                    exceldf = pd.read_excel(datDFexcel, index_col=[0, 1], header=[0,1], dtype=DatDF._dtypes)  # FIXME: Load from excel needs to know how deep column levels go
                    inst.df = DU.compare_pickle_excel(inst.df, exceldf, f'DatDF[{inst.name}]')  # Returns either pickle or excel df depending on input
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
        # for attrdict in [dic.__dict__ for dic in [dat, dat.Logs, dat.Entropy] if type(dic) == dict]:
        #     for coladdress in attrdict:
        #         self.add_dat_attr(dat.datnum, coladdress, getattr(dat, coladdress), datname=dat.datname)  # add_dat_attr checks if value can be added

        coladdress = []
        attrdict = dat.__dict__
        for attrname in attrdict:
            coladdress = [attrname]
            self.add_dat_attr_recursive(dat, coladdress, attrdict[attrname])
        return None

    def add_dat_attr_recursive(self, dat, coladdress, attrvalue):
        ret = self.add_dat_attr(dat.datnum, coladdress, attrvalue, datname=dat.datname)
        if ret is True:
            return None
        elif ret is False:
            datattr = DatDF._get_dat_attr_from_coladdress(dat, coladdress)
            if hasattr(datattr, '__dict__'):
                for attr in datattr.__dict__:
                    self.add_dat_attr_recursive(dat, coladdress+[attr], getattr(datattr, attr))
            return None

    @staticmethod
    def _get_dat_attr_from_coladdress(dat, coladdress:list):
        attr = getattr(dat, coladdress[0])
        if len(coladdress) > 1:
            for attrname in coladdress[1:]:
                attr = getattr(attr, attrname)
        return attr

    @DU.temp_reset_index
    def add_dat_attr(self, datnum, coladdress, attrvalue, datname='base'):
        """Adds single value to dataframe, performs checks on values being entered"""
        self.df.set_index(['datnum', 'datname'], inplace=True)
        if type(coladdress) != list:
            coladdress = [coladdress]
        if DatDF.allowable_attrvalue(self.df, coladdress, attrvalue) is True:  # Don't want to fill dataframe with big things
            if tuple(coladdress) not in self.df.columns:
                ans = CU.option_input(f'There is currently no column for "{coladdress}", would you like to add one?\n', {'yes': True, 'no': False})
                if ans is True:
                    self.df = DU.add_new_col(self.df, coladdress)  # add new column now to avoid trying to add new
                    # col and row at the same time
                else:
                    # region Verbose DatDF add_dat_attr
                    if cfg.verbose is True:
                        verbose_message(f'Verbose[DatDF][add_dat_attr] - Not adding "{coladdress}" "to df')
                    # endregion
                    return None
            else:
                if (datnum, datname) in self.df.index and not pd.isna(self.df.loc[(datnum, datname), coladdress]):
                    inp = input(
                        f'{coladdress} is currently {self.df.loc[(datnum, datname), coladdress]}, do you want to overwrite with {attrvalue}?')
                    if inp in {'yes', 'y'}:
                        pass
                    else:
                        # region Verbose DatDF add_dat_attr
                        if cfg.verbose is True:
                            verbose_message(
                                f'Verbose[DatDF][add_dat_attr] - Not overwriting "{coladdress}" for dat{datnum}[{datname}] in df')
                        # endregion

                        return None



            self.df.at[(datnum, datname), tuple(coladdress)] = attrvalue
            return True
        else:
            return False

    def change_in_excel(self):
        """Lets user change df in excel. Does not save changes by default!!!"""
        self.df = DU.change_in_excel(self.filepathexcel)
        return None

    @staticmethod
    def allowable_attrvalue(df, coladdress, attrvalue) -> bool:
        """Returns true if allowed in df"""
        if type(attrvalue) in {str, int, float, np.float32, np.float64}:  # Right sort of data to add
            if coladdress not in [['datnum'], ['datname'], ['dfname']]:  # Not necessary to add these  #TODO: Check this works?
                if DatDF.check_dtype(df, coladdress, attrvalue) is True:
                    return True
        # region Verbose DatDF allowable_attrvalue
        if cfg.verbose is True:
            verbose_message(f'Verbose[DatDF][allowable_attrvalue] - Type "{type(attrvalue)}" not allowed in DF')
        # endregion
        return False


    @staticmethod
    def check_dtype(df, coladdress, attrvalue):
        """Checks if dtype matches if column exists and that only addressing one column"""
        if type(coladdress) == list and len(coladdress) == 1:  # If only 1 long it needs to be in string form
            # to work in df[address]
            coladdress = coladdress[0]
        if tuple(coladdress) in df.columns or (type(coladdress) == str and coladdress in df.columns):
            if type(attrvalue) == str:
                t = 'object'
            elif type(attrvalue) == int:
                t = float
            else:
                t = type(attrvalue)
            try:
                if t == df[coladdress].dtype:
                    return True
            except AttributeError:
                # region Verbose DatDF check_dtype
                if cfg.verbose is True:
                    verbose_message(f'Verbose[DatDF][check_dtype] - {coladdress} does not specify a single column')
                # endregion
                return False
            else:
                return CU.option_input(f'Trying to add {coladdress}="{attrvalue}" with dtype={t} where current '
                            f'dtype is {df[coladdress].dtype}, would you like to continue?', {'yes': True, 'no': False})
        else:
            return True  # i.e. add new column

    @DU.temp_reset_index
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

    def infodict(self, datnum, datname):  # FIXME: Doesn't return the same infodict that make_dat provides.
        """Returns infodict for named dat so pickle can be made again"""
        _dat_exists_in_df(datnum, datname, self)
        vals = [val for val in self.df.loc[(datnum, datname)]]
        keys = self.df.columns
        return dict(zip(keys, vals))
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

    def get_path(self, datnum, datname=None):
        """Returns path to pickle of dat specified by datnum [, dfname]"""
        if datnum in self.df.index.levels[0]:
            if datname is not None and (datnum, datname) in self.df.index:
                path = self.df.at[(datnum, datname), 'picklepath']
            else:
                path = self.df.at[
                    datnum, 'picklepath']  # FIXME: How does index look for multi index without second entry
        else:
            raise ValueError(f'No dat exists with datnum={datnum}, dfname={datname}')
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
