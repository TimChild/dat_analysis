from __future__ import annotations

import src.DatObject.Make_Dat

import pickle
import pandas as pd
from src.DFcode import DFutil as DU
from bisect import bisect
import os
import src.CoreUtil as CU
import datetime
import shutil

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.DataStandardize.BaseClasses import ConfigBase

pd.DataFrame.set_index = DU.protect_data_from_reindex(pd.DataFrame.set_index)  # Protect from deleting columns of data


def get_setupdf_id(name, config: ConfigBase):
    return f'[{config.dir_name}]{name}'


class SetupDF(object):
    """
    Pandas Dataframe holding experiment setup data (such as divider info, current amp settings, other comments)
    """
    __instance = DU.inst_dict()
    bobs = 50  # number of BOB I/Os

    def __getnewargs_ex__(self):
        """When loading from pickle, this is passed into __new__"""
        args = None
        kwargs = {'frompickle': True}
        return (args,), kwargs

    def __new__(cls, *args, **kwargs):
        # If loading from pickle then just return that instance
        if 'frompickle' in kwargs.keys() and kwargs['frompickle'] is True:
            inst = super(SetupDF, cls).__new__(cls)
            return inst

        # Either get config from args passed in, or use the default one from Main_Config
        config = kwargs.get('config', None)
        if config is None:
            config = src.DatObject.Make_Dat.default_config

        name = 'setup'  # in case I want to add name to setup df later
        full_id = get_setupdf_id(name, config)  # Gets config specific ID

        # If already exists then just return the current instance
        if full_id in SetupDF.__instance and SetupDF.__instance[
            full_id] is not None:  # If already existing instance return that instance.
            inst = SetupDF.__instance[full_id]
            inst.loaded = True
            inst.exists = True
            return inst

        # If here then need to look for existing file
        setupDFpath = config.Directories.dfsetupdir  # already corrected for shortcuts etc
        setupDF_pkl_path = os.path.join(setupDFpath, 'setup.pkl')
        inst = DU.load_from_pickle(setupDF_pkl_path, SetupDF)  # Gets inst if it exists otherwise returns None
        if inst is not None:
            inst.loaded = True
            inst.filepathpkl = setupDF_pkl_path
            SetupDF.__instance[full_id] = inst
        else:
            inst = object.__new__(cls)
            inst.loaded = False
            SetupDF.__instance[full_id] = inst
        inst.exists = False  # Need to set some things in __init__
        return SetupDF.__instance[full_id]

    def __init__(self, config=None):
        if self.exists is False:  # If already exists in current environment (i.e. already initialized)
            # self.config_name, self.filepath, self._dfbackupdir, self.wavenames, self._default_columns, self._default_data, self._dtypes = self.set_defaults()
            self.config_name, self.filepath, self._dfbackupdir, self.wavenames, self._default_columns, self._default_data, self._dtypes = self.set_defaults(
                config)
            self.name = 'setup'
            if self.loaded is False:
                self.df = pd.DataFrame(self._default_data, index=[0], columns=self._default_columns)
                self.set_dtypes()
                # self.df.set_index(['datnumplus'], inplace=True)  # sets multi index
                self.save()
            else:
                filepath_excel = os.path.join(self.filepath, f'{self.name}.xlsx')
                if os.path.isfile(filepath_excel):  # If excel of df only exists
                    self.df = DU.getexceldf(filepath_excel, comparisondf=self.df, dtypes=self._dtypes)

    def set_defaults(self, config: ConfigBase):
        """Sets defaults from config"""
        if config is None:
            config = src.DatObject.Make_Dat.default_config
        self.config_name = config.dir_name
        self.filepath = config.Directories.dfsetupdir
        self._dfbackupdir = config.Directories.dfbackupdir
        wns = []
        for l in config.get_exp_names_dict().values():
            for v in l:
                wns.append(v)
        self.wavenames = wns

        # self.config_name = cfg.current_config.__name__.split('.')[-1]
        # self.filepath = cfg.dfsetupdir
        # self._dfbackupdir = cfg.dfbackupdir
        # self.wavenames = cfg.common_wavenames
        self._default_columns = ['datetime', 'datnumplus'] + [name for name in self.wavenames]
        self._default_data = [['Wednesday, January 1, 2020 00:00:00', 0] + [1.0 for _ in range(len(self.wavenames))]]
        self._dtypes = dict(zip(self._default_columns, [object, int] + [float for i in range(
            len(self.wavenames))]))  # puts into form DataFrame can use
        # Can use 'converters' to make custom converter functions if necessary
        return self.config_name, self.filepath, self._dfbackupdir, self.wavenames, self._default_columns, self._default_data, self._dtypes

    def change_in_excel(self):
        """Lets user change df in excel. Does not save changes by default!!!"""
        filepath_excel = os.path.join(CU.get_full_path(self.filepath), f'{self.name}.xlsx')
        self.df = DU.change_in_excel(filepath_excel)
        return None

    # def load(self):
    #     """Loads from named pickle and excel then asks which to keep"""
    #     SetupDF.__instance = None
    #     inst = SetupDF.__new__(SetupDF)
    #     return inst

    @DU.temp_reset_index
    def save(self, backup=True):
        """saves df to pickle and excel at self.filepath locations (which default to where it was loaded from)"""
        self.df.set_index(['datnumplus'], inplace=True)
        self.df.sort_index(inplace=True)  # Always save with datnum as index in order
        filepath_excel = os.path.join(CU.get_full_path(self.filepath), f'{self.name}.xlsx')
        filepath_pkl = os.path.join(CU.get_full_path(self.filepath), f'{self.name}.pkl')

        if backup is True:
            self.backup()

        self.df.to_excel(filepath_excel)  # Can use pandasExcelWriter if need to do more fancy saving
        with open(filepath_pkl, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        return None

    def backup(self):
        """Saves copy of current pickle and excel to backup directory under current date"""
        if self.name is not None:
            backup_folder = CU.get_full_path(self._dfbackupdir)
            backup_dir = os.path.join(backup_folder, str(datetime.date.today()))
            os.makedirs(backup_dir, exist_ok=True)
            excel_path = os.path.join(CU.get_full_path(self.filepath), f'{self.name}.xlsx')
            pkl_path = os.path.join(CU.get_full_path(self.filepath), f'{self.name}.pkl')
            backup_name = 'SETUP_' + datetime.datetime.now().strftime('%H-%M_') + self.name
            if os.path.isfile(excel_path):
                shutil.copy2(excel_path, os.path.join(backup_dir, backup_name + '.xlsx'))
            if os.path.isfile(pkl_path):
                shutil.copy2(pkl_path, os.path.join(backup_dir, backup_name + '.pkl'))
        return None

    @DU.temp_reset_index
    def add_row(self, datetime, datnumplus, data):
        alldata = dict({'datetime': datetime, 'datnumplus': datnumplus}, **data)
        rowindex = self.df.last_valid_index() + 1
        for key, value in alldata.items():
            if key not in self.df.columns:  # If doesn't already exist ask for input
                inp = input(f'"{key}" not in SetupDF, do you want to add it?')
                if inp not in ['y', 'yes']:  # if don't want to add then loop back to next entry
                    continue
            self.df.loc[rowindex, key] = value

    @DU.temp_reset_index
    def get_valid_row(self, datnum, datetime=None) -> pd.Series:  # TODO: make work for entering time instead of datnum
        self.df.sort_values(by=['datnumplus'], inplace=True)
        rowindex = bisect(self.df['datnumplus'], datnum) - 1  # Returns closest higher index than datnum in datnumplus
        return self.df.loc[rowindex]

    @DU.temp_reset_index
    def set_dtypes(self):
        for key, value in self._dtypes.items():
            if type(value) == type:
                self.df[key] = self.df[key].astype(value)

    @staticmethod
    def killinstance():
        SetupDF.__instance = None
