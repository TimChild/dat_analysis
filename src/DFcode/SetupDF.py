
from src import config as cfg
import os
import pickle
import pandas as pd
from src.DFcode import DFutil
from bisect import bisect


pd.DataFrame.set_index = DFutil.protect_data_from_reindex(pd.DataFrame.set_index)  # Protect from deleting columns of data

class SetupDF(object):
    """
    Pandas Dataframe holding experiment setup data (such as divider info, current amp settings, other comments)
    """
    __instance = None
    bobs = 50  # number of BOB I/Os

    wavenames = cfg.commonwavenames
    _default_columns = ['datetime', 'datnumplus'] + [name for name in wavenames]
    _default_data = [['Wednesday, January 1, 2020 00:00:00', 0] + [None for _ in range(len(wavenames))]]
    _dtypes = [object, int]+[float for i in range(len(wavenames))]
    _dtypes = dict(zip(_default_columns, _dtypes))  # puts into form DataFrame can use
    # Can use 'converters' to make custom converter functions if necessary



    def __getnewargs_ex__(self):
        """When loading from pickle, this is passed into __new__"""
        args = None
        kwargs = {'frompickle': True}
        return (args,), kwargs

    def __new__(cls, *args, **kwargs):
        if 'frompickle' in kwargs.keys() and kwargs['frompickle'] is True:
            return super(SetupDF, cls).__new__(cls)
        if SetupDF.__instance is not None:  # If already existing instance return that instance.
            return SetupDF.__instance
        setupDFpath = cfg.dfsetupdirpkl
        setupDFexcel = cfg.dfsetupdirexcel

        inst = DFutil.load_from_pickle(setupDFpath, cls)  # Gets inst if it exists otherwise returns None
        if inst is not None:
            if os.path.isfile(setupDFexcel):
                exceldf = pd.read_excel(setupDFexcel, index_col=0, header=0, dtype=SetupDF._dtypes)
                assert exceldf.index.name == 'datnumplus'
                inst.df = DFutil.compare_pickle_excel(inst.df, exceldf, f'SetupDF')  # Returns either pickle or excel df depending on input
            inst.loaded = True
            SetupDF.__instance = inst
        else:
            inst = object.__new__(cls)
            inst.loaded = False
            SetupDF.__instance = inst
        return SetupDF.__instance

    def __init__(self):
        if self.loaded is False:
            self.df = pd.DataFrame(SetupDF._default_data, index=[0], columns=SetupDF._default_columns)
            self.set_dtypes()
            # self.df.set_index(['datnumplus'], inplace=True)  # sets multi index
            self.save()
        else:
            pass

    def save(self):
        """saves df to pickle and excel"""
        self.df.reset_index(inplace=True)
        self.df.set_index(['datnumplus'], inplace=True)
        self.df.sort_index(inplace=True)  # Always save with datnum as index in order
        self.df.to_excel(cfg.dfsetupdirexcel)  # Can use pandasExcelWriter if need to do more fancy saving
        with open(cfg.dfsetupdirpkl, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        return None

    @DFutil.temp_reset_index
    def add_row(self, datetime, datnumplus, data):
        alldata = dict({'datetime': datetime, 'datnumplus': datnumplus}, **data)
        rowindex = self.df.last_valid_index()+1
        for key, value in alldata.items():
            if key not in self.df.columns:  # If doesn't already exist ask for input
                inp = input(f'"{key}" not in SetupDF, do you want to add it?')
                if inp not in ['y', 'yes']:  # if don't want to add then loop back to next entry
                    continue
            self.df.loc[rowindex, key] = value

    def get_valid_row(self, datnum, datetime = None) -> pd.Series:  # TODO: make work for entering time instead of datnum
        self.df.sort_values(by=['datnumplus'], inplace=True)
        rowindex = bisect(self.df['datnumplus'])-1  # Returns closest higher index than datnum in datnumplus
        return self.df.loc[rowindex]

    def set_dtypes(self):
        for key, value in SetupDF._dtypes.items():
            if type(value) == type:
                self.df[key] = self.df[key].astype(value)

    @staticmethod
    def killinstance():
        SetupDF.__instance = None




if __name__ == '__main__':
    SetupDF.killinstance
