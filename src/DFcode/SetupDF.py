from src import config
from src import config as cfg
import os
import pickle
import pandas as pd
from src.DFcode import DFutil
from singleton_decorator import singleton


class SetupDF(object):
    """
    Pandas Dataframe holding experiment setup data (such as divider info, current amp settings, other comments)
    """
    __instance = None
    bobs = 50  # number of BOB I/Os
    _default_columns = ['datetime', 'datnumplus'] + [f'BOB{i}' for i in range(bobs + 1)]
    _default_data = [['Wednesday, January 1, 2020 00:00:00', 0] + [None for i in range(bobs+1)]]
    _dtypes = [object, float]+[float for i in range(bobs+1)]
    _dtypes = dict(zip(_default_columns, _dtypes))  # puts into form DataFrame can use
    # Can use 'converters' to make custom converter functions if necessary

    setupDFpath = os.path.join(cfg.dfdir, 'setup/setup.pkl')
    setupDFexcel = os.path.join(cfg.dfdir, 'setup/setup.xlsx')

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
        setupDFpath = SetupDF.setupDFpath
        setupDFexcel = SetupDF.setupDFexcel

        inst = DFutil.load_from_pickle(setupDFpath, cls)  # Gets inst if it exists otherwise returns None
        if inst is not None:
            if os.path.isfile(setupDFexcel):
                exceldf = pd.read_excel(setupDFexcel, index_col=0, header=0, dtype=SetupDF._dtypes)
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
            self.df.set_index(['datetime', 'datnumplus'])  # sets multi index
            self.save()
        else:
            pass

    def save(self):
        """saves df to pickle and excel"""
        self.df.to_excel(SetupDF.setupDFexcel)  # Can use pandasExcelWriter if need to do more fancy saving
        with open(SetupDF.setupDFpath, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        return None


    def set_dtypes(self):
        for key, value in SetupDF._dtypes.items():
            if type(value) == type:
                self.df[key] = self.df[key].astype(value)

    @staticmethod
    def killinstance():
        SetupDF.__instance = None

if __name__ == '__main__':
    SetupDF.killinstance
