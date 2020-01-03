from src import config
from src import config as cfg
import os
import pickle
import pandas as pd
from src.DFcode import DFutil
import inspect

@DFutil.singleton
class SetupDF(object):
    """
    Pandas Dataframe holding experiment setup data (such as divider info, current amp settings, other comments)
    """
    __instance_dict = {}  # Keeps track of whether SetupPD exists or not
    bobs = 50  # number of BOB I/Os
    _default_columns = ['datetime', 'datnumplus'] + [f'BOB{i}' for i in range(bobs + 1)]
    _default_data = ['Wednesday, January 1, 2020 00:00:00', 0] + [None for i in range(bobs+1)]
    _dtypes = [object, float]+[float for i in range(bobs+1)]
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
        setupDFpath = os.path.join(cfg.dfdir, 'setup.pkl')
        setupDFexcel = os.path.join(cfg.dfdir, 'setup.xlsx')

        pickleinst = DFutil.load_from_pickle(setupDFpath, cls)


    def __init__(self):
        pass


    # def __new__(cls, **kwargs):
    #     if inspect.stack()[1][3] == '__new__':  # If loading from pickle in this loop, don't start an infinite loop
    #         return super(DatDF, cls).__new__(cls)
    #     if 'dfname' in kwargs.keys() and kwargs['dfname'] is not None:
    #         name = kwargs['dfname']
    #     else:
    #         name = 'default'
    #     datDFpath = os.path.join(cfg.dfdir, f'{name}.pkl')
    #     datDFexcel = os.path.join(cfg.dfdir, f'{name}.xlsx')
    #     # TODO: Can add later way to load different versions, or save to a different version etc. Or backup by week or something
    #     if name not in cls.__instance_dict:  # If named datDF doesn't already exist
    #         if os.path.isfile(datDFpath):  # check if saved version exists
    #             with open(datDFpath, 'rb') as f:
    #                 inst = pickle.load(f)  # FIXME: This loops back to beginning of __new__, not sure why?!
    #             inst.loaded = True
    #             if not isinstance(inst, cls):  # Check if loaded version is actually a datPD
    #                 raise TypeError(f'File saved at {datDFpath} is not of the type {cls}')
    #             if os.path.isfile(datDFexcel):  # If excel of df only exists
    #                 tempdf = pd.read_excel(datDFexcel, index_col=[0,1], header=0, dtype=DatDF._dtypes)
    #                 if not DatDF.compare_to_df(inst.df, tempdf):
    #                     inp = input(f'datDF[{name}] has a different pickle and excel version of DF '
    #                                 f'do you want to use excel version?')
    #                     if inp.lower() in {'y', 'yes'}:  # Replace pickledf with exceldf
    #                         inst.df = tempdf
    #             cls.__instance_dict[name] = inst
    #         else:
    #             inst = object.__new__(cls)  # Otherwise create a new instance
    #             inst.loaded = False
    #             cls.__instance_dict[name] = inst
    #     else:
    #         # region Verbose DatDF __new__
    #         cls.__instance_dict[name].loaded = True  # loading from existing
    #         if cfg.verbose is True:
    #             verbose_message('DatPD already exists, returned same instance')
    #         # endregion
    #
    #     return cls.__instance_dict[name]  # Return the instance to __init__
    #
    # def __init__(self, **kwargs):
    #     if self.loaded is False:  # If not loaded from file need to create it
    #         mux = pd.MultiIndex.from_arrays([[0], ['base']], names=['datnum', 'datname'])  # Needs at least one row of data to save
    #         self.df = pd.DataFrame(DatDF._default_data, index=mux, columns=DatDF._default_columns)
    #         self.set_dtypes()
    #         # self.df = pd.DataFrame(columns=['time', 'picklepath'])  # TODO: Add more here
    #         if 'dfname' in kwargs.keys():
    #             name = kwargs['dfname']
    #         else:
    #             name = 'default'
    #         self.name = name
    #         self.save()  # So overwrites without asking (by here you have decided to overwrite anyway)
    #     else:  # Probably don't need to do much if loaded from file
    #         pass
    #     # region Verbose DatDF __init__
    #     if cfg.verbose is True:
    #         verbose_message('End of init of DatDF')
    #     # endregion
    #
    # def set_dtypes(self):
    #     for key, value in DatDF._dtypes.items():
    #         if type(value) == type:
    #             self.df[key] = self.df[key].astype(value)