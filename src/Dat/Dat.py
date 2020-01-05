import inspect
from typing import List, Tuple, Union, NamedTuple

from src import config as cfg
from src.CoreUtil import verbose_message


class Dat(object):
    """Overall Dat object which contains general information about dat, more detailed info should be put
    into a subclass. Everything in this overall class should be useful for 99% of dats

    Init only puts Dat in DF but doesn't save DF"""

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
            dattype = infodict['dattypes']
        except:
            dattype = 'none'  # Can't check if str is in None, but can check if in 'none'
        self.datnum = datnum
        if 'datname' in infodict:
            self.datname = datname
        else:
            self.datname = 'base'
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

        # TODO: These should be classes inside of the overall class s.t. the dat object typing is not overcrowded
        if 'i_sense' in dattype:
            self.i_sense = infodict[
                'i_sense']  # type: np.ndarray  # Charge sensor current in nA  # TODO: Do I want to move this to a subclass?
        if 'entropy' in dattype:
            # TODO: Then init subclass entropy dat here??
            # self.__init_subclass__(Entropy_Dat)
            pass
        self.dfname = dfname


    def instr_vals(self, name: str, data: List[NamedTuple]):
        if data is not None:
            for ntuple in data:  # data should be a List of namedtuples for instrument, First field should be ID (e.g. 1 or x)
                evalstr = f'self.{name}{ntuple[0]} = {ntuple}'
                exec(evalstr)
        return None
