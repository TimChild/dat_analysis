import inspect

import src.Configs.Main_Config as cfg
from src.CoreUtil import verbose_message
from src.DatCode.Logs import Logs
from src.DatCode.Data import Data
from src.DatCode.Instruments import Instruments
from src.DatCode.Entropy import Entropy
from src.DatCode.Transition import Transition
from src.DatCode.Pinch import Pinch
from src.DatCode.DCbias import DCbias
from src.DatCode.Li_theta import Li_theta
import numpy as np
import src.PlottingFunctions as PF
import src.DatCode.Datutil as DU
from datetime import datetime
import matplotlib.pyplot as plt
import sys

class Dat(object):
    """Overall Dat object which contains general information about dat, more detailed info should be put
    into a subclass. Everything in this overall class should be useful for 99% of dats

    Init only puts Dat in DF but doesn't save DF"""
    __version = '1.1'
    """
    Version history
        1.1 -- Added version to dat, also added Li_theta
    """

    # def __new__(cls, *args, **kwargs):
    #     return object.__new__(cls)

    def __getattr__(self, name):  # __getattribute__ overrides all, __getattr__ overrides only missing attributes
        # Note: This affects behaviour of hasattr(). Hasattr only checks if getattr returns a value, not whether
        # attribute was defined previously.
        raise AttributeError(f'Attribute {name} does not exist. Maybe want to implement getting attrs from datDF here')

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
            super().__setattr__(name, value)

        else:
            super().__setattr__(name, value)

    def __init__(self, datnum: int, datname, infodict: dict, dfname='default'):
        """Constructor for dat"""
        self.version = Dat.__version
        try:
            self.dattype = set(infodict['dattypes'])
        except KeyError:
            self.dattype = {'none'}  # Can't check if str is in None, but can check if in ['none']
        self.datnum = datnum
        if 'datname' in infodict:
            self.datname = datname
        else:
            self.datname = 'base'
        self.picklepath = None
        self.hdf_path = infodict.get('hdfpath', None)
        self.Logs = Logs(infodict)
        self.Instruments = Instruments(infodict)
        self.Data = Data(infodict)

        if 'transition' in self.dattype:
            self._reset_transition()
        if 'entropy' in self.dattype and self.Data.entx is not None:
            self._reset_entropy()
        if 'pinch' in self.dattype:
            self.Pinch = Pinch(self.Data.x_array, self.Data.current)
        if 'dcbias' in self.dattype:
            self._reset_dcbias()
        if 'li_theta' in self.dattype:
            self._reset_li_theta()

        self.dfname = dfname
        self.date_initialized = datetime.now().date()

    def _reset_li_theta(self):
        self.Li_theta = Li_theta(self.hdf_path, self.Data.li_theta_keys, self.Data.li_multiplier)

    def _reset_transition(self):
        try:
            self.Transition = Transition(self.Data.x_array, self.Data.i_sense)
            self.dattype = set(self.dattype)  # Required while I transition to using a set for dattype
            self.dattype.add('transition')
        except:
            e = sys.exc_info()[0]
            print(f'Error while calculating Transition: {e}')

    def _reset_entropy(self):
        try:
            try:
                mids = self.Transition.fit_values.mids
                thetas = self.Transition.fit_values.thetas
            except AttributeError:
                raise ValueError('Mids is now a required parameter for Entropy. Need to pass in some mid values relative to x_array')
            self.Entropy = Entropy(self.Data.x_array, self.Data.entx, mids, enty=self.Data.enty, thetas=thetas)
            self.dattype = set(self.dattype)  # Required while I transition to using a set for dattype
            self.dattype.add('entropy')
        except:
            e = sys.exc_info()[0]
            print(f'Error while calculating Entropy: {e}')

    def _reset_dcbias(self):
        try:
            self.DCbias = DCbias(self.Data.x_array, self.Data.y_array, self.Data.i_sense, self.Transition.fit_values)
            self.dattype = set(self.dattype)  # Required while I transition to using a set for dattype
            self.dattype.add('dcbias')
        except:
            e = sys.exc_info()[0]
            print(f'Error while calculating DCbias: {e}')


    def plot_standard_info(self, mpl_backend='qt', raw_data_names=[], fit_attrs=None, dfname='default', **kwargs):
        extra_info = {'duration': self.Logs.time_elapsed, 'temp': self.Logs.temps.get('mc', np.nan)*1000}
        if fit_attrs is None:
            fit_attrs = {}
            if 'transition' in self.dattype and 'dcbias' not in self.dattype:
                fit_attrs['Transition'] = ['amps', 'thetas']
            if 'dcbias' in self.dattype:
                fit_attrs['Transition']=['thetas']
            if 'entropy' in self.dattype:
                fit_attrs['Entropy']=['dSs']
                fit_attrs['Transition']=['amps']
            if 'pinch' in self.dattype:
                pass
        PF.standard_dat_plot(self, mpl_backend=mpl_backend, raw_data_names=raw_data_names, fit_attrs=fit_attrs, dfname=dfname, **kwargs)

    def display(self, data, ax=None, xlabel: str = None, ylabel: str = None, swapax=False, norm=None, colorscale=True,
                axtext=None, dim=None,**kwargs):
        """Just displays 1D or 2D data using x and y array of dat. Can pass in option kwargs"""
        x = self.Logs.x_array
        y = self.Logs.y_array
        if dim is None:
            dim = self.Logs.dim
        if xlabel is None:
            xlabel = self.Logs.x_label
        if ylabel is None:
            ylabel = self.Logs.y_label
        if swapax is True:
            x = y
            y = self.Logs.x_array
            data = np.swapaxes(data, 0, 1)
        if axtext is None:
            axtext = f'Dat{self.datnum}'
        ax = PF.get_ax(ax)
        if dim == 2:
            PF.display_2d(x, y, data, ax, norm, colorscale, xlabel, ylabel, axtext=axtext, **kwargs)
        elif dim == 1:
            PF.display_1d(x, data, ax, xlabel, ylabel, axtext=axtext, **kwargs)
        else:
            raise ValueError('No value of "dim" present to determine which plotting to use')
        return ax

    def display1D_slice(self, data, yval, ax=None, xlabel: str = None, yisindex=False, fontsize=10, textpos=(0.1, 0.8),
                        **kwargs) -> (plt.Axes, int):
        """

        @param data: 2D data
        @type data: np.ndarray
        @param yval: real or index value of y to slice at
        @type yval: Union[int, float]
        @param ax: Axes
        @type ax: plt.Axes
        @param xlabel:
        @type xlabel: str
        @param yisindex: Whether yval is real or index
        @type yisindex: bool
        @param fontsize:
        @type fontsize:
        @param textpos: tuple of proportional coords
        @type textpos: tuple
        @param kwargs:
        @type kwargs:
        @return: Returns axes with 1D slice and index of y value used
        @rtype: (plt.Axes, int)
        """
        """Returns 1D plot of 2D data (takes 2D data as input) and index of the y value used"""
        # TODO: make work for vertical slice
        ax = PF.get_ax(ax)
        if yisindex is False:
            idy, yval = DU.get_id_from_val(self.Data.y_array, yval)
        else:
            idy = yval
            yval = self.Data.y_array[idy]
        data = data[idy]
        if 'axtext' in kwargs.keys() and kwargs['axtext']:
            axtext = f'Dat={self.datnum}\n@{yval:.1f}mV'
            kwargs['axtext'] = axtext
        if 'textpos' in kwargs.keys() and kwargs['textpos']:
            kwargs['textpos'] = textpos
        self.display(data, ax, xlabel, dim=1, **kwargs)
        return ax, idy
