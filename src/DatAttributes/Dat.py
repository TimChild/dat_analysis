import inspect

import src.Configs.Main_Config as cfg
from src.CoreUtil import get_data_index
from src.DatAttributes import Logs, Data, Instruments, Entropy, Transition, Pinch, DCbias, Li_theta
import numpy as np
import src.PlottingFunctions as PF
from datetime import datetime
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

#
# class Dat(object):
#     """Overall Dat object which contains general information about dat, more detailed info should be put
#     into a subclass. Everything in this overall class should be useful for 99% of dats
#
#     Init only puts Dat in DF but doesn't save DF"""
#     version = '2.0'
#     """
#     Version history
#         1.1 -- Added version to dat, also added Li_theta
#         1.2 -- added self.config_name which stores name of config file used when initializing dat.
#         1.3 -- Can call _reset_transition() with fit_function=func now.  Can also stop auto initialization by adding
#             dattype = {'suppress_auto_calculate'}
#         2.0 -- All dat attributes now how .version and I am going to try update this version every time any other version changes
#     """
#
#     # def __new__(cls, *args, **kwargs):
#     #     return object.__new__(cls)
#
#     def __getattr__(self, name):  # __getattribute__ overrides all, __getattr__ overrides only missing attributes
#         # Note: This affects behaviour of hasattr(). Hasattr only checks if getattr returns a value, not whether
#         # attribute was defined previously.
#         raise AttributeError(f'Attribute {name} does not exist. Maybe want to implement getting attrs from datDF here')
#
#     def __setattr__(self, name, value):
#         # region Verbose Dat __setattr__
#         if cfg.verbose is True:
#             verbose_message(
#                 f'in override setattr. Being called from {inspect.stack()[1][3]}, hasattr is {hasattr(self, name)}')
#         # endregion
#         if not hasattr(self, name) and inspect.stack()[1][3] != '__init__':  # Inspect prevents this override
#             # affecting init
#             # region Verbose Dat __setattr__
#             if cfg.verbose is True:
#                 verbose_message(
#                     'testing setattr override')  # TODO: implement writing change to datPD at same time, maybe with a check?
#             # endregion
#             super().__setattr__(name, value)
#
#         else:
#             super().__setattr__(name, value)
#
#     def __init__(self, datnum: int, datname, infodict: dict, dfname='default'):
#         """Constructor for dat"""
#         self.version = Dat.version
#         self.config_name = cfg.current_config.__name__.split('.')[-1]
#         try:
#             self.dattype = set(infodict['dattypes'])
#         except KeyError:
#             self.dattype = {'none'}  # Can't check if str is in None, but can check if in ['none']
#         self.datnum = datnum
#         self.datname = datname
#         self.picklepath = None
#         self.hdf_path = infodict.get('hdfpath', None)
#         try:
#             self.Logs = Logs.Logs(infodict)
#         except Exception as e:
#             logger.warning(f'Error setting "Logs" for dat{self.datnum}: {e}')
#         try:
#             self.Instruments = Instruments.Instruments(infodict)
#         except Exception as e:
#             logger.warning(f'Error setting "Instruments" for dat{self.datnum}: {e}')
#         try:
#             self.Data = Data.Data(infodict)
#         except Exception as e:
#             logger.warning(f'Error setting "Data" for dat{self.datnum}: {e}')
#
#         if 'transition' in self.dattype and 'suppress_auto_calculate' not in self.dattype:
#             self._reset_transition()
#         if 'entropy' in self.dattype and self.Data.entx is not None and 'suppress_auto_calculate' not in self.dattype:
#             self._reset_entropy()
#         if 'pinch' in self.dattype and 'suppress_auto_calculate' not in self.dattype:
#             self.Pinch = Pinch.Pinch(self.Data.x_array, self.Data.current)
#         if 'dcbias' in self.dattype and 'suppress_auto_calculate' not in self.dattype:
#             self._reset_dcbias()
#         if 'li_theta' in self.dattype and 'suppress_auto_calculate' not in self.dattype:
#             self._reset_li_theta()
#
#         self.dfname = dfname
#         self.date_initialized = datetime.now().date()
#
#     def _reset_li_theta(self):
#         self.Li_theta = Li_theta.Li_theta(self.hdf_path, self.Data.li_theta_keys, self.Data.li_multiplier)
#
#     def _reset_transition(self, fit_function=None):
#         try:
#             self.Transition = Transition.Transition(self.Data.x_array, self.Data.i_sense, fit_function=fit_function)
#             self.dattype = set(self.dattype)  # Required while I transition to using a set for dattype
#             self.dattype.add('transition')
#         except Exception as e:
#             print(f'Error while calculating Transition: {e}')
#
#     def _reset_entropy(self):
#         try:
#             try:
#                 mids = self.Transition.fit_values.mids
#                 thetas = self.Transition.fit_values.thetas
#             except AttributeError:
#                 raise ValueError('Mids is now a required parameter for Entropy. Need to pass in some mid values relative to x_array')
#             self.Entropy = Entropy.Entropy(self.Data.x_array, self.Data.entx, mids, enty=self.Data.enty, thetas=thetas)
#             self.dattype = set(self.dattype)  # Required while I transition to using a set for dattype
#             self.dattype.add('entropy')
#         except Exception as e:
#             print(f'Error while calculating Entropy: {e}')
#
#     def _reset_dcbias(self):
#         try:
#             self.DCbias = DCbias.DCbias(self.Data.x_array, self.Data.y_array, self.Data.i_sense, self.Transition.fit_values)
#             self.dattype = set(self.dattype)  # Required while I transition to using a set for dattype
#             self.dattype.add('dcbias')
#         except Exception as e:
#             print(f'Error while calculating DCbias: {e}')
#
#
#     def plot_standard_info(self, mpl_backend='qt', raw_data_names=[], fit_attrs=None, dfname='default', **kwargs):
#         extra_info = {'duration': self.Logs.time_elapsed, 'temp': self.Logs.temps.get('mc', np.nan)*1000}
#         if fit_attrs is None:
#             fit_attrs = {}
#             if 'transition' in self.dattype and 'dcbias' not in self.dattype:
#                 fit_attrs['Transition'] = ['amps', 'thetas']
#             if 'dcbias' in self.dattype:
#                 fit_attrs['Transition']=['thetas']
#             if 'entropy' in self.dattype:
#                 fit_attrs['Entropy']=['dSs']
#                 fit_attrs['Transition']=['amps']
#             if 'pinch' in self.dattype:
#                 pass
#         PF.standard_dat_plot(self, mpl_backend=mpl_backend, raw_data_names=raw_data_names, fit_attrs=fit_attrs, dfname=dfname, **kwargs)
#
#     def display(self, data, ax=None, xlabel: str = None, ylabel: str = None, swapax=False, norm=None, colorscale=True,
#                 axtext=None, dim=None,**kwargs):
#         """Just displays 1D or 2D data using x and y array of dat. Can pass in option kwargs"""
#         x = self.Logs.x_array
#         y = self.Logs.y_array
#         if dim is None:
#             dim = self.Logs.dim
#         if xlabel is None:
#             xlabel = self.Logs.x_label
#         if ylabel is None:
#             ylabel = self.Logs.y_label
#         if swapax is True:
#             x = y
#             y = self.Logs.x_array
#             data = np.swapaxes(data, 0, 1)
#         if axtext is None:
#             axtext = f'Dat{self.datnum}'
#         ax = PF.get_ax(ax)
#         if dim == 2:
#             PF.display_2d(x, y, data, ax, norm, colorscale, xlabel, ylabel, axtext=axtext, **kwargs)
#         elif dim == 1:
#             PF.display_1d(x, data, ax, xlabel, ylabel, axtext=axtext, **kwargs)
#         else:
#             raise ValueError('No value of "dim" present to determine which plotting to use')
#         return ax
#
#     def display1D_slice(self, data, yval, ax=None, xlabel: str = None, yisindex=False, fontsize=10, textpos=(0.1, 0.8),
#                         **kwargs) -> (plt.Axes, int):
#         """
#
#         @param data: 2D data
#         @type data: np.ndarray
#         @param yval: real or index value of y to slice at
#         @type yval: Union[int, float]
#         @param ax: Axes
#         @type ax: plt.Axes
#         @param xlabel:
#         @type xlabel: str
#         @param yisindex: Whether yval is real or index
#         @type yisindex: bool
#         @param fontsize:
#         @type fontsize:
#         @param textpos: tuple of proportional coords
#         @type textpos: tuple
#         @param kwargs:
#         @type kwargs:
#         @return: Returns axes with 1D slice and index of y value used
#         @rtype: (plt.Axes, int)
#         """
#         """Returns 1D plot of 2D data (takes 2D data as input) and index of the y value used"""
#         # TODO: make work for vertical slice
#         ax = PF.get_ax(ax)
#         if yisindex is False:
#             idy = get_data_index(self.Data.y_array, yval)
#         else:
#             idy = yval
#             yval = self.Data.y_array[idy]
#         data = data[idy]
#         if 'axtext' in kwargs.keys() and kwargs['axtext']:
#             axtext = f'Dat={self.datnum}\n@{yval:.1f}mV'
#             kwargs['axtext'] = axtext
#         if 'textpos' in kwargs.keys() and kwargs['textpos']:
#             kwargs['textpos'] = textpos
#         self.display(data, ax, xlabel, dim=1, **kwargs)
#         return ax, idy
#



##################################################


if __name__ == '__main__':
    pass
