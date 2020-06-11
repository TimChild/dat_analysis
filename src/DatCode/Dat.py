import inspect

import src.Configs.Main_Config as cfg
from src.CoreUtil import verbose_message
import src.CoreUtil as CU
from src.DatCode import Logs, Data, Instruments, Entropy, Transition, Pinch, DCbias, Li_theta
import numpy as np
import src.PlottingFunctions as PF
import src.DatCode.Datutil as DU
from datetime import datetime
import matplotlib.pyplot as plt
import src.DatHDF.Util as HU
import h5py
import os
import logging
from dictor import dictor
from src.DatCode import DatAttribute as DA
import src.Exp_to_standard as E2S
import abc
logger = logging.getLogger(__name__)


BASE_ATTRS = ['datnum', 'datname', 'dfname', 'dat_id', 'dattypes', 'config_name', 'date_initialized']


class Dat(object):
    """Overall Dat object which contains general information about dat, more detailed info should be put
    into a subclass. Everything in this overall class should be useful for 99% of dats

    Init only puts Dat in DF but doesn't save DF"""
    version = '2.0'
    """
    Version history
        1.1 -- Added version to dat, also added Li_theta
        1.2 -- added self.config_name which stores name of config file used when initializing dat.
        1.3 -- Can call _reset_transition() with fit_function=func now.  Can also stop auto initialization by adding 
            dattype = {'suppress_auto_calculate'}
        2.0 -- All dat attributes now how .version and I am going to try update this version every time any other version changes
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
        self.version = Dat.version
        self.config_name = cfg.current_config.__name__.split('.')[-1]
        try:
            self.dattype = set(infodict['dattypes'])
        except KeyError:
            self.dattype = {'none'}  # Can't check if str is in None, but can check if in ['none']
        self.datnum = datnum
        self.datname = datname
        self.picklepath = None
        self.hdf_path = infodict.get('hdfpath', None)
        self.Logs = Logs(infodict)
        self.Instruments = Instruments(infodict)
        self.Data = Data(infodict)

        if 'transition' in self.dattype and 'suppress_auto_calculate' not in self.dattype:
            self._reset_transition()
        if 'entropy' in self.dattype and self.Data.entx is not None and 'suppress_auto_calculate' not in self.dattype:
            self._reset_entropy()
        if 'pinch' in self.dattype and 'suppress_auto_calculate' not in self.dattype:
            self.Pinch = Pinch(self.Data.x_array, self.Data.current)
        if 'dcbias' in self.dattype and 'suppress_auto_calculate' not in self.dattype:
            self._reset_dcbias()
        if 'li_theta' in self.dattype and 'suppress_auto_calculate' not in self.dattype:
            self._reset_li_theta()

        self.dfname = dfname
        self.date_initialized = datetime.now().date()

    def _reset_li_theta(self):
        self.Li_theta = Li_theta(self.hdf_path, self.Data.li_theta_keys, self.Data.li_multiplier)

    def _reset_transition(self, fit_function=None):
        try:
            self.Transition = Transition(self.Data.x_array, self.Data.i_sense, fit_function=fit_function)
            self.dattype = set(self.dattype)  # Required while I transition to using a set for dattype
            self.dattype.add('transition')
        except Exception as e:
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
        except Exception as e:
            print(f'Error while calculating Entropy: {e}')

    def _reset_dcbias(self):
        try:
            self.DCbias = DCbias(self.Data.x_array, self.Data.y_array, self.Data.i_sense, self.Transition.fit_values)
            self.dattype = set(self.dattype)  # Required while I transition to using a set for dattype
            self.dattype.add('dcbias')
        except Exception as e:
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




##################################################


def get_dat_id(datnum, datname, dfname):
    """Returns unique dat_id within one experiment. (i.e. specific to whichever DF the dat is a member of)"""
    name = f'Dat{datnum}'
    if datname != 'base':
        name += f'[{datname}]'
    if dfname != 'default':
        name += f'_{dfname}'
    return name


class NewDat(object):
    """Overall Dat object which contains general information about dat, more detailed info should be put
    into DatAttribute classes. Everything in this overall class should be useful for 99% of dats

    Init only puts Dat in DF but doesn't save DF"""
    version = '3.0'
    """
    Version history
        1.1 -- Added version to dat, also added Li_theta
        1.2 -- added self.config_name which stores name of config file used when initializing dat.
        1.3 -- Can call _reset_transition() with fit_function=func now.  Can also stop auto initialization by adding 
            dattype = {'suppress_auto_calculate'}
        2.0 -- All dat attributes now how .version and I am going to try update this version every time any other version changes
        3.0 -- Moving to HDF based save files
    """

    def __init__(self, datnum: int, datname, dat_hdf, dfname='default', Data=None, Logs=None, Instruments=None, Entropy=None, Transition=None, DCbias=None):
        """Constructor for dat"""
        self.version = NewDat.version
        self.config_name = cfg.current_config.__name__.split('.')[-1]
        self.dattype = None
        self.datnum = datnum
        self.datname = datname
        self.hdf = dat_hdf

        self.dfname = dfname
        self.date_initialized = datetime.now().date()

        self.Data = Data
        self.Logs = Logs
        self.Instruments = Instruments
        self.Entropy = Entropy
        self.Transition = Transition
        self.DCbias = DCbias

    def __del__(self):
        self.hdf.close()  # Close HDF when object is destroyed

    def _reset_transition(self, fit_function=None):
        pass
        # try:
        #     self.Transition = Transition(self.Data.x_array, self.Data.i_sense, fit_function=fit_function)
        #     self.dattype = set(self.dattype)  # Required while I transition to using a set for dattype
        #     self.dattype.add('transition')
        # except Exception as e:
        #     print(f'Error while calculating Transition: {e}')

    def _reset_entropy(self):
        pass
        # try:
        #     try:
        #         mids = self.Transition.fit_values.mids
        #         thetas = self.Transition.fit_values.thetas
        #     except AttributeError:
        #         raise ValueError('Mids is now a required parameter for Entropy. Need to pass in some mid values relative to x_array')
        #     self.Entropy = Entropy(self.Data.x_array, self.Data.entx, mids, enty=self.Data.enty, thetas=thetas)
        #     self.dattype = set(self.dattype)  # Required while I transition to using a set for dattype
        #     self.dattype.add('entropy')
        # except Exception as e:
        #     print(f'Error while calculating Entropy: {e}')

    def _reset_dcbias(self):
        pass
        # try:
        #     self.DCbias = DCbias(self.Data.x_array, self.Data.y_array, self.Data.i_sense, self.Transition.fit_values)
        #     self.dattype = set(self.dattype)  # Required while I transition to using a set for dattype
        #     self.dattype.add('dcbias')
        # except Exception as e:
        #     print(f'Error while calculating DCbias: {e}')


class NewDatLoader(abc.ABC):
    def __init__(self, datnum=None, datname=None, dfname=None, file_path=None):
        if file_path is not None:
            assert all([datnum is None, datname is None, dfname is None])
            self.hdf = h5py.File(file_path, 'r+')
        else:
            assert datnum is not None
            datname = datname if datname else 'base'
            dfname = dfname if dfname else 'default'
            self.hdf = h5py.File(HU.get_dat_hdf_path(get_dat_id(datnum, datname, dfname)))

        # Base attrs
        self.datnum = None
        self.datname = None
        self.dfname = None
        self.dat_id = None
        self.dattypes = None
        self.config_name = None
        self.date_initialized = None

        self.get_Base_attrs()
        self.Data = Data.NewData(self.hdf)
        self.Logs = Logs.NewLogs(self.hdf)
        self.Instruments = None # TODO: Replace with Instruments
        # self.Instruments = Instruments.NewInstruments(self.hdf)

    def get_Base_attrs(self):
        for key in BASE_ATTRS:
            setattr(self, key, self.hdf.attrs.get(key, None))

    @abc.abstractmethod
    def build_dat(self) -> NewDat:
        """Override to add checks for Entropy/Transition etc"""
        return NewDat(self.datnum, self.datname, self.hdf, self.dfname, Data=self.Data, Logs=self.Logs, Instruments=self.Instruments)


class NewDatBuilder(abc.ABC):
    def __init__(self, datnum, datname, dfname='default', load_overwrite='load'):
        # Init with basic info at least - enough to Identify DatHDF
        assert load_overwrite in ['load', 'overwrite']
        overwrite = True if load_overwrite == 'overwrite' else False
        # Base attrs for Dat
        self.datnum = datnum
        self.datname = datname
        self.dfname = dfname
        self.config_name = cfg.current_config.__name__.split('.')[-1]
        self.date_initialized = datetime.now().date()
        self.dat_id = get_dat_id(datnum, datname, dfname)
        self.dattypes = None

        self.hdf_path = HU.get_dat_hdf_path(self.dat_id, dir=None, overwrite=overwrite)  # Location of My HDF which will store everything to do with dat
        self.hdf = h5py.File(self.hdf_path, 'r+')

        self.copy_exp_hdf()  # Will copy Experiment HDF if not already existing

        # Init General Dat attributes to None
        self.Data = None
        self.Logs = None
        self.Instruments = None

        # Basic Inits which are sufficient if data exists in HDF already. Otherwise need to be built elsewhere
        self.init_Data()
        self.init_Logs()
        self.init_Instruments()

    def init_Base(self):
        """ For storing Base info in HDF attrs
        Note that dattypes won't be set here!"""
        hdf = self.hdf
        for attr, val in zip(BASE_ATTRS, [self.datnum, self.datname, self.dfname, self.dat_id, self.dattypes, self.config_name, self.date_initialized]):
            hdf.attrs[attr] = val

    @abc.abstractmethod
    def set_dattypes(self, value=None):
        """Reminder to set dattypes attr in HDF at some point"""
        self.dattypes = value if value else self.dattypes
        self.hdf['dattypes'] = self.dattypes

    def copy_exp_hdf(self):
        if 'Exp_measured_data' not in self.hdf.keys() or 'Exp_metadata' not in self.hdf.keys():  # Only if first time
            hdfpath = CU.get_full_path(os.path.join(cfg.ddir, f'dat{self.datnum:d}.h5'))
            if os.path.isfile(CU.get_full_path(hdfpath)):  # Only if original HDF exists
                e_data = self.hdf.create_group('Exp_measured_data')
                with h5py.File(CU.get_full_path(hdfpath), 'r') as hdf:
                    for key in hdf.keys():
                        if isinstance(hdf[key], h5py.Dataset) and key not in self.hdf['Exp_measured_data'].keys():  # Only once
                            ds = hdf[key]
                            e_data[key] = ds[:]
                        elif isinstance(hdf[key], h5py.Group) and key not in self.hdf.keys():
                            hdf.copy(hdf[key], self.hdf, 'Exp_metadata')
                self.hdf.flush()  # writes changes to file
            else:
                raise FileNotFoundError(f'Did not find HDF at {hdfpath}')

    def init_Data(self, setup_dict=None, exp_hdf=None):
        """
        @param setup_dict: dict formatted as {<standard_name>:[<exp_name(s), multiplier(s), offset(s)*], ...}  *optional
            there MUST be a multiplier and offset for every possible exp_name
        @type setup_dict: Dict[list]
        @return: Sets attributes in Data
        @rtype: None
        """
        self.Data = self.Data if self.Data else Data.NewData(self.hdf)
        if None not in [setup_dict, exp_hdf]:
            for item in setup_dict.items():
                standard_name = item[0]
                info = item[1]
                exp_names = CU.ensure_list(info[0])  # All possible names in exp
                exp_name, index = match_name_in_exp_hdf(exp_names, exp_hdf)  # First name which matches a dataset
                multiplier = info[1][index]
                offset = info[2][index] if len(info) == 3 else 0
                if multiplier == 1 and offset == 0:  # Just link to exp data
                    self.Data.link_data(standard_name, exp_name)
                else:  # duplicate and alter dataset before saving in HDF
                    full_name = 'Exp_'+exp_name
                    data = self.Data.group.get(full_name)[:]  # Get copy of exp Data
                    data = data*multiplier+offset  # Adjust as necessary
                    self.Data.set_data(standard_name, data)  # Store as new data in HDF

    @staticmethod
    def _init_logs_set_simple_attrs(grp, js):
        # Simple attrs
        grp.attrs['comments'] = dictor(js, 'comment', '')
        grp.attrs['filenum'] = dictor(js, 'filenum', 0)
        grp.attrs['x_label'] = dictor(js, 'axis_labels.x', 'None')
        grp.attrs['y_label'] = dictor(js, 'axis_labels.y', 'None')
        grp.attrs['current_config'] = dictor(js, 'current_config', None)
        grp.attrs['time_completed'] = dictor(js, 'time_completed', None)
        grp.attrs['time_elapsed'] = dictor(js, 'time_elapsed', None)

    @staticmethod
    def _init_logs_set_srss(group, json):
        for i in range(1, cfg.current_config.instrument_num['srs'] + 1 + 1):
            if f'SRS_{i}' in json.keys():
                srs_dict = dictor(json, f'SRS_{i}', checknone=True)
                srs_data = E2S.srs_from_json(srs_dict, i)  # Converts to my standard
                srs_id, srs_tuple = DA.get_key_ntuple('srs', i)  # Gets named tuple to store
                ntuple = CU.data_to_NamedTuple(srs_data, srs_tuple)  # Puts data into named tuple

                srs_group = group.require_group(f'srss/{srs_id}')  # Make group in HDF
                for key in ntuple:
                    srs_group.attrs[key] = ntuple[key]  # Store as attrs of group in HDF
            else:
                logger.info(f'No "SRS_{i}" found in json')

    @staticmethod
    def _init_logs_set_babydac(group, json):
        if 'BabyDAC' in json.keys():
            """dac dict should be stored in format:
                    visa_address: ...
                    
            
            """ # TODO: Fill this in
            dacs_group = group.create_group('dacs')
            bdac_dict = dictor(json, 'BabyDAC')
            Logs.save_simple_dict_to_hdf(dacs_group, bdac_dict)

        else:
            logger.info(f'No "BabyDAC" found in json')

    @staticmethod
    def _init_logs_set_fastdac(group, json):
        if 'FastDAC 1' in json.keys():
            """fdac dict should be stored in format:
                            visa_address: ...
                            SamplingFreq:
                            DAC#{<name>}: <val>
                            ADC#: <val>

                            ADCs not currently required
                            """
            fdac_group = group.create_group('FastDAC 1')
            fdac_json = json['FastDAC 1']
            Logs.save_simple_dict_to_hdf(fdac_group, fdac_json)
        else:
            logger.info(f'No "FastDAC" found in json')

    def init_Logs(self, json=None):
        self.Logs = self.Logs if self.Logs else Logs.NewLogs(self.hdf)
        if json is not None:
            group = self.Logs.group

            # Simple attrs
            self._init_logs_set_simple_attrs(group, json)

            # Instr attrs
            self._init_logs_set_srss(group, json)
            self._init_logs_set_babydac(group, json)
            self._init_logs_set_fastdac(group, json)

            # TODO: add mags
            # for i in range(1, cfg.current_config.instrument_num['mags']+1+1):
            #     if f'Mag...':
            #         pass

            self.hdf.flush()
            self.Logs.get_from_HDF()

    def init_Instruments(self):
        assert self.Logs is not None
        # TODO: copy links from relevant groups in logs to Instruments
        pass

    @abc.abstractmethod
    def build_dat(self) -> NewDat:
        """Override if passing more info to NewDat"""
        return NewDat(self.datnum, self.datname, self.hdf, self.dfname, Data=self.Data, Logs=self.Logs, Instruments=self.Instruments)


def match_name_in_exp_hdf(exp_names, exp_hdf):
    """
    Returns the first name from exp_names which is a dataset in exp_hdf

    @param exp_names: list of expected names in exp_dataset
    @type exp_names: Union[str, list]
    @param exp_hdf: The experiment hdf (or group) to look for datasets in
    @type exp_hdf: Union[h5py.File, h5py.Group]
    @return: First name which is a dataset or None if not found
    @rtype: Union[str, None]

    """
    exp_names = CU.ensure_list(exp_names)
    for i, name in enumerate(exp_names):
        if name in exp_hdf.keys() and isinstance(exp_hdf[name], h5py.Dataset):
            return name, i
    logger.warning(f'[{exp_names}] not found in [{exp_hdf.name}]')
    return None

if __name__ == '__main__':
    pass
