"""These should only use data which is in a standard form (i.e. using DataStandardize first)"""

import abc
import os
from datetime import datetime
from typing import Union, Type, Optional
import h5py
import numpy as np
import subprocess
from dictor import dictor
from src.Builders import Util
from src.DatObject.Attributes import Transition as T, Data, Instruments, Entropy as E, Other, Logs as L, AWG, SquareEntropy as SE
from src.DatObject import DatHDF
from src import HDF_Util as HDU, CoreUtil as CU
from src.DataStandardize import Standardize_Util as E2S
import logging

logger = logging.getLogger(__name__)


class NewDatBuilder(abc.ABC):
    """Base DatHDF builder class. Only contains the core DatObject Logs, Data, Instruments, Other. Any others should be
    added in a subclass of this
    This is only for creating the necessary HDF file to be able to Load a dat from it. Use a DatLoader to get an actual
    dat object"""

    def __init__(self, datnum, datname, hdfdir, overwrite=False):
        # Init with basic info at least - enough to Identify DatHDF
        # Base attrs for Dat
        self.datnum = datnum
        self.datname = datname
        # self.config_name = cfg.current_config.__name__.split('.')[-1]  # TODO: Remove this
        self.date_initialized = datetime.now().date()
        self.dat_id = CU.get_dat_id(datnum, datname)
        self.dattypes = None

        self.hdf_path = HDU.get_dat_hdf_path(self.dat_id, hdfdir,
                                             overwrite=overwrite)  # Location of My HDF which will store everything to do with dat
        self.hdf = h5py.File(self.hdf_path, 'r+')  # Open file in Read/Write mode

        # Init General Dat attributes to None
        self.Data: Data.NewData = None
        self.Logs: L.NewLogs = None
        self.Instruments: Instruments.NewInstruments = None
        self.Other: Other.Other = None

    def copy_exp_hdf(self, ddir):
        """Copy experiment HDF data into my HDF file if not done already"""
        self.Data = self.Data if self.Data else Data.NewData(self.hdf)  # Will init Data from Dat HDF if already exists,
        if 'Exp_measured_data' not in self.hdf.keys() or 'Exp_metadata' not in self.hdf.keys():  # Only if first time
            hdfpath = os.path.join(ddir, f'dat{self.datnum:d}.h5')
            if os.path.isfile(hdfpath):  # Only if original HDF exists
                e_data = self.hdf.create_group('Exp_measured_data')
                with h5py.File(hdfpath, 'r') as hdf:
                    for key in hdf.keys():
                        if isinstance(hdf[key], h5py.Dataset) and key not in e_data.keys():  # Only once
                            ds = hdf[key]
                            e_data[key] = ds[:]  # Make full copy of data to my HDF with prefix so it's obvious
                        elif isinstance(hdf[key],
                                        h5py.Group) and key not in self.hdf.keys():  # TODO: Check I'm actually copying metadata group and nothing else
                            hdf.copy(hdf[key], self.hdf, 'Exp_metadata')  # Make full copy of group to my HDF
                self.Data.set_links_to_measured_data()
                self.hdf.flush()  # writes changes to my HDF to file
            else:
                raise FileNotFoundError(f'Did not find HDF at {hdfpath}')

    def set_base_attrs_HDF(self):
        """ For storing Base info in HDF attrs"""
        for attr, val in zip(DatHDF.BASE_ATTRS,
                             [self.datnum, self.datname, self.dat_id, self.dattypes, self.date_initialized]):
            HDU.set_attr(self.hdf, attr, val)

    @abc.abstractmethod
    def set_dattypes(self, value=None):
        """Reminder to set dattypes attr in HDF at some point"""
        self.dattypes = value if value else self.dattypes
        HDU.set_attr(self.hdf, 'dattypes', self.dattypes)

    def init_Data(self, setup_dict=None):
        """ Sets up Data in my HDF from data Exp_measured_data using setup_dict(). Otherwise blank init
        @param setup_dict: dict formatted as {<standard_name>:[<exp_name(s), multiplier(s), offset(s)*], ...}  *optional
            there MUST be a multiplier and offset for every possible exp_name
        @type setup_dict: Dict[list]
        @return: Sets attributes in Data
        @rtype: None
        """
        self.Data = self.Data if self.Data else Data.NewData(self.hdf)  # Will init Data from Dat HDF if already exists,
        # otherwise will be blank init
        if setup_dict is not None:  # For initializing data into Dat HDF (Exp_data should already be located in
            # 'Exp_measured_data' inside Dat HDF

            # Initialize Data in Dat HDF with multipliers/offsets from setup_dict
            Data.init_Data(self.Data, setup_dict)

            # Flush changes to disk
            self.hdf.flush()

    def init_Logs(self, sweep_logs=None):
        self.Logs = self.Logs if self.Logs else L.NewLogs(self.hdf)
        if sweep_logs is not None:
            group = self.Logs.group

            # Store full sweeplogs
            HDU.set_attr(group, 'Full sweeplogs', sweep_logs)

            # Simple attrs
            InitLogs.set_simple_attrs(group, sweep_logs)

            # Instr attrs  # TODO: maybe want these part of subclass

            InitLogs.set_srss(group, sweep_logs)  # SRSs are top level keys in sweep logs
            InitLogs.set_babydac(group, dictor(sweep_logs, 'BabyDAC', None))
            InitLogs.set_fastdac(group, dictor(sweep_logs, 'FastDAC', None))
            InitLogs.set_awg(group, dictor(sweep_logs, 'FastDAC.AWG', None))
            InitLogs.set_temps(group, dictor(sweep_logs, 'Temperatures', None))

            # TODO: add mags
            # for i in range(1, cfg.current_config.instrument_num['mags']+1+1):
            #     if f'Mag...':
            #         pass

            self.hdf.flush()

    def init_Instruments(self):
        assert self.Logs is not None
        # TODO: copy links from relevant groups in logs to Instruments
        self.Instruments = self.Instruments if self.Instruments else Instruments.NewInstruments(self.hdf)

    def init_Other(self):
        self.Other = self.Other if self.Other else Other.Other(self.hdf)

    @abc.abstractmethod
    def check_built(self, additional_dat_attrs: list = None, additional_dat_names: list = None):
        """Just a quick check that all relevant datAttrs have actually be initialized at some point
        Subclasses should add additional datattrs,names to super().check_built() call"""
        ada = additional_dat_attrs
        adn = additional_dat_names
        # Quick check that either both are none, or neither are none and have same length
        assert any([ada is not None and adn is not None and len(ada) == len(adn), ada is None and adn is None])
        dat_attrs = [self.Data, self.Logs, self.Instruments, self.Other]
        dat_attr_names = ['Data', 'Logs', 'Instruments', 'Other']
        if ada is not None:
            dat_attrs += ada
            dat_attr_names += adn
        still_none = []
        for dat_attr, name in zip(dat_attrs, dat_attr_names):
            if dat_attr is None:
                still_none.append(name)
        if still_none:
            logger.warning(f'[{still_none}] are still None')
            return False
        else:
            return True

    def set_initialized(self):
        self.hdf.attrs['initialized'] = True


class BasicDatBuilder(NewDatBuilder):

    def set_dattypes(self, value=None):
        super().set_dattypes(value)

    def check_built(self, additional_dat_attrs: list = None, additional_dat_names: list = None):
        super().check_built(additional_dat_attrs=None, additional_dat_names=None)


class TransitionDatBuilder(NewDatBuilder):
    """For building dats which may have Transition or AWG (Arbitrary Wave Generator) data"""

    def __init__(self, datnum, datname, hdfdir, overwrite=False):
        super().__init__(datnum, datname, hdfdir, overwrite)
        self.Transition: Union[T.NewTransitions, None] = None
        self.AWG: Union[AWG.AWG, None] = None

    def set_dattypes(self, value=None):
        """Just need to remember to call this to set dattypes in HDF"""
        super().set_dattypes(value)

    def init_Transition(self):
        self.Transition = self.Transition if self.Transition else T.NewTransitions(self.hdf)
        x = self.Data.get_dataset('x_array')
        y = self.Data.get_dataset('y_array')
        i_sense = self.Data.get_dataset('i_sense')
        init_isense_data(self.Transition.group, x, y, i_sense)

    def init_AWG(self, logs_group, data_group):
        """For initializing Arbitrary Wave Generator info"""
        self.AWG = self.AWG if self.AWG else AWG.AWG(self.hdf)
        AWG.init_AWG(self.AWG.group, logs_group, data_group)

    def check_built(self, additional_dat_attrs: list = None, additional_dat_names: list = None):
        ada = [self.Transition, self.AWG]
        if additional_dat_attrs is not None:
            ada += additional_dat_attrs
        adn = ['Transition', 'AWG']
        if additional_dat_names is not None:
            adn += additional_dat_names
        super().check_built(additional_dat_attrs=ada, additional_dat_names=adn)


class SquareDatBuilder(TransitionDatBuilder):
    """For building dats which have Square Entropy"""
    def __init__(self, datnum, datname, hdfdir, overwrite=False):
        super().__init__(datnum, datname, hdfdir, overwrite)
        self.SquareEntropy: Optional[SE.SquareEntropy] = None

    def init_SquareEntropy(self):
        self.SquareEntropy = self.SquareEntropy if self.SquareEntropy else SE.SquareEntropy(self.hdf)
        x = self.Data.get_dataset('x_array')
        y = self.Data.get_dataset('y_array')
        i_sense = self.Data.get_dataset('i_sense')
        init_isense_data(self.SquareEntropy.group, x, y, i_sense)

    def check_built(self, additional_dat_attrs: list = None, additional_dat_names: list = None):
        ada = [self.SquareEntropy]
        if additional_dat_attrs is not None:
            ada += additional_dat_attrs
        adn = ['SquareEntropy']
        if additional_dat_names is not None:
            adn += additional_dat_names
        super().check_built(additional_dat_attrs=ada, additional_dat_names=adn)




class EntropyDatBuilder(TransitionDatBuilder):
    """For building dats which may have Entropy (or anything from Transition Builder)"""

    def __init__(self, datnum, datname, hdfdir, overwrite=False):
        super().__init__(datnum, datname, hdfdir, overwrite)
        self.Entropy: Union[E.NewEntropy, None] = None

    def init_Entropy(self, centers):
        """If centers is passed as None, then Entropy.data (entr) is not initialized"""
        self.Entropy = self.Entropy if self.Entropy else E.NewEntropy(self.hdf)
        x = self.Data.get_dataset('x_array')
        y = self.Data.get_dataset('y_array')
        entx = self.Data.get_dataset('entx')
        enty = self.Data.get_dataset('enty')
        init_entropy_data(self.Entropy.group, x, y, entx, enty, centers=centers)

    def check_built(self, additional_dat_attrs: list = None, additional_dat_names: list = None):
        ada = [self.Entropy]
        if additional_dat_attrs is not None:
            ada += additional_dat_attrs
        adn = ['Entropy']
        if additional_dat_names is not None:
            adn += additional_dat_names
        super().check_built(additional_dat_attrs=ada, additional_dat_names=adn)


def get_builder(dattypes) -> Type[NewDatBuilder]:
    """Returns the class of the appropriate builder"""
    if dattypes is None:
        return BasicDatBuilder
    elif 'square entropy' in dattypes:
        return SquareDatBuilder
    elif 'entropy' in dattypes:
        return EntropyDatBuilder
    elif 'transition' in dattypes:
        return TransitionDatBuilder
    elif 'AWG' in dattypes:
        return TransitionDatBuilder
    elif 'none_given' in dattypes:
        return BasicDatBuilder
    else:
        raise NotImplementedError(f'No builder found for {dattypes}')


class InitLogs(object):
    """Class to contain all functions required for setting up Logs in HDF (so that Logs DA can get_from_hdf())"""
    BABYDAC_KEYS = ['com_port',
                    'DAC#{<name>}']  # TODO: not currently using DAC#{}, should figure out a way to check this
    FASTDAC_KEYS = ['SamplingFreq', 'MeasureFreq', 'visa_address', 'AWG', 'numADCs', 'DAC#{<name>}', 'ADC#']


    @staticmethod
    def check_key(k, expected_keys):
        if k in expected_keys:
            return
        elif k[0:3] in ['DAC', 'ADC']:  # TODO: This should be checked better
            return
        else:
            logger.warning(f'Unexpected key in logs: k = {k}, Expected = {expected_keys}')
            return

    @staticmethod
    def set_babydac(group, babydac_json):
        """Puts info into Dat HDF"""
        """dac dict should be stored in format:
                        com_port: ...
                        DAC#{<name>}: <val>
                """
        if babydac_json is not None:
            for k, v in babydac_json.items():
                InitLogs.check_key(k, InitLogs.BABYDAC_KEYS)
            HDU.set_attr(group, 'BabyDACs', babydac_json)
        else:
            logger.info(f'No "BabyDAC" found in json')

    @staticmethod
    def set_fastdac(group, fdac_json):
        """Puts info into Dat HDF"""  # TODO: Make work for more than one fastdac
        """fdac dict should be stored in format:
                                visa_address: ...
                                SamplingFreq:
                                DAC#{<name>}: <val>
                                ADC#: <val>
    
                                ADCs not currently required
                                """
        if fdac_json is not None:
            for k, v in fdac_json.items():
                InitLogs.check_key(k, InitLogs.FASTDAC_KEYS)
            HDU.set_attr(group, 'FastDACs', fdac_json)
        else:
            logger.info(f'No "FastDAC" found in json')

    @staticmethod
    def set_awg(group, awg_json):
        """Put info into Dat HDF

        Args:
            group (h5py.Group): Group to put AWG NamedTuple in
            awg_json (dict): From standardized Exp sweeplogs

        Returns:
            None
        """

        if awg_json is not None:
            # Simplify and shorten names
            awg_data = E2S.awg_from_json(awg_json)

            # Store in NamedTuple
            ntuple = Util.data_to_NamedTuple(awg_data, L.AWGtuple)
            HDU.set_attr(group, 'AWG', ntuple)
        else:
            logger.info(f'No "AWG" added')

    @staticmethod
    def set_srss(group, json):
        """Sets SRS values in Dat HDF from either full sweeplogs or minimally json which contains SRS_{#} keys"""
        srs_ids = [key[4] for key in json.keys() if key[:3] == 'SRS']

        for num in srs_ids:
            if f'SRS_{num}' in json.keys():
                srs_data = E2S.srs_from_json(json, num)  # Converts to my standard
                ntuple = Util.data_to_NamedTuple(srs_data, L.SRStuple)  # Puts data into named tuple
                srs_group = group.require_group(f'srss')  # Make sure there is an srss group
                HDU.set_attr(srs_group, f'srs{num}', ntuple)  # Save in srss group
            else:
                logger.error(f'No "SRS_{num}" found in json')  # Should not get to here

    @staticmethod
    def set_temps(group, temp_json):
        """Sets Temperatures in DatHDF from temperature part of sweeplogs"""
        if temp_json:
            temp_data = E2S.temp_from_json(temp_json)
            ntuple = Util.data_to_NamedTuple(temp_data, L.TEMPtuple)
            HDU.set_attr(group, 'Temperatures', ntuple)
        else:
            logger.warning('No "Temperatures" added')

    @staticmethod
    def set_simple_attrs(group, json):
        """Sets top level attrs in Dat HDF from sweeplogs"""
        group.attrs['comments'] = dictor(json, 'comment', '')
        group.attrs['filenum'] = dictor(json, 'filenum', 0)
        group.attrs['x_label'] = dictor(json, 'axis_labels.x', 'None')
        group.attrs['y_label'] = dictor(json, 'axis_labels.y', 'None')
        group.attrs['current_config'] = dictor(json, 'current_config', None)
        group.attrs['time_completed'] = dictor(json, 'time_completed', None)
        group.attrs['time_elapsed'] = dictor(json, 'time_elapsed', None)


def init_entropy_data(group: h5py.Group, x: Union[h5py.Dataset, np.ndarray], y: Union[h5py.Dataset, np.ndarray, None],
                      entx: Union[h5py.Dataset, np.ndarray], enty: Union[h5py.Dataset, np.ndarray, None],
                      centers: Union[None, list, np.ndarray]):
    """Convert from standardized experiment data to data stored in dat HDF
    i.e. start with x and possible y array (1D or 2D), and entropy x and possibly y (if just entx assume it is entr,
    otherwise calculate entr with both of them and calculate phase angle)"""

    dg = group.require_group('Data')
    enty = enty if enty is not None else np.nan
    if y is None:
        y = np.nan  # Can't store None in HDF
        centers = None  # Don't need to center 1D data

    if centers is not None:
        if not isinstance(enty, (np.ndarray, h5py.Dataset)):
            entr = CU.center_data(x, entx, centers) if centers is not None else entx
            angle = 0.0
        else:
            entr, angle = E.calc_r(entx, enty, x, centers)
    else:
        entr = np.nan
        angle = np.nan

    for data, name in zip([x, y, entx, enty, entr], ['x', 'y', 'entropy_x', 'entropy_y', 'entropy_r']):
        if isinstance(data, h5py.Dataset):
            logger.info(f'Creating link to {name} only in Transition.Data')
        else:
            logger.info(f'Creating data for {name} in Transition.Data')
        dg[name] = data
    group.attrs['angle'] = angle
    group.file.flush()


def init_isense_data(group: h5py.Group, x: Union[h5py.Dataset, np.ndarray],
                     y: Union[h5py.Dataset, np.ndarray, None], i_sense: Union[h5py.Dataset, np.ndarray]):
    """
        Convert from standardized experiment data to data stored in Dat attribute group of HDF
        Args:
            group (h5py.Group):
            x (Union(np.ndarray, h5py.Dataset)):
            y (Union(np.ndarray, h5py.Dataset)):
            i_sense (Union(np.ndarray, h5py.Dataset)):

        Returns:
            (None):
        """
    tdg = group.require_group('Data')
    y = y if y is not None else np.nan  # can't store None in HDF
    for data, name in zip([x, y, i_sense], ['x', 'y', 'i_sense']):
        if isinstance(data, h5py.Dataset):
            logger.info(f'Creating link to {name} only in {group.name}.Data')
            tdg[name] = data
        elif np.isnan(y):
            logger.info(f'No {name} data, np.nan stored in {group.name}.Data.{name}')
            tdg[name] = data
        else:
            raise ValueError(f'data for {name} is invalid: data = {data}')
    group.file.flush()
