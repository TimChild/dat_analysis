import abc
import os
from datetime import datetime
from typing import Union, Type
import numpy as np
import h5py
from dictor import dictor
from src import CoreUtil as CU
from src.Configs import Main_Config as cfg
from src.DatAttributes import Entropy as E, Transition as T, Data, Logs, Instruments
from src.DatBuilder import DatHDF
from src.DatBuilder.Util import match_name_in_group
from src.HDF import Util as HU


class NewDatBuilder(abc.ABC):
    """Base DatHDF builder class. Only contains the core DatAttributes Logs, Data, Instruments. Any others should be
    added in a subclass of this"""
    def __init__(self, datnum, datname, hdfdir, overwrite=False):
        # Init with basic info at least - enough to Identify DatHDF
        # Base attrs for Dat
        self.datnum = datnum
        self.datname = datname
        # self.config_name = cfg.current_config.__name__.split('.')[-1]  # TODO: Remove this
        self.date_initialized = datetime.now().date()
        self.dat_id = DatHDF.get_dat_id(datnum, datname)
        self.dattypes = None

        self.hdf_path = HU.get_dat_hdf_path(self.dat_id, hdfdir, overwrite=overwrite)  # Location of My HDF which will store everything to do with dat
        self.hdf = h5py.File(self.hdf_path, 'r+')  # Open file in Read/Write mode

        # Init General Dat attributes to None
        self.Data = None  # type: Data.NewData
        self.Logs = None  # type: Logs.NewLogs
        self.Instruments = None  # type: Instruments.NewInstruments

        # Basic Inits which are sufficient if data exists in HDF already. Otherwise need to be built elsewhere
        self.init_Data()
        self.init_Logs()
        self.init_Instruments()

    def copy_exp_hdf(self, ddir):
        """Copy experiment HDF data into my HDF file if not done already"""
        if 'Exp_measured_data' not in self.hdf.keys() or 'Exp_metadata' not in self.hdf.keys():  # Only if first time
            hdfpath = os.path.join(ddir, f'dat{self.datnum:d}.h5')
            if os.path.isfile(hdfpath):  # Only if original HDF exists
                e_data = self.hdf.create_group('Exp_measured_data')
                with h5py.File(hdfpath, 'r') as hdf:
                    for key in hdf.keys():
                        if isinstance(hdf[key], h5py.Dataset) and key not in e_data.keys():  # Only once
                            ds = hdf[key]
                            e_data[key] = ds[:]  # Make full copy of data to my HDF with prefix so it's obvious
                        elif isinstance(hdf[key], h5py.Group) and key not in self.hdf.keys():  # TODO: Check I'm actually copying metadata group and nothing else
                            hdf.copy(hdf[key], self.hdf, 'Exp_metadata')  # Make full copy of group to my HDF
                self.Data.set_links_to_measured_data()
                self.hdf.flush()  # writes changes to my HDF to file
            else:
                raise FileNotFoundError(f'Did not find HDF at {hdfpath}')

    def init_Base(self):
        """ For storing Base info in HDF attrs
        Note: dattypes won't be set here!"""
        hdf = self.hdf
        for attr, val in zip(DatHDF.BASE_ATTRS, [self.datnum, self.datname, self.dat_id, self.dattypes, self.config_name, self.date_initialized]):
            hdf.attrs[attr] = val

    @abc.abstractmethod
    def set_dattypes(self, value=None):
        """Reminder to set dattypes attr in HDF at some point"""
        self.dattypes = value if value else self.dattypes
        self.hdf['dattypes'] = str(self.dattypes)  # TODO: Fix how this is saved

    def init_Data(self, setup_dict=None):
        """
        @param setup_dict: dict formatted as {<standard_name>:[<exp_name(s), multiplier(s), offset(s)*], ...}  *optional
            there MUST be a multiplier and offset for every possible exp_name
        @type setup_dict: Dict[list]
        @return: Sets attributes in Data
        @rtype: None
        """
        self.Data = self.Data if self.Data else Data.NewData(self.hdf)  # Will init Data from Dat HDF if already exists, otherwise will be blank init
        if setup_dict is not None:  # For initializing data into Dat HDF (Exp_data should already be located in 'Exp_measured_data' inside Dat HDF
            dg = self.Data.group
            for item in setup_dict.items():  # Use Data.get_setup_dict to create
                standard_name = item[0]  # The standard name used in rest of this analysis
                info = item[1]  # The possible names, multipliers, offsets to look for in exp data  (from setupDF)
                exp_names = CU.ensure_list(info[0])  # All possible names in exp
                exp_names = [f'Exp_{name}' for name in exp_names]  # stored with prefix in my Data folder
                exp_name, index = match_name_in_group(exp_names, dg)  # First name which matches a dataset in exp
                multiplier = info[1][index]  # Get the correction multiplier
                offset = info[2][index] if len(info) == 3 else 0  # Get the correction offset or default to zero
                if multiplier == 1 and offset == 0:  # Just link to exp data
                    self.Data.link_data(standard_name, exp_name, dg)  # Hard link to data (so not duplicated in HDF file)
                else:  # duplicate and alter dataset before saving in HDF
                    data = dg.get(exp_name)[:]  # Get copy of exp Data
                    data = data*multiplier+offset  # Adjust as necessary
                    self.Data.set_data(standard_name, data)  # Store as new data in HDF
            self.hdf.flush()
        self.Data.get_from_HDF()  # Set up Data attrs (doesn't do much for Data)

    def init_Logs(self, json=None):
        self.Logs = self.Logs if self.Logs else Logs.NewLogs(self.hdf)
        if json is not None:
            group = self.Logs.group

            # Simple attrs
            Logs._init_logs_set_simple_attrs(group, json)

            # Instr attrs  # TODO: maybe want these part of subclass

            Logs._init_logs_set_srss(group, json)  # SRS_1 etc are in top level of sweeplogs anyway
            Logs._init_logs_set_babydac(group, dictor(json, 'BabyDAC', None))
            Logs._init_logs_set_fastdac(group, dictor(json, 'FastDAC', None))

            # TODO: add mags
            # for i in range(1, cfg.current_config.instrument_num['mags']+1+1):
            #     if f'Mag...':
            #         pass

            self.hdf.flush()
        self.Logs.get_from_HDF()  # To put the values stored in Dat HDF into Logs attrs

    def init_Instruments(self):
        assert self.Logs is not None
        # TODO: copy links from relevant groups in logs to Instruments
        self.Instruments = self.Instruments if self.Instruments else Instruments.NewInstruments(self.hdf)
        self.Instruments.get_from_HDF()
        pass

    @abc.abstractmethod
    def build_dat(self) -> DatHDF.DatHDF:
        """Override if passing more info to NewDat (like any other DatAttributes"""
        return DatHDF(self.datnum, self.datname, self.hdf, Data=self.Data, Logs=self.Logs, Instruments=self.Instruments)


class NewDatLoader(abc.ABC):
    def __init__(self, datnum=None, datname=None, file_path=None):
        if file_path is not None:
            assert all([datnum is None, datname is None])
            self.hdf = h5py.File(file_path, 'r+')
        else:
            assert datnum is not None
            datname = datname if datname else 'base'
            self.hdf = h5py.File(HU.get_dat_hdf_path(DatHDF.get_dat_id(datnum, datname)))

        # Base attrs
        self.datnum = None
        self.datname = None
        self.dat_id = None
        self.dattypes = None
        self.config_name = None
        self.date_initialized = None

        self.get_Base_attrs()
        self.Data = Data.NewData(self.hdf)
        self.Logs = Logs.NewLogs(self.hdf)
        self.Instruments = None  # TODO: Replace with Instruments
        # self.Instruments = Instruments.NewInstruments(self.hdf)

    def get_Base_attrs(self):
        for key in DatHDF.BASE_ATTRS:
            setattr(self, key, self.hdf.attrs.get(key, None))
            self.dattypes = set(self.dattypes)

    @abc.abstractmethod
    def build_dat(self) -> DatHDF.DatHDF:
        """Override to add checks for Entropy/Transition etc"""
        return DatHDF.DatHDF(self.datnum, self.datname, self.hdf, Data=self.Data, Logs=self.Logs, Instruments=self.Instruments)


class TransitionDatBuilder(NewDatBuilder):
    """For building dats which may have any of Transition"""

    def __init__(self, datnum, datname, hdfdir, overwrite=False):
        super().__init__(datnum, datname, hdfdir, overwrite)
        self.Transition: Union[T.NewTransitions, None] = None

    def set_dattypes(self, value=None):
        """Just need to remember to call this to set dattypes in HDF"""
        super().set_dattypes(value)

    def init_Transition(self):
        self.Transition = self.Transition if self.Transition else T.NewTransitions(self.hdf)
        x = self.Data.get_dataset('x_array')
        y = self.Data.get_dataset('y_array')
        i_sense = self.Data.get_dataset('i_sense')
        T.init_transition_data(self.Transition.group, x, y, i_sense)
        self.Transition.get_from_HDF()

    def build_dat(self):
        return DatHDF.DatHDF(self.datnum, self.datname, self.hdf, self.Data, self.Logs,
                                            self.Instruments, Transition=self.Transition)


class TransitionDatLoader(NewDatLoader):
    """For loading dats which may have any of Entropy, Transition, DCbias"""
    def __init__(self, datnum=None, datname=None, file_path=None):
        super().__init__(datnum, datname, file_path)
        if 'transition' in self.dattypes:
            self.Transition = T.NewTransitions(self.hdf)

    def build_dat(self) -> DatHDF:
        return DatHDF.DatHDF(self.datnum, self.datname, self.hdf, self.Data, self.Logs,
                                            self.Instruments, Transition=self.Transition)


class EntropyDatBuilder(TransitionDatBuilder):
    """For building dats which may have any of Entropy, Transition, DCbias"""
    def __init__(self, datnum, datname, hdfdir, overwrite=False):
        super().__init__(datnum, datname, hdfdir, overwrite)
        self.Entropy: Union[E.NewEntropy, None] = None

    def init_Entropy(self, center_ids):
        """If center_ids is passed as None, then Entropy.data (entr) is not initialized"""
        self.Entropy = self.Entropy if self.Entropy else E.NewEntropy(self.hdf)
        x = self.Data.get_dataset('x_array')
        y = self.Data.get_dataset('y_array')
        entx = self.Data.get_dataset('entx')
        enty = self.Data.get_dataset('enty')
        E.init_entropy_data(self.Entropy.group, x, y, entx, enty, center_ids=center_ids)
        self.Entropy.get_from_HDF()

    def build_dat(self):
        return DatHDF.DatHDF(self.datnum, self.datname, self.hdf, self.Data, self.Logs,
                                            self.Instruments, Entropy=self.Entropy, Transition=self.Transition)


class EntropyDatLoader(TransitionDatLoader):
    """For loading dats which may have any of Entropy, Transition, DCbias"""
    def __init__(self, datnum=None, datname=None, file_path=None):
        super().__init__(datnum, datname, file_path)
        if 'entropy' in self.dattypes:
            self.Entropy = E.NewEntropy(self.hdf)

    def build_dat(self) -> DatHDF:
        return DatHDF.DatHDF(self.datnum, self.datname, self.hdf, self.Data, self.Logs,
                                            self.Instruments, Entropy=self.Entropy, Transition=self.Transition)


def get_builder(dattypes) -> Type[NewDatBuilder]:
    """Returns the class of the appropriate builder"""  # TODO: Make this less specific to my analysis
    if dattypes is None:
        return NewDatBuilder
    elif 'entropy' in dattypes:
        return EntropyDatBuilder
    elif 'transition' in dattypes:
        return TransitionDatBuilder
    else:
        raise NotImplementedError(f'No builder found for {dattypes}')


def get_loader(dattypes) -> Type[NewDatLoader]:
    """Returns the class of the appropriate loader"""
    if dattypes is None:
        return NewDatLoader
    elif 'entropy' in dattypes:
        return EntropyDatLoader
    elif 'transition' in dattypes:
        return TransitionDatLoader
    else:
        raise NotImplementedError(f'No loader found for {dattypes}')