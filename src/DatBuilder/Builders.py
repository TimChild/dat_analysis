import abc
import os
from datetime import datetime
from typing import Union, Type
import h5py
from dictor import dictor
import src.DatBuilder.Util
from src import CoreUtil as CU
from src.DatAttributes import Entropy as E
from src.DatAttributes import AWG
from src.DatAttributes import Other
from src.DatAttributes import Transition as T
from src.DatAttributes import Data
from src.DatAttributes import Logs
from src.DatAttributes import Instruments
from src.DatBuilder import DatHDF

from src.HDF import Util as HDU


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
        self.dat_id = src.DatBuilder.Util.get_dat_id(datnum, datname)
        self.dattypes = None

        self.hdf_path = HDU.get_dat_hdf_path(self.dat_id, hdfdir,
                                             overwrite=overwrite)  # Location of My HDF which will store everything to do with dat
        self.hdf = h5py.File(self.hdf_path, 'r+')  # Open file in Read/Write mode

        # Init General Dat attributes to None
        self.Data: Data.NewData = None
        self.Logs: Logs.NewLogs = None
        self.Instruments: Instruments.NewInstruments = None
        self.Other: Other.Other = None

        # Basic Inits which are sufficient if data exists in HDF already. Otherwise need to be built elsewhere
        self.init_Data()
        self.init_Logs()
        self.init_Instruments()
        self.init_Other()

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
        self.Data.get_from_HDF()  # Set up Data attrs (doesn't do much for Data)

    def init_Logs(self, json=None):
        self.Logs = self.Logs if self.Logs else Logs.NewLogs(self.hdf)
        if json is not None:
            group = self.Logs.group

            # Simple attrs
            Logs.InitLogs.set_simple_attrs(group, json)

            # Instr attrs  # TODO: maybe want these part of subclass

            Logs.InitLogs.set_srss(group, json)  # SRS_1 etc are in top level of sweeplogs anyway
            Logs.InitLogs.set_babydac(group, dictor(json, 'BabyDAC', None))
            Logs.InitLogs.set_fastdac(group, dictor(json, 'FastDAC', None))
            Logs.InitLogs.set_awg(group, dictor(json, 'FastDAC', None))

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

    def init_Other(self):
        self.Other = self.Other if self.Other else Other.Other(self.hdf)
        self.Other.get_from_HDF()

    @abc.abstractmethod
    def build_dat(self, **kwargs) -> DatHDF.DatHDF:
        """Override if passing more info to NewDat (like any other DatAttributes"""
        return DatHDF.DatHDF(self.datnum, self.datname, self.hdf, Data=self.Data, Logs=self.Logs,
                             Instruments=self.Instruments, Other=Other, **kwargs)


class NewDatLoader(abc.ABC):
    def __init__(self, datnum=None, datname=None, file_path=None, hdfdir=None):
        if file_path is not None:
            assert all([datnum is None, datname is None])
            self.hdf = h5py.File(file_path, 'r+')
        else:
            assert datnum is not None
            assert hdfdir is not None
            datname = datname if datname else 'base'
            dat_id = src.DatBuilder.Util.get_dat_id(datnum, datname)
            self.hdf = h5py.File(HDU.get_dat_hdf_path(dat_id, hdfdir_path=hdfdir), 'r+')

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
        self.Other = Other.Other(self.hdf)
        # self.Instruments = Instruments.NewInstruments(self.hdf)

    def get_Base_attrs(self):
        for key in DatHDF.BASE_ATTRS:
            val = HDU.get_attr(self.hdf, key, default=None)
            setattr(self, key, val)

    @abc.abstractmethod
    def build_dat(self, *args, **kwargs) -> DatHDF.DatHDF:
        """Override to add checks for Entropy/Transition etc"""
        return DatHDF.DatHDF(self.datnum, self.datname, self.hdf, *args, Data=self.Data, Logs=self.Logs,
                             Instruments=self.Instruments, Other=self.Other, **kwargs)


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
        T.init_transition_data(self.Transition.group, x, y, i_sense)
        self.Transition.get_from_HDF()

    def init_AWG(self, logs_group, data_group):
        """For initializing Arbitrary Wave Generator info"""
        self.AWG = self.AWG if self.AWG else AWG.AWG(self.hdf)
        AWG.init_AWG(self.AWG.group, logs_group, data_group)
        self.AWG.get_from_HDF()

    def build_dat(self, **kwargs):
        return super().build_dat(Transition=self.Transition, AWG=self.AWG, **kwargs)


class TransitionDatLoader(NewDatLoader):
    """For loading dats which may have any of Entropy, Transition, DCbias"""

    def __init__(self, datnum=None, datname=None, file_path=None, hdfdir=None):
        super().__init__(datnum, datname, file_path, hdfdir)
        if 'transition' in self.dattypes:
            self.Transition = T.NewTransitions(self.hdf)
        else:
            self.Transition = None

        if 'AWG' in self.dattypes:
            self.AWG = AWG.AWG(self.hdf)
        else:
            self.AWG = None

    def build_dat(self, **kwargs) -> DatHDF:
        return super().build_dat(Transition=self.Transition, AWG=self.AWG, **kwargs)


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

    def build_dat(self, **kwargs):
        return super().build_dat(Entropy=self.Entropy, **kwargs)


class EntropyDatLoader(TransitionDatLoader):
    """For loading dats which may have any of Entropy, Transition, DCbias"""

    def __init__(self, datnum=None, datname=None, file_path=None, hdfdir=None):
        super().__init__(datnum, datname, file_path, hdfdir)
        if 'entropy' in self.dattypes:
            self.Entropy = E.NewEntropy(self.hdf)

    def build_dat(self, **kwargs) -> DatHDF:
        return super().build_dat(Entropy=self.Entropy, **kwargs)


def get_builder(dattypes) -> Type[NewDatBuilder]:
    """Returns the class of the appropriate builder"""
    if dattypes is None:
        return NewDatBuilder
    elif 'entropy' in dattypes:
        return EntropyDatBuilder
    elif 'transition' in dattypes:
        return TransitionDatBuilder
    elif 'AWG' in dattypes:
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
    elif 'AWG' in dattypes:
        return TransitionDatLoader
    else:
        raise NotImplementedError(f'No loader found for {dattypes}')
