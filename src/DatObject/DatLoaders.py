import abc
from typing import Type

import h5py

from src import CoreUtil as CU
from src import HDF_Util as HDU
from src.DatObject import DatHDF
from src.DatObject.Attributes import Data, Logs, Other, Transition as T, AWG, Entropy as E


class NewDatLoader(abc.ABC):
    def __init__(self, datnum=None, datname=None, file_path=None, hdfdir=None):
        if file_path is not None:
            assert all([datnum is None, datname is None])
            self.hdf = h5py.File(file_path, 'r+')
        else:
            assert datnum is not None
            assert hdfdir is not None
            datname = datname if datname else 'base'
            dat_id = CU.get_dat_id(datnum, datname)
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


class BasicDatLoader(NewDatLoader):
    """For loading the base dat with only Logs, Data, Instruments, Other"""
    def build_dat(self, *args, **kwargs) -> DatHDF.DatHDF:
        return super().build_dat()


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


class EntropyDatLoader(TransitionDatLoader):
    """For loading dats which may have any of Entropy, Transition, DCbias"""

    def __init__(self, datnum=None, datname=None, file_path=None, hdfdir=None):
        super().__init__(datnum, datname, file_path, hdfdir)
        if 'entropy' in self.dattypes:
            self.Entropy = E.NewEntropy(self.hdf)

    def build_dat(self, **kwargs) -> DatHDF:
        return super().build_dat(Entropy=self.Entropy, **kwargs)


def get_loader(dattypes) -> Type[NewDatLoader]:
    """Returns the class of the appropriate loader"""
    if dattypes is None:
        return BasicDatLoader
    elif 'entropy' in dattypes:
        return EntropyDatLoader
    elif 'transition' in dattypes:
        return TransitionDatLoader
    elif 'AWG' in dattypes:
        return TransitionDatLoader
    elif 'none_given' in dattypes:
        return BasicDatLoader
    else:
        raise NotImplementedError(f'No loader found for {dattypes}')