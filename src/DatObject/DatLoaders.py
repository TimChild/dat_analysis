import abc
from typing import Type
from dataclasses import dataclass, field, InitVar
import h5py

from src import CoreUtil as CU
from src import HDF_Util as HDU
from src.DatObject import DatHDF
from src.DatObject.Attributes import Data, Logs, Other, Transition as T, AWG, Entropy as E, SquareEntropy as SE


@dataclass
class NewDatLoader(abc.ABC):
    # Info for finding file only
    file_path: InitVar[str] = None  # Directly provide path to file
    hdfdir: InitVar[str] = None  # Or provide path to HDF directory along with datnum, [datname]
    # Above makes the below attribute after confirming filepath to HDF
    hdf_path: str = field(default=None, init=False, repr=False)

    # Info used to find find and loaded from HDF (i.e. used to find file, then overwritten from file)
    datnum: int = field(default=None)
    datname: str = field(default='base')

    # Info only loaded from HDF (i.e. stored in HDF)
    dat_id: str = field(default=None)
    dattypes: str = field(default=None)
    date_initialized: str = field(default=None, init=False)  # TODO: what is the type for this?

    def __post_init__(self, file_path, hdfdir):
        if file_path is not None:  # Either load from filepath directly
            assert all([self.datnum is None, self.datname is None])  # Either use file_path OR datnum/name
            self.hdf_path = HDU.check_hdf_path(file_path)
        else:  # Or look for datnum, datname in hdfdir
            assert self.datnum is not None
            assert hdfdir is not None
            dat_id = CU.get_dat_id(self.datnum, self.datname)
            self.hdf_path = HDU.check_hdf_id(dat_id, hdfdir_path=hdfdir)

        # Gets attrs saved in HDF (i.e. datnum, datname, etc)
        self.get_Base_attrs()
        self.Data = Data.Data(self.hdf_path)
        self.Logs = Logs.OldLogs(self.hdf_path)
        self.Instruments = None  # TODO: Replace with Instruments
        self.Other = Other.Other(self.hdf_path)

    def get_Base_attrs(self):
        with h5py.File(self.hdf_path, 'r') as f:
            for key in DatHDF.BASE_ATTRS:
                val = HDU.get_attr(f, key, default=None)
                setattr(self, key, val)

    @abc.abstractmethod
    def build_dat(self, *args, **kwargs) -> DatHDF.DatHDF:
        """Override to add checks for Entropy/Transition etc"""
        return DatHDF.DatHDF(self.datnum, self.datname, self.hdf, *args, Data=self.Data, Logs=self.Logs,
                             Other=self.Other, **kwargs)


class BasicDatLoader(NewDatLoader):
    """For loading the base dat with only Logs, Data, Instruments, Other"""
    def build_dat(self, *args, **kwargs) -> DatHDF.DatHDF:
        return super().build_dat()


class TransitionDatLoader(NewDatLoader):
    """For loading dats which may have any of Entropy, Transition, DCbias"""

    def __init__(self, datnum=None, datname=None, file_path=None, hdfdir=None):
        super().__init__(datnum, datname, file_path, hdfdir)
        if 'transition' in self.dattypes:
            self.Transition = T.OldTransitions(self.hdf)
        else:
            self.Transition = None

        if 'AWG' in self.dattypes:
            self.AWG = AWG.AWG(self.hdf)
        else:
            self.AWG = None

    def build_dat(self, **kwargs) -> DatHDF:
        return super().build_dat(Transition=self.Transition, AWG=self.AWG, **kwargs)


class SquareEntropyDatLoader(TransitionDatLoader):
    """For loading dats which have Square Entropy and anything from Transition"""
    def __init__(self, datnum=None, datname=None, file_path=None, hdfdir=None):
        super().__init__(datnum, datname, file_path, hdfdir)
        if 'square entropy' in self.dattypes:
            self.SquareEntropy = SE.SquareEntropy(self.hdf)

    def build_dat(self, **kwargs) -> DatHDF:
        return super().build_dat(SquareEntropy=self.SquareEntropy, **kwargs)


class EntropyDatLoader(TransitionDatLoader):
    """For loading dats which may have any of Entropy"""

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
    elif 'square entropy' in dattypes:
        return SquareEntropyDatLoader
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