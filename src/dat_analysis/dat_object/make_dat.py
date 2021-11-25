"""This is where exp_specific is used to get data into a standard form to be passed to DatBuilders which
make DatHDFs"""
from __future__ import annotations
import os
import re
import logging
from singleton_decorator import singleton
from typing import TYPE_CHECKING, Union, Iterable, Tuple, List, Optional, Type
import threading

from dat_analysis.dat_object.dat_hdf import DatHDF, get_dat_id, DatHDFBuilder
import dat_analysis.hdf_util as HDU

if TYPE_CHECKING:
    from dat_analysis.data_standardize.base_classes import Exp2HDF

# from dat_analysis.data_standardize.exp_specific.Sep20 import SepExp2HDF
from dat_analysis.data_standardize.exp_specific.Feb21 import Feb21Exp2HDF
from dat_analysis.data_standardize.exp_specific.FebMar21 import FebMar21Exp2HDF
from dat_analysis.data_standardize.exp_specific.May21 import May21Exp2HDF
from dat_analysis.data_standardize.exp_specific.Nov21 import Nov21Exp2HDF
from dat_analysis.data_standardize.exp_specific.Nov21_LD import Nov21Exp2HDF_LD

# default_Exp2HDF = SepExp2HDF
# default_Exp2HDF = Feb21Exp2HDF
# default_Exp2HDF = FebMar21Exp2HDF
# default_Exp2HDF = Nov21Exp2HDF
default_Exp2HDF = Nov21Exp2HDF_LD


CONFIGS = {
    'febmar21': FebMar21Exp2HDF,
    'may21': May21Exp2HDF,
    'nov21': Nov21Exp2HDF,
}

logger = logging.getLogger(__name__)


# sync_lock = threading.Lock()


def get_newest_datnum(last_datnum=None, exp2hdf=Nov21Exp2HDF):
    """Get the newest datnum that already exists in Experiment data directory
    (last_datnum is useful to pass if the location of dats changes after a certain datnum for example)
    """
    last_datnum = last_datnum if last_datnum else 0
    exp2hdf = exp2hdf(last_datnum)
    data_directory = exp2hdf.SysConfig.Directories.ddir
    files = os.listdir(data_directory)
    datnums = [re.findall(r'\d+', f.split('.')[0]) for f in files]  # Split the part before the extension
    datnums = [int(num[0]) for num in datnums if num]  # Convert first found number to string if there were numbers found
    return max(datnums)


@singleton
class DatHandler(object):
    """
    Holds onto references to open dats (so that I don't try open the same datHDF more than once). Will return
    same dat instance if already open.
    Can also see what dats are open, remove individual dats from DatHandler, or clear all dats from DatHandler
    """
    open_dats = {}
    lock = threading.Lock()
    sync_lock = threading.Lock()

    def get_dat(self, datnum: int, datname='base', overwrite=False,
                init_level='min',
                exp2hdf: Optional[Union[str, Type[Exp2HDF]]] = None) -> DatHDF:
        if isinstance(exp2hdf, str):
            if exp2hdf.lower() not in CONFIGS:
                raise KeyError(f'{exp2hdf} not found in {CONFIGS.keys()}')
            exp2hdf = CONFIGS[exp2hdf.lower()](datnum=datnum, datname=datname)
        elif hasattr(exp2hdf, 'ExpConfig'):  # Just trying to check it is an Exp2HDF without importing
            exp2hdf = exp2hdf(datnum=datnum, datname=datname)
        elif exp2hdf is None:
            exp2hdf = default_Exp2HDF(datnum=datnum, datname=datname)
        else:
            raise RuntimeError(f"Don't know how to interpret {exp2hdf}")

        full_id = self._full_id(exp2hdf.ExpConfig.dir_name, datnum, datname)  # For temp local storage
        if not overwrite and full_id in self.open_dats:
            return self.open_dats[full_id]

        with self.lock:  # Only 1 thread should be making choices about deleting or creating new HDFs
            path = exp2hdf.get_datHDF_path()
            self._ensure_dir(path)
            if overwrite:
                self._delete_hdf(path)
                if full_id in self.open_dats:
                    del self.open_dats[full_id]

            if full_id not in self.open_dats:  # Need to open or create DatHDF
                if os.path.isfile(path):
                    self.open_dats[full_id] = self._open_hdf(path)
                else:
                    self._check_exp_data_exists(exp2hdf)
                    builder = DatHDFBuilder(exp2hdf, init_level)
                    self.open_dats[full_id] = builder.build_dat()
            return self.open_dats[full_id]

    def get_dats(self, datnums: Union[Iterable[int], Tuple[int, int]], datname='base', overwrite=False, init_level='min',
                 exp2hdf=None) -> List[DatHDF]:
        """Convenience for loading multiple dats at once, just calls get_dat multiple times"""
        # TODO: Make this multiprocess/threaded especially if overwriting or if dat does not already exist!
        if type(datnums) == tuple and len(datnums) == 2:
            datnums = range(*datnums)
        return [self.get_dat(num, datname=datname, overwrite=overwrite, init_level=init_level, exp2hdf=exp2hdf)
                for num in datnums]

    @staticmethod
    def _ensure_dir(path):
        dir_path = os.path.dirname(path)
        if not os.path.isdir(dir_path):
            logger.warning(f'{dir_path} was not found, being created now')
            os.makedirs(dir_path, exist_ok=True)

    @staticmethod
    def _delete_hdf(path: str):
        """Should delete hdf if it exists"""
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    @staticmethod
    def _open_hdf(path: str):
        hdf_container = HDU.HDFContainer.from_path(path)
        dat = DatHDF(hdf_container)
        return dat

    @staticmethod
    def _full_id(dir_name: str, datnum, datname):
        return f'{dir_name}:{get_dat_id(datnum, datname)}'

    def _check_exp_data_exists(self, exp2hdf: Exp2HDF):
        exp_path = exp2hdf.get_exp_dat_path()
        if os.path.isfile(exp_path):
            return True
        else:
            with self.sync_lock:
                if not os.path.isfile(exp_path):  # Might be there by time lock is released
                    exp2hdf.synchronize_data()  # Tries to synchronize data from server then check for path again.
                if os.path.isfile(exp_path):
                    return True
                raise FileNotFoundError(f'No experiment data found for dat{exp2hdf.datnum} at {os.path.abspath(exp_path)}')

    def list_open_dats(self):
        return self.open_dats

    def clear_dats(self):
        self.open_dats = {}

    def remove(self, dat: DatHDF):
        """Remove a single dat from stashed dats"""
        with self.lock:
            dat_id = self.get_open_dat_id(dat)
            if dat_id:
                del self.open_dats[dat_id]

    def get_open_dat_id(self, dat: DatHDF):
        if dat:
            for k, v in self.open_dats.items():
                if dat == v:
                    return k


get_dat = DatHandler().get_dat
get_dats = DatHandler().get_dats


if __name__ == '__main__':
    print(get_newest_datnum())
