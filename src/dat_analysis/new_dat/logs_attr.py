"""
2022-06: Making a simpler interface for DatHDF files
Generally will be useful to have an easy way to navigate through Logs

This should not require specific things to exist though (i.e. should not be a problem to initialize without
temperature data, and maybe raise a reasonable error if asked for temperature data?)

Not sure yet how I want to handle the possibility of different logs existing...
-- Hard coding has the advantage of typing, but will be misleading as to what Logs actually exist
    -- But I could just make a method which tries to get all the different logs excepting any missing
-- Dynamically adding attributes for Logs would be nice in Jupyter/console, but bad for static typing...
"""
from __future__ import annotations
from ..hdf_file_handler import HDFFileHandler
from dataclasses import dataclass
import logging

from ..hdf_util import get_attr, HDFStoreableDataclass, NotFoundInHdfError


logger = logging.getLogger(__name__)


class Logs:
    def __init__(self, hdf_path: str, path_to_logs_group: str):
        self._hdf_path = hdf_path
        self._group_path = path_to_logs_group

    @property
    def hdf_read(self):
        return HDFFileHandler(self._hdf_path, 'r', internal_path=self._group_path)  # with self.hdf_read as f: ...

    @property
    def hdf_write(self):
        return HDFFileHandler(self._hdf_path, 'r+', internal_path=self._group_path)  # with self.hdf_write as f: ...

    @property
    def logs_keys(self):
        all_keys = self._get_all_keys()
        keys = set(all_keys) - set()
        return keys

    @property
    def dacs(self):
        """All DACs connected (i.e. babydacs, fastdacs, ... combined)"""
        keys = self.logs_keys
        dacs = {}
        for i in range(1, 10):  # Up to 10 FastDACs
            if f'FastDAC{i}' in keys:
                fd = self.get_fastdac(i)
                dacs.update(fd.dacs)

        for i in range(1, 10):  # Up to 10 BabyDACs
            pass  # TODO: add BabyDacs
        return dacs

    @property
    def temperatures(self):
        temps = self._get_temps()
        return temps

    @property
    def x_label(self):
        with self.hdf_read as f:
            label = f['General'].attrs.get('x_label', None)
        return label

    @property
    def y_label(self):
        with self.hdf_read as f:
            label = f['General'].attrs.get('y_label', None)
        return label

    @property
    def measure_freq(self):
        with self.hdf_read as f:
            freq = f['General'].attrs.get('measure_freq', None)
        return freq

    def get_fastdac(self, num=1) -> FastDAC:
        """Load entry for FastDAC logs (i.e. dict of Dac channels where keys are labels or channel num)"""
        fd_logs = None
        with self.hdf_read as f:
            try:
                fd_logs = FastDAC.from_hdf(f, f'FastDAC{num}')
            except NotFoundInHdfError:
                logger.warning(f'FastDAC{num} not found in DatHDF')
            except Exception as e:
                logger.warning(f'need to set exceptions I should catch here')
                raise e
        return fd_logs

    def _get_temps(self):
        temps = None
        if 'Temperatures' in self.logs_keys:
            with self.hdf_read as f:
                temps = Temperatures.from_hdf(f, 'Temperatures')
        return temps

    def _get_all_keys(self):
        """Get all keys that are groups or attrs of top group"""
        keys = []
        with self.hdf_read as f:
            keys.extend(f.attrs.keys())
            keys.extend(f.keys())
        return keys


@dataclass
class FastDAC(HDFStoreableDataclass):
    dac_vals: dict
    dac_names: dict
    adcs: dict
    sample_freq: float
    measure_freq: float
    AWG: dict
    visa_address: str

    @property
    def dacs(self):
        """Dict of {dacname: dacval}"""
        return {name: val for name, val in zip(self.dac_names.values(), self.dac_vals.values())}


@dataclass
class Temperatures(HDFStoreableDataclass):
    fiftyk: float
    fourk: float
    magnet: float
    still: float
    mc: float



