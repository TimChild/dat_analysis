"""
2022-06: Trying to make a simpler Dat interface.

The aim of the DatHDF class is to provide an easy interface to the HDF files that are made here (i.e. not the experiment
files directly which differ too much from experiment to experiment)
"""
from .data_attr import Data
from .logs_attr import Logs

from ..hdf_util import HDFFileHandler, NotFoundInHdfError


class DatHDF:

    # TODO: I think this actually might want to subclass from h5py.File? And then just wrap all calls to the h5py.File
    # TODO: with something that opens the file in the right mode first
    # TODO: Just need to think a little more about if it is OK to be accessing the file a lot


    def __init__(self, hdf_path: str):
        self._hdf_path = hdf_path
        self._check_existing_hdf()

        self.Logs = Logs(hdf_path, '/Logs')
        self.Data = Data(hdf_path, '/Data')
        # self.Analysis = ... maybe add some shortcuts to standard analysis stuff? Fitting etc

    @property
    def hdf_read(self):
        return HDFFileHandler(self._hdf_path, 'r')  # with self.hdf_read as f: ...

    @property
    def hdf_write(self):
        return HDFFileHandler(self._hdf_path, 'r+')  # with self.hdf_write as f: ...

    def _check_existing_hdf(self):
        """Check the hdf_path points to an HDF file that contains expected groups/attrs"""
        with self.hdf_read as f:
            keys = f.keys()
            for k in ['Logs', 'Data']:
                if k not in keys:
                    raise NotFoundInHdfError(f'Did not find {k} in {self._hdf_path}')





