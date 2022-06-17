"""
2022-06: Making a simpler interface for DatHDF objects.
Still require a general way to interact with datasets in the HDF file
"""
from ..hdf_file_handler import HDFFileHandler


class Data:
    def __init__(self, hdf_path: str, path_to_logs_group: str):
        self._hdf_path = hdf_path
        self._group_path = path_to_logs_group

    @property
    def hdf_read(self):
        return HDFFileHandler(self._hdf_path, 'r', internal_path=self._group_path)  # with self.hdf_read as f: ...

    @property
    def hdf_write(self):
        return HDFFileHandler(self._hdf_path, 'r+', internal_path=self._group_path)  # with self.hdf_write as f: ...
