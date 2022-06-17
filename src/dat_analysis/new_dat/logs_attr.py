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
from ..hdf_file_handler import HDFFileHandler


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
