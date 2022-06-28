"""
2022-06: Making a simpler interface for DatHDF objects.
Still require a general way to interact with datasets in the HDF file
"""
import h5py

from ..hdf_file_handler import HDFFileHandler

_NOT_SET = object()


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

    @property
    def x(self):
        return self.get_data('x_array', None)

    @property
    def y(self):
        return self.get_data('y_array', None)

    @property
    def data_keys(self):
        all_keys = self._get_all_data_keys()
        keys = list(sorted(set(all_keys) - {'x_array', 'y_array', 'sweepgates_x', 'sweepgates_y'}))
        return keys

    def get_data(self, key, default=_NOT_SET):
        keys = self._get_all_data_keys()
        if key in keys:
            data = self._load_data(key)
        elif default is not _NOT_SET:
            data = default
        else:
            data = None
        return data

    def _get_all_data_keys(self):
        with self.hdf_read as f:
            data_keys = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]
        return data_keys

    def _load_data(self, path_in_data_group: str):
        with self.hdf_read as f:
            data = f[path_in_data_group][:]
        return data
