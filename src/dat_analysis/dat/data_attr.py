"""
2022-06: Making a simpler interface for DatHDF objects.
Still require a general way to interact with datasets in the HDF file
"""
import h5py
import numpy as np

from ..hdf_file_handler import HDFFileHandler
from ..hdf_util import NotFoundInHdfError


_NOT_SET = object()


class Data:
    def __init__(self, hdf_path: str, path_to_logs_group: str):
        self._hdf_path = hdf_path
        self._group_path = path_to_logs_group
        self._x = None
        self._y = None

    @property
    def hdf_read(self):
        return HDFFileHandler(self._hdf_path, "r")  # with self.hdf_read as f: ...

    @property
    def hdf_write(self):
        return HDFFileHandler(self._hdf_path, "r+")  # with self.hdf_write as f: ...

    @property
    def x(self):
        if self._x is None:
            self._x = self.get_data("x_array", None)
        return self._x

    @property
    def y(self):
        if self._y is None:
            self._y = self.get_data("y_array", None)
        return self._y

    @property
    def data_keys(self):
        all_keys = self._get_all_data_keys()
        keys = list(
            sorted(
                set(all_keys) - {"x_array", "y_array", "sweepgates_x", "sweepgates_y"}
            )
        )
        return keys

    def get_data(self, key, default=_NOT_SET):
        keys = self._get_all_data_keys()
        if key in keys:
            data = self._load_data(key)
        elif default is not _NOT_SET:
            data = default
        else:
            raise NotFoundInHdfError(
                f"{key} not found. Existing keys are {self.data_keys}"
            )
        return data

    def set_data(self, key: str, data: np.ndarray, subgroup=None):
        assert isinstance(key, str)
        assert isinstance(data, np.ndarray)
        if subgroup is None:
            subgroup = "calculated"
        with self.hdf_write as f:
            dg = f.get(self._group_path)
            group = dg.require_group(subgroup)
            if key in group.keys():
                if group[key].shape == data.shape:
                    group[key][...] = data  # Modify the existing data (more efficient)
                else:
                    del group[key]  # Remove old data so it can be replaced
                    group.create_dataset(key, data=data)
            else:
                group.create_dataset(key, data=data)

    def _get_all_data_keys(self):
        """Recursively search all sub-directories of group to find datasets"""
        data_keys = []

        def _get_keys(group: h5py.Group, prepend_path: str):
            for k in group.keys():
                if isinstance(group[k], h5py.Dataset):
                    data_keys.append("/".join([prepend_path, k]))
                elif isinstance(group[k], h5py.Group):
                    _get_keys(group[k], prepend_path="/".join([prepend_path, k]))

        with self.hdf_read as f:
            f = f.get(self._group_path)
            _get_keys(f, "")
        data_keys = [k[1:] for k in data_keys]  # Remove the leading '/'
        return data_keys

    def _load_data(self, path_in_data_group: str):
        with self.hdf_read as f:
            f = f.get(self._group_path)
            data = f[path_in_data_group][:]
        return data
