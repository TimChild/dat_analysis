"""Utility functions for Dat"""
import os
from typing import Tuple

import h5py
import numpy as np

from src.DatCode.Entropy import _get_values_at_max


def get_id_from_val(data1d, value):
    """Returns closest ID of data to value, and returns true value"""
    return min(enumerate(data1d), key=lambda x: abs(x[1] - value))  # Gets the position of the


def get_data(hdf_path, wavename) -> np.ndarray:
    """
    Returns array of data from hdf file if given path to hdf file

    @param hdf_path: Path to file
    @type hdf_path: str
    @param wavename: name of wave in hdf file
    @type wavename: str
    @return: array of dat
    @rtype: np.ndarray
    """
    if os.path.isfile(hdf_path):
        hdf = h5py.File(hdf_path, 'r')
        if wavename in hdf.keys():
            array = hdf[wavename][:]
        else:
            print(f'WARNING: wavename [{wavename}] not found in hdf')
            array = None
    else:
        print(f'WARNING: hdf not found at {hdf_path}')
        array = None
    return array


def calc_r(x, y, phase=None):
    """Calculates R from x and y, and returns R, phase"""
    if phase is None:
        x_max, y_max, _ = _get_max_and_sign_of_max(x, y)  # Gets max of x and y at same location
        # and which was bigger
        phase = np.arctan(y_max / x_max)
    r = np.array([x * np.cos(phase) + y * np.sin(phase) for x, y in zip(x, y)])
    return r, phase


def _get_max_and_sign_of_max(x, y) -> Tuple[float, float, np.array]:
    """Returns value of x, y at the max position of the larger of the two and which was larger...
     i.e. x and y value at index=10 if max([x,y]) is x at x[10] and 'x' because x was larger"""

    if max(np.abs(x)) > max(np.abs(y)):
        which = 'x'
        x_max, y_max = _get_values_at_max(x, y)
    else:
        which = 'y'
        y_max, x_max = _get_values_at_max(y, x)
    return x_max, y_max, which