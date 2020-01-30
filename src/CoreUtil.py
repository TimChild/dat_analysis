import os
import sys
from typing import List, NamedTuple, Union, Dict, Tuple

import h5py
import numpy as np
import pandas as pd

from src import config as cfg


def verbose_message(printstr: str, forcelevel=None, forceon=False):
    """Prints verbose message if global verbose is True"""
    level = stack_size()  # TODO: set level by how far into stack the function is being called from so that prints can be formatted nicer
    if cfg.verbose is True or forceon is True and level < cfg.verboselevel:
        print(f'{printstr.rjust(level + len(printstr))}')
    return None


def add_infodict_Logs(infodict: dict = None, xarray: np.array = None, yarray: np.array = None, xlabel: str = None,
                      ylabel: str = None,
                      dim: int = None, srss: dict = None, mags: List[NamedTuple] = None,
                      temperatures: NamedTuple = None, time_elapsed: float = None, time_completed=None,
                      dacs: dict = None, dacnames: dict = None, hdfpath=None) -> dict:
    """Makes dict with all info to pass to Dat. Useful for typehints"""
    if infodict is None:
        infodict = {}
    infodict['Logs'] = {'x_array': xarray, 'y_array': yarray, 'axis_labels': {'x': xlabel, 'y': ylabel}, 'dim': dim,
                        'srss': srss, 'mags': mags, 'temperatures': temperatures, 'time_elapsed': time_elapsed,
                        'time_completed': time_completed, 'dacs': dacs, 'dacnames': dacnames, 'hdfpath': hdfpath}
    return infodict


def open_hdf5(dat, path='') -> h5py.File:
    fullpath = os.path.join(path, 'dat{0:d}.h5'.format(dat))
    return h5py.File(fullpath, 'r')


def center_data_2D(data2d: np.array, center_ids: np.array) -> np.array:
    """Centers 2D data given id's of alignment, and returns the aligned 2D data with the same shape as original"""
    data = data2d
    xarray = np.linspace(-np.average(center_ids), data.shape[0] - np.average(center_ids), data.shape[
        0])  # Length of original data centered on middle of aligned data (not centered at 0)
    aligned_2d = np.array(
        [np.interp(xarray, np.arange(data.shape[0]) - mid_id, data_1d, left=np.nan, right=np.nan) for data_1d, mid_id in
         zip(data2d, center_ids)])  # Interpolated data after shifting data to be aligned at 0
    return aligned_2d


def average_data(data2d: np.array, center_ids: np.array) -> Tuple[np.array, np.array]:
    """Takes 2D data and the center(id's) of that data and returns averaged data and standard deviations"""
    aligned_2d = center_data_2D(data2d, center_ids)
    averaged = np.array([np.average(aligned_2d[:, i]) for i in range(aligned_2d.shape[0])])  # averaged 1D data
    stderrs = np.array([np.std(aligned_2d[:, i]) for i in range(aligned_2d.shape[0])])  # stderr of 1D data
    return averaged, stderrs


@DeprecationWarning
def average_repeats(dat, returndata: str = 'i_sense', centerdata: np.array = None, retstd=False) -> Union[
    List[np.array], List[np.array]]:
    """Takes dat object and returns (xarray, averaged_data, centered by charge transition by default)"""
    if centerdata is None:
        if dat.transitionvalues.x0s is not None:  # FIXME: put in try except here if there are other default ways to center
            centerdata = dat.transitionvalues.x0s
        else:
            raise AttributeError('No data available to use for centering')

    xlen = dat.x_array[-1] - dat.x_array[0]
    xarray = np.array(list(np.linspace(-(xlen / 2), (xlen / 2), len(dat.x_array))),
                      dtype=np.float32)  # FIXME: Should think about making this create an array smaller than original with more points to avoid data degredation
    midvals = centerdata
    if returndata in dat.__dict__.keys():
        data = dat.__getattribute__(returndata)
        if data.shape[1] != len(dat.x_array) or data.shape[0] != len(dat.y_array):
            raise ValueError(f'Shape of Data is {data.shape} which is not compatible with x_array={len(dat.x_array)} '
                             f'and y_array={len(dat.y_array)}')
    else:
        raise ValueError(f'Returndata = \'{returndata}\' does not exist in dat... it is case sensitive by the way')
    matrix = np.array([np.interp(xarray, dat.x_array - midvals[i], data[i]) for i in range(len(dat.y_array))],
                      dtype=np.float32)  # interpolated data after shifting data to be centred at 0
    averaged = np.array([np.average(matrix[:, i]) for i in range(len(xarray))],
                        dtype=np.float32)  # average the data over number of rows of repeat
    stderrs = np.array([np.std(matrix[:, i]) for i in range(len(xarray))],
                       dtype=np.float32)  # stderr of each averaged datapoint
    #  All returning np.float32 because lmfit doesn't like float64

    ret = [xarray, averaged]
    if retstd is True:
        ret += [stderrs]
    return ret


def stack_size():
    frame = sys._getframe(1)
    i = 0
    while frame:
        frame = frame.f_back
        i += 1
    return i


def option_input(question: str, answerdict: Dict):
    """answerdict should be ['ans':return] format. Then this will ask user and return whatever is in 'return'"""
    answerdict = {k.lower(): v for k, v in answerdict.items()}
    for long, short in zip(['yes', 'no', 'overwrite', 'load'], ['y', 'n', 'o', 'l']):
        if long in answerdict.keys():
            answerdict[short] = answerdict[long]
    inp = input(question)
    while True:
        if inp.lower() in answerdict.keys():
            ret = answerdict[inp]
            break
        else:
            inp = input(f'Answer dictionary is {answerdict.items()}. Please enter a new answer.')
    return ret


def get_data_index(data1d, val):
    """Returns index position of nearest data value in 1d data"""
    index, _ = min(enumerate(data1d), key=lambda x: abs(x[1] - val))
    return index
