import os
import sys
from typing import List, NamedTuple, Union, Dict

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


def add_infodict_Logs(infodict: dict = None, xarray: np.array = None, yarray: np.array = None, xlabel: str = None, ylabel: str = None,
                      dim: int = None, srss: dict = None, mags: List[NamedTuple] = None,
                      temperatures: NamedTuple = None, time_elapsed: float = None, time_completed=None,
                      dacs:dict = None, dacnames: dict = None) -> dict:
    """Makes dict with all info to pass to Dat. Useful for typehints"""
    if infodict is None:
        infodict = {}
    infodict['Logs'] = {'xarray': xarray, 'yarray': yarray, 'axis_labels': {'x': xlabel, 'y': ylabel}, 'dim': dim,
        'srss': srss, 'mags': mags, 'temperatures': temperatures, 'time_elapsed': time_elapsed,
        'time_completed': time_completed, 'dacs': dacs, 'dacnames': dacnames}
    return infodict


def open_hdf5(dat, path='') -> h5py.File:
    fullpath = os.path.join(path, 'dat{0:d}.h5'.format(dat))
    return h5py.File(fullpath, 'r')


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


def add_col_label(df, new_col, on_cols, level=1):
    def _new_level_emptycols(df, level=1, address='top'):
        if level == 1:
            return dict(zip(df.columns, np.repeat('', df.shape[1])))
        else:
            if address == 'full':
                return dict(zip([x for x in df.columns], np.repeat('', df.shape[1])))
            elif address == 'top':
                return dict(zip([x[0] for x in df.columns], np.repeat('', df.shape[1])))
            else:
                raise ValueError(f'Address "{address}" is not valid, choose "top" or "full"')

    def _existing_level_cols(df, level=1, address='top'):
        newcols = [x[level] for x in list(df.columns)]
        if address == 'top':
            newcols = dict(zip([x[0] for x in df.columns], newcols))
        elif address == 'full':
            newcols = dict(zip(df.columns.levels, newcols))
        return newcols

    def _newcols_generator(dfinternal, level):
        if isinstance(dfinternal.columns, pd.Index) and not isinstance(dfinternal.columns,
                                                                       pd.MultiIndex):  # if only 1D index, must be asking for new column level
            newcolsfn = _new_level_emptycols
        elif len(dfinternal.columns.levels) - 1 < level:  # if asking for new level
            newcolsfn = _new_level_emptycols
        else:  # column labels already exist
            newcolsfn = _existing_level_cols
        return newcolsfn

    dfinternal = df[:]  # shallow copy to prevent changing later df's
    if level == 0:
        raise ValueError("Using level 0 will overwrite main column titles")
    if type(on_cols) != list:
        on_cols = [on_cols]
    if type(on_cols[0]) == tuple:  # if fully addressing with tuples
        address = 'full'
    else:
        address = 'top'

    newcolsfn = _newcols_generator(df, level)  # Either gets _new... or _existing... colnames
    newcols = newcolsfn(dfinternal, level, address=address)

    for col in on_cols:  # Set new values of columns
        newcols[col] = new_col
    if isinstance(dfinternal.columns, pd.Index) and not isinstance(dfinternal.columns, pd.MultiIndex):
        colarray = [list(dfinternal.columns)]
    else:
        colarray = []
        for i in [x for x in range(len(dfinternal.columns.levels)) if x != level]:
            colarray.append([x[i] for x in dfinternal.columns])
    colarray.append(list(newcols.values()))
    dfinternal.columns = pd.MultiIndex.from_arrays(colarray)
    return dfinternal


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
