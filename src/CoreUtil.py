import copy
import functools
import os
import sys
from typing import List, NamedTuple, Union, Dict, Tuple

import h5py
import lmfit as lm
import numpy as np
import pandas
import win32com.client
import re
import numbers
import pandas as pd
import src.Characters as Char
from src.Configs import Main_Config as cfg, Main_Config


# TODO: This shouldn't use the current config, it should use whatever config was used for dat or datdf etc... hmm
def path_replace(path):
    """For replacing chunk of path using cfg.path_replace in case experiment file has been moved for example"""
    if cfg.path_replace is not None:
        pattern, repl = cfg.path_replace
        if pattern and repl:
            pre, match, post = path.rpartition(pattern)
            path = ''.join((pre, repl if match else match, post))
    return path


def get_full_path(path):
    """
    Fixes paths if files have been moved by returning the full path even if there is a shortcut somewhere along the way,
    or replacing parts using the current config file cfg.path_replace

    @param path: Possibly old path to a file or folder (i.e. something may have been moved and replaced with a shortcut)
    @type path: str
    @return: Correct path to file or shortcut taking into account shortcuts
    @rtype: str
    """
    def _split_path(path):
        """carefully returns head, tail of path"""
        assert len(path) > 0
        head_path, tail_path = os.path.split(path)
        if tail_path == '':  # If was already point to a directory
            head_path, tail_path = os.path.split(head_path)
        return head_path, tail_path

    path = path_replace(path)  # Fixes path if necessary (e.g. if experiment has been moved)
    o_path = path
    tail_path = ''
    if os.path.exists(path):
        return path
    else:

        while True:
            if os.path.isfile(path+'.lnk') is True:
                break
            path, tail = _split_path(path)
            if tail == '':  # Must have got to top of file path and not found a shortcut
                print(f'WARNING[CU.get_full_path]: Path [{o_path}] does not exist and contains no shortcuts')
                return o_path
                # raise ValueError(f'{path+tail_path} is not valid and contains no shortcut links either')
            if tail_path != '':  # Otherwise lose track of if the path was to a file
                tail_path = os.path.join(tail, tail_path)
            else:
                tail_path = tail
        target = _get_shortcut_target(path)
    return os.path.join(target, tail_path)


def _get_shortcut_target(path):
    """
    Returns target of shortcut file at given path (where path points to the expected name of directory)

    @param path: Path to directory which may be replaced with shortcut
    @return: Target path of shortcut with same name as directory specified if it exists
    @raise: ValueError if no shortcut exists
    """
    shell = win32com.client.Dispatch("WScript.Shell")
    path = path + '.lnk'  # If it's a shortcut instead of a folder it will appear as a .lnk file
    if os.path.isfile(path) is True:
        shortcut = shell.CreateShortCut(path)
    else:
        raise ValueError(f'Path "{path}" is not a shortcut link')
    return shortcut.TargetPath


def verbose_message(printstr: str, forcelevel=None, forceon=False):
    """Prints verbose message if global verbose is True"""
    level = stack_size()  # TODO: set level by how far into stack the function is being called from so that prints can be formatted nicer
    if cfg.verbose is True or forceon is True and level < cfg.verboselevel:
        print(f'{printstr.rjust(level + len(printstr))}')
    return None


def add_infodict_Logs(infodict: dict = None, xarray: np.array = None, yarray: np.array = None, x_label: str = None,
                      y_label: str = None,
                      dim: int = None, srss: dict = None, mags: dict = None,
                      temperatures: NamedTuple = None, time_elapsed: float = None, time_completed=None,
                      dacs: dict = None, dacnames: dict = None, fdacs: dict = None, fdacnames: dict = None, fdacfreq: float = None, comments: str = None) -> dict:
    """Makes dict with all info to pass to Dat. Useful for typehints"""
    if infodict is None:
        infodict = {}
    infodict['Logs'] = {'x_array': xarray, 'y_array': yarray, 'axis_labels': {'x': x_label, 'y': y_label}, 'dim': dim,
                        'srss': srss, 'mags': mags, 'temperatures': temperatures, 'time_elapsed': time_elapsed,
                        'time_completed': time_completed, 'dacs': dacs, 'dacnames': dacnames, 'fdacs': fdacs, 'fdacnames': fdacnames, 'fdacfreq': fdacfreq, 'comments': comments}
    return infodict


def open_hdf5(dat, path='') -> h5py.File:
    fullpath = os.path.join(path, 'dat{0:d}.h5'.format(dat))
    return h5py.File(fullpath, 'r')


def center_data_2D(data2d: np.array, center_ids: np.array) -> np.array:
    """Centers 2D data given id's of alignment, and returns the aligned 2D data with the same shape as original"""
    data = data2d
    xarray = np.linspace(-np.average(center_ids), data.shape[1] - np.average(center_ids), data.shape[
        1])  # Length of original data centered on middle of aligned data (not centered at 0)
    aligned_2d = np.array(
        [np.interp(xarray, np.arange(data.shape[1]) - mid_id, data_1d, left=np.nan, right=np.nan) for data_1d, mid_id in
         zip(data2d, center_ids)])  # Interpolated data after shifting data to be aligned at 0
    return aligned_2d


def average_data(data2d: np.array, center_ids: np.array) -> Tuple[np.array, np.array]:
    """Takes 2D data and the center(id's) of that data and returns averaged data and standard deviations"""
    aligned_2d = center_data_2D(data2d, center_ids)
    averaged = np.array([np.average(aligned_2d[:, i]) for i in range(aligned_2d.shape[1])])  # averaged 1D data
    stderrs = np.array([np.std(aligned_2d[:, i]) for i in range(aligned_2d.shape[1])])  # stderr of 1D data
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

    if 'yes' in answerdict.keys() and cfg.yes_to_all is True:
        print(f'Automatically answered "{question}":\n"yes"')
        return answerdict['yes']

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


def data_to_NamedTuple(data: dict, named_tuple) -> NamedTuple:
    """Given dict of key: data and a named_tuple with the same keys, it returns the filled NamedTuple
    If data is not stored then a cfg._warning string is set"""
    tuple_dict = named_tuple.__annotations__  # Get ordered dict of keys of namedtuple
    for key in tuple_dict.keys():  # Set all values to None so they will default to that if not entered
        tuple_dict[key] = None
    for key in set(data.keys()) & set(tuple_dict.keys()):  # Enter valid keys values
        tuple_dict[key] = data[key]
    if set(data.keys()) - set(tuple_dict.keys()):  # If there is something left behind
        cfg.warning = f'data keys not stored: {set(data.keys()) - set(tuple_dict.keys())}'
        # region Verbose  data_to_NamedTuple
        if cfg.verbose is True:
            print(
                f'WARNING: The following data is not being stored: {set(data.keys()) - set(tuple_dict.keys())}')
        # endregion
    else:
        cfg.warning = None
    ntuple = named_tuple(**tuple_dict)
    return ntuple


def set_kwarg_if_none(key, value, kwargs) -> dict:
    """Only updates or adds key, value if current value is None or doesn't exist"""
    if kwargs.get(key, None) is None:  # If key doesn't exist, or value is None
        kwargs[key] = value  # Set value
    return kwargs


def ensure_list(data) -> list:
    if type(data) == str:
        return [data]
    elif type(data) == list:
        return data
    elif type(data) == tuple:
        return list(data)
    else:
        raise ValueError('Either not a list, or not implemented yet')


def ensure_set(data) -> set:
    if type(data) == set:
        return data
    else:
        return set(ensure_list(data))


def data_index_from_width(x_array, mid_val, width) -> Tuple[int, int]:
    """Returns (low, high) index of data around mid_val (being careful about size of data"""
    low_index = round(get_data_index(x_array, mid_val-width/2))
    high_index = round(get_data_index(x_array, mid_val+width/2))
    if low_index < 0:
        low_index = 0
    if high_index > len(x_array)-1:
        high_index = -1
    return low_index, high_index


def edit_params(params: lm.Parameters, param_name, value=None, vary=None, min_val=None, max_val=None) -> lm.Parameters:
    """
    Returns a deepcopy of parameters with values unmodified unless specified

    @param params:  single lm.Parameters
    @type params:  lm.Parameters
    @param param_name:  which parameter to vary
    @type param_name:  str
    @param value: initial or fixed value
    @type value: float
    @param vary: whether it's varied
    @type vary: bool
    @param min_val: min value
    @type min_val: Union[float, None]
    @param max_val: max value
    @type max_val: Union[float, None]
    @return: single lm.Parameters
    @rtype: lm.Parameters
    """

    params = copy.deepcopy(params)
    if min_val is None:
        min_val = params[param_name].min
    if max_val is None:
        max_val = params[param_name].max
    if value is None:
        value = params[param_name].value
    if vary is None:
        vary = params[param_name].vary
    params[param_name].vary = vary
    params[param_name].value = value
    params[param_name].min = min_val
    params[param_name].max = max_val
    return params


def sig_fig(val, sf=5):
    """
    Rounds to given given significant figures - taken from https://stackoverflow.com/a/59888924/12620905

    @param val: int, float, array, of values to round. Handles np.nan,
    @param sf: How many significant figures to round to.
    """
    def sig_fig_array(val, sf): # Does the actual rounding part of int, float, array
        x = np.asarray(val)
        x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (sf - 1))
        mags = 10 ** (sf - 1 - np.floor(np.log10(x_positive)))
        return np.round(x * mags) / mags

    if not isinstance(val, (numbers.Number, pd.Series, pd.DataFrame, np.ndarray)):
        return val
    elif type(val) == bool:
        return val
    if isinstance(val, pd.DataFrame):
        num_dtypes = (float, int)
        for col in val.columns:
            if val[col].dtype in num_dtypes:  # Don't try to apply to strings for example
                val[col] = val[col].apply(lambda x: sig_fig_array(x, sf))  # Apply sig fig function to column
        return val
    elif type(val) == int:
        return int(sig_fig_array(val, sf))  # cast back to int afterwards
    else:
        return sig_fig_array(val, sf)


def fit_info_to_df(fits, uncertainties=False, sf=4, index=None):
    """
    Takes list of fits and puts all fit params into a dataframe optionally with index labels. Also adds reduced chi sq

    @param fits: list of fit results
    @type fits: List[lm.model.ModelResult]
    @param uncertainties: whether to show +- uncertainty in table. If so, values in table will be strings to sig fig given
    @type uncertainties: bool
    @param sf: how many sig fig to give values to if also showing uncertainties, otherwise full values
    @type sf: int
    @param index: list to use as index of dataframe
    @type index: List
    @return: dataframe of fit info with index and reduced chi square
    @rtype: pd.DataFrame
    """

    columns = ['index'] + list(fits[0].best_values.keys()) + ['reduced_chi_sq']
    if index is None or len(index) != len(fits):
        index = range(len(fits))
    if uncertainties is False:
        data = [[ind] + list(fit.best_values.values()) + [fit.redchi] for i, (ind, fit) in enumerate(zip(index, fits))]
    elif uncertainties is True:
        keys = fits[0].best_values.keys()
        data = [[ind] + [str(sig_fig(fit.params[key].value, sf))+Char.PM+str(sig_fig(fit.params[key].stderr, 2))
                         for key in keys] + [fit.redchi] for i, (ind, fit) in enumerate(zip(index, fits))]
    else:
        raise NotImplementedError
    return pd.DataFrame(data=data, columns=columns)



def switch_config_decorator_maker(config, folder_containing_experiment=None):
    """
    Decorator Maker - makes a decorator which switches to given config and back again at the end

    @param config: config file to switch to temporarily
    @type config: module
    @return: decorator which will switch to given config temporarily
    @rtype: function
    """

    def switch_config_decorator(func):
        """
        Decorator - Switches config before a function call and returns it back to starting state afterwards

        @param func: Function to wrap
        @type func: function
        @return: Wrapped Function
        @rtype: function
        """

        @functools.wraps(func)
        def func_wrapper(*args, **kwargs):
            if config != cfg.current_config:  # If config does need changing
                old_config = cfg.current_config  # save old config module (probably current experiment one)
                old_folder_containing_experiment = cfg.current_folder_containing_experiment
                cfg.set_all_for_config(config, folder_containing_experiment)
                result = func(*args, **kwargs)
                cfg.set_all_for_config(old_config, old_folder_containing_experiment)  # Return back to original state
            else:  # Otherwise just run func like normal
                result = func(*args, **kwargs)
            return result

        return func_wrapper

    return switch_config_decorator


def wrapped_call(decorator, func):
    result = decorator(func)()
    return result