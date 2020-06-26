import copy
import functools
import os
from typing import List, NamedTuple, Dict, Tuple
from slugify import slugify

import lmfit as lm
import numpy as np
import win32com.client
import numbers
import pandas as pd
import logging
import scipy.io as sio
import scipy.signal
import src.Characters as Char
from src import Constants as Const
from src.Configs import Main_Config as cfg

logger = logging.getLogger(__name__)


def set_default_logging():
    logging.basicConfig(level=logging.INFO, format=f'%(threadName)s %(funcName)s %(lineno)d %(message)s')


def plan_to_remove(func):
    """Wrapper for functions I am planning to remove"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.warning(f'Planning to deprecate {func.__name__}')
        return func(*args, **kwargs)
    return wrapper


def _path_replace(path, path_replace):
    """For replacing chunk of path using cfg.path_replace in case experiment file has been moved for example"""
    if path_replace is not None:
        pattern, repl = path_replace
        if pattern and repl:
            pre, match, post = path.rpartition(pattern)
            if match is not None:
                logger.warning(f'Planning to remove this function: replacing {pattern} with {repl} in {path}. Match was {match}')
            path = ''.join((pre, repl if match else match, post))
    return path


def get_full_path(path, path_replace=None):
    """
    Fixes paths if files have been moved by returning the full path even if there is a shortcut somewhere along the way,
    or replacing parts using the current config file cfg.path_replace (i.e. if changing where data is stored on PC)

    @param path: Possibly old path to a file or folder (i.e. something may have been moved and replaced with a shortcut)
    @type path: str
    @param path_replace:(pattern, repl) to fix the path to the experiment folder
    @type path_replace: Union[Tuple[str, str], None]
    @return: Correct path to file or shortcut taking into account shortcuts
    @rtype: str
    """
    def _split_path(p):
        """carefully returns head, tail of path"""
        assert len(p) > 0
        hp, tp = os.path.split(p)
        if tp == '':  # If was already point to a directory
            hp, tp = os.path.split(hp)
        return hp, tp

    path = _path_replace(path, path_replace)  # Fixes path if necessary (e.g. if experiment has been moved)
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


@plan_to_remove
def verbose_message(printstr: str, forcelevel=None, forceon=False):
    """Prints verbose message if global verbose is True"""
    level = 0  # removed function for this
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


def center_data_2D(data2d: np.array, center_ids: np.array) -> np.array:  # TODO: Time this, and improve it by making the interpolation a vector operation (or multiprocess it)
    # TODO: Also is it faster to do this if I force float.16 or something?
    """Centers 2D data given id's of alignment, and returns the aligned 2D data with the same shape as original"""
    data = data2d
    xarray = np.linspace(-np.average(center_ids), data.shape[1] - np.average(center_ids), data.shape[
        1])  # Length of original data centered on middle of aligned data (not centered at 0)

    # scipy.interpolate.interp1d(xarray, data2d, axis=-1, bounds_error=False, fill_value=np.NaN, assume_sorted=True)  Think more about this. Is this bit even slow?
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
    """
    Returns index position(s) of nearest data value(s) in 1d data

    @param data1d: data to compare values
    @type data1d: np.ndarray
    @param val: value(s) to find index positions of
    @type val: Union[float, list, tuple, np.ndarray]
    @return: index value(s)
    @rtype: Union[int, List[int]]
    """
    val = np.asarray(val)
    if val.ndim == 0:
        index, _ = min(enumerate(data1d), key=lambda x: abs(x[1] - val))
    elif val.ndim == 1:
        index = [get_data_index(data1d, v) for v in val]
    else:
        raise ValueError('ERROR[get_data_index]: val must be 0D or 1D points')
    return index


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
    @type param_name:  Union[str, list]
    @param value: initial or fixed value
    @type value: Union[float, None, list]
    @param vary: whether it's varied
    @type vary: Union[bool, None, list]
    @param min_val: min value
    @type min_val: Union[float, None, list]
    @param max_val: max value
    @type max_val: Union[float, None, list]
    @return: single lm.Parameters
    @rtype: lm.Parameters
    """
    
    def _make_array(val):
        if val is None:
            val = as_array([None]*len(param_names))
        else:
            val = as_array(val)
            assert len(val) == len(param_names)
        return val

    def as_array(val):
        val = np.asarray(val)
        if val.ndim == 0:
            val = np.array([val])
        return val

    params = copy.deepcopy(params)
    param_names = as_array(param_name)

    values = _make_array(value)
    varys = _make_array(vary)
    min_vals = _make_array(min_val)
    max_vals = _make_array(max_val)

    for param_name, value, vary, min_val, max_val in zip(param_names, values, varys, min_vals, max_vals):
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
        val = copy.deepcopy(val)
        num_dtypes = (float, int)
        for col in val.columns:
            if val[col].dtype in num_dtypes:  # Don't try to apply to strings for example
                val[col] = val[col].apply(lambda x: sig_fig_array(x, sf))  # Apply sig fig function to column
        return val
    elif type(val) == int:
        return int(sig_fig_array(val, sf))  # cast back to int afterwards
    else:
        return sig_fig_array(val, sf).astype(np.float32)


def fit_info_to_df(fits, uncertainties=False, sf=4, index=None):
    """
    Takes list of fits and puts all fit params into a dataframe optionally with index labels. Also adds reduced chi sq

    @param fits: list of fit results
    @type fits: List[lm.model.ModelResult]
    @param uncertainties: whether to show +- uncertainty in table. If so, values in table will be strings to sig fig
    given. 2 will return uncertainties only in df.
    @type uncertainties: Union[bool, int]
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
    if uncertainties == 0:
        data = [[ind] + list(fit.best_values.values()) + [fit.redchi] for i, (ind, fit) in enumerate(zip(index, fits))]
    elif uncertainties == 1:
        keys = fits[0].best_values.keys()
        data = [[ind] + [str(sig_fig(fit.params[key].value, sf))+Char.PM+str(sig_fig(fit.params[key].stderr, 2))
                         for key in keys] + [fit.redchi] for i, (ind, fit) in enumerate(zip(index, fits))]
    elif uncertainties == 2:
        keys = fits[0].best_values.keys()
        data = [[ind] + [fit.params[key].stderr for key in keys] + [fit.redchi] for i, (ind, fit) in enumerate(zip(index, fits))]
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


@plan_to_remove
def print_verbose(text, verbose):
    """Only print if verbose is True"""
    if verbose is True:
        print(text)


def get_alpha(mV, T):
    """
    From known temp for given gate voltage broadeneing, return the lever arm value (alpha)
    @param mV: Broadening of transition in mV
    @type mV: Union[np.ndarray, list[float], float]
    @param T: Known temperature of electrons in mK
    @type T: Union[np.ndarray, list[float], float]
    @return: lever arm (alpha) in SI units
    @rtype: float
    """
    mV = np.asarray(mV)  # in mV
    T = np.asarray(T)  # in mK
    kb = Const.kb  # in mV/K
    if mV.ndim == 0 and T.ndim == 0:  # If just single value each
        alpha = kb*T/1000/mV  # *1000 to K
    elif mV.ndim == 1 and T.ndim == 1:
        line = lm.models.LinearModel()
        fit = line.fit(T/1000, x=mV)
        intercept = fit.best_values['intercept']
        if np.abs(intercept) > 0.01:  # Probably not a good fit, should go through 0K
            logger.warning(f'Intercept of best fit of T vs mV is {intercept*1000:.2f}mK')
        slope = fit.best_values['slope']
        alpha = slope*kb
    else:
        raise NotImplemented
    return alpha


def ensure_params_list(params, data, verbose=True):
    """
    Make sure params is a list of lm.Parameters which matches the y dimension of data if it is 2D

    @param params: possible params, list of params, list of 1 param
    @type params: Union[list, lm.Parameters]
    @param data: data going to be fit
    @type data: np.ndarray
    @return: list of params which is right length for data
    @rtype: list[lm.Parameters]
    """
    if isinstance(params, lm.Parameters):
        if data.ndim == 2:
            params = [params] * data.shape[0]
        elif data.ndim == 1:
            params = [params]
        else:
            raise NotImplementedError
    elif isinstance(params, list):
        if data.ndim == 1:
            if len(params) != 1:
                print_verbose(f'Wrong length list of params. Only using first of parameters', verbose)
                params = [params[0]]
        elif data.ndim == 2:
            if len(params) != data.shape[0]:
                print_verbose(f'Wrong length list of params. Making params list multiple of first param', verbose)
                params = [params[0]]*data.shape[0]
        else:
            raise NotImplementedError
    else:
        raise ValueError(f'[{params}] is not a supported parameter list/object')
    return params


def bin_data(data, bin_size):
    """
    Reduces size of dataset by binning data with given bin_size. Works for 1D, 2D or list of datasets
    @param data: Either single 1D or 2D data, or list of dataset
    @type data: Union[np.ndarray, list]
    @param bin_size: bin_size (will drop the last values that don't fit in bin)
    @type bin_size: Union[float, int]
    @return: list of binned datasets, or single binned dataset
    @rtype: Union[list[np.ndarray], np.ndarray]
    """
    def _bin_1d(d, bin1d):
        d = np.asarray(d)
        assert d.ndim == 1
        new_data = []
        s = 0
        while s+bin1d <= len(d):
            new_data.append(np.average(d[s:s + bin1d]))
            s += bin1d
        return np.array(new_data).astype(np.float32)

    def _bin_2d(d, bin2d):
        d = np.asarray(d)
        if d.ndim == 1:
            return _bin_1d(d, bin2d)
        elif d.ndim == 2:
            return np.array([_bin_1d(row, bin2d) for row in d])

    bin_size = int(bin_size)
    if bin_size <= 1:
        return data
    else:
        if isinstance(data, (list, tuple)):  # Possible list of datasets
            if len(data) > bin_size*10:  # Probably just a dataset that isn't an np.ndarray
                print(f'WARNING[CU.bin_data]: data passed in was a list with len [{len(data)}].'
                      f' Assumed this to be a 1D dataset rather than list of datasets.'
                      f' Making data an np.ndarray first will prevent this warning message in the future')
                return _bin_2d(data, bin_size)
            else:
                return [_bin_2d(data_set, bin_size) for data_set in data]
        elif isinstance(data, np.ndarray):
            if data.ndim not in [1, 2]:
                raise NotImplementedError(f'ERROR[CU.bin_data]:Only 1D or 2D data supported for binning.'
                                          f' Data passed had ndim = [{data.ndim}')
            else:
                return _bin_2d(data, bin_size)
        else:
            print(f'WARNING[CU.bin_data]: Bad datatype [{type(data)}] passed in. Returned None')
            return None


def del_kwarg(name, kwargs):
    """
    Deletes name(s) from kwargs if present in kwargs
    @param kwargs: kwargs to try deleting args from
    @type kwargs: dict
    @param name: name or names of kwargs to delete
    @type name: Union[str, List[str]]
    @return: None
    @rtype: None
    """
    def del_1_kwarg(n, ks):
        try:
            del ks[n]
        except KeyError:
            pass
    names = np.atleast_1d(name)
    for name in names:
        del_1_kwarg(name, kwargs)


def sub_poly_from_data(x, z, fits):
    """
    Subtracts polynomial terms from data if they exist (i.e. will sub up to quadratic term)
    @param x: x data
    @type x: np.ndarray
    @param z: y data
    @type z: np.ndarray
    @param fits: lm fit(s) which has best values for up to quad term (const, lin, quad)
    @type fits: Union[list[lm.model.ModelResult], lm.model.ModelResult]
    @return: tuple of x, y or list of x, y tuples
    @rtype: Union[tuple[np.ndarray], list[tuple[np.ndarray]]]
    """

    def _sub_1d(x1d, z1d, fit1d):
        mid = fit1d.best_values.get('mid', 0)
        const = fit1d.best_values.get('const', 0)
        lin = fit1d.best_values.get('lin', 0)
        quad = fit1d.best_values.get('quad', 0)

        x1d = x1d - mid
        subber = lambda x, y: y - quad * x ** 2 - lin * x - const
        z1d = subber(x1d, z1d)
        return x1d, z1d

    x = np.asarray(x)
    z = np.asarray(z)
    assert x.ndim == 1
    assert z.ndim in [1, 2]
    assert isinstance(fits, (lm.model.ModelResult, list, tuple, np.ndarray))
    if z.ndim == 2:
        x = np.array([x] * z.shape[0])
        if not isinstance(fits, (list, tuple, np.ndarray)):
            fits = [fits] * z.shape[0]
        return x[0], np.array([_sub_1d(x1, z1, fit1)[1] for x1, z1, fit1 in zip(x, z, fits)])

    elif z.ndim == 1:
        return _sub_1d(x, z, fits)


def _save_to_checks(datas, names, file_path, fp_ext=None):
    assert type(datas) == list
    assert type(names) == list
    base, tail = os.path.split(file_path)
    if base != '':
        assert os.path.isdir(base)  # Check points to existing folder
    if fp_ext is not None:
        if tail[-(len(fp_ext)):] != fp_ext:
            tail += fp_ext  # add extension if necessary
            logger.warning(f'added "{fp_ext}" to end of file_path provided to make [{file_path}]')
            file_path = os.path.join(base, tail)
    return file_path


def save_to_mat(datas, names, file_path):
    file_path = _save_to_checks(datas, names, file_path, fp_ext='.mat')
    mat_data = dict(zip(names, datas))
    sio.savemat(file_path, mat_data)
    logger.info(f'saved [{names}] to [{file_path}]')


def save_to_txt(datas, names, file_path):
    file_path = _save_to_checks(datas, names, file_path, fp_ext='.txt')
    for data, name in zip(datas, names):
        path, ext = os.path.splitext(file_path)
        fp = path+f'_{slugify(name)}'+ext  # slugify ensures filesafe name
        np.savetxt(fp, data)
        logger.info(f'saved [{name}] to [{fp}]')


def remove_nans(nan_data, other_data = None):
    """Removes np.nan values from 1D data, and removes corresponding values from 'other_data' if passed"""
    assert isinstance(nan_data, (np.ndarray, pd.Series))
    assert nan_data.ndim == 1
    if other_data is not None:
        assert isinstance(other_data, (np.ndarray, pd.Series))
        assert nan_data.shape == other_data.shape
    mask = ~np.isnan(nan_data)
    nans_removed = np.sum(mask)-len(nan_data)
    if nans_removed > 0:
        logger.info(f'Removed {nans_removed} np.nans')
    if other_data is not None:
        return nan_data[mask], other_data[mask]
    else:
        return nan_data[mask]


def check_dat_xor_args(dat, args) -> bool:
    """Check that either dat exists or all args exist"""
    args = ensure_list(args)
    return (dat is None) ^ (all(arg is None for arg in [args]))


def get_nested_attr_default(obj, attr_path, default):
    """Trys getting each attr separated by . otherwise returns default
    @param obj: object to look for attributes in
    @param attr_path: attribute path to look for (e.g. "Logs.x_label")
    @type attr_path: str
    @param default: value to default to in case of error or None
    @type default: any
    @return: Value of attr or default
    @rtype: any
    """
    attrs = attr_path.split('.')
    val = obj
    for attr in attrs:
        val = getattr(val, attr, None)
        if val is None:
            break
    if val is None:
        return default
    else:
        return val


def power_spectrum(data, meas_freq, normalization=1):
    """
    Computes power spectrum and returns freq, power spec

    @param data: data to calculate power spectrum of
    @type data: np.ndarray
    @param meas_freq: frequency of measurement (not just sample rate)
    @type meas_freq: float
    @param normalization: Multiply data by this before calculating (i.e. if comparing power spec with different
    current amp settings)
    @return: frequencies, power spectrum
    @rtype: List[np.ndarray, np.ndarray]
    """
    freq, power = scipy.signal.periodogram(data*normalization, fs=meas_freq)
    return freq, power