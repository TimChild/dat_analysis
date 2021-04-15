import src.Characters as Characters
import logging
import os
import logging
import numpy as np
import scipy.signal
from scipy import io as sio
from slugify import slugify
from typing import List, Tuple, Iterable, Union, Dict
import plotly.graph_objs as go
import json

logger = logging.getLogger(__name__)

from src.CoreUtil import get_data_index, get_matching_x, edit_params, sig_fig, bin_data, decimate, FIR_filter, \
    get_sweeprate, bin_data_new, get_bin_size, mean_data, resample_data, run_multithreaded, run_multiprocessed, \
    ensure_list
from src.HDF_Util import NotFoundInHdfError

ARRAY_LIKE = Union[np.ndarray, List, Tuple]


def set_default_logging():
    # logging.basicConfig(level=logging.INFO, format=f'%(threadName)s %(funcName)s %(lineno)d %(message)s')
    # logging.basicConfig(level=logging.INFO, force=True, format=f'%(levelname)s:%(module)s:%(lineno)d:%(funcName)s:%(message)s')
    root_logger = logging.getLogger()
    root_logger.handlers = []  # Probably a bad thing to be doing...
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        f'%(thread)d:%(process)d:%(levelname)s:%(module)s:%(lineno)d:%(funcName)s:%(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    # root_logger.setLevel(logging.DEBUG)


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
        fp = path + f'_{slugify(name)}' + ext  # slugify ensures filesafe name
        np.savetxt(fp, data)
        logger.info(f'saved [{name}] to [{fp}]')


def power_spectrum(data, meas_freq, normalization=1):
    """
    Computes power spectrum and returns (freq, power spec)
    Args:
        data (): 1D data to calculate spectrum for
        meas_freq (): Frequency of measurement (not sample rate)
        normalization (): Multiplies data before calculating

    Returns:

    """
    freq, power = scipy.signal.periodogram(data * normalization, fs=meas_freq)
    return freq, power


def dac_step_freq(x_array=None, freq=None, dat=None):
    if dat:
        assert all([x_array is None, freq is None])
        x_array = dat.Data.x_array
        freq = dat.Logs.Fastdac.measure_freq

    full_x = abs(x_array[-1] - x_array[0])
    num_x = len(x_array)
    min_step = 20000 / 2 ** 16
    req_step = full_x / num_x
    step_every = min_step / req_step
    step_t = step_every / freq
    step_hz = 1 / step_t
    return step_hz


def data_from_json(filepath: str) -> Dict[str, np.ndarray]:
    with open(filepath, 'r') as f:
        s = f.read()
    js = json.loads(s)
    for k in js:
        js[k] = np.array(js[k])
    return js


def fig_from_json(filepath: str) -> go.Figure:
    with open(filepath, 'r') as f:
        s = f.read()
    fig = go.Figure(json.loads(s))
    return fig



if __name__ == '__main__':
    from src.DatObject.Make_Dat import get_dat, get_dats

