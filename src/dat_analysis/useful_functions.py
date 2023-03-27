"""
Simple general functions that are useful in analysis but NOT used in any other modules

WARNING: IMPORTING FROM HERE IN ANY OTHER MODULE IS LIKELY TO CAUSE CIRCULAR IMPORT ISSUES

NOTE: If any of the functions here become useful in any other module in dat_analysis, the function should be moved to
core_util.py and then imported here for backwards compatability and ease of access"""

import os
import logging
from dataclasses import dataclass
from deprecation import deprecated

import numpy as np
import scipy.signal
from scipy import io as sio
from slugify import slugify
from typing import List, Tuple, Iterable, Union, Dict, Optional
import plotly.graph_objs as go
import json
from igorwriter import IgorWave
import io
from pathlib import Path

logger = logging.getLogger(__name__)

from .core_util import (
    get_data_index,
    get_matching_x,
    edit_params,
    sig_fig,
    decimate,
    FIR_filter,
    get_sweeprate,
    bin_data,
    get_bin_size,
    mean_data,
    center_data,
    resample_data,
    ensure_list,
    order_list,
    my_round,
    Data1D,
    Data2D,
)
from .hdf_util import NotFoundInHdfError

ARRAY_LIKE = Union[np.ndarray, List, Tuple]


def set_default_logging(level_override=None):
    # logging.basicConfig(level=logging.INFO, format=f'%(threadName)s %(funcName)s %(lineno)d %(message)s')
    # logging.basicConfig(level=logging.INFO, force=True, format=f'%(levelname)s:%(module)s:%(lineno)d:%(funcName)s:%(message)s')
    root_logger = logging.getLogger()
    root_logger.handlers = []  # Probably a bad thing to be doing...
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        f"%(thread)d:%(process)d:%(levelname)s:%(module)s:%(lineno)d:%(funcName)s:%(message)s"
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    if level_override:
        root_logger.setLevel(level_override)


def _save_to_checks(datas, names, file_path, fp_ext=None):
    assert type(datas) == list
    assert type(names) == list
    base, tail = os.path.split(file_path)
    if base != "":
        assert os.path.isdir(base)  # Check points to existing folder
    if fp_ext is not None:
        if tail[-(len(fp_ext)) :] != fp_ext:
            tail += fp_ext  # add extension if necessary
            logger.warning(
                f'added "{fp_ext}" to end of file_path provided to make [{file_path}]'
            )
            file_path = os.path.join(base, tail)
    return file_path


def save_to_mat(datas: list[np.ndarray], names: list[str], file_path: str):
    file_path = _save_to_checks(datas, names, file_path, fp_ext=".mat")
    mat_data = dict(zip(names, datas))
    sio.savemat(file_path, mat_data)
    logger.info(f"saved [{names}] to [{file_path}]")


def save_to_txt(datas: list[np.ndarray], names: list[str], file_path: str):
    file_path = _save_to_checks(datas, names, file_path, fp_ext=".txt")
    for data, name in zip(datas, names):
        path, ext = os.path.splitext(file_path)
        fp = path + f"_{slugify(name)}" + ext  # slugify ensures filesafe name
        np.savetxt(fp, data)
        logger.info(f"saved [{name}] to [{fp}]")


def fig_to_igor_itx(f: go.Figure, filepath: str):
    """Save data from a figure to Igor .itx file
    Note: Adds extra info to waves than `save_to_itx` has alone (axis labels)
    """
    d = data_from_plotly_fig(f)

    datas, names = list(), list()
    axes = {"x": [], "y": []}
    for k in d:
        if not k.endswith("_x") and not k.endswith("_y"):
            datas.append(np.asanyarray(d[k]))
            names.append(k)

            for dim in ["x", "y"]:
                if f"{k}_{dim}" in d:
                    axes[dim].append(np.asanyarray(d[f"{k}_{dim}"]))
                else:
                    axes[dim].append(None)

    x_units = f.layout.xaxis.title.text
    x_units = x_units if x_units else "not set"

    # TODO: Implement saving the y-axis label as the datascale_units for 1D traces (needs to be implemented in save_to_igor_itx)
    y_units = f.layout.yaxis.title.text
    y_units = y_units if y_units else "not set"

    save_to_igor_itx(
        file_path=filepath,
        xs=axes["x"],
        datas=datas,
        names=names,
        ys=axes["y"],
        x_labels=x_units,
        y_labels=y_units,
    )


@deprecated(deprecated_in="3.2.0", details="Part of the Data class now")
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
    """Calculates the frequency of the true DAC steps (i.e. 20V/2^16 is smallest DAC step size)"""
    if dat:
        assert all([x_array is None, freq is None])
        x_array = dat.Data.x_array
        freq = dat.Logs.Fastdac.measure_freq

    full_x = abs(x_array[-1] - x_array[0])
    num_x = len(x_array)
    min_step = 20000 / 2**16
    req_step = full_x / num_x
    step_every = min_step / req_step
    step_t = step_every / freq
    step_hz = 1 / step_t
    return step_hz


@dataclass
class IgorSaveInfo:
    x: np.ndarray
    data: np.ndarray
    name: str
    x_label: str
    y_label: str
    y: Optional[np.ndarray] = None


def save_multiple_save_info_to_itx(file_path: str, save_infos: List[IgorSaveInfo]):
    """
    Save multiple save infos to a single .itx file
    Args:
        file_path ():
        save_infos ():

    Returns:

    """
    save_to_igor_itx(
        file_path=file_path,
        xs=[s.x for s in save_infos],
        ys=[s.y for s in save_infos],
        datas=[s.data for s in save_infos],
        names=[s.name for s in save_infos],
        x_labels=[s.x_label for s in save_infos],
        y_labels=[s.y_label for s in save_infos],
    )


def save_to_igor_itx(
    file_path: str,
    xs: List[np.ndarray],
    datas: List[np.ndarray],
    names: List[str],
    ys: Optional[List[np.ndarray]] = None,
    x_labels: Optional[Union[str, List[str]]] = None,
    y_labels: Optional[Union[str, List[str]]] = None,
):
    """Save data to a .itx file which can be dropped into Igor

    Args:
        file_path: filepath to save .itx
        xs: x_arrays for each data
        datas: data arrays
        names: name for each data
        ys: y_arrays for each data
        x_labels: units for x-axis
        y_labels: units for y-axis (2D only)
    """
    # TODO: Implement saving datascale_units for 1D data

    def check_axis_linear(
        arr: np.ndarray, axis: str, name: str, current_waves: list
    ) -> bool:
        if arr.shape[-1] > 1 and not np.all(np.isclose(np.diff(arr), np.diff(arr)[0])):
            logger.warning(
                f"{file_path}: Igor doesn't support a non-linear {axis}-axis. Saving as separate wave"
            )
            axis_wave = IgorWave(arr, name=name + f"_{axis}")
            current_waves.append(axis_wave)
            return False
        else:
            return True

    if x_labels is None or isinstance(x_labels, str):
        x_labels = [x_labels] * len(datas)
    if y_labels is None or isinstance(y_labels, str):
        y_labels = [y_labels] * len(datas)
    if ys is None:
        ys = [None] * len(datas)
    assert all([len(datas) == len(list_) for list_ in [xs, names, x_labels, y_labels]])

    waves = []
    for x, y, data, name, x_label, y_label in zip(
        xs, ys, datas, names, x_labels, y_labels
    ):
        wave = IgorWave(data, name=name)
        # wave.set_datascale(units='<1D data units here>')  # TODO: Implement saving datascale_units for 1D data
        if x is not None:
            if check_axis_linear(x, "x", name, waves):
                wave.set_dimscale("x", x[0], np.mean(np.diff(x)), units=x_label)
        if y is not None:
            if check_axis_linear(y, "y", name, waves):
                wave.set_dimscale("y", y[0], np.mean(np.diff(y)), units=y_label)
        elif y_label is not None:
            wave.set_datascale(y_label)
        waves.append(wave)

    with open(file_path, "w") as fp:
        for wave in waves:
            wave.save_itx(
                fp, image=True
            )  # Image = True hopefully makes np and igor match in x/y


def data_from_plotly_fig(f: go.Figure) -> Dict[str, np.ndarray]:
    all_data = {}
    for i, d in enumerate(f.data):
        name = getattr(d, "name", None)
        if name is None:
            name = f"data{i}"
        elif name in all_data.keys():
            name = name + f"_{i}"
        if "z" in d:  # Then it is 2D
            all_data[name] = np.array(getattr(d, "z")).astype(np.float32)
            all_data[name + "_y"] = np.array(getattr(d, "y")).astype(np.float32)
        else:
            all_data[name] = np.array(getattr(d, "y")).astype(np.float32)
        all_data[name + "_x"] = np.array(getattr(d, "x")).astype(np.float32)
    return all_data


def data_from_json(filepath: str) -> Dict[str, np.ndarray]:
    with open(filepath, "r") as f:
        s = f.read()
    js = json.loads(s)
    for k in js:
        js[k] = np.array(js[k], dtype=np.float32)
    return js


def fig_from_json(filepath: str) -> go.Figure:
    with open(filepath, "r") as f:
        s = f.read()
    fig = go.Figure(json.loads(s))
    return fig


def fig_to_data_json(fig: go.Figure, filepath: str) -> bool:
    """Saves all data in figure to json file"""
    data = data_from_plotly_fig(fig)
    filepath = (
        filepath if os.path.splitext(filepath)[-1] == ".json" else f"{filepath}.json"
    )
    return data_dict_to_json(data, filepath)


def data_dict_to_json(data_dict: dict, filepath: str) -> bool:
    """Saves dict of arrays to json"""
    with open(filepath, "w+") as f:
        json.dump(data_dict, f, default=lambda arr: arr.tolist())
    return True


def data_to_json(datas: List[np.ndarray], names: List[str], filepath: str) -> dict:
    """
    Saves list of data arrays to a json file with names.
    Args:
        datas ():
        names ():
        filepath ():

    Returns:
        dict: Dict of data that was saved to json file

    """
    data_dict = {name: data for name, data in zip(names, datas)}
    data_dict_to_json(data_dict, filepath)
    return data_dict
