"""
Collection of functions for saving/loading/transforming data between different types including extracting from plotly
figures
"""
import json
import os
from dataclasses import dataclass
from typing import Optional, List, Union, Dict
import logging

import numpy as np
from igorwriter import IgorWave
from plotly import graph_objs as go
from scipy import io as sio
from slugify import slugify

logger = logging.getLogger(__name__)


######## Saving/Loading data ############
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
        # Note: atol and rtol of np.isclose may need further fine-tuning to properly detect what is linear/non-linear
        if arr.shape[-1] > 1 and not np.all(np.isclose(np.diff(arr), np.diff(arr)[0], rtol=0.0005)):
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


def _save_to_checks(datas, names, file_path, fp_ext=None):
    assert type(datas) == list
    assert type(names) == list
    base, tail = os.path.split(file_path)
    if base != "":
        assert os.path.isdir(base)  # Check points to existing folder
    if fp_ext is not None:
        if tail[-(len(fp_ext)):] != fp_ext:
            tail += fp_ext  # add extension if necessary
            logger.warning(
                f'added "{fp_ext}" to end of file_path provided to make [{file_path}]'
            )
            file_path = os.path.join(base, tail)
    return file_path


def data_from_json(filepath: str) -> Dict[str, np.ndarray]:
    with open(filepath, "r") as f:
        s = f.read()
    js = json.loads(s)
    for k in js:
        js[k] = np.array(js[k], dtype=np.float32)
    return js


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


######### Extracting/Saving/Loading Data from Figures ############
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
