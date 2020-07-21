from src.Scripts.StandardImports import *
import napari
import numpy as np
from dataclasses import dataclass, field
from skimage import measure
from typing import Union, List, Tuple
import napari.layers
from src.Scripts.Jul14_SquareWave import SquareTransitionModel, SquareWave, SquarePlotInfo, TransitionModel
from src.DatObject.Attributes.Transition import i_sense

import logging

logger = logging.getLogger(__name__)


def get_line(image, start, end):
    return measure.profile_line(image, start, end, mode='nearest')



class View(object):
    def __init__(self):
        self.viewer = napari.Viewer()
        self.datas: Union[None, List[napari.layers.image.image.Image]] = None  # List of datas added to viewer
        self.x_infos: Union[None, List[Tuple]] = None  # List of x_infos (start, end)s for profile plots

        # For profiles
        self.plines_layer = None
        self.profile_fig = None
        self._profile_id = None
        self._profiles = None

    def add_data(self, data, x_array=None):
        if self.datas is None:
            self.datas = []
            self.x_infos = []

        scales = list((100 / s for s in data.shape[-2:]))
        if data.ndim > 2:
            scales = [1] * (data.ndim - 2) + scales
        # scales = (1,1)
        # scales[-2] = -scales[-2]  # Want y array increasing upwards not downwards
        # translates = np.zeros(len(scales))
        # translates[-2] = 100

        if x_array is not None:
            self.x_infos.append((x_array[0], x_array[-1]))
        else:
            self.x_infos.append(None)
        self.datas.append(self.viewer.add_image(data, scale=tuple(scales)))  # , translate=tuple(translates)))

    def add_profile(self, data_id=0):
        if self.profile_fig is None:
            self.profile_fig, _ = self.viewer.window.add_docked_figure(area='right', initial_width=500)
            self._profile_id = 0
            self._profiles = []
        else:
            self._profile_id += 1
        data_layer = self.datas[data_id]
        front_data = get_front_data(data_layer)
        self._profiles.append(
            self.profile_fig[self._profile_id, 0].plot(get_line(front_data, [0, 0], [0, front_data.shape[-1]]), marker_size=0))
        self._attach_dims_callback()

    def _attach_dims_callback(self):
        def update_lines(*args):
            data_layer = self.datas[0]  # TODO: Change to be for current self.datas
            front_data = get_front_data(data_layer)
            for profile in self._profiles:
                profile.set_data(get_line(front_data, [0, 0], [0, front_data.shape[-1]]), marker_size=0)
            for ax in self.profile_fig.plot_widgets:
                ax.autoscale()

        self.viewer.dims.events.connect(update_lines)


def get_front_data(data_layer):
    if data_layer.ndim > 2:
        front_data = data_layer.data[data_layer.coordinates[:-2]]
    else:
        front_data = data_layer.data[:]
    return front_data


# @line.mouse_drag_callbacks.append
# def print_coords(shape_layer, event):
#     print(shape_layer.data)
#     yield
#     while event.type == 'mouse_move':
#         print(shape_layer.data)
#         yield



@dataclass
class Data2D:
    x: np.ndarray
    data: np.ndarray
    y: np.ndarray = None

    def __post_init__(self):
        if self.y is None:
            self.y = np.arange(self.data.shape[-2])


if __name__ == '__main__':
    get_ipython().enable_gui('qt')

    dat = get_dat(500)
    # sqw = SquareWave(dat.Logs.Fastdac.measure_freq, -10, 10, 1, 0.25, 500)
    mid = 0
    theta = 0.5
    amp = 0.5
    lin = 0.01
    const = 8

    x = np.linspace(-10, 10, 200)

    amps = np.linspace(0.1, 0.9, num=8)
    thetas = np.linspace(0.5, 2.5, num=21)
    lins = np.linspace(0, 0.02, num=11)


    a, t, l, xs = np.meshgrid(amps, thetas, lins, x, indexing='ij')
    data = i_sense(xs, mid, t, a, l, const)

    # Add repeated y axis to make 2D image easier to interpret
    data_repeated = np.moveaxis(np.repeat([data], 100, axis=0), 0, -2)[:]

    # tmodel = TransitionModel(mid=0, amp=a, theta=t, lin=l, const=8)



    v = View()
    v.add_data(data_repeated)

    v.add_profile()
    #
    # fig = v.viewer.window.add_docked_figure(area='right', initial_width=500)
    # lines = v.viewer.add_shapes(name='Profile lines')
    # lines.add([[0, 0, 0, 50, 5], [0, 0, 0, 50, 95], [*data_repeated.shape[0:-2], 50, 95], [*data_repeated.shape[0:-2], 50, 5]],
    #           edge_width=5, edge_color='red', face_color='blue', opacity=0.5)

