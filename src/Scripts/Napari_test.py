import napari
import numpy as np

from skimage import measure
from typing import Union, List, Tuple
import napari.layers
import logging

logger = logging.getLogger(__name__)


def get_line(image, start, end):
    if np.any([[np.isnan(v) for v in p] for p in [start, end]]):
        ret = np.linspace(0, 10, 100)
    else:
        ret = measure.profile_line(image, start, end, mode='nearest')
    return ret


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

        scales = list((100/s for s in data.shape[-2:]))
        if data.ndim > 2:
            scales = [1]*(data.ndim-2)+scales
        # scales = (1,1)
        # scales[-2] = -scales[-2]  # Want y array increasing upwards not downwards
        # translates = np.zeros(len(scales))
        # translates[-2] = 100

        if x_array is not None:
            self.x_infos.append((x_array[0], x_array[-1]))
        else:
            self.x_infos.append(None)
        self.datas.append(self.viewer.add_image(data, scale=tuple(scales))) #, translate=tuple(translates)))

    def add_profile(self, data_id=0):
        if self.profile_fig is None:
            self.profile_fig, _ = self.viewer.window.add_docked_figure(area='right', initial_width=500)
            self._profile_id = 0
            self.plines_layer = self.viewer.add_shapes(name='Profile lines')
            self._profiles = []
            self._attach_callback()

        else:
            self._profile_id += 1
        data_layer = self.datas[data_id]
        data = data_layer.data[data_layer.coordinates[:-2]]  # Get 2D data that's at front
        coords = [[0, 10], [0, data.shape[-1]-10]]  # TODO: Needs to change when get real scaling

        self.plines_layer.add(coords, shape_type='line', edge_width=0.3, edge_color='red', face_color='red')
        self.plines_layer.scale = self.datas[data_id].scale[-2:]
        self.plines_layer.mode = 'select'
        self.plines_layer._highlight_width = 0.03
        self._profiles.append(
            self.profile_fig[self._profile_id, 0].plot(get_line(data, *self.plines_layer.data[-1]), marker_size=0))


    def _attach_callback(self):
        def profile_lines(image, shape_layer):
            if shape_layer == self.plines_layer:
                for i, (profile, pline) in enumerate(zip(self._profiles, self.plines_layer.data)):
                    profile.set_data(get_line(image, *pline), marker_size=0)
                    self.profile_fig[i, 0].autoscale()
            else:
                logger.warning('Trying to call profile lines with shape_layer which is not plines_layer')

        def update_lines(*args):
            data_layer = self.datas[0]  # TODO: Change to be for current self.datas
            front_data = get_front_data(data_layer)
            profile_lines(front_data, self.plines_layer)

        @self.plines_layer.mouse_drag_callbacks.append
        def profile_lines_drag(plines_layer, event):
            data_layer = self.datas[0]  # TODO: Change to be for current self.datas
            front_data = get_front_data(data_layer)
            profile_lines(front_data, plines_layer)
            yield
            while event.type == 'mouse_move':
                print(plines_layer.data)
                profile_lines(front_data, plines_layer)
                yield

        self.viewer.dims.events.connect(update_lines)


def get_front_data(data_layer):
    if data_layer.ndim > 2:
        front_data = data_layer.data[data_layer.coordinates[-3]]
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


if __name__ == '__main__':
    rand_data = np.random.random((200, 100))
    with napari.gui_qt():
        v = View()
        v.add_data(rand_data)
    # v.add_profile()

