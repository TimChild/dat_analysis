import os
import sys
import plotly.offline
from PyQt5 import QtWebEngineWidgets
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QMainWindow, QApplication


# qt5 backend for Ipython AFTER importing QtWebEngineWidgets which has to be imported first
try:
    from IPython import get_ipython
    ip = get_ipython()  # This function exists at runtime if in Ipython kernel
    ip.enable_gui('qt5')
except:
    print('\n\n\nERROR when trying to enable qt5 backend support of IPython\n\n\n')
    pass


class PlotlyViewer(QtWebEngineWidgets.QWebEngineView, QMainWindow):
    def __init__(self, fig, exec=False):
        # Create a QApplication instance or use the existing one if it exists
        self.fig = fig

        self.app = QApplication.instance() if QApplication.instance() else QApplication(sys.argv)
        super().__init__()

        self.file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp.html"))
        plotly.offline.plot(self.fig, filename=self.file_path, auto_open=False)
        self.load(QUrl.fromLocalFile(self.file_path))
        self.setWindowTitle("Plotly Viewer")
        self.show()

        if exec:
            self.appc_()

    def closeEvent(self, event):
        os.remove(self.file_path)

    def draw(self):
        plotly.offline.plot(self.fig, filename=self.file_path, auto_open=False)
        self.load(QUrl.fromLocalFile(self.file_path))


import plotly.graph_objects as go
import numpy as np


def get_figure(datas, xs, ys=None, ids=None, titles=None, labels=None, xlabel='', ylabel='',
               plot_kwargs=None):
    """
    Get plotly figure with data in layers and a slider to change between them
    Args:
        ylabel (str): Y label
        xlabel (str): X label
        plot_kwargs (dict): Args to pass into either go.Heatmap or go.Scatter depending on 2D or 1D
        xs (Union(np.ndarray, list)): x_array for each slider position or single array to use for all
        ys (Union(np.ndarray, list)): y_array for each slider position or single array to use for all
        datas (list): list of datas for each slider position (i.e. can be list of list of arrays or just list of arrays)
        ids (list): ID for slider
        titles (list): Title at slider position
        labels (List[str]): Label for each trace per step (i.e. four lines per step, four labels)

    Returns:
        (go.Figure): Plotly figure
    """
    if plot_kwargs is None:
        plot_kwargs = {}
    assert type(datas) == list

    if ys is None:
        datas_per_step = np.atleast_2d(datas[0]).shape[0]
    else:
        datas_per_step = 1

    ids = ids if ids else range(len(datas))
    titles = titles if titles else range(len(datas))
    labels = labels if labels is not None else [None]*datas_per_step

    if isinstance(xs, np.ndarray):
        xs = [xs] * len(datas)

    if ys is not None:
        if isinstance(ys, np.ndarray):
            ys = [ys] * len(datas)

    fig = go.Figure()
    if ys is not None:
        for data, x, y in zip(datas, xs, ys):
            fig.add_trace(
                go.Heatmap(
                    visible=False,
                    x=x,
                    y=y,
                    z=data,
                    **plot_kwargs
                ))
            fig.data[0].visible = True
    else:
        for data, x in zip(datas, xs):
            x = np.atleast_2d(x)
            data = np.atleast_2d(data)
            if x.shape[0] == 1 and data.shape[0] != 1:
                x = np.tile(x, (data.shape[0], 1))
            for x, d, label in zip(x, data, labels):
                plot_kwargs['mode'] = plot_kwargs.pop('mode', 'lines')
                fig.add_trace(go.Scatter(x=x, y=d, visible=False, name=label, **plot_kwargs))

        for i in range(datas_per_step):
            fig.data[i].visible = True

    steps = []
    for i, (id, title) in enumerate(zip(ids, titles)):
        step = dict(
            method='update',
            args=[{'visible': [False] * len(fig.data)},
                  {'title': f'{title}'},
                  plot_kwargs],
            label=f'{id}'
        )
        for j in range(datas_per_step):
            step['args'][0]['visible'][i*datas_per_step+j] = True  # Toggle i'th trace to visible
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={'prefix': ''},
        pad={'t': 50},
        steps=steps
    )]
    fig.update_layout(sliders=sliders, title=titles[0],
                      xaxis_title=xlabel, yaxis_title=ylabel)
    return fig


def fig_setup(fig: go.Figure, title=None, x_label=None, y_label=None, legend_title=None):
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, legend_title=legend_title)


def add_vertical(fig, x):
    fig.update_layout(shapes=[dict(type='line', yref='paper', y0=0, y1=1, xref='x', x0=x, x1=x)])


def add_horizontal(fig, y):
    fig.update_layout(shapes=[dict(type='line', yref='y', y0=y, y1=y, xref='paper', x0=0, x1=1)])


# TODO: Make classes and functions which return a plotly figure with numbered subplots
# TODO: in a way that I can use the 'axs' to address the plots.
# TODO: Will that actually be easier in the future? Or should I just hold onto traces more carefully?
# def get_fig(num=None, rows=None, cols=None):
#     if all([v is None for v in [num, rows, cols]]):
#         fig = go.Figure()
#         ax = None
#         return


# class Ax:
#     def __init__(self, fig, num):
#         self.fig = fig
#         self.num = num
#
#     @property
#     def row(self):
#         return
#
#
# def add_line(fig, x, z, x_label=None, y_label=None, label=None, mode='lines', **kwargs) -> go.Scatter:
#     trace = go.Scatter(mode=mode, x=x, y=z, name=label)
#
#
# def plot_1d(x, z, x_label=None, y_label=None, title=None, fig=None):
#     if fig is None:
#         fig = go.Figure()
#     trace = go.Scatter(x=x, y=z, labels={'x': x_label, 'y': y_label})


if __name__ == '__main__':
    num = 10
    xs = [np.tile(np.linspace(0, 10, 100), (5,1)) for i in range(num)]
    datas = [np.sin(x) for x in xs]

    fig = get_figure(datas, xs, ids=None, titles=None, labels=['1', '2', '3', '4', '5'], xlabel='xlabel',
                     ylabel='ylabel')
    v = PlotlyViewer(fig)
