import os
import sys
import plotly.offline
from PyQt5 import QtWebEngineWidgets
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QMainWindow, QApplication


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
            self.app.exec_()

    def closeEvent(self, event):
        os.remove(self.file_path)

    def draw(self):
        plotly.offline.plot(self.fig, filename=self.file_path, auto_open=False)
        self.load(QUrl.fromLocalFile(self.file_path))


import plotly.graph_objects as go
import numpy as np


def get_figure(datas, xs, ys=None, ids=None, titles=None, fixed_axes=False, xlabel='', ylabel=''):
    """
    Get plotly figure with data in layers and a slider to change between them
    Args:
        xs ():
        ys ():
        datas ():
        ids ():
        titles ():
        fixed_axes ():

    Returns:
        (go.Figure): Plotly figure
    """
    assert type(datas) == list

    ids = ids if ids else range(len(datas))
    titles = titles if titles else range(len(datas))

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
                ))
            fig.data[0].visible = True
    else:
        for data, x in zip(datas, xs):
            data = np.atleast_2d(data)
            fig.add_trace(go.Scatter(x=x, y=data, visible=False, mode='lines+markers'))
            fig.data[0].visible = True

    steps = []
    for i, (id, title) in enumerate(zip(ids, titles)):
        step = dict(
            method='update',
            args=[{'visible': [False] * len(fig.data)},
                  {'title': f'{title}'}],
            label=f'{id}'
        )
        step['args'][0]['visible'][i] = True  # Toggle i'th trace to visible
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={'prefix': ''},
        pad={'t': 50},
        steps=steps
    )]

    fig.update_layout(sliders=sliders)
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title=ylabel)
    return fig
