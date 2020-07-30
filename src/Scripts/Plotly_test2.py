import numpy as np
import plotly.graph_objs as go
import plotly.offline
import matplotlib.pyplot as plt

fig = go.Figure()
fig.add_scatter(x=np.random.rand(100), y=np.random.rand(100), mode='markers',
                marker={'size': 30, 'color': np.random.rand(100), 'opacity': 0.6,
                        'colorscale': 'Viridis'})

import os, sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QUrl
from PyQt5 import QtWebEngineWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import QCoreApplication


class PlotlyViewer(QtWebEngineWidgets.QWebEngineView, QMainWindow):
    def __init__(self, fig, exec=True):
        # Create a QApplication instance or use the existing one if it exists
        self.fig = fig

        self.app = QApplication.instance() if QApplication.instance() else QApplication(sys.argv)
        super().__init__()

        # self.timer = QTimer()
        # self.timer.timeout.connect(lambda: None)
        # self.timer.start(100)

        self.file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp.html"))
        plotly.offline.plot(self.fig, filename=self.file_path, auto_open=False)
        self.load(QUrl.fromLocalFile(self.file_path))
        self.setWindowTitle("Plotly Viewer")
        self.show()
        # QTimer.singleShot(0, self.draw)

        if exec:
            self.app.exec_()

    def closeEvent(self, event):
        os.remove(self.file_path)

    def draw(self):
        plotly.offline.plot(self.fig, filename=self.file_path, auto_open=False)
        self.load(QUrl.fromLocalFile(self.file_path))
        # self.update()
        # QCoreApplication.processEvents()


win = PlotlyViewer(fig, exec=False)
# Need to import matplotlib.pyplot to turn some interactive mode on.. Not sure why or what it does


# import matplotlib as mpl
# mpl.use('qt5agg')
# fig, ax = plt.subplots(1)