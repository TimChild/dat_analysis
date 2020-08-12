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