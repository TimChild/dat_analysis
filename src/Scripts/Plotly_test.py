import plotly.express as px
import plotly
import plotly.io as pio
import nbformat
import plotly.graph_objects as go
from plotly.offline import iplot, iplot_mpl

# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('qt5agg')


import os, sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QUrl
from PyQt5 import QtWebEngineWidgets
# from PyQt5.QtCore import QEventLoop


class PlotlyViewer(QtWebEngineWidgets.QWebEngineView):
    def __init__(self, fig, exec=True):
        # Create a QApplication instance or use the existing one if it exists
        self.app = QApplication.instance() if QApplication.instance() else QApplication(sys.argv)

        super().__init__()

        self.file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp.html"))
        plotly.offline.plot(fig, filename=self.file_path, auto_open=False)
        self.load(QUrl.fromLocalFile(self.file_path))
        self.setWindowTitle("Plotly Viewer")
        self.show()

        if exec:
            self.app.exec_()

    def closeEvent(self, event):
        os.remove(self.file_path)



import numpy as np
import threading
pio.renderers.default = 'nteract'

x = np.linspace(0, 2*np.pi, 100)
z = np.sin(x)

fig = px.line(x=x, y=z, labels=dict(x='xlabel', y='ylabel'))
fig.add_scatter(x=[4], y=[0.8])
iplot(fig)
# w = PlotlyViewer(fig, exec=True)

# x = np.linspace(0, 10, 100)
# y = np.random.random(100)*5+x
#
# import src.CoreUtil as CU
# CU.save_to_txt([np.array([x, y])], ['linear_noise'], 'temp.txt')
# np.savetxt('temp.txt', (x, y), fmt='%.4f', delimiter=', ', newline='\n', header='x, y')
#
# import pandas as pd
# a = pd.read_csv('temp.txt', header=1)
#
# df = pd.DataFrame(data=np.array([x, y]).T, columns=['x', 'y'])
#
# df.to_excel('temp.xlsx', index=False)

#
# scatter = go.Scatter(x=[9], y=[1], showlegend=True, name='')
#
# fig.add_scatter(x= [0.001000], y= [0.546399], error_x=dict(type = 'percent', value= 5), error_y=dict(type = 'percent', value = 20), showlegend=True, name='Scatter')
# ax: plt.Axes
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1)
#
# ax.cla()
# ax.scatter([0.001000], [0.546399], label='Scatter', marker='x')  # Just adds point
# ax.errorbar([0.001000], [0.546399], xerr=0.0005, yerr=0.1, label='Scatter', marker='x', capsize=10)  # Adds point and errorbar
# ax.legend()  # Updates the legend, or adds a legend if not already present
#
