from src.Scripts.StandardImports import *
import numpy as np
import plotly.graph_objs as go
import plotly.offline
import os, sys


fig = go.Figure()
x = np.linspace(0, 10, 100)
fig.add_scatter(x=x, y=np.sin(x), mode='markers',
                marker={'size': 30, 'color': np.linspace(0, 1, 100), 'opacity': 0.6,
                        'colorscale': 'Viridis'})


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


# win = PlotlyViewer(fig, exec=False)
# # Need to import matplotlib.pyplot to turn some interactive mode on.. Not sure why or what it does
#
# fig.add_scatter(x=[5], y=[5])
# fig.add_scatter(x=[3], y=[3])
# win.draw()

import plotly.graph_objects as go
import numpy as np

# Create figure
fig = go.Figure()

# Add traces, one for each slider step
for step in np.arange(0, 5, 0.1):
    fig.add_trace(
        go.Scatter(
            visible=False,
            line=dict(color="#00CED1", width=6),
            name="𝜈 = " + str(step),
            x=np.arange(0, 10, 0.01),
            y=np.sin(step * np.arange(0, 10, 0.01))))

# Make 10th trace visible
fig.data[10].visible = True

# Create and add slider
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=10,
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders
)

win = PlotlyViewer(fig, exec=False)

print('hello')