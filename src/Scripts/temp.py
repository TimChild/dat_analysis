from src.Scripts.StandardImports import *

from scipy.interpolate import interp1d

from src.Plotting.Plotly.PlotlyUtil import PlotlyViewer
import plotly.graph_objects as go
import plotly.io as pio


if __name__ == '__main__':
    dats = get_dats(range(69, 73+1))