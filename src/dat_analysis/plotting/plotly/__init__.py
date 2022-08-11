import plotly.io as pio
from .plotly_util import make_slider_figure, add_vertical, add_horizontal
from . import hover_info
from .dat_plotting import OneD, TwoD
from ...core_util import Data1D, Data2D

pio.renderers.default = 'browser'
