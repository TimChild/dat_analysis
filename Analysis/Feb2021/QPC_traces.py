"""
Mostly just getting a quick idea of what the QPC traces look like
"""

from src.dat_object.make_dat import get_dat, get_dats, DatHDF
from src.characters import DELTA, THETA, PM
from src.plotting.plotly.dat_plotting import OneD, TwoD
import src.useful_functions as U
from src.analysis_tools.transition import do_transition_only_calc
from src.analysis_tools.general_fitting import calculate_fit

import numpy as np
import lmfit as lm
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
from itertools import product
from typing import Tuple, List, Optional, Union, Dict
from functools import partial
from progressbar import progressbar
from dataclasses import dataclass

from concurrent.futures import ProcessPoolExecutor

pio.renderers.default = 'browser'


datnums = list(range(6791, 6810+1))


if __name__ == '__main__':
    dats = get_dats(datnums)

    plotter = OneD(dats=dats)
    fig = plotter.figure(ylabel='Current /nA', title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: CS QPC Traces')
    for dat in dats:
        trace = plotter.trace(data=dat.Data.get_data('i_sense'), x=dat.Data.get_data('x'), mode='lines', name=f'Dat{dat.datnum}: ESC={dat.Logs.fds["ESC"]:.0f}mV')
        fig.add_trace(trace)

    fig.show()