"""
Using varying fridge temperature at various Gamma to see if we can detect lever arm change.
I.e. go from gamma broadened at 50mK to thermally broadened at 500mK. Should be able to get a good measure of lever arm
when thermally broadened, and the idea is that the lever arm won't change with temperature since gamma will be staying
fixed.

"""
from src.DatObject.Make_Dat import get_dat, get_dats, DatHDF
from src.Dash.DatPlotting import OneD, TwoD
import src.UsefulFunctions as U

import numpy as np
import lmfit as lm
import plotly.graph_objects as go
import plotly.io as pio
from typing import Tuple

pio.renderers.default = 'browser'

# Series of dats taken at 50, 500, 400, 300, 200, 100, 50mK, at each temp data taken at 6 places from weakly coupled
# to gamma broadened, At each position, there are three scans. 1 with no Heating bias, then with + - ~100% heating
# bias.
NO_HEAT_DATS = list(range(5371, 5946+1, 3))
POS_HEAT_DATS = list(range(5372, 5946+1, 3))
NEG_HEAT_DATS = list(range(5373, 5946+1, 3))


if __name__ == '__main__':
    dats = get_dats(NO_HEAT_DATS + POS_HEAT_DATS + NEG_HEAT_DATS)
