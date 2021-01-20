from __future__ import annotations
import plotly
import plotly.graph_objs as go
import numpy as np

import src.UsefulFunctions as UF
from src.DatObject.Make_Dat import DatHandler
from src.DataStandardize.ExpSpecific import Sep20
from src.Dash.DatPlotting import OneD, TwoD


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.DatObject import DatHDF

get_dat = DatHandler().get_dat


class SquareEntropyPlotter:

    def __init__(self, dat: DatHDF):
        self.dat: DatHDF = dat
        self.one_plotter: OneD = OneD(dat)

    def plot_raw(self) -> go.Figure:
        z = self.dat.Data.i_sense
        z = z[0]
        # z = np.mean(z, axis=0)

        fig = self.one_plotter.plot(z, mode='lines', title='Row 0 of I_sense')
        return fig

    def plot_cycled(self) -> go.Figure:
        z = self.dat.SquareEntropy.default_Output.cycled
        z = z[0]
        # z = np.mean(z, axis=0)

        fig = self.one_plotter.plot(z, mode='lines', title='Row 0 of cycled')

        return fig

    def plot_avg(self) -> go.Figure:
        z = self.dat.SquareEntropy.default_Output.averaged
        return self.one_plotter.plot(z, mode='lines')


if __name__ == '__main__':

    chosen_dats = {
        8797: '100mK Similar',
        8710: '50mK Similar',
        8808: '100mK Different',
        8721: '50mK Different'
    }

    datnums = list(chosen_dats.keys())

    dats = [get_dat(num, datname='s2e', exp2hdf=Sep20.SepExp2HDF) for num in datnums]

    plotters = [SquareEntropyPlotter(dat) for dat in dats]
    for plotter in plotters[0:1]:
        plotter.plot_raw().show(renderer='browser')
        plotter.plot_cycled().show(renderer='browser')
        plotter.plot_avg().show(renderer='browser')

