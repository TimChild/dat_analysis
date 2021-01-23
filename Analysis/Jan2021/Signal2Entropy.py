from __future__ import annotations
import plotly
import plotly.graph_objs as go
import numpy as np

import src.UsefulFunctions as UF
from src.DatObject.Make_Dat import DatHandler
from src.DataStandardize.ExpSpecific import Sep20
from src.Dash.DatPlotting import OneD, TwoD
from src.AnalysisTools import DCbias

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

        fig = self.one_plotter.plot(z, mode='lines', title=f'Dat{self.dat.datnum}: Row 0 of I_sense')
        return fig

    def plot_cycled(self) -> go.Figure:
        z = self.dat.SquareEntropy.default_Output.cycled
        x = self.dat.SquareEntropy.x
        z = z[0]

        fig = self.one_plotter.figure(title=f'Dat{self.dat.datnum}: Row 0 of cycled')
        for row, label in zip(z, ['v0_0', 'vP', 'v0_1', 'vM']):
            fig.add_trace(self.one_plotter.trace(row, name=label, x=x, mode='lines'))

        return fig

    def plot_avg(self) -> go.Figure:
        z = self.dat.SquareEntropy.default_Output.averaged
        x = self.dat.SquareEntropy.x
        fig = self.one_plotter.figure(title=f'Dat{self.dat.datnum}: Centered and Averaged I_sense')

        for row, label in zip(z, ['v0_0', 'vP', 'v0_1', 'vM']):
            fig.add_trace(self.one_plotter.trace(row, name=label, x=x, mode='lines'))
        return fig

    def plot_entropy_signal(self) -> go.Figure:
        z = self.dat.SquareEntropy.avg_entropy_signal
        x = self.dat.SquareEntropy.x

        fit_info = self.dat.Entropy.avg_fit
        fit_x = self.dat.Entropy.avg_x
        fit = fit_info.eval_fit(fit_x)

        fig = self.one_plotter.figure(title=f'Dat{self.dat.datnum}: Average Entropy Signal')

        fig.add_trace(self.one_plotter.trace(data=z, x=x, mode='lines', name='Entropy Signal'))
        fig.add_trace(self.one_plotter.trace(data=fit, x=fit_x, mode='lines', name='Fit'))

        self.one_plotter.add_textbox(fig, text=f'Fit Values:\n'
                                               f'dS={fit_info.best_values.dS:.3f}',
                                     position='TR')
        return fig

    def plot_integrated_entropy(self) -> go.Figure:
        z = self.dat.Entropy.integrated_entropy
        x = self.dat.Entropy.avg_x

        fig = self.one_plotter.figure(title=f'Dat{self.dat.datnum}: Average Integrated Entropy')
        fig.add_trace(self.one_plotter.trace(data=z, x=x, mode='lines'))
        return fig


if __name__ == '__main__':

    chosen_dats = {
        8797: '100mK Similar',
        8710: '50mK Similar',
        8808: '100mK Different',
        8721: '50mK Different'
    }

    dc_bias_dats = {
        100: DatHandler().get_dats((4284, 4295), datname='s2e', exp2hdf=Sep20.SepExp2HDF),
        50: DatHandler().get_dats((8593, 8599), datname='s2e', exp2hdf=Sep20.SepExp2HDF),
    }

    dc_bias_infos = {k: DCbias.DCbiasInfo.from_dats(dc_bias_dats[k], bias_key=bias_key, force_centered=False)
                     for k, bias_key in zip(dc_bias_dats, ['R2T(10M)', 'R2T/0.001'])}

    datnums = list(chosen_dats.keys())

    dats = [get_dat(num, datname='s2e', exp2hdf=Sep20.SepExp2HDF, overwrite=False) for num in datnums]

    for dat in dats:
        if dat.Entropy._integration_info_exists('s2e') is False:
            if dat.datnum in [8797, 8808]:
                temp = 100
            elif dat.datnum in [8710, 8721]:
                temp = 50
            else:
                raise NotImplementedError(f'Dont know temp of dat{dat.datnum}')
            dcinfo = dc_bias_infos[temp]
            dat.Entropy.set_integration_info(dc_info=dcinfo, name='s2e')


    plotters = [SquareEntropyPlotter(dat) for dat in dats]
    for plotter in plotters[0:1]:
        # plotter.plot_raw().show(renderer='browser')
        # plotter.plot_cycled().show(renderer='browser')
        # plotter.plot_avg().show(renderer='browser')
        plotter.plot_entropy_signal().show(renderer='browser')
        plotter.plot_integrated_entropy().show(renderer='browser')





