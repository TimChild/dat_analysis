import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.signal import savgol_filter
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Dict, Callable, Optional
from dataclasses import dataclass

from FinalFigures.Gamma.plots import dndt_2d
from src.Dash.DatPlotting import OneD, TwoD
from Analysis.Feb2021.NRG_comparison import NRG_func_generator, NRGData
import src.UsefulFunctions as U

p1d = OneD(dat=None)
p2d = TwoD(dat=None)


@dataclass
class Data2D:
    x: np.ndarray
    y: np.ndarray
    data: np.ndarray


@dataclass
class Data1D:
    x: np.ndarray
    data: np.ndarray


class Nrg2D:
    temperature = 50  # in sweepgate mV so ~100mK
    gs = np.linspace(temperature / 10, temperature * 30, 100)
    g_over_ts = gs / temperature
    ylabel = 'G/T'

    def __init__(self, which_data: str, which_x: str):
        self.which_data = which_data
        self.which_x = which_x

    def _get_title(self):
        if self.which_x == 'sweepgate':
            x_part = 'Sweep Gate'
        elif self.which_x == 'occupation':
            x_part = 'Occupation'
        else:
            raise NotImplementedError
        if self.which_data == 'dndt':
            data_part = 'dN/dT'
        elif self.which_data == 'occupation':
            data_part = 'Occupation'
        else:
            raise NotImplementedError
        return f'NRG {data_part} vs {x_part}'

    def _get_x(self):
        if self.which_x == 'sweepgate':
            return np.linspace(-100, 100, 201)
        elif self.which_x == 'occupation':
            raise NotImplementedError

    def _get_x_label(self) -> str:
        if self.which_x == 'sweepgate':
            return "Sweep Gate (mV)"
        elif self.which_x == 'occupation':
            raise NotImplementedError

    def _data(self) -> Data2D:
        """Generate the 2D data for NRG"""
        nrg_dndt_func = NRG_func_generator(self.which_data)
        x = self._get_x()
        data = np.array([nrg_dndt_func(x, 0, g, self.temperature)
                         for g in self.gs])
        data2d = Data2D(x=x, y=self.g_over_ts, data=data)
        return data2d

    def plot(self, data2d: Optional[Data2D] = None) -> go.Figure:
        if data2d is None:
            data2d = self._data()
        fig = p2d.plot(x=data2d.x, y=data2d.y, data=data2d.data,
                       xlabel=self._get_x_label(), ylabel=self.ylabel, title=self._get_title())
        return fig

    def run(self) -> go.Figure:
        data2d = self._data()
        fig = self.plot(data2d)
        # TOOD: Save data here
        # U.save_to_igor_itx()
        return fig


class NrgOccVsGate2D:


if __name__ == '__main__':
    from src.DatObject.Make_Dat import get_dats, get_dat


    # NRG dN/dT vs sweep gate (fixed T varying G)
    Nrg2D(which_data='dndt', which_x='sweepgate').run().show()

    # NRG Occupation vs sweep gate (fixed T varying G)
    Nrg2D(which_data='occupation', which_x='sweepgate').run().show()





    ##################

    nrg = NRGData.from_mat()
    nrg_dndt = nrg.dndt
    nrg_occ = nrg.occupation

    ts = nrg.ts
    ts2d = np.tile(ts, (nrg_occ.shape[-1], 1)).T

    vs_occ_interpers = [interp1d(x=occ, y=dndt, bounds_error=False, fill_value=np.nan) for occ, dndt in zip(nrg_occ, nrg_dndt)]

    new_occ = np.linspace(0, 1, 200)

    new_dndt = np.array([interp(x=new_occ) for interp in vs_occ_interpers])

    fig = p2d.plot(new_dndt, x=new_occ, y=0.001/ts,
                   xlabel='Occupation', ylabel='G/T',
                   title='NRG dN/dT vs Occ')
    fig.update_yaxes(type='log')
    fig.update_layout(yaxis=dict(range=[np.log10(v) for v in (0.1, 30)]))
    fig.show()


    fig = p2d.plot(nrg_dndt, x=nrg.ens, y=0.001/ts,
                   xlabel='Energy', ylabel='G/T',
                   title='NRG dN/dT vs Energy')
    fig.update_yaxes(type='log')
    fig.update_layout(yaxis=dict(range=[np.log10(v) for v in (0.1, 30)],),
                      xaxis=dict(range=[-0.03, 0.03]))
    fig.show()


    fig = p2d.plot(nrg_occ, x=nrg.ens, y=0.001/ts,
                   xlabel='Energy', ylabel='G/T',
                   title='NRG Occupation vs Energy')
    fig.update_yaxes(type='log')
    fig.update_layout(yaxis=dict(range=[np.log10(v) for v in (0.1, 30)],),
                      xaxis=dict(range=[-0.03, 0.03]))
    fig.show()


