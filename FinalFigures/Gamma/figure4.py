import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.signal import savgol_filter
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple

from FinalFigures.Gamma.plots import dndt_2d
from src.Dash.DatPlotting import OneD, TwoD
from Analysis.Feb2021.NRG_comparison import NRG_func_generator, NRGData
import src.UsefulFunctions as U

p1d = OneD(dat=None)
p2d = TwoD(dat=None)


def data_nrg_vs_gate_2d(which: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the 2D data for NRG dN/dT vs Sweepgate plot"""
    temperature = 50  # in sweepgate mV so ~100mK
    sweep_x = np.linspace(-100, 100, 201)
    gs = np.linspace(temperature/10, temperature*30, 100)
    nrg_dndt_func = NRG_func_generator(which)

    g_over_ts = gs/temperature

    data = np.array([nrg_dndt_func(sweep_x, 0, g, temperature)
                     for g in gs])

    return sweep_x, g_over_ts, data


def plot_nrg_dndt_vs_gate_2d(x: np.ndarray, y: np.ndarray, dndt: np.ndarray) -> go.Figure:
    fig = p2d.plot(x=x, y=y, data=dndt, xlabel='Sweep Gate (mV)', ylabel='G/T', title='NRG dN/dT vs Sweep Gate')
    return fig




if __name__ == '__main__':
    from src.DatObject.Make_Dat import get_dats, get_dat

    # NRG dN/dT vs sweep gate
    xs, gts, dndt = data_nrg_vs_gate_2d(which='dndt')
    fig = plot_nrg_dndt_vs_gate_2d(xs, gts, dndt)
    fig.show()

    # NRG Occupation vs sweep gate
    xs, gts, dndt = data_nrg_vs_gate_2d(which='occupation')
    fig = plot_nrg_dndt_vs_gate_2d(xs, gts, dndt)
    fig.show()

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


