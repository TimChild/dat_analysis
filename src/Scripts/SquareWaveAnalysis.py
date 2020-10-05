import plotly.graph_objects as go
import numpy as np
from src.Plotting.Plotly import PlotlyUtil as PlU
from src import CoreUtil as CU
from src.Scripts.StandardImports import get_dats


def avg_transition_sw(dats, x_range):
    fig = go.Figure()

    for dat in dats:
        wavelen = dat.Logs.AWG.wave_len
        freq = dat.Logs.AWG.measureFreq / wavelen

        x = dat.Data.x_array
        indexs = CU.get_data_index(x, x_range)
        x = x[indexs[0]:indexs[1]]
        z = np.mean(dat.Data.Exp_cscurrent_2d[indexs[0]:indexs[1] - (indexs[1] - indexs[0]) % wavelen], axis=0)
        z = np.reshape(z, (-1, wavelen))
        z = np.mean(z, axis=0)
        z = z - np.mean(z)
        lin_x = np.linspace(0, 1, wavelen)
        fig.add_trace(go.Scatter(x=lin_x, y=z, name=f'{dat.datnum}'))

    PlU.fig_setup(fig,
                  title=f'Single wave averaged over all rows from {x_range[0]:.0f}mV to {x_range[-1]:.0f}mV - wavelen={wavelen}samples, freq={freq:.1f}Hz',
                  x_label='Fraction of single cycle', y_label='Delta Current /nA')
    fig.update_yaxes(showspikes=True)
    fig.show(renderer="browser")
    return fig

# x_range = (-500, 500)
# dats = get_dats([3911, 3912])
# avg_transition_sw(dats, x_range)

