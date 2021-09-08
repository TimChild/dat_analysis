import plotly.graph_objects as go
import numpy as np
from src import core_util as CU


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

    fig.update_layout(
        title=f'Single wave averaged over all rows from {x_range[0]:.0f}mV to {x_range[-1]:.0f}mV - wavelen={wavelen}samples, freq={freq:.1f}Hz',
        xaxis_title='Fraction of single cycle', yaxis_title='Delta Current /nA')
    fig.update_yaxes(showspikes=True)
    # fig.show(renderer="browser")
    return fig


# x_range = (-500, 500)
# dats = get_dats([3911, 3912])
# avg_transition_sw(dats, x_range)


if __name__ == '__main__':
    from src.dat_object.make_dat import DatHandler as DH
    from src.data_standardize.exp_specific.Sep20 import Fixes
    import logging

    logging.root.setLevel(level=logging.WARNING)

    # dats = DH.get_dats([7031, 7032, 7033])
    all_dats = DH.get_dats([7031, 7032, 7033])
    for dat in all_dats:
        Fixes.fix_magy(dat)

    # fig = avg_transition_sw(dats, [-10000, 10000])
    # fig.write_html('Testing ramp square wave.html')
