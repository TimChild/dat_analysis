from scipy.interpolate import interp1d
import numpy as np
import plotly.graph_objects as go

import src.UsefulFunctions as U
from src.Characters import DELTA
from src.DatObject.Make_Dat import get_dats, get_dat
from src.Dash.DatPlotting import OneD, TwoD

import plotly.io as pio
pio.renderers.default = 'browser'

p1d = OneD(dat=None)
p2d = TwoD(dat=None)


def plot_dndt_without_zero_offset(data_json_path: str,
                                  max_dndt: float = 1) -> go.Figure:
    """
    Re-plot the scaled dNdTs from Dash with data starting at zero instead of being somewhere else
    Args:
        data_json_path ():
        max_dndt (): To rescale the dN/dT data back into something like nA instead of 0 -> 1

    Returns:

    """
    def get_offset(d):
        return np.nanmean(d[:round(d.shape[-1] / 10)])  # Use first 10% to find 'zero'

    def rescale(d, offset_, max_):
        d = d - offset_
        d = d / np.nanmax(d) * max_
        return d

    all_data = U.data_from_json(data_json_path)
    x = all_data['Data - i_sense_cold_x']
    data_dndt = all_data['Scaled Data - dndt']
    nrg_dndt = all_data['Scaled NRG dndt']
    # occupation = all_data['Scaled NRG i_sense']

    fig = p1d.figure(xlabel='Sweep Gate (mV)', ylabel=f'{DELTA}I (nA)')

    # Plot data with offset subtracted
    offset = get_offset(data_dndt)
    data = rescale(data_dndt, offset, max_dndt)
    fig.add_trace(p1d.trace(data=data, x=x, mode='lines', name='Data'))

    # Plot nrg with offset subtracted and scaled to data
    offset = get_offset(nrg_dndt)
    data = rescale(nrg_dndt, offset, max_dndt)
    fig.add_trace(p1d.trace(data=data, x=x, mode='lines', name='NRG'))

    return fig


if __name__ == '__main__':
    fig = plot_dndt_without_zero_offset(data_json_path='downloaded_data_jsons/dat2167_gamma_broadened_expected_theta.json',
                                        max_dndt=0.02555)
    fig.update_layout(title='Gamma Broadened - Linearly extrapolated lever arm from weakly coupled')
    fig.show()

    