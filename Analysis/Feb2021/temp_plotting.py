import src.UsefulFunctions as U
from src.DatObject.Make_Dat import get_dat, get_dats, DatHDF
from scipy.interpolate import interp1d
import numpy as np
from src.Dash.DatPlotting import OneD, TwoD

import plotly.io as pio
pio.renderers.default = 'browser'


if __name__ == '__main__':
    all_data = U.data_from_json('Data_2213vsNRG.json')

    x = all_data['Data - i_sense_cold_x']
    data_dndt = all_data['Scaled Data - dndt']
    nrg_dndt = all_data['Scaled NRG dndt']
    occupation = all_data['Scaled NRG occupation']

    print(x.shape, data_dndt.shape, nrg_dndt.shape, occupation.shape)

    interp_range = np.where(np.logical_and(occupation < 0.99, occupation > 0.01))
    interp_data = occupation[interp_range]
    interp_x = x[interp_range]

    interper = interp1d(x=interp_x, y=interp_data, assume_sorted=True)

    occ_x = interper(x)

    plotter = OneD(dat=None)

    fig = plotter.figure(xlabel='Occupation', ylabel='Arbitrary', title='dN/dT vs Occupation at Temp/Gamma = 0.04 (Temp = 4e-5 in NRG)')
    fig.add_trace(plotter.trace(x=occ_x, data=(data_dndt-0.2)*1.2, name='Data', mode='lines+markers'))
    fig.add_trace(plotter.trace(x=occ_x, data=nrg_dndt, name='NRG', mode='lines'))
    fig.show()
