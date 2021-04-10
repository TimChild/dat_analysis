from src.DatObject.Make_Dat import DatHDF, get_dat, get_dats
from src.Dash.DatPlotting import OneD, TwoD
from Analysis.Feb2021.common import _get_transition_fit_func_params

import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

pio.renderers.default = 'browser'


if __name__ == '__main__':
    dats = get_dats((5352, 5357+1))

    for dat in dats[-1:]:
        plotter = OneD(dat=dat)
        x = dat.Transition.x
        data = dat.Transition.data
        func, params = _get_transition_fit_func_params(x, data[0], 'i_sense', theta=None, gamma=0)
        fits = dat.Transition.get_row_fits(name='i_sense', fit_func=func, initial_params=None, check_exists=False,
                                           overwrite=False)
        thetas = [fit.best_values.theta for fit in fits]
        sweepgates_y = dat.Data.get_data('sweepgates_y')
        y = np.linspace(sweepgates_y[0][1]/10, sweepgates_y[0][2]/10, dat.Data.get_data('y').shape[0])

        fig = plotter.plot(data=thetas, x=y, xlabel='HQPC bias /nA', ylabel='Theta /mV', mode='markers+lines',
                           title=f'Dat{dat.datnum}: MC temp={dat.Logs.temps.mc*1000:.1f}mK DCBias thetas')

        fig.show()

