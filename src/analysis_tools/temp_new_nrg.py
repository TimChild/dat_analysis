from src.analysis_tools.nrg import NRGData, nrg_func

import numpy as np
import plotly.graph_objects as go
import scipy.io as sio
import logging
import plotly.io as pio

from src.plotting.plotly.dat_plotting import OneD, TwoD

logger = logging.getLogger(__name__)

pio.renderers.default = 'browser'

p1d = OneD(dat=None)
p2d = TwoD(dat=None)


def test_plot_various_params():
    from src.analysis_tools.nrg import NRG_func_generator

    data = sio.loadmat('../../resources/NRGResultsNew.mat')

    d = NRGData.from_mat()
    # t1 = time.time()
    # # f = NRG_func_generator_new('i_sense')
    # print(time.time()-t1)

    x = np.linspace(-200, 200, 1001)

    old_func = NRG_func_generator('i_sense')
    fig = go.Figure()
    for c, g, theta in zip([1, 2, 3], [10, 10, 10], [10, 10, 10]):
        l=0
        o=0
        a=1
        c=1
        m=0
        arr = old_func(x, m, g, theta, amp=a, lin=l, occ_lin=o, const=c)
        fig.add_trace(go.Scatter(x=x, y=arr, mode='lines', name='Old', line=dict(dash='dash')))
        arr = nrg_func(x, m, g, theta, amp=a, lin=l, occ_lin=o, const=c, data_name='i_sense')
        fig.add_trace(go.Scatter(x=x, y=arr, mode='lines', name='New'))
    fig.show()



if __name__ == '__main__':
    pass
