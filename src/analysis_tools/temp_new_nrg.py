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



if __name__ == '__main__':
    from src.analysis_tools.nrg import NRG_func_generator

    data = sio.loadmat('../../resources/NRGResultsNew.mat')

    d = NRGData.from_new_mat()
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
    # from src.dat_object.Attributes.Transition import i_sense_digamma
    # fig = go.Figure()
    # nrg = NRGData.from_new_mat()
    # for i, (x, d, t, g) in enumerate(zip(nrg.ens, nrg.dndt, nrg.ts, nrg.gs)):
    #     # if i in [0, 10, 11, 15, 20, 30, 39]:
    #     if i in [11, 15, 20, 30, 39]:
    #     # if i == 20:
    #         digamma_x = np.linspace(x[0]*5, x[-1]*5, num=200)
    #         digamma = i_sense_digamma(digamma_x, 0, g, t, 1, 0, 0)*-1 + 0.5
    #         digamma_x = scale_x(digamma_x*10000, 0, g*10000, 0, inverse=False)
    #         fig.add_trace(p1d.trace(x=digamma_x, data=digamma, name=f'digamma {t/g:.2f}', mode='lines'))
    #         fig.add_trace(p1d.trace(x=x, data=d, name=f'nrg {t/g:.2f}'))
    #
    # fig.show()
