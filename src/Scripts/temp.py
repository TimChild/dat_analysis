# from src.DataStandardize.ExpSpecific.Aug20 import add_temp_magy
from src.Scripts.StandardImports import *
from src.Plotting.Plotly.PlotlyUtil import PlotlyViewer as PV

import plotly.graph_objects as go

if __name__ == '__main__':
    # dats = get_dats(range(337, 400))
    # dat = dats[0]
    #
    # add_temp_magy(dats)
    #
    # datas, xs, ids, titles = list(), list(), list(), list()
    # traces = list()
    # for dat in dats:
    #     dat.Other.magy: float
    #     data = CU.decimate(dat.Transition.avg_data, dat.Logs.Fastdac.measure_freq, 30)
    #     x = np.linspace(dat.Data.x_array[0], dat.Data.x_array[-1], data.shape[-1])
    #     x, data = CU.sub_poly_from_data(x, data, dat.Transition.avg_fit)
    #     datas.append(data)
    #     xs.append(x)
    #     ids.append(dat.datnum)
    #     name = f'Dat{dat.datnum}: Field={dat.Other.magy:.0f}mT, Bias={dat.Logs.fds["R2T(10M)"]:.0f}mV'
    #     titles.append(name)
    #     traces.append(go.Scatter(x=x, y=data, mode='lines', name=name))
    # xlabel = 'LP*2/mV'
    # ylabel = 'Current /nA'
    #
    # # fig = PL.get_figure(datas, xs, ids=ids, titles=titles, xlabel=xlabel, ylabel=ylabel)
    #
    # fig = go.Figure()
    # fig.add_traces(traces)
    #
    # v = PV(fig)
    #
    #
    # P.display_2d()

    get_dat(85)