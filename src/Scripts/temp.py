# from src.DataStandardize.ExpSpecific.Aug20 import add_temp_magy
from typing import List, Union, Tuple

from src.Scripts.StandardImports import *
from src.Plotting.Plotly.PlotlyUtil import PlotlyViewer as PV
from progressbar import progressbar
import plotly.graph_objects as go
from src.DataStandardize.ExpSpecific.Sep20 import Fixes
from src.Plotting.Plotly import PlotlyUtil as PlU
from src.Scripts.SquareEntropyAnalysis import *
import src.Scripts.SquareEntropyAnalysis as EA
import src.DatObject.Attributes.SquareEntropy as SE

analysis_params = EA_params(bin_data=True, num_per_row=400,
                            sub_line=False, sub_line_range=(-4000, -500),
                            int_entropy_range=(600, 1000),
                            allowed_amp_range=(0.8, 1.2), default_amp=1.05,
                            allowed_dT_range=(1, 15), default_dT=5.96,
                            CT_fit_range=(None, None),
                            fit_param_edit_kwargs=dict())



if __name__ == '__main__':
    # dats = get_dats((1428, 1452 + 1))  # Scan along transition at -95 -> -105mT in 1mT steps, multiple repeats at each
    dats = get_dats((1428, 1430 + 1), overwrite=False)
    for dat in dats:
        Fixes.fix_magy(dat)

    recalculate = True

    for dat in progressbar(dats):
        if not hasattr(dat.Other, 'time_processed') or recalculate is True:
            datas = EA.EA_datas()
            all_values = EA.EA_values()
            for row in dat.SquareEntropy.Processed.outputs.cycled:
                entropy_data = SE.entropy_signal(row)

                data = EA.EA_data(x=dat.SquareEntropy.Processed.outputs.x, trans_data=row, entropy_data=entropy_data)
                values = EA.EA_value()
                EA.calculate_CT_values(data, values,
                                       analysis_params)  # Calculates things like dT from transition data. Can change CT_fit_range to fit mostly to inner averaged data for example
                EA.calculate_integrated(data, values, analysis_params)  # Will follow instructions from analysis_params
                EA.calculate_fit(data, values,
                                 analysis_params)  # Can add any edit params kwargs here, or uses from analysis_params

                if analysis_params.bin_data:
                    EA.bin_datas(data,
                                 analysis_params.num_per_row)  # Bins data in place (bin because this data is much reduced from fastdac speed due to averaging already)

                datas.append(data)
                all_values.append(values)

            EA.save_to_dat(dat, datas, all_values, analysis_params)  # Saves everything to DatHDF. Most will be loaded automatically, data needs to be loaded with EA.get_data(dat)






    # datas = EA_datas.from_dats(dats)
    # datas.add_fit_to_entropys([p[0].Other.EA_values.efit_info for p in dat_pairs])
    #
    # dat = dats[0]
    #
    # fig1 = PlU.get_figure(datas=datas.trans_datas, xs=datas.xs, ids=titles.ids, titles=titles.trans, labels=['v0_0', 'vp', 'v0_1', 'vm'],
    #                       xlabel=f'{dat.Logs.x_label}', ylabel='Current /nA', plot_kwargs={'mode': 'lines+markers'})
    # fig2 = PlU.get_figure(datas=datas.entropy_datas, xs=datas.xs, ids=titles.ids, titles=titles.entropy, labels=['fit', 'data'],
    #                       xlabel=f'{dat.Logs.x_label}', ylabel='Current /nA', plot_kwargs={'mode': 'lines+markers'})
    # fig3 = PlU.get_figure(datas=datas.integrated_datas, xs=datas.xs, ids=titles.ids, titles=titles.integrated,
    #                       xlabel=f'{dat.Logs.x_label}', ylabel='Entropy /kB', plot_kwargs={'mode': 'lines+markers'})
    #
    # for fig, name in zip([fig1, fig2, fig3], ['AveragedCS_vs_Channel_bias', 'Average_Entropy_With_Fit', 'Averaged_Integrated_Entropy']):
    #     fig.update_layout(hovermode = 'x unified',
    #                  title=dict(y=0.95,x=0.5,xanchor='center',yanchor='top', font=dict(size=12)))
    #     fig.show()






















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

    # get_dat(85)
