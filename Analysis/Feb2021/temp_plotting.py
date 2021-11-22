"""
Sep 21 -- Don't remember exactly what I was doing with this, I think just some early plotting with Occupation on x-axis
Not worth salvaging anything from here.
"""
import dat_analysis.useful_functions as U
from dat_analysis.dat_object.make_dat import get_dats
from scipy.interpolate import interp1d
import numpy as np
from dat_analysis.plotting.plotly.dat_plotting import OneD

import plotly.io as pio
pio.renderers.default = 'browser'


if __name__ == '__main__':
    # all_data = U.data_from_json('Data_2213vsNRG.json')
    all_data = U.data_from_json('Data_2213vsNRG_dndt_matched.json')

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
    fig = plotter.figure(xlabel='Occupation', ylabel='Arbitrary', title='dN/dT vs Occupation at Temp/Gamma = 0.19 (Temp = 1.9e-4 in NRG)')
    fig.add_trace(plotter.trace(x=occ_x, data=(data_dndt-0.2)*1.2, name='Data', mode='lines+markers'))
    fig.add_trace(plotter.trace(x=occ_x, data=nrg_dndt, name='NRG', mode='lines'))
    fig.show()
    #
    #
    # nrg = NRGData.from_mat()
    # occs = nrg.occupation
    # dndts = nrg.dndt
    # ts = nrg.ts
    #
    # fig = plotter.figure(xlabel='Occupation', ylabel='Arbitrary', title='dN/dT vs Occupation for various NRG T')
    #
    # # Plot Data dN/dT
    # fig.add_trace(plotter.trace(x=occ_x, data=(data_dndt*1.40-0.25), mode='lines+markers', name='Data'))
    #
    # for i in range(40, 47, 1):
    #     fig.add_trace(plotter.trace(data=dndts[i]/np.max(dndts[i]), x=occs[i], name=f'T = {ts[i]:.2g}', mode='lines'))
    #
    # [plotter.add_line(fig, v, color='black', linetype='dash') for v in [0, 1]]
    #
    # fig.update_layout(template='simple_white')
    # fig.show()
    #
    # fig.write_image('dndt_vs_Occ_many.svg')
    #

    dats = get_dats([2102, 7046, 7084, 7094])  # Last CD, This CD slow sweep, This CD same sweep, This CD fast sweep

    plotter = OneD(dats=dats)
    plotter.RESAMPLE_METHOD = 'downsample'  # So jumps are still obvious instead of binning
    plotter.MAX_POINTS = 2000

    single_figs = []
    row = 0
    fig = plotter.figure(xlabel='ACC /mV', ylabel='Current /nA', title=f'Comparing Transition Noise: Row{row}')
    for dat in dats:
        fig.add_trace(plotter.trace(data=dat.Transition.data[row], x=dat.Transition.x/100,
                                    name=f'Dat{dat.datnum}: {dat.Logs.sweeprate/100:.1f}mV/s', mode='lines'))

        single_fig = plotter.figure(xlabel=dat.Logs.xlabel, ylabel=dat.Logs.ylabel,
                                    title=f'Dat{dat.datnum}: Few rows of transition data')
        for r in range(10):
            single_fig.add_trace(plotter.trace(x=dat.Transition.x, data=dat.Transition.data[r], name=f'Row: {r}', mode='lines'))

        single_figs.append(single_fig)

    fig.show()
    for i, f in enumerate(single_figs):
        f.write_html(f'figs/temp{i}.html')
        U.fig_to_igor_itx(f, f'figs/data/temp{i}.itx')
    #     f.show()


