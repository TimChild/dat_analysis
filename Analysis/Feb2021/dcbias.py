"""
Sep 21 -- Nothing else worth salvaging from here. Only experiment specific analysis left here
"""

from progressbar import progressbar
import plotly.io as pio
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import copy

from dat_analysis.dat_object.make_dat import get_dat, get_dats
from dat_analysis.plotting.plotly.common_plots import dcbias_single_dat, dcbias_multi_dat
from Analysis.Feb2021.common import sort_by_temps
from dat_analysis.core_util import order_list

pio.renderers.default = 'browser'

if __name__ == '__main__':
    def temp_calc(datnum):
        """Just so I can pass into a multiprocessor"""
        get_dat(datnum).Transition.get_fit(check_exists=False)
        return datnum

    # Single Dat DCBias
    # # dats = get_dats((5352, 5357+1))
    # # dats = get_dats([5726, 5727, 5731])
    # # dats = get_dats([6432, 6434, 6435, 6436, 6437, 6439, 6440])
    # dats = get_dats([6447])
    # fig_x_func = lambda dat: np.linspace(dat.Data.get_data('sweepgates_y')[0][1]/10,
    #                                  dat.Data.get_data('sweepgates_y')[0][2]/10,
    #                                  dat.Data.get_data('y').shape[0])
    # x_label = "HQPC bias /nA"
    #
    # # fig_x_func = lambda dat: dat.Data.get_data('y')
    # # x_label = dats[-1].Logs.ylabel
    #
    # for dat in dats[-1:]:
    #     fig = dcbias_single_dat(dat, fig_x_func=fig_x_func, x_label=x_label)
    #     fig.show()

    # Multi Dat DCbias
    # all_dats = get_dats((6449, 6456 + 1))
    # all_dats = get_dats((6912, 6963 + 1))
    all_dats = get_dats((6960, 6963 + 1))
    # all_dats = get_dats((7437, 7844 + 1))
    # with ProcessPoolExecutor() as pool:
    #     dones = pool.map(temp_calc,  [dat.datnum for dat in all_dats])
    #     for num in dones:
    #         print(num)

    dats_by_temp = sort_by_temps(all_dats)
    figs = []
    for temp, dats in progressbar(dats_by_temp.items()):
        dats = order_list(dats, [dat.Logs.fds['HO1/10M'] for dat in dats])
        fig = dcbias_multi_dat(dats)
        fig.update_layout(title=f'Dats{min([dat.datnum for dat in dats])}-{max([dat.datnum for dat in dats])}: DC bias '
                                f'at {np.nanmean([dat.Logs.temps.mc*1000 for dat in dats]):.0f}mK')
        fig.data[0].update(name=f'{temp:.0f}mK')
        figs.append(fig)

    for fig in figs:
        fig.show()


    multi_fig = copy.copy(figs[0])
    for fig in figs[1:]:
        multi_fig.add_trace(fig.data[0])

    multi_fig.show()

