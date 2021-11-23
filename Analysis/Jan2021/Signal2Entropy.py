from __future__ import annotations
import numpy as np

from dat_analysis.dat_object.make_dat import get_dats
from dat_analysis.data_standardize.exp_specific import Sep20
from dat_analysis.analysis_tools import dcbias
from dat_analysis.hdf_util import NotFoundInHdfError

from dat_analysis.plotting.plotly.dat_plotting import OneD

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass


def set_integration_info(dc_info: dcbias.DCbiasInfo, dat):
    dat.Entropy.set_integration_info(dc_info=dc_info)


def entropy_fit_sp_start(dat, sp_start: int):
    pp = dat.SquareEntropy.get_ProcessParams(setpoint_start=sp_start, save_name='sps_50')
    out = dat.SquareEntropy.get_Outputs(name='sps_50', process_params=pp)
    fit = dat.Entropy.get_fit(which='avg', name='sps_50', x=out.x, data=out.average_entropy_signal, check_exists=False)
    return fit


if __name__ == '__main__':

    chosen_dats = {
        8797: '100mK Similar',
        8710: '50mK Similar',
        8808: '100mK Different',
        8721: '50mK Different'
    }

    dc_bias_dats = {
        100: get_dats((4284, 4295), datname='s2e', exp2hdf=Sep20.SepExp2HDF),
        50: get_dats((8593, 8599), datname='s2e', exp2hdf=Sep20.SepExp2HDF),
    }

    dc_bias_infos = {k: dcbias.DCbiasInfo.from_dats(dc_bias_dats[k], bias_key=bias_key, force_centered=False)
                     for k, bias_key in zip(dc_bias_dats, ['R2T(10M)', 'R2T/0.001'])}

    # dats = [get_dat(num, datname='s2e', exp2hdf=Sep20.SepExp2HDF, overwrite=False) for num in datnums]
    #
    # for dat in dats:
    #     if dat.Entropy._integration_info_exists('default') is False:
    #         if dat.datnum in [8797, 8808]:
    #             temp = 100
    #         elif dat.datnum in [8710, 8721]:
    #             temp = 50
    #         else:
    #             raise NotImplementedError(f'Dont know temp of dat{dat.datnum}')
    #         dcinfo = dc_bias_infos[temp]
    #         dat.Entropy.set_integration_info(dc_info=dcinfo)

    # plotters = [SquareEntropyPlotter(dat) for dat in dats]
    # for plotter in plotters[0:1]:
    #     # plotter.plot_raw().show(renderer='browser')
    #     # plotter.plot_cycled().show(renderer='browser')
    #     # plotter.plot_avg().show(renderer='browser')
    #     plotter.plot_entropy_avg().show(renderer='browser')
    #     plotter.plot_integrated_entropy_avg().show(renderer='browser')

    # dats_100 = get_dats((8516, 8534+1), overwrite=False, exp2hdf=Sep20.SepExp2HDF)
    # dats_50 = get_dats((8600, 8626+1), overwrite=False, exp2hdf=Sep20.SepExp2HDF)

    dats_100 = get_dats((8796, 8816+1), overwrite=False, exp2hdf=Sep20.SepExp2HDF)
    dats_50 = get_dats((8710, 8729+1), overwrite=False, exp2hdf=Sep20.SepExp2HDF)
    for all_dats, temp in zip([dats_100, dats_50], [100, 50]):
        dc_info = dc_bias_infos[temp]
        for dat in all_dats:
            try:
                info = dat.Entropy.integration_info
            except NotFoundInHdfError:
                set_integration_info(dc_info, dat)

    plotter = OneD(dats=dats_100)
    # fig = plotter.figure(xlabel='LCB /mV', ylabel='Entropy /kB', title=None)
    fig = plotter.figure(xlabel='LCT /mV', ylabel='Entropy /kB', title=None)
    for all_dats in [dats_100, dats_50]:
        plotter = OneD(dats=all_dats)
        fits = [entropy_fit_sp_start(dat, 50) for dat in all_dats]
        data = np.array([dat.Entropy.avg_fit.best_values.dS for dat in all_dats])
        data_50 = np.array([fit.best_values.dS for fit in fits])
        # x = np.array([dat.Logs.fds['LCB'] for dat in dats])
        x = np.array([dat.Logs.fds['LCT'] for dat in all_dats])
        fig.add_trace(plotter.trace(data=data, x=x, mode='markers', name='sp_0'))
        fig.add_trace(plotter.trace(data=data_50, x=x, mode='markers', name='sp_50'))

    fig.show(renderer='browser')







