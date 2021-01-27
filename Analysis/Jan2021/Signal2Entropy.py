from __future__ import annotations

from Plotting.Plotly.AttrSpecificPlotting import SquareEntropyPlotter
from src.DatObject.Make_Dat import DatHandler
from src.DataStandardize.ExpSpecific import Sep20
from src.AnalysisTools import DCbias

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass

get_dat = DatHandler().get_dat

if __name__ == '__main__':

    chosen_dats = {
        8797: '100mK Similar',
        8710: '50mK Similar',
        8808: '100mK Different',
        8721: '50mK Different'
    }

    dc_bias_dats = {
        100: DatHandler().get_dats((4284, 4295), datname='s2e', exp2hdf=Sep20.SepExp2HDF),
        50: DatHandler().get_dats((8593, 8599), datname='s2e', exp2hdf=Sep20.SepExp2HDF),
    }

    dc_bias_infos = {k: DCbias.DCbiasInfo.from_dats(dc_bias_dats[k], bias_key=bias_key, force_centered=False)
                     for k, bias_key in zip(dc_bias_dats, ['R2T(10M)', 'R2T/0.001'])}

    datnums = list(chosen_dats.keys())

    dats = [get_dat(num, datname='s2e', exp2hdf=Sep20.SepExp2HDF, overwrite=False) for num in datnums]

    for dat in dats:
        if dat.Entropy._integration_info_exists('default') is False:
            if dat.datnum in [8797, 8808]:
                temp = 100
            elif dat.datnum in [8710, 8721]:
                temp = 50
            else:
                raise NotImplementedError(f'Dont know temp of dat{dat.datnum}')
            dcinfo = dc_bias_infos[temp]
            dat.Entropy.set_integration_info(dc_info=dcinfo)

    plotters = [SquareEntropyPlotter(dat) for dat in dats]
    for plotter in plotters[0:1]:
        # plotter.plot_raw().show(renderer='browser')
        # plotter.plot_cycled().show(renderer='browser')
        # plotter.plot_avg().show(renderer='browser')
        plotter.plot_entropy_signal().show(renderer='browser')
        plotter.plot_integrated_entropy().show(renderer='browser')





