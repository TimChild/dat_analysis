from __future__ import annotations
from src.DatObject.Make_Dat import get_dat, get_dats
import src.UsefulFunctions as U
from src.DataStandardize.ExpSpecific.Feb21 import Feb21Exp2HDF, Feb21ExpConfig
from src.DataStandardize.ExpSpecific.FebMar21 import FebMar21Exp2HDF, FebMar21ExpConfig

from src.DataStandardize.ExpConfig import ExpConfigGroupDatAttribute, ExpConfigBase
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import plotly.graph_objs as go
import numpy as np
import lmfit as lm
from typing import TYPE_CHECKING, Iterable, Optional, Tuple

if TYPE_CHECKING:
    from src.DatObject.Make_Dat import DatHDF
    from src.DatObject.Attributes.DatAttribute import FitInfo


def _fix_temps(dats):
    for dat in dats:
        dat.del_hdf_item('Experiment Config')
        exp_config = ExpConfigGroupDatAttribute(dat, exp_config=Feb21ExpConfig(dat.datnum))
        dat.ExpConfig = exp_config
    return True


def _fix_logs(dats):
    for dat in dats:
        try:
            tc = dat.Logs.time_completed
        except Exception as e:
            print(f'Fixing Dat{dat.datnum} which raised {e} when looking for dat.Logs.time_completed')
            logs = dat.Logs
            del dat.Logs
        try:
            tc = dat.Logs.time_completed
        except Exception as e:
            print(f'Dat{dat.datnum} not fixed: raised {e}')


def calc_transition(datnum: int, exp2hdf=FebMar21Exp2HDF) -> FitInfo:
    dat = get_dat(datnum, exp2hdf=exp2hdf, overwrite=False)
    try:
        tfit = dat.Transition.avg_fit
    except Exception as e:
        print(f'Dat{datnum} raised {e}')
        tfit = None
    return tfit


def theta_vs_fridge_temp_fig(thetas: Iterable[float], temps: Iterable[float], datnums: Iterable[int],
                             lower_fit_temp_limit: Optional[float] = None) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(mode='markers+lines+text', x=temps, y=thetas, text=datnums, name='Avg Thetas'))

    if lower_fit_temp_limit is not None:
        line = lm.models.LinearModel()
        fit_thetas, fit_temps = np.array(
            [(theta, temp) for theta, temp in zip(thetas, temps) if temp > lower_fit_temp_limit]).T
        fit = line.fit(fit_thetas, x=fit_temps, nan_policy='omit')
        x_range = np.linspace(0, max(temps), int(max(temps)) + 1)
        fig.add_trace(go.Scatter(mode='lines', x=x_range, y=fit.eval(x=x_range),
                                 name=f'Fit to temps > {lower_fit_temp_limit}'))

    fig.update_layout(xaxis_title='Temp /mK', yaxis_title='Theta /mV',
                      title=f'Dats{min(datnums)}-{max(datnums)}: Transition vs Fridge temp')

    return fig


def _thetas_from_datnums(datnums: Iterable[int], overwrite=False, exp2hdf=FebMar21Exp2HDF) -> Tuple[float]:
    dats = get_dats(list(datnums), exp2hdf=exp2hdf, overwrite=overwrite)
    thetas = [dat.Transition.avg_fit.best_values.theta for dat in dats]
    return tuple(thetas)


if __name__ == '__main__':
    # dats = get_dats((613, 624+1), exp2hdf=Feb21Exp2HDF)
    # thetas_ = [dat.Transition.avg_fit.best_values.theta for dat in dats]
    # temps_ = [300, 275, 250, 225, 200, 175, 150, 125, 100, 75, 50, 10]
    # datnums_ = [dat.datnum for dat in dats]
    # fig = theta_vs_fridge_temp_fig(thetas=thetas_, temps=temps_, datnums=datnums_, lower_fit_temp_limit=160)
    # fig.show(renderer='browser')

    pos2 = {
        300: (824, 825),
        275: (836, 837),
        250: (848, 849),
        225: (860, 861),
        200: (872, 873, 877, 878),
        175: (889, 890),
        150: (891, 892),
        125: (913, 914),
        100: (926, 927),
        75: (938, 939, 946, 947),
        50: (958, 959),
        30: (970, 971),
        10: (982, 983),
    }

    pos3 = {
        300: (826, 827),
        275: (838, 839),
        250: (850, 851),
        225: (862, 863),
        200: (874, 875, 879, 880),
        175: (891, 892),
        150: (893, 894),
        125: (915, 916),
        100: (928, 929),
        75: (940, 941, 948, 949),
        50: (960, 961),
        30: (972, 973),
        10: (984, 985),
    }
    pos3badtemps = [300, 275, 150]

    pos = pos3
    bads = pos3badtemps

    for k in bads:
        pos.pop(k)

    pos_thetas = {temp: _thetas_from_datnums(datnums) for temp, datnums in pos.items()}
    temps_, thetas_ = list(zip(*[(k, v) for k in pos_thetas for v in pos_thetas[k]]))
    datnums_ = [v for k in pos for v in pos[k]]
    # thetas_ = [dat.Transition.avg_fit.best_values.theta for dat in dats]
    # temps_ = [300, 275, 250, 225, 200, 175, 150, 125, 100, 75, 50, 10]
    # datnums_ = [dat.datnum for dat in dats]
    fig = theta_vs_fridge_temp_fig(thetas=thetas_, temps=temps_, datnums=datnums_, lower_fit_temp_limit=60)
    fig.show(renderer='browser')

    # # pool = mp.Pool(processes=6)
    # pool = ProcessPoolExecutor(max_workers=6)
    # datnums = range(822, 990+1)
    # # datnums = range(822, 825)
    # chunked_datnums = np.array_split(datnums, 10)
    # for dnums in chunked_datnums:
    #     fits = list(pool.map(calc_transition, dnums))
    #     for num, fit in zip(dnums, fits):
    #         if fit is not None:
    #             # print(f'Dat{num}:\n'
    #             #       f'{fit.best_values}\n')
    #             pass
    #         else:
    #             print(f'#####################################\n'
    #                   f'Dat{num}: Fit failed\n'
    #                   f'#####################################\n')

# from concurrent.futures import ProcessPoolExecutor
#
# def do_calc(datnum):
#     dat = get_dat(datnum)
#     fit = dat.Transition.avg_fit  # This runs all the necessary transition fits
#
# if __name__ == '__main__':
#     pool = ProcessPoolExecutor(max_workers=6)
#     datnums = range(800, 901)
#     pool.map(do_calc, datnums)


