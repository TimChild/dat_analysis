from __future__ import annotations
from src.DatObject.Make_Dat import get_dat, get_dats
import src.UsefulFunctions as U
from src.DataStandardize.ExpSpecific.Feb21 import Feb21Exp2HDF, Feb21ExpConfig
from src.DataStandardize.ExpConfig import ExpConfigGroupDatAttribute, ExpConfigBase
import multiprocessing as mp
import plotly.graph_objs as go
import numpy as np
import lmfit as lm
from typing import TYPE_CHECKING, Iterable, Optional

# TRICKING PYCHARM TO TRY TO IMPORT A BUNCH OF THINGS. NOTHING ACTUALLY DONE IN MAIN


if TYPE_CHECKING:
    from src.DatObject.Make_Dat import DatHDF
    from src.DatObject.Attributes.DatAttribute import FitInfo

def _fix_temps(dats):
    for dat in dats:
        dat.del_hdf_item('Experiment Config')
        exp_config = ExpConfigGroupDatAttribute(dat, exp_config=Feb21ExpConfig(dat.datnum))
        dat.ExpConfig = exp_config
    return True


def calc_transition(datnum: int) -> FitInfo:
    dat = get_dat(datnum, exp2hdf=Feb21Exp2HDF)
    tfit = dat.Transition.avg_fit
    return tfit


def theta_vs_fridge_temp_fig(thetas: Iterable[float], temps: Iterable[float], datnums: Iterable[int],
                             lower_fit_temp_limit: Optional[float] = None) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(mode='markers+lines+text', x=temps, y=thetas, text=datnums, name='Avg Thetas'))

    if lower_fit_temp_limit is not None:
        line = lm.models.LinearModel()
        fit_thetas, fit_temps = np.array([(theta, temp) for theta, temp in zip(thetas, temps) if temp > lower_fit_temp_limit]).T
        fit = line.fit(fit_thetas, x=fit_temps, nan_policy='omit')
        x_range = np.linspace(0, max(temps), int(max(temps))+1)
        fig.add_trace(go.Scatter(mode='lines', x=x_range, y=fit.eval(x=x_range),
                                 name=f'Fit to temps > {lower_fit_temp_limit}'))

    fig.update_layout(xaxis_title='Temp /mK', yaxis_title='Theta /mV',
                      title=f'Dats{min(datnums)}-{max(datnums)}: Transition vs Fridge temp')

    return fig


if __name__ == '__main__':
    pass