from __future__ import annotations

from _md5 import md5

from progressbar import progressbar
from src.DatObject.Make_Dat import get_dat, get_dats
import src.UsefulFunctions as U
from src.DataStandardize.ExpSpecific.Feb21 import Feb21Exp2HDF, Feb21ExpConfig
from src.DataStandardize.ExpConfig import ExpConfigGroupDatAttribute, ExpConfigBase
import multiprocessing as mp
import plotly.graph_objs as go
import numpy as np
import lmfit as lm
from typing import TYPE_CHECKING, Iterable, Optional
from src.DatObject.Attributes.Transition import i_sense_digamma, i_sense, i_sense_digamma_quad
from src.UsefulFunctions import edit_params
from src.DatObject.Attributes.SquareEntropy import square_wave_time_array, integrate_entropy
import logging
logger = logging.getLogger(__name__)


def hash_data_test(data: np.ndarray):
    if data.ndim == 1:
        data = data[~np.isnan(data)]  # Because fits omit NaN data so this will make data match the fit data.
    return md5(data.tobytes()).hexdigest()

print(hash_data_test(np.array([np.nan, np.nan, 1,2,3])))
print(hash_data_test(np.array([np.nan, np.nan, np.nan,2,3])))




# theta = dats[0].SquareEntropy.avg_fit.best_values.theta
# fit = dats[0].SquareEntropy.get_fit(which='avg', which_fit='transition', transition_part='cold', check_exists=False)
# params = fit.params
# params.add('g', value=0, vary=True, min=-50, max=1000)
# new_pars = edit_params(params, param_name='theta', value=theta, vary=False) # Based on some old DC bias scans
#
# print(narrow_fit(
#     dats[0],
#     400,
#     initial_params=new_pars,
#     fit_func=i_sense_digamma,
#     check_exists=False).best_values.amp)
#
# print(narrow_fit(
#     dats[0],
#     400,
#     initial_params=new_pars,
#     fit_func=i_sense_digamma,
#     check_exists=True).best_values.amp)
#
# amp_digamma_ = [narrow_fit(
#     dat,
#     400,
#     initial_params=new_pars,
#     fit_func=i_sense_digamma,
#     check_exists=False).best_values.amp
# for dat in progressbar(dats)]

