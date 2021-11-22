"""
Sep 21 -- Common analysis functions used throughout experiment.
Useful functions have been moved.
"""
from typing import List, Optional, Dict

import numpy as np
from progressbar import progressbar
import logging

from dat_analysis import useful_functions as U
from dat_analysis.analysis_tools.entropy import _get_deltaT

from dat_analysis.dat_object.dat_hdf import DatHDF

from dat_analysis.dat_object.make_dat import get_dat

logger = logging.getLogger(__name__)


def set_sf_from_transition(entropy_datnums, transition_datnums, fit_name, integration_info_name, dt_from_self=False,
                           fixed_dt=None, fixed_amp=None, experiment_name: Optional[str] = None):
    for enum, tnum in progressbar(zip(entropy_datnums, transition_datnums)):
        edat = get_dat(enum, exp2hdf=experiment_name)
        tdat = get_dat(tnum, exp2hdf=experiment_name)
        try:
            _set_amplitude_from_transition_only(edat, tdat, fit_name, integration_info_name, dt_from_self=dt_from_self,
                                                fixed_dt=fixed_dt, fixed_amp=fixed_amp)
        except (TypeError, U.NotFoundInHdfError):
            print(f'Failed to set scaling factor for dat{enum} using dat{tnum}')


def _set_amplitude_from_transition_only(entropy_dat: DatHDF, transition_dat: DatHDF, fit_name, integration_info_name,
                                        dt_from_self,
                                        fixed_dt=None, fixed_amp=None):
    ed = entropy_dat
    td = transition_dat
    # for k in ['ESC', 'ESS', 'ESP']:
    if fixed_dt is None:
        dt = _get_deltaT(ed, from_self=dt_from_self, fit_name=fit_name)
    else:
        dt = fixed_dt

    if fixed_amp is None:
        for k in ['ESP']:
            if ed.Logs.fds[k] != td.Logs.fds[k]:
                raise ValueError(f'Non matching FDS for entropy_dat {ed.datnum} and transition_dat {td.datnum}: \n'
                                 f'entropy_dat fds = {ed.Logs.fds}\n'
                                 f'transition_dat fds = {td.Logs.fds}')
        amp = td.Transition.get_fit(name=fit_name).best_values.amp
    else:
        amp = fixed_amp
    ed.Entropy.set_integration_info(dT=dt,
                                    amp=amp if amp is not None else np.nan,
                                    name=integration_info_name,
                                    overwrite=True)
    return True


def sort_by_temps(dats: List[DatHDF]) -> Dict[float, List[DatHDF]]:
    d = {
        temp: [dat for dat in dats if np.isclose(dat.Logs.temps.mc * 1000, temp, atol=25)]
        for temp in [500, 400, 300, 200, 100, 50, 10]}
    for k in list(d.keys()):
        if len(d[k]) == 0:
            d.pop(k)
    return d


def sort_by_coupling(dats: List[DatHDF]) -> Dict[float, List[DatHDF]]:
    d = {
        gate: [dat for dat in dats if np.isclose(dat.Logs.fds['ESC'], gate, atol=5)]
        for gate in set([U.my_round(dat.Logs.fds['ESC'], base=10) for dat in dats])}
    for k in d:
        if len(d[k]) == 0:
            d.pop(k)
    return d


