from __future__ import annotations

import numpy as np
import lmfit as lm
from typing import Union, List, TYPE_CHECKING, Dict, Optional, Tuple
import logging

from dat_analysis.analysis_tools.general_fitting import FitInfo
from dat_analysis.analysis_tools.nrg import NrgUtil, NRGParams
from dat_analysis.plotting.plotly import OneD, TwoD, hover_info

from dat_analysis.dat_object.make_dat import get_dat, get_dats
import dat_analysis.useful_functions as U

if TYPE_CHECKING:
    from dat_analysis.dat_object.dat_hdf import DatHDF
    import plotly.graph_objects as go

p1d = OneD(dat=None)
p2d = TwoD(dat=None)


class LeverArmComparison:
    class DataInfo:
        full_data: np.ndarray
        full_x: np.ndarray

        smoothed_data: np.ndarray
        smoothed_x: np.ndarray

        rough_center: float

        fit_data: np.ndarray
        fit_x: np.ndarray

        fit: FitInfo

        def __init__(self, dat: DatHDF, xlabel: str):
            self.dat = dat
            self.xlabel = xlabel

        def fig(self) -> go.Figure:
            return p1d.figure(xlabel=self.xlabel, ylabel='Current (nA)', title=f'Dat{self.dat.datnum}: Transition Fit')

        def data_trace(self):
            return p1d.trace(data=self.fit_data, x=self.fit_x, name='Data', mode='markers')

        def fit_trace(self):
            return p1d.trace(data=self.fit.eval_fit(x=self.fit_x), x=self.fit_x, name='Fit', mode='lines')

        def traces(self) -> List[go.Scatter]:
            return [self.data_trace(), self.fit_trace()]

        def print_fit_params(self):
            print(f'Fit Params for Dat{self.dat.datnum}:')
            print(f'{self.fit.best_values}')

    vp_info: DataInfo
    acc_info: DataInfo

    def __init__(self, vp_dat: DatHDF, acc_dat: DatHDF):
        self.vp_dat = vp_dat
        self.acc_dat = acc_dat

    def _fit(self, data: np.ndarray, x: np.ndarray) -> FitInfo:
        dat = self.vp_dat if self.vp_dat else self.acc_dat  # Just need any dat to get dat.Transition.get_fit....
        return dat.Transition.get_fit(data=data, x=x, calculate_only=True)

    def vp_transition_fit(self) -> DataInfo:
        dat = self.vp_dat
        self.vp_info = self.DataInfo(dat, xlabel='V_P (mV)')
        vp = self.vp_info
        vp.full_data = dat.Data.i_sense
        vp.full_x = dat.Data.x

        vp.smoothed_data, vp.smoothed_x = U.resample_data(vp.full_data, vp.full_x, max_num_pnts=100,
                                                          resample_method='bin')
        vp.rough_center = vp.smoothed_x[np.nanargmin(np.diff(vp.smoothed_data))]

        ids = U.get_data_index(vp.full_x, [vp.rough_center - 4, vp.rough_center + 4])
        vp.fit_data = vp.full_data[ids[0]:ids[1]]
        vp.fit_x = vp.full_x[ids[0]:ids[1]]

        vp.fit = self._fit(vp.fit_data, vp.fit_x)
        return self.vp_info

    def acc_transition_fit(self) -> DataInfo:
        dat = self.acc_dat
        self.acc_info = self.DataInfo(dat, xlabel='ACC*100 (mV)')
        acc = self.acc_info

        acc.full_data = dat.Data.i_sense
        acc.full_x = dat.Data.x

        acc.fit_data, acc.fit_x = dat.Transition.get_avg_data(acc.full_x, acc.full_data,
                                                              centers=dat.Transition.get_centers(), return_x=True,
                                                              check_exists=False)

        acc.fit = self._fit(acc.fit_data, acc.fit_x)
        return self.acc_info


if __name__ == '__main__':
    lever_comparison = LeverArmComparison(
        vp_dat=get_dat(1260),
        acc_dat=get_dat(1270)
    )
    vp = lever_comparison.vp_transition_fit()
    acc = lever_comparison.acc_transition_fit()

    for d in [vp, acc]:
        fig = d.fig()
        fig.add_trace(d.data_trace())
        fig.add_trace(d.fit_trace())
        # fig.show()
        d.print_fit_params()

