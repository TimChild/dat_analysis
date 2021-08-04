from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from typing import Tuple, Optional, List, TYPE_CHECKING
from scipy.interpolate import interp1d
import lmfit as lm

import src.analysis_tools.nrg
from src.analysis_tools.nrg import NRGParams, NrgUtil
from src.characters import DELTA
from src.plotting.plotly.dat_plotting import OneD, TwoD, Data2D, Data1D
from src.analysis_tools.nrg import NRG_func_generator
from src.analysis_tools.nrg import NRGParams, NrgUtil, get_x_of_half_occ
import src.useful_functions as U
from temp import get_avg_entropy_data, get_avg_i_sense_data, _center_func

if TYPE_CHECKING:
    from src.dat_object.make_dat import DatHDF

p1d = OneD(dat=None)
p2d = TwoD(dat=None)

p1d.TEMPLATE = 'simple_white'
p2d.TEMPLATE = 'simple_white'

NRG_OCC_FIT_NAME = 'csq_forced_theta'
CSQ_DATNUM = 2197


class StrongGammaNoise:
    def __init__(self, dat: DatHDF):
        self.dat = dat

        # The averaged data which does show some signal, but quite small given the number of repeats and duration
        self.avg_data = get_avg_entropy_data(self.dat, lambda _: False, CSQ_DATNUM)
        self.avg_data.x = self.avg_data.x / 100  # Convert to real mV

        # Single trace of data to show how little signal for a single scan
        self.data = Data1D(data=self.dat.SquareEntropy.entropy_signal[0], x=self.dat.SquareEntropy.x)
        self.data.x = self.data.x / 100  # Convert to real mV

    def print_repeats_and_duration(self):
        repeats = self.dat.Data.i_sense.shape[0]
        duration = self.dat.Logs.time_elapsed
        print(f'Averaged Entropy data for Dat{self.dat.datnum} is for {repeats} repeats taken over {duration / 60:.1f} minutes')

    def print_single_sweep_duration(self):
        single_duration = self.dat.Logs.time_elapsed / self.dat.Data.i_sense.shape[0]
        print(f'Single sweep Entropy data for Dat{self.dat.datnum} was roughly {single_duration:.1f} seconds long')

    def save_to_itx(self):
        U.save_to_igor_itx('sup_fig1_strong_gamma_noise.itx',
                           xs=[self.data.x],
                           datas=[self.data.data],
                           names=['single_strong_dndt'],
                           x_labels=['Sweepgate (mV)'],
                           y_labels=['Delta I (nA)']
                           )

    def plot(self) -> go.Figure:
        # Plot single data sweep
        fig = p1d.figure(xlabel='Sweepgate (mV)', ylabel=f'{DELTA}I (nA)',
                         title=f'Dat{dat.datnum}: Single entropy sweep showing measurement is small compared to noise')
        fig.add_trace(p1d.trace(data=self.data.data, x=self.data.x, mode='markers'))
        return fig


if __name__ == '__main__':
    from src.dat_object.make_dat import get_dat, get_dats

    # Strongly coupled gamma measurement being close to noise floor
    dat = get_dat(2213)

    strong_gamma_noise = StrongGammaNoise(dat=dat)
    strong_gamma_noise.print_repeats_and_duration()
    strong_gamma_noise.print_single_sweep_duration()
    strong_gamma_noise.save_to_itx()
    strong_gamma_noise.plot().show()





