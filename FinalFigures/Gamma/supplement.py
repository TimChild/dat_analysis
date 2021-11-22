from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from typing import Optional, TYPE_CHECKING
from progressbar import progressbar

from dat_analysis.analysis_tools.general_fitting import FitInfo
from dat_analysis.dat_analysis.characters import DELTA
from dat_analysis.plotting.plotly.dat_plotting import OneD, TwoD
from dat_analysis.core_util import Data1D
from dat_analysis.analysis_tools.nrg import NRGParams, NrgUtil
import dat_analysis.useful_functions as U
from temp import get_avg_entropy_data, get_avg_i_sense_data, get_linear_theta, get_initial_params, get_2d_data

if TYPE_CHECKING:
    from dat_analysis.dat_object.make_dat import DatHDF

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
        print(
            f'Averaged Entropy data for Dat{self.dat.datnum} is for {repeats} repeats taken over {duration / 60:.1f} minutes')

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


# if __name__ == '__main__':
#     from dat_analysis.dat_object.make_dat import get_dat, get_dats
#
#     # Strongly coupled gamma measurement being close to noise floor
#     dat = get_dat(2213)
#
#     strong_gamma_noise = StrongGammaNoise(dat=dat)
#     strong_gamma_noise.print_repeats_and_duration()
#     strong_gamma_noise.print_single_sweep_duration()
#     strong_gamma_noise.save_to_itx()
#     strong_gamma_noise.plot().show()
#
#     fig = strong_gamma_noise.plot()
#     fig.write_image('Sup1.pdf')


def run_dat_init(dat: DatHDF, overwrite: bool = False) -> FitInfo:
    def lin_theta_slope_name() -> str:
        if 5000 < dat.datnum < 6000:
            return 'dats5000+'
        elif dat.datnum > 7000:
            return 'dats7000+'
        elif 1500 < dat.datnum < 2500:
            return 'normal'
        else:
            raise NotImplementedError

    lin_name = lin_theta_slope_name()

    avg_data = get_avg_i_sense_data(dat, None, center_func=lambda _: False)
    pars = get_initial_params(avg_data, which='nrg')

    theta = get_linear_theta(dat, which_params=lin_name)
    if 45 < dat.Logs.temps.mc * 1000 < 55:
        theta = theta / 2
    pars['theta'].value = theta
    pars['theta'].vary = False
    pars['g'].value = 5
    pars['g'].max = theta * 50  # limit of NRG data
    pars['g'].min = theta / 10000  # limit of NRG data

    # if abs((x := avg_data.x)[-1] - x[0]) > 1500:  # If it's a wider scan, only fit over middle 1500
    #     cond = np.where(np.logical_and(x > -750, x < 750))
    #     avg_data.x, avg_data.data = x[cond], avg_data.data[cond]

    fit = dat.NrgOcc.get_fit(which='avg', name='forced_theta',
                             initial_params=pars,
                             data=avg_data.data, x=avg_data.x,
                             calculate_only=False, check_exists=False, overwrite=overwrite)
    return fit


def get_dndt_vs_occ(dat: DatHDF) -> Data1D:
    data = get_avg_entropy_data(dat, lambda _: False, None)
    fit = dat.NrgOcc.get_fit(name='forced_theta')
    nrg = NrgUtil(NRGParams.from_lm_params(fit.params))
    data.x = nrg.get_occupation_x(data.x)
    data.data = data.data / np.nanmax(U.decimate(data.data, measure_freq=dat.Logs.measure_freq, numpnts=100))
    return data


def get_expected_dndt_vs_occ(dat: DatHDF) -> Data1D:
    fit = dat.NrgOcc.get_fit(name='forced_theta')
    nrg = NrgUtil(NRGParams.from_lm_params(fit.params))
    real_data = get_avg_entropy_data(dat, lambda _: False, None)
    data = nrg.data_from_params(x=real_data.x, which_data='dndt', which_x='occupation')
    data.data = data.data / np.max(data.data)
    return data


def plot_occupation_fit(dat: DatHDF) -> go.Figure:
    data = get_avg_i_sense_data(dat, None, lambda _: False, overwrite=False, hot_or_cold='cold')
    fit = dat.NrgOcc.get_fit(name='forced_theta')
    fig = p1d.figure(xlabel='Sweepgate', ylabel='Delta I', title=f'Dat{dat.datnum}: Checking NRG Transition Fit')
    fig.add_trace(p1d.trace(x=data.x, data=data.data, mode='lines', name='Data'))
    fig.add_trace(p1d.trace(x=data.x, data=fit.eval_fit(data.x), mode='lines', name='Fit'))
    return fig


def plot_csq_trace(dat: DatHDF, cutoff: Optional[float] = None) -> Data1D:
    plotter = OneD(dat=dat)
    plotter.TEMPLATE = 'simple_white'
    fig = plotter.figure(xlabel='CSQ Gate (mV)', ylabel='Current (nA)', title='CS current vs CSQ gate')
    x = dat.Data.x
    data = dat.Data.i_sense
    if cutoff:
        upper_lim = U.get_data_index(x, cutoff)
        x, data = x[:upper_lim], data[:upper_lim]
    fig.add_trace(plotter.trace(data=data, x=x))
    fig.show()
    return Data1D(x=x, data=data)


if __name__ == '__main__':
    from dat_analysis.dat_object.make_dat import get_dat, get_dats

    dats = get_dats([5774, 5775, 5777])  # Gamma broadened entropy with 300, 100, 50uV CS bias
    names = ['Usual settings (300uV bias)', '100uV CS bias', '50uV CS bias']
    igor_names = ['gamma_300uv_csbias', 'gamma_100uv_csbias', 'gamma_50uv_csbias']
    filename = f'sup_gamma_varying_csbias.itx'

    # dats = get_dats([7356, 7428, 7845])  # Normal, 1.5 instead of 2.5nA heating, 50mK (and corresponding heat) vs 100mK
    # names = ['Usual settings', 'Reduced heating bias', 'Lower Fridge Temp']
    # igor_names = ['gamma_100mk_normalheat', 'gamma_100mk_lowheat', 'gamma_50mk_normalheat']
    # filename = f'sup_gamma_varying_heat.itx'

    for dat in progressbar(dats):
        run_dat_init(dat, overwrite=False)

    # for dat in dats:
    #     plot_occupation_fit(dat).show()

    save_infos = []
    for dat, name, igor_name in zip(dats, names, igor_names):
        fig = p1d.figure(xlabel='Occupation', ylabel=f'Scaled {DELTA}I (a.u.)', title=name)
        real_data = get_dndt_vs_occ(dat)
        fit_data = get_expected_dndt_vs_occ(dat)
        for data, trace_name in zip([real_data, fit_data], ['Data', 'Fit']):
            fig.add_trace(p1d.trace(data=data.data, x=data.x, mode='lines', name=trace_name))
        # fig.show()
        save_infos.append(U.IgorSaveInfo(
            x=real_data.x, data=real_data.data, name=igor_name,
            x_label='Occupation', y_label=f'Scaled {DELTA}I (a.u.)'
        ))
        save_infos.append(U.IgorSaveInfo(
            x=real_data.x, data=fit_data.data, name=f'{igor_name}_fit',
            x_label='Occupation', y_label=f'Scaled {DELTA}I (a.u.)'
        ))

    U.save_multiple_save_info_to_itx(filename, save_infos)



    ############################# Weak signal in Gamma broadened
    save_infos = []
    dat = get_dat(2213)
    data2d = get_2d_data(dat, 'entropy')
    data2d.x = data2d.x/100  # Convert to real mV

    data = Data1D(data=np.nanmean(data2d.data, axis=0), x=data2d.x)
    data_single = Data1D(data=data2d.data[0], x=data2d.x)

    fig = p1d.plot(data.data, x=data.x, xlabel='V_D (mV)', ylabel='Delta I (nA)', mode='lines', trace_name='Avg Data')
    fig.add_trace(p1d.trace(data_single.data, x=data_single.x, mode='markers', name='Single sweep'))
    fig.show()

    fig = p2d.plot(data2d.data, data2d.x, data2d.y, 'V_D (mV)', 'Repeats')
    fig.show()

    save_infos.append(U.IgorSaveInfo(
        x=data2d.x, data=data2d.data, name='sup_weak_signal_2d', x_label='V_D (mV)', y_label='Repeats', y=data2d.y
    ))
    for d, name_append in zip([data, data_single], ['avg', 'single']):
        save_infos.append(U.IgorSaveInfo(
            x=d.x, data=d.data, name=f'sup_weak_signal_{name_append}', x_label='V_D (mV)', y_label='Delta I (nA)'
        )
        )

    U.save_multiple_save_info_to_itx(file_path='Sup_weak_signal.itx', save_infos=save_infos)

    ################  CSQ Map
    dat = get_dat(2197)
    data = plot_csq_trace(dat, cutoff=-165)
    U.save_to_igor_itx(file_path='Sup_csq_map.itx',
                       xs = [data.x],
                       datas= [data.data],
                       names= ['csq_map'],
                       x_labels=['V_C (mV)'],
                       y_labels=['I_{CS}'],
                       )

