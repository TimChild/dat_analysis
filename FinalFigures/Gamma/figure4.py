from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from typing import Tuple, Optional, List
from scipy.interpolate import interp1d
import lmfit as lm

import dat_analysis.analysis_tools.nrg
from dat_analysis.dat_analysis.characters import DELTA
from dat_analysis.plotting.plotly.dat_plotting import OneD, TwoD
from dat_analysis.core_util import Data1D, Data2D
from dat_analysis.analysis_tools.nrg import NRG_func_generator
from dat_analysis.analysis_tools.nrg import NRGParams, NrgUtil, get_x_of_half_occ
import dat_analysis.useful_functions as U
from temp import get_avg_entropy_data, get_avg_i_sense_data, _center_func

kb = 0.08617

p1d = OneD(dat=None)
p2d = TwoD(dat=None)

p1d.TEMPLATE = 'simple_white'
p2d.TEMPLATE = 'simple_white'

NRG_OCC_FIT_NAME = 'forced_theta'
# NRG_OCC_FIT_NAME = 'csq_forced_theta'
CSQ_DATNUM = None
# CSQ_DATNUM = 2197

GAMMA_EXPECTED_THETA_PARAMS = NRGParams(
    gamma=23.4352,
    theta=4.5,
    center=78.4,
    amp=0.675,
    lin=0.00121,
    const=7.367,
    lin_occ=0.0001453,
)

MORE_GAMMA_EXPECTED_THETA_PARAMS = NRGParams(
    gamma=59.0125,
    theta=4.5,
    center=21.538,
    amp=0.580,
    lin=0.00109,
    const=7.207,
    lin_occ=0.0001884,
)

MOST_GAMMA_EXPECTED_THETA_PARAMS = NRGParams(
    gamma=109.6544,
    theta=4.5,
    center=-56.723,
    amp=0.481,
    lin=0.00097,
    const=7.214,
    lin_occ=0.00097,
)

EQUAL_GAMMA_THETA_PARAMS = NRGParams(
    gamma=5.7764,
    theta=4.2,
    center=8.825,
    amp=0.784,
    lin=0.00137,
    const=7.101,
    lin_occ=0.0000485,
)

GAMMA_FORCE_FIT_PARAMS = NRGParams(
    gamma=21.5358,
    theta=9.597,
    center=90.782,
    amp=0.667,
    lin=0.00121,
    const=7.378,
    lin_occ=0.0001217,
)

THERMAL_HOT_FIT_PARAMS = NRGParams(
    gamma=0.4732,
    theta=4.672,
    center=7.514,
    amp=0.939,
    lin=0.00152,
    const=7.205,
    lin_occ=-0.0000358,
)

PARAM_DATNUM_DICT = {
    2164: THERMAL_HOT_FIT_PARAMS,
    2121: EQUAL_GAMMA_THETA_PARAMS,
    2167: GAMMA_EXPECTED_THETA_PARAMS,
    2170: MORE_GAMMA_EXPECTED_THETA_PARAMS,
    2213: MOST_GAMMA_EXPECTED_THETA_PARAMS,
}


class Nrg2DPlots:
    """For generating 2D NRG data (as well as plotting)"""
    # temperature = 4.5  # in sweepgate mV so ~100mK
    temperature = 1  # in sweepgate mV so ~100mK
    gs = np.linspace(temperature / 10, temperature * 30, 100)
    g_over_ts = gs / temperature
    ylabel = 'G/T'

    def __init__(self, which_data: str, which_x: str):
        self.which_data = which_data
        self.which_x = which_x

    def _get_title(self):
        if self.which_x == 'sweepgate':
            x_part = '$V_D$'
        elif self.which_x == 'occupation':
            x_part = 'Occupation'
        else:
            raise NotImplementedError
        if self.which_data == 'dndt':
            data_part = 'dN/dT'
        elif self.which_data == 'occupation':
            data_part = 'Occupation'
        else:
            raise NotImplementedError
        return f'NRG {data_part} vs {x_part}'

    def _get_x(self):
        if self.which_x == 'sweepgate':
            return np.linspace(-200, 200, 1001)
        elif self.which_x == 'occupation':
            raise NotImplementedError

    def _get_x_label(self) -> str:
        if self.which_x == 'sweepgate':
            return "$V_D$ (mV)"
        elif self.which_x == 'occupation':
            raise NotImplementedError

    def _data(self) -> Data2D:
        """Generate the 2D data for NRG"""
        nrg_dndt_func = NRG_func_generator(self.which_data)
        x = self._get_x()
        data = np.array([nrg_dndt_func(x, 0, g, self.temperature)
                         for g in self.gs])
        if self.which_data == 'dndt':  # Rescale to cumsum = ln2
            data = data / np.sum(data, axis=-1)[:, None] * np.log(2)

        occupation_func = NRG_func_generator('occupation')
        occupations = np.array([occupation_func(x, 0, g, self.temperature) for g in self.gs])
        # Set 0.5 Occupation to be at x = 0
        centered_data = self._center_2d_data(x, data, occupations)
        # data2d = Data2D(x=x, y=self.g_over_ts, data=data)
        centered_data.y = self.g_over_ts
        centered_data.x = centered_data.x * 0.0001  # Convert back to NRG units
        return centered_data

    @staticmethod
    def _center_2d_data(x, data2d, occupation2d) -> Data2D:
        assert all(a.ndim == 2 for a in [data2d, occupation2d])
        new_data = []
        for data, occupation in zip(data2d, occupation2d):
            idx = U.get_data_index(occupation, 0.5)  # Get values near to occ = 0.5
            interper = interp1d(occupation[idx - 1:idx + 2], x[idx - 1:idx + 2], bounds_error=False,
                                fill_value=(0, 1))  # will include occ = 0.5
            half_x = interper(0.5)
            new_x = x - half_x

            data_interper = interp1d(new_x, data, bounds_error=False, fill_value=(data[0], data[-1]))
            new_data.append(data_interper(x))
        return Data2D(x=x, y=np.arange(len(new_data)), data=np.array(new_data))

    def plot(self, data2d: Optional[Data2D] = None) -> go.Figure:
        if data2d is None:
            data2d = self._data()
        fig = p2d.plot(x=data2d.x, y=data2d.y, data=data2d.data,
                       xlabel=self._get_x_label(), ylabel=self.ylabel, title=self._get_title(),
                       plot_type='heatmap'
                       )
        fig.update_yaxes(type='log')
        return fig

    def run(self, save_name: Optional[str] = None, name_prefix: Optional[str] = None) -> go.Figure:
        data2d = self._data()
        fig = self.plot(data2d)
        if save_name:
            assert name_prefix is not None
            U.save_to_igor_itx(f'{save_name}.itx',
                               xs=[data2d.x],
                               ys=[data2d.y],
                               datas=[data2d.data],
                               names=[f'{name_prefix}_data_2d'],
                               x_labels=['$V_D$ (mV)'],
                               y_labels=['Gamma/T'])
            U.save_to_txt(datas=[data2d.data, data2d.x, data2d.y], names=['dndt', 'x', 'y'], file_path='temp.txt')
        return fig


class Nrg1DPlots:
    def __init__(self, which_plot: str, params_from_fitting: bool = False):
        self.which_plot = which_plot
        self.params_from_fitting = params_from_fitting

    def _get_params_from_dat(self, datnum, fit_which: str = 'i_sense', hot_or_cold: str = 'cold') -> NRGParams:
        dat = get_dat(datnum)
        orig_fit = dat.NrgOcc.get_fit(name=NRG_OCC_FIT_NAME)
        if fit_which == 'i_sense':
            data = get_avg_i_sense_data(dat, CSQ_DATNUM, center_func=_center_func, hot_or_cold=hot_or_cold)
            if hot_or_cold == 'hot':  # Then allow theta to vary, but hold gamma const
                params = U.edit_params(orig_fit.params, ['g', 'theta'], [None, None], vary=[False, True])
            else:
                params = orig_fit.params
            new_fit = dat.NrgOcc.get_fit(calculate_only=True, x=data.x, data=data.data, initial_params=params)
        elif fit_which == 'entropy':
            data = get_avg_entropy_data(dat, center_func=_center_func, csq_datnum=CSQ_DATNUM)
            new_fit = NrgUtil(inital_params=orig_fit).get_fit(data.x, data.data, which_data='dndt')
        else:
            raise NotImplementedError
        params = NRGParams.from_lm_params(new_fit.params)
        return params

    def get_params(self, fit_hot: bool = False) -> NRGParams:
        hot_or_cold = 'hot' if fit_hot else 'cold'
        if self.which_plot == 'weak':
            if self.params_from_fitting:
                params = self._get_params_from_dat(datnum=2164, fit_which='i_sense', hot_or_cold=hot_or_cold)
            else:
                params = THERMAL_HOT_FIT_PARAMS
        elif self.which_plot == 'strong':
            if self.params_from_fitting:
                params = self._get_params_from_dat(datnum=2170, fit_which='i_sense', hot_or_cold=hot_or_cold)
                # params = self._get_params_from_dat(datnum=2213, fit_which='i_sense', hot_or_cold=hot_or_cold)
            else:
                params = GAMMA_EXPECTED_THETA_PARAMS
        else:
            raise NotImplementedError
        return params

    def get_real_data(self, occupation=False) -> Data1D:
        if self.which_plot == 'weak':
            dat = get_dat(2164)
        elif self.which_plot == 'strong':
            dat = get_dat(2170)
            # dat = get_dat(2213)
        else:
            raise NotImplementedError
        if occupation:
            data = get_avg_i_sense_data(dat, CSQ_DATNUM, center_func=_center_func, hot_or_cold='cold')
        else:
            data = get_avg_entropy_data(dat, center_func=_center_func, csq_datnum=CSQ_DATNUM)
        return Data1D(x=data.x, data=data.data)

    def nrg_data(self, params: NRGParams, x: np.ndarray,
                 which_data: str = 'dndt', which_x: str = 'occupation',
                 real_data: Optional[np.ndarray] = None,
                 which_fit_data: str = 'i_sense') -> Data1D:
        """For NRG data only"""
        nrg_generator = NrgUtil(inital_params=params)
        if real_data is not None:
            fit = nrg_generator.get_fit(x=x, data=real_data, which_data=which_fit_data)
            params = NRGParams.from_lm_params(fit.params)
        return nrg_generator.data_from_params(params, x=x, which_data=which_data, which_x=which_x)

    def plot(self, real_data: Optional[Data1D] = None, nrg_data: Optional[Data1D] = None,
             params: Optional[NRGParams] = None,
             x_label: str = 'Occupation',
             ) -> go.Figure:
        if nrg_data is None:
            nrg_data = self.nrg_data(params=params, x=real_data.x,
                                     which_data='dndt', which_x='occupation',
                                     real_data=real_data.data)
        fig = p1d.figure(xlabel=x_label, ylabel=f'{DELTA}I (nA)')
        if real_data is not None:
            fig.add_trace(p1d.trace(data=real_data.data, x=real_data.x, mode='lines+markers', name='Data'))
            # Rescale NRG dndt to match data dndt in amplitude. /len(...) to help account for different x axis
            nrg_data.data = nrg_data.data / (np.nanmax(nrg_data.data) / np.nanmax(real_data.data))
            # nrg_data.data = nrg_data.data / (np.nanmean(nrg_data.data) / np.nanmean(real_data.data))
        fig.add_trace(p1d.trace(data=nrg_data.data, x=nrg_data.x, mode='lines', name='NRG'))
        return fig

    def run(self, save_name: Optional[str] = None, name_prefix: Optional[str] = None,
            occupation_x_axis: bool = False, fit_hot: bool = False) -> go.Figure:
        params = self.get_params(fit_hot=fit_hot)
        real_dndt = self.get_real_data()
        occ_data = self.get_real_data(occupation=True)
        real_dndt.x = occ_data.x
        if occupation_x_axis:
            real_dndt.x = NrgUtil().get_occupation_x(real_dndt.x, params=params)
            x_short = 'occ'
            x_label = 'Occupation'
            which_x = 'occupation'
        else:
            x_short = 'sweepgate'
            x_label = 'Sweepgate (mV)'
            which_x = 'sweepgate'
        nrg_data = self.nrg_data(params=params, x=real_dndt.x,
                                 which_data='dndt', which_x=which_x)

        if not occupation_x_axis:
            real_dndt.x = real_dndt.x / 100  # Convert to real mV
            nrg_data.x = nrg_data.x / 100

        # Switch to occupation as x axis
        fig = self.plot(real_data=real_dndt, nrg_data=nrg_data, x_label=x_label)
        if save_name:
            assert name_prefix is not None
            U.save_to_igor_itx(f'{save_name}.itx', xs=[data.x for data in [real_dndt, nrg_data]],
                               datas=[data.data for data in [real_dndt, nrg_data]],
                               names=[f'{name_prefix}_data_vs_{x_short}', f'{name_prefix}_nrg_vs_{x_short}'],
                               x_labels=[x_label] * 2,
                               y_labels=[f'{DELTA}I (nA)'] * 2)
        return fig


class ScaledDndtPlots:
    def __init__(self, which_plot: str):
        self.which_plot = which_plot

    def _get_datas(self) -> Tuple[List[Data1D], List[float], List[float]]:
        datas = []
        if self.which_plot == 'data':
            gammas = []
            thetas = []
            for k in PARAM_DATNUM_DICT:
                dat = get_dat(k)
                data = get_avg_entropy_data(dat, center_func=_center_func, csq_datnum=CSQ_DATNUM)
                occ_data = get_avg_i_sense_data(dat, CSQ_DATNUM, center_func=_center_func)
                data.x = occ_data.x
                data.x -= dat.NrgOcc.get_x_of_half_occ(fit_name=NRG_OCC_FIT_NAME)
                fit = dat.NrgOcc.get_fit(name=NRG_OCC_FIT_NAME)
                # data.x = data.x/fit.best_values.theta*kb*0.1  # So that difference in lever arm is taken into account
                gammas.append(fit.best_values.g)
                thetas.append(fit.best_values.theta)
                rescale = max(fit.best_values.g, fit.best_values.theta)
                datas.append(Data1D(x=data.x / rescale, data=data.data * rescale))
        elif self.which_plot == 'nrg':
            gts = [0.1, 1, 5, 10, 25]
            theta = 1
            thetas = [theta] * len(gts)
            gammas = list(np.array(gts) * theta)
            for gamma in gammas:
                x_width = max([gamma, theta]) * 15
                pars = NRGParams(gamma=gamma, theta=theta)
                data = NrgUtil().data_from_params(params=pars,
                                                  x=np.linspace(-x_width, x_width, 501), which_data='dndt',
                                                  which_x='sweepgate')
                data.x -= get_x_of_half_occ(params=pars.to_lm_params())
                data.x *= 0.0001  # Convert back to NRG units
                data.data = data.data/np.sum(data.data)/np.mean(np.diff(data.x))*np.log(2)  # Convert to real entropy
                rescale = max(gamma, theta)
                data.x = data.x / rescale
                data.data = data.data * rescale
                datas.append(data)

        else:
            raise NotImplementedError

        return datas, gammas, thetas

    def plot(self, datas: List[Data1D], gammas: List[float], thetas: List[float]) -> go.Figure:
        fig = p1d.figure(xlabel=r'$V_D \text{/max(Gamma, T) (a.u.)}$', ylabel=f'{DELTA}I*max(Gamma, T)',
                         title=f'{DELTA}I vs Sweep gate for various Gamma/T')
        for data, gamma, theta in zip(datas, gammas, thetas):
            gt = gamma / theta
            fig.add_trace(p1d.trace(x=data.x, data=data.data, mode='lines', name=f'{gt:.2f}'))
        fig.update_layout(legend_title='Gamma/Theta')
        if self.which_plot == 'data':
            fig.update_xaxes(range=[-15, 15])
            # fig.update_xaxes(range=[-15/4.5*kb*0.1, 15/4.5*kb*0.1])
        elif self.which_plot == 'nrg':
            fig.update_xaxes(range=[-0.0015, 0.0015])
        else:
            raise NotImplementedError

        return fig

    def run(self, save_name: Optional[str] = None, name_prefix: Optional[str] = None) -> go.Figure:
        datas, gammas, thetas = self._get_datas()
        fig = self.plot(datas, gammas, thetas)
        if save_name:
            U.save_to_igor_itx(f'{save_name}.itx', xs=[data.x for data in datas], datas=[data.data for data in datas],
                               names=[f'{name_prefix}_scaled_dndt_g{g / t:.2f}' for g, t in zip(gammas, thetas)],
                               x_labels=[r'$V_D$/max(Gamma, T)'] * len(datas),
                               y_labels=[f'{DELTA}I*max(Gamma, T)'] * len(datas))

        return fig


if __name__ == '__main__':
    from dat_analysis.dat_object.make_dat import get_dat

    # # NRG dN/dT vs sweep gate (fixed T varying G)
    # Nrg2DPlots(which_data='dndt', which_x='sweepgate').run(save_name='fig4_nrg_dndt_2d', name_prefix='dndt').show()
    # # fig = Nrg2D(which_data='dndt', which_x='sweepgate').run()
    #
    # # NRG Occupation vs sweep gate (fixed T varying G)
    # Nrg2DPlots(which_data='occupation', which_x='sweepgate').run(save_name='fig4_nrg_occ_2d', name_prefix='occ').show()
    # # fig = Nrg2D(which_data='occupation', which_x='sweepgate').run()
    #
    # # Data Vs NRG thermally broadened
    # # Nrg1DPlots(which_plot='weak', params_from_fitting=False).run(save_name='fig4_weak_data_vs_nrg',
    # #                                                                name_prefix='weak',
    # #                                                              occupation_x_axis=True).show()
    # Nrg1DPlots(which_plot='weak', params_from_fitting=True).run(save_name='fig4_weak_data_vs_nrg',
    #                                                             name_prefix='weak',
    #                                                             occupation_x_axis=False,
    #                                                             fit_hot=True).show()
    #
    # # # # Data Vs NRG gamma broadened (with expected Theta)
    # # Nrg1DPlots(which_plot='strong', params_from_fitting=False).run(save_name='fig4_strong_data_vs_nrg',
    # #                                                                name_prefix='strong',
    # #                                                                occupation_x_axis=True).show()
    # Nrg1DPlots(which_plot='strong', params_from_fitting=True).run(save_name='fig4_strong_data_vs_nrg',
    #                                                               name_prefix='strong',
    #                                                               occupation_x_axis=False).show()
    #
    # # Scaled dN/dT Data
    # ScaledDndtPlots(which_plot='data').run(save_name='fig4_scaled_data_dndt', name_prefix='data').show()
    #
    # # # Scaled dN/dT NRG
    # ScaledDndtPlots(which_plot='nrg').run(save_name='fig4_scaled_nrg_dndt', name_prefix='nrg').show()
    #
    # Supplement data
    ScaledDndtPlots(which_plot='data').run(save_name='Sup_scaled_data_comparison_normal',
                                           name_prefix='normal').show()


    # fig.layout.title = None
    # fig.show()
    # fig.write_image(f'Sup_scaled_dndt_alpha.pdf')

