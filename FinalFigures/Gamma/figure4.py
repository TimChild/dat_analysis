from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.signal import savgol_filter
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Dict, Callable, Optional, Union, List
import lmfit as lm
from dataclasses import dataclass
import logging

from FinalFigures.Gamma.plots import dndt_2d
from src.Characters import DELTA
from src.AnalysisTools.fitting import FitInfo, calculate_fit
from src.Dash.DatPlotting import OneD, TwoD
from Analysis.Feb2021.NRG_comparison import NRG_func_generator, NRGData
from src.AnalysisTools.nrg import NRGParams
import src.UsefulFunctions as U

p1d = OneD(dat=None)
p2d = TwoD(dat=None)

p1d.TEMPLATE = 'simple_white'
p2d.TEMPLATE = 'simple_white'


# @dataclass
# class NRGParams:
#     gamma: float
#     theta: float
#     center: Optional[float] = 0
#     amp: Optional[float] = 1
#     lin: Optional[float] = 0
#     const: Optional[float] = 0
#     lin_occ: Optional[float] = 0
#
#     def to_lm_params(self, which_data: str = 'i_sense', x: Optional[np.ndarray] = None,
#                      data: Optional[np.ndarray] = None) -> lm.Parameters:
#         if x is None:
#             x = [-1000, 1000]
#         if data is None:
#             data = [-10, 10]
#
#         lm_pars = lm.Parameters()
#         lm_pars.add_many(
#             ('mid', self.center, True, np.nanmin(x), np.nanmax(x), None, None),
#             ('theta', self.theta, False, 0.5, 200, None, None),
#             ('g', self.gamma, True, 0.2, 4000, None, None),
#         )
#
#         if which_data == 'i_sense':  # then add other necessary parts
#             lm_pars.add_many(
#                 ('amp', self.amp, True, 0.1, 3, None, None),
#                 ('lin', self.lin, True, 0, 0.005, None, None),
#                 ('occ_lin', self.lin_occ, True, -0.0003, 0.0003, None, None),
#                 ('const', self.const, True, np.nanmin(data), np.nanmax(data), None, None),
#             )
#         return lm_pars
#
#     @classmethod
#     def from_lm_params(cls, params: lm.Parameters) -> NRGParams:
#         d = {}
#         for k1, k2 in zip(['gamma', 'theta', 'center', 'amp', 'lin', 'const', 'lin_occ'],
#                           ['g', 'theta', 'mid', 'amp', 'lin', 'const', 'occ_lin']):
#             par = params.get(k2, None)
#             if par is not None:
#                 v = par.value
#             elif k1 == 'gamma':
#                 v = 0  # This will cause issues if not set somewhere else, but no better choice here.
#             else:
#                 v = 0 if k1 != 'amp' else 1  # Most things should default to zero except for amp
#             d[k1] = v
#         return cls(**d)


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


@dataclass
class Data2D:
    x: np.ndarray
    y: np.ndarray
    data: np.ndarray


@dataclass
class Data1D:
    x: np.ndarray
    data: np.ndarray


class NrgGenerator:
    """For generating 1D NRG Data"""
    """
    What do I want:
    In all cases, the x-axis of data may want to be Sweepgate or Occupation
    
    1. NRG data to fit dat (might need to give some initial params)
        a. Might want dN/dT, Occupation, etc
    2. NRG data given parameters
    """

    nrg = NRGData.from_mat()

    def __init__(self, inital_params: Optional[NRGParams] = None):
        """
        Args:
            inital_params (): For running later fits
        """
        self.inital_params = inital_params if inital_params else NRGParams(gamma=1, theta=1)

    def get_occupation_x(self, orig_x: np.ndarray, params: NRGParams) -> np.ndarray:
        occupation = self.data_from_params(params=params, x=orig_x,
                                           which_data='occupation', which_x='sweepgate').data
        # TODO: Might need to think about what happens when occupation is 0 or 1 in the tails
        # interp_range = np.where(np.logical_and(occupation < 0.999, occupation > 0.001))
        #
        # interp_data = occupation[interp_range]
        # interp_x = orig_x[interp_range]
        #
        # interper = interp1d(x=interp_x, y=interp_data, assume_sorted=True, bounds_error=False)
        #
        # occ_x = interper(orig_x)
        return occupation

    def data_from_params(self, params: Optional[NRGParams] = None,
                         x: Optional[np.ndarray] = None,
                         which_data: str = 'dndt',
                         which_x: str = 'sweepgate') -> Data1D:
        """Return 1D NRG data using parameters only"""
        if x is None:
            x = np.linspace(-1000, 1000, 1001)
        if params is None:
            params = self.inital_params

        nrg_func = NRG_func_generator(which_data)
        nrg_data = nrg_func(x=x, mid=params.center, g=params.gamma, theta=params.theta,
                            amp=params.amp, lin=params.lin, const=params.const, occ_lin=params.lin_occ)
        if which_x == 'occupation':
            x = self.get_occupation_x(x, params)
        return Data1D(x=x, data=nrg_data)

    def data_from_fit(self, x: np.ndarray, data: np.ndarray,
                      initial_params: Optional[Union[NRGParams, lm.Parameters]] = None,
                      which_data: str = 'dndt',
                      which_x: str = 'sweepgate',
                      which_fit_data: str = 'i_sense',
                      ) -> Data1D:
        fit = self.get_fit(x=x, data=data, initial_params=initial_params, which_data=which_fit_data)
        params = NRGParams.from_lm_params(fit.params)
        return self.data_from_params(params, x=x, which_data=which_data, which_x=which_x)

    def get_fit(self, x: np.ndarray, data: np.ndarray,
                initial_params: Optional[Union[NRGParams, lm.Parameters]] = None,
                which_data: str = 'i_sense'
                ) -> FitInfo:
        if initial_params is None:
            initial_params = self.inital_params

        if isinstance(initial_params, lm.Parameters):
            lm_pars = initial_params
        else:
            lm_pars = initial_params.to_lm_params(which_data=which_data, x=x, data=data)

        fit = calculate_fit(x=x, data=data, params=lm_pars, func=NRG_func_generator(which=which_data),
                            method='powell')
        return fit


class Nrg2DPlots:
    """For generating 2D NRG data (as well as plotting)"""
    temperature = 4.5  # in sweepgate mV so ~100mK
    gs = np.linspace(temperature / 10, temperature * 30, 100)
    g_over_ts = gs / temperature
    ylabel = 'G/T'

    def __init__(self, which_data: str, which_x: str):
        self.which_data = which_data
        self.which_x = which_x

    def _get_title(self):
        if self.which_x == 'sweepgate':
            x_part = 'Sweep Gate'
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
            return np.linspace(-800, 800, 201)
        elif self.which_x == 'occupation':
            raise NotImplementedError

    def _get_x_label(self) -> str:
        if self.which_x == 'sweepgate':
            return "Sweep Gate (mV)"
        elif self.which_x == 'occupation':
            raise NotImplementedError

    def _data(self) -> Data2D:
        """Generate the 2D data for NRG"""
        nrg_dndt_func = NRG_func_generator(self.which_data)
        x = self._get_x()
        data = np.array([nrg_dndt_func(x, 0, g, self.temperature)
                         for g in self.gs])
        if self.which_data == 'dndt':  # Rescale to cumsum = 1
            data = data / np.sum(data, axis=-1)[:, None]
        data2d = Data2D(x=x, y=self.g_over_ts, data=data)
        return data2d

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
                               x_labels=['Sweep Gate (mV)'],
                               y_labels=['Gamma/T'])
        # TODO: Save data here
        # U.save_to_igor_itx()
        return fig


class Nrg1DPlots:
    def __init__(self, which_plot: str, params_from_fitting: bool = False):
        self.which_plot = which_plot
        self.params_from_fitting = params_from_fitting

    def _get_params_from_dat(self, datnum, fit_name: str, transition_part: str = 'cold',
                             theta_override: Optional[float] = None) -> NRGParams:
        dat = get_dat(datnum)
        out = dat.SquareEntropy.get_Outputs(name=fit_name)
        init_params = NRGParams.from_lm_params(dat.SquareEntropy.get_fit(fit_name=fit_name).params)
        if theta_override:
            init_params.theta = theta_override
        fit = NrgGenerator(inital_params=init_params).get_fit(x=out.x, data=out.transition_part(transition_part),
                                                              which_data='i_sense')
        params = NRGParams.from_lm_params(fit.params)
        print(f'New Params for Dat{dat.datnum}:\n{params}')
        return params

    def get_params(self) -> NRGParams:
        if self.which_plot == 'weak':
            if self.params_from_fitting:
                params = self._get_params_from_dat(datnum=2164, fit_name='forced_theta_linear', transition_part='hot',
                                                   theta_override=4.672)
            else:
                params = THERMAL_HOT_FIT_PARAMS
        elif self.which_plot == 'strong':
            if self.params_from_fitting:
                params = self._get_params_from_dat(datnum=2170, fit_name='forced_theta_linear_non_csq')
            else:
                params = GAMMA_EXPECTED_THETA_PARAMS
        else:
            raise NotImplementedError
        return params

    def get_real_data(self) -> Data1D:
        if self.which_plot == 'weak':
            dat = get_dat(2164)
            out = dat.SquareEntropy.get_Outputs(name='forced_theta_linear')
            x = out.x
            return Data1D(x=x, data=out.average_entropy_signal)
        if self.which_plot == 'strong':
            dat = get_dat(2170)
            out = dat.SquareEntropy.get_Outputs(name='forced_theta_linear')
            x = out.x
            return Data1D(x=x, data=out.average_entropy_signal)
        else:
            raise NotImplementedError

    def nrg_data(self, params: NRGParams, x: np.ndarray,
                 which_data: str = 'dndt', which_x: str = 'occupation',
                 real_data: Optional[np.ndarray] = None,
                 which_fit_data: str = 'i_sense') -> Data1D:
        nrg_generator = NrgGenerator(inital_params=params)
        if real_data is not None:
            fit = nrg_generator.get_fit(x=x, data=real_data, which_data=which_fit_data)
            params = NRGParams.from_lm_params(fit.params)
        return nrg_generator.data_from_params(params, x=x, which_data=which_data, which_x=which_x)

    def plot(self, real_data: Optional[Data1D] = None, nrg_data: Optional[Data1D] = None,
             params: Optional[NRGParams] = None
             ) -> go.Figure:
        if nrg_data is None:
            nrg_data = self.nrg_data(params=params, x=real_data.x,
                                     which_data='dndt', which_x='occupation',
                                     real_data=real_data.data)
        fig = p1d.figure(xlabel='Occupation', ylabel=f'{DELTA}I (nA)')
        if real_data is not None:
            fig.add_trace(p1d.trace(data=real_data.data, x=real_data.x, mode='lines+markers', name='Data'))
            # Rescale NRG dndt to match data dndt in amplitude. /len(...) to help account for different x axis
            nrg_data.data = nrg_data.data / (np.nanmax(nrg_data.data) / np.nanmax(real_data.data))
            # nrg_data.data = nrg_data.data / (np.nanmean(nrg_data.data) / np.nanmean(real_data.data))
        fig.add_trace(p1d.trace(data=nrg_data.data, x=nrg_data.x, mode='lines', name='NRG'))
        return fig

    def run(self, save_name: Optional[str] = None, name_prefix: Optional[str] = None) -> go.Figure:
        params = self.get_params()
        real_dndt = self.get_real_data()
        nrg_data = self.nrg_data(params=params, x=real_dndt.x,
                                 which_data='dndt', which_x='occupation')
        # Switch to occupation as x axis
        real_dndt.x = NrgGenerator().get_occupation_x(real_dndt.x, params=params)
        fig = self.plot(real_data=real_dndt, nrg_data=nrg_data)
        if save_name:
            assert name_prefix is not None
            U.save_to_igor_itx(f'{save_name}.itx', xs=[data.x for data in [real_dndt, nrg_data]],
                               datas=[data.data for data in [real_dndt, nrg_data]],
                               names=[f'{name_prefix}_data_vs_occ', f'{name_prefix}_nrg_vs_occ'],
                               x_labels=['Occupation']*2,
                               y_labels=[f'{DELTA}I (nA)']*2)
        return fig


class ScaledDndtPlots:
    def __init__(self, which_plot: str):
        self.which_plot = which_plot

    def _get_datas(self) -> Tuple[List[Data1D], List[float], List[float]]:
        datas = []
        if self.which_plot == 'data':
            fit_name = 'forced_theta_linear_non_csq'
            gammas = []
            thetas = []
            for k in PARAM_DATNUM_DICT:
                dat = get_dat(k)
                out = dat.SquareEntropy.get_Outputs(name=fit_name)
                p = PARAM_DATNUM_DICT[k]
                gammas.append(p.gamma)
                thetas.append(p.theta)
                rescale = max(p.gamma, p.theta)
                datas.append(Data1D(x=out.x/rescale, data=out.average_entropy_signal*rescale))
        elif self.which_plot == 'nrg':
            gts = [0.1, 1, 5, 10, 25]
            theta = 5
            thetas = [5] * len(gts)
            gammas = list(np.array(gts) * theta)
            for gamma in gammas:
                x_width = max([gamma, theta]) * 15
                data = NrgGenerator().data_from_params(params=NRGParams(gamma=gamma, theta=theta),
                                                       x=np.linspace(-x_width, x_width, 501), which_data='dndt',
                                                       which_x='sweepgate')
                data.data = data.data / np.sum(data.data*x_width)
                rescale = max(gamma, theta)
                data.x = data.x/rescale
                data.data = data.data*rescale
                datas.append(data)

        else:
            raise NotImplementedError

        return datas, gammas, thetas

    def plot(self, datas: List[Data1D], gammas: List[float], thetas: List[float]) -> go.Figure:
        fig = p1d.figure(xlabel='Sweep Gate/max(Gamma, T)', ylabel=f'{DELTA}I*max(Gamma, T)',
                         title=f'{DELTA}I vs Sweep gate for various Gamma/T')
        for data, gamma, theta in zip(datas, gammas, thetas):
            gt = gamma/theta
            fig.add_trace(p1d.trace(x=data.x, data=data.data, mode='lines', name=f'{gt:.2f}'))
        fig.update_layout(legend_title='Gamma/Theta')
        fig.update_xaxes(range=[-10, 10])
        return fig

    def run(self, save_name: Optional[str] = None, name_prefix: Optional[str] = None) -> go.Figure:
        datas, gammas, thetas = self._get_datas()
        fig = self.plot(datas, gammas, thetas)
        if save_name:
            U.save_to_igor_itx(f'{save_name}.itx', xs=[data.x for data in datas], datas=[data.data for data in datas],
                               names=[f'{name_prefix}_scaled_dndt_g{g/t:.2f}' for g, t  in zip(gammas, thetas)],
                               x_labels=['Sweep Gate/max(Gamma, T)']*len(datas),
                               y_labels=[f'{DELTA}I*max(Gamma, T)']*len(datas))

        return fig


if __name__ == '__main__':
    from src.DatObject.Make_Dat import get_dats, get_dat

    # NRG dN/dT vs sweep gate (fixed T varying G)
    Nrg2DPlots(which_data='dndt', which_x='sweepgate').run(save_name='fig4_nrg_dndt_2d', name_prefix='dndt').show()
    # fig = Nrg2D(which_data='dndt', which_x='sweepgate').run()

    # NRG Occupation vs sweep gate (fixed T varying G)
    Nrg2DPlots(which_data='occupation', which_x='sweepgate').run(save_name='fig4_nrg_occ_2d', name_prefix='occ').show()
    # fig = Nrg2D(which_data='occupation', which_x='sweepgate').run()

    # # Data Vs NRG thermally broadened
    Nrg1DPlots(which_plot='weak', params_from_fitting=False).run(save_name='fig4_weak_data_vs_nrg',
                                                                   name_prefix='weak').show()
    # Nrg1DPlots(which_plot='weak', params_from_fitting=True).run(save_name='fig4_weak_data_vs_nrg',
    #                                                                name_prefix='weak').show()

    # # Data Vs NRG gamma broadened (with expected Theta)
    Nrg1DPlots(which_plot='strong', params_from_fitting=False).run(save_name='fig4_strong_data_vs_nrg',
                                                                   name_prefix='strong').show()
    Nrg1DPlots(which_plot='strong', params_from_fitting=True).run(save_name='fig4_strong_data_vs_nrg',
                                                                   name_prefix='strong').show()

    # # Scaled dN/dT Data
    ScaledDndtPlots(which_plot='data').run(save_name='fig4_scaled_data_dndt', name_prefix='data').show()

    # # Scaled dN/dT NRG
    ScaledDndtPlots(which_plot='nrg').run(save_name='fig4_scaled_nrg_dndt', name_prefix='nrg').show()

    ##################
    #
    # nrg = NRGData.from_mat()
    # nrg_dndt = nrg.dndt
    # nrg_occ = nrg.occupation
    #
    # ts = nrg.ts
    # ts2d = np.tile(ts, (nrg_occ.shape[-1], 1)).T
    #
    # vs_occ_interpers = [interp1d(x=occ, y=dndt, bounds_error=False, fill_value=np.nan) for occ, dndt in zip(nrg_occ, nrg_dndt)]
    #
    # new_occ = np.linspace(0, 1, 200)
    #
    # new_dndt = np.array([interp(x=new_occ) for interp in vs_occ_interpers])
    #
    # fig = p2d.plot(new_dndt, x=new_occ, y=0.001/ts,
    #                xlabel='Occupation', ylabel='G/T',
    #                title='NRG dN/dT vs Occ')
    # fig.update_yaxes(type='log')
    # fig.update_layout(yaxis=dict(range=[np.log10(v) for v in (0.1, 30)]))
    # fig.show()
    #
    #
    # fig = p2d.plot(nrg_dndt, x=nrg.ens, y=0.001/ts,
    #                xlabel='Energy', ylabel='G/T',
    #                title='NRG dN/dT vs Energy')
    # fig.update_yaxes(type='log')
    # fig.update_layout(yaxis=dict(range=[np.log10(v) for v in (0.1, 30)],),
    #                   xaxis=dict(range=[-0.03, 0.03]))
    # fig.show()
    #
    #
    # fig = p2d.plot(nrg_occ, x=nrg.ens, y=0.001/ts,
    #                xlabel='Energy', ylabel='G/T',
    #                title='NRG Occupation vs Energy')
    # fig.update_yaxes(type='log')
    # fig.update_layout(yaxis=dict(range=[np.log10(v) for v in (0.1, 30)],),
    #                   xaxis=dict(range=[-0.03, 0.03]))
    # fig.show()
    #
    #
