from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from typing import Tuple, Optional, List
from scipy.interpolate import interp1d

from src.analysis_tools.nrg import NRGParams, NrgUtil
from src.characters import DELTA
from src.plotting.plotly.dat_plotting import OneD, TwoD, Data2D, Data1D
from src.analysis_tools.nrg import NRG_func_generator
from src.analysis_tools.nrg import NRGParams
import src.useful_functions as U
from temp import get_avg_entropy_data

p1d = OneD(dat=None)
p2d = TwoD(dat=None)

p1d.TEMPLATE = 'simple_white'
p2d.TEMPLATE = 'simple_white'



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
            return np.linspace(-200, 200, 1001)
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
        if self.which_data == 'dndt':  # Rescale to cumsum = ln2
            data = data / np.sum(data, axis=-1)[:, None] * np.log(2)

        occupation_func = NRG_func_generator('occupation')
        occupations = np.array([occupation_func(x, 0, g, self.temperature) for g in self.gs])
        # Set 0.5 Occupation to be at x = 0
        centered_data = self._center_2d_data(x, data, occupations)
        # data2d = Data2D(x=x, y=self.g_over_ts, data=data)
        centered_data.y = self.g_over_ts
        centered_data.x = centered_data.x*0.0001  # Convert back to NRG units
        return centered_data

    @staticmethod
    def _center_2d_data(x, data2d, occupation2d) -> Data2D:
        assert all(a.ndim == 2 for a in [data2d, occupation2d])
        new_data = []
        for data, occupation in zip(data2d, occupation2d):
            idx = U.get_data_index(occupation, 0.5)  # Get values near to occ = 0.5
            interper = interp1d(occupation[idx-1:idx+2], x[idx-1:idx+2], bounds_error=False,
                                fill_value=(0, 1))  # will include occ = 0.5
            half_x = interper(0.5)
            new_x = x-half_x

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
                               x_labels=['Sweep Gate (mV)'],
                               y_labels=['Gamma/T'])
            U.save_to_txt(datas=[data2d.data, data2d.x, data2d.y], names=['dndt', 'x', 'y'], file_path='temp.txt')
        return fig


class Nrg1DPlots:
    def __init__(self, which_plot: str, params_from_fitting: bool = False):
        self.which_plot = which_plot
        self.params_from_fitting = params_from_fitting

    def _get_params_from_dat(self, datnum, fit_name: str, transition_part: str = 'cold',
                             theta_override: Optional[float] = None) -> NRGParams:
        dat = get_dat(datnum)
        # out = dat.SquareEntropy.get_Outputs(name=fit_name)
        # init_params = NRGParams.from_lm_params(dat.SquareEntropy.get_fit(fit_name=fit_name).params)
        # if theta_override:
        #     init_params.theta = theta_override
        # fit = NrgUtil(inital_params=init_params).get_fit(x=out.x, data=out.transition_part(transition_part),
        #                                                       which_data='i_sense')
        # params = NRGParams.from_lm_params(fit.params)
        # print(f'New Params for Dat{dat.datnum}:\n{params}')
        params = NRGParams.from_lm_params(dat.NrgOcc.get_fit(name='forced_theta').params)
        return params

    def get_params(self) -> NRGParams:
        if self.which_plot == 'weak':
            if self.params_from_fitting:
                # params = self._get_params_from_dat(datnum=2164, fit_name='forced_theta_linear', transition_part='hot',
                #                                    theta_override=4.672)
                params = self._get_params_from_dat(datnum=2164, fit_name=None)
            else:
                params = THERMAL_HOT_FIT_PARAMS
        elif self.which_plot == 'strong':
            if self.params_from_fitting:
                # params = self._get_params_from_dat(datnum=2170, fit_name='forced_theta_linear_non_csq')
                params = self._get_params_from_dat(datnum=2170, fit_name=None)
            else:
                params = GAMMA_EXPECTED_THETA_PARAMS
        else:
            raise NotImplementedError
        return params

    def get_real_data(self) -> Data1D:
        if self.which_plot == 'weak':
            dat = get_dat(2164)
            data = get_avg_entropy_data(dat, center_func=lambda dat: True if dat.Logs.dacs['ESC'] < -250 else False,
                                        overwrite=False)
            # out = dat.SquareEntropy.get_Outputs(name='forced_theta_linear')
            # x = out.x
            # return Data1D(x=x, data=out.average_entropy_signal)
            return Data1D(x=data.x, data=data.data)
        if self.which_plot == 'strong':
            dat = get_dat(2170)
            data = get_avg_entropy_data(dat, center_func=lambda dat: True if dat.Logs.dacs['ESC'] < -250 else False,
                                        overwrite=False)
            # out = dat.SquareEntropy.get_Outputs(name='forced_theta_linear')
            # x = out.x
            # return Data1D(x=x, data=out.average_entropy_signal)
            return Data1D(x=data.x, data=data.data)
        else:
            raise NotImplementedError

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
        real_dndt.x = NrgUtil().get_occupation_x(real_dndt.x, params=params)
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
                datas.append(Data1D(x=out.x / rescale, data=out.average_entropy_signal * rescale))
        elif self.which_plot == 'nrg':
            gts = [0.1, 1, 5, 10, 25]
            theta = 5
            thetas = [theta] * len(gts)
            gammas = list(np.array(gts) * theta)
            for gamma in gammas:
                x_width = max([gamma, theta]) * 15
                data = NrgUtil().data_from_params(params=NRGParams(gamma=gamma, theta=theta),
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
    from src.dat_object.make_dat import get_dat

    # # NRG dN/dT vs sweep gate (fixed T varying G)
    # Nrg2DPlots(which_data='dndt', which_x='sweepgate').run(save_name='fig4_nrg_dndt_2d', name_prefix='dndt').show()
    # # fig = Nrg2D(which_data='dndt', which_x='sweepgate').run()
    #
    # # NRG Occupation vs sweep gate (fixed T varying G)
    # Nrg2DPlots(which_data='occupation', which_x='sweepgate').run(save_name='fig4_nrg_occ_2d', name_prefix='occ').show()
    # # fig = Nrg2D(which_data='occupation', which_x='sweepgate').run()

    # TODO: Need to figure out why this isn't fitting the data properly
    # TODO: Plot these with Occupation on x axis and sweepgate on x axis and decide which is better for figure 5
    # Data Vs NRG thermally broadened
    Nrg1DPlots(which_plot='weak', params_from_fitting=False).run(save_name='fig4_weak_data_vs_nrg',
                                                                   name_prefix='weak').show()
    Nrg1DPlots(which_plot='weak', params_from_fitting=True).run(save_name='fig4_weak_data_vs_nrg',
                                                                   name_prefix='weak').show()

    # # Data Vs NRG gamma broadened (with expected Theta)
    Nrg1DPlots(which_plot='strong', params_from_fitting=False).run(save_name='fig4_strong_data_vs_nrg',
                                                                   name_prefix='strong').show()
    Nrg1DPlots(which_plot='strong', params_from_fitting=True).run(save_name='fig4_strong_data_vs_nrg',
                                                                   name_prefix='strong').show()

    # TODO: Need to center the data properly using Occ = 0.5 as center
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
