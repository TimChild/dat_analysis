from __future__ import annotations
from typing import Callable
from scipy.interpolate import RectBivariateSpline, interp1d
import scipy.io
from dataclasses import dataclass
import numpy as np
import plotly.io as pio
from functools import lru_cache
import lmfit as lm
import pandas as pd
from itertools import product
import time
import logging

from src.DatObject.Make_Dat import get_dat, DatHDF, get_dats
from src.Dash.DatPlotting import OneD, TwoD
from src.AnalysisTools.fitting import calculate_fit, get_data_in_range
from Analysis.Feb2021.common import do_entropy_calc, data_from_output
import src.UsefulFunctions as U

pio.renderers.default = "browser"
logger = logging.getLogger(__name__)


# ### Just here as a hint -- taken from fitting page
# def NRG_fitter() -> Callable:
#     NRG = scipy.io.loadmat('NRGResults.mat')
#     occ = NRG["Occupation_mat"]
#     ens = np.reshape(NRG["Ens"], 401)
#     ts = np.reshape(NRG["Ts"], 70)
#     ens = np.flip(ens)
#     occ = np.flip(occ, 0)
#     interp = RectBivariateSpline(ens, np.log10(ts), occ, kx=1, ky=1)
#
#     def interpNRG(x, logt, dx=1, amp=1, center=0, lin=0, const=0):
#         ens = np.multiply(np.add(x, center), dx)
#         curr = [interp(en, logt)[0][0] for en in ens]
#         scaled_current = np.multiply(curr, amp)
#         scaled_current += const + np.multiply(lin, x)
#         return scaled_current
#
#     return interpNRG


def NRG_func_generator(which='i_sense') -> Callable:
    """
    Use this to generate the fitting function (i.e. to generate the equivalent of i_sense().
    It just makes sense in this case to do the setting up of the NRG data within a generator function.

    Note: RectBivariateSpline is a more efficient interp2d function for input data where x and y form a grid
    Args:

    Returns:

    """
    nrg = NRGData.from_mat()
    nrg_gamma = 0.001
    x_ratio = -1000  # Some arbitrary ratio to make NRG equivalent to i_sense/digamma x scaling (e.g. to get same theta)
    if which == 'i_sense':
        z = 1 - nrg.occupation
    elif which == 'occupation':
        z = nrg.occupation
    elif which == 'dndt':
        z = nrg.dndt
    elif which == 'entropy':
        z = nrg.entropy
    elif which == 'int_dndt':
        z = nrg.int_dndt
    elif which == 'conductance':
        z = nrg.conductance
    else:
        raise NotImplementedError(f'{which} not implemented')
    interper = RectBivariateSpline(x=nrg.ens * x_ratio, y=np.log10(nrg.ts / nrg_gamma),
                                   z=z.T, kx=1, ky=1)
    # 1-occupation to be comparable to CS data which decreases for increasing occupation
    # Log10 to help make y data more uniform for interper. Should not make a difference to fit values

    def nrg_func(x, mid, g, theta, amp=1, lin=0, const=0, occ_lin=0):
        """


        Args:
            x (): sweep gate
            mid (): center
            g (): gamma broadening
            theta (): thermal broadening
            amp (): charge sensor amplitude
            lin (): sweep gate charge sensor cross capacitance
            const (): charge sensor current average
            occ_lin (): screening of sweep gate/charge sensor cross capacitance due to increased occupation

        Returns:

        """
        x_scaled = (x - mid - g*(-1.76567) - theta*(-1)) / g   # To rescale varying temperature data with G instead (
        # double G is really half T). Note: The -g*(...) - theta*(...) is just to make the center roughly near OCC =
        # 0.5 (which is helpful for fitting only around the transition) x_scaled = (x - mid) / g

        # Note: the fact that NRG_gamma = 0.001 is taken into account with x_ratio above
        interped = interper(x_scaled, np.log10(theta / g)).flatten()
        if which == 'i_sense':
            interped = amp * (1 + occ_lin * (x - mid)) * interped + lin * (x - mid) + const - amp / 2
        # Note: (occ_lin*x)*Occupation is a linear term which changes with occupation,
        # not a linear term which changes with x
        return interped

    return nrg_func


@dataclass
class NRGData:
    ens: np.ndarray
    ts: np.ndarray
    conductance: np.ndarray
    dndt: np.ndarray
    entropy: np.ndarray
    occupation: np.ndarray
    int_dndt: np.ndarray

    @classmethod
    @lru_cache
    def from_mat(cls, path=r'D:\GitHub\dat_analysis\dat_analysis\resources\NRGResults.mat') -> NRGData:
        import os
        print(os.path.abspath('.'))
        data = scipy.io.loadmat(path)
        return cls(
            ens=data['Ens'].flatten(),
            ts=data['Ts'].flatten(),
            conductance=data['Conductance_mat'].T,
            dndt=data['DNDT_mat'].T,
            entropy=data['Entropy_mat'].T,
            occupation=data['Occupation_mat'].T,
            int_dndt=data['intDNDT_mat'].T,
        )


# def calculate_NRG_fit(dat: DatHDF):
#     def calculate_output(dat: DatHDF):
#         if dat.Logs.fds['ESC'] >= -240:  # Gamma broadened so no centering
#             logger.info(f'Dat{dat.datnum}: Calculating SPS.005 without centering')
#             do_entropy_calc(dat.datnum, save_name='SPS.005', setpoint_start=0.005, csq_mapped=False,
#                             center_for_avg=False)
#         else:  # Not gamma broadened so needs centering
#             logger.info(f'Dat{dat.datnum}: Calculating SPS.005 with centering')
#             do_entropy_calc(dat.datnum, save_name='SPS.005', setpoint_start=0.005, csq_mapped=False,
#                             center_for_avg=True,
#                             t_func_name='i_sense')
#     if 'SPS.005' not in dat.SquareEntropy.Output_names():
#         calculate_output(dat)
#     out = dat.SquareEntropy.get_Outputs(name='SPS.005')
#     data = data_from_output(out, 'i_sense_cold')
#     x = out.x
#     x, data = get_data_in_range(x, data, width=500)
#     params = lm.Parameters()
#     params.add_many(
#         ('mid', mid, True, -500, 200, None, None),
#         ('theta', theta, True, 0.5, 50, None, None),
#         ('amp', amp, True, 0.1, 3, None, None),
#         ('lin', lin, True, 0, 0.005, None, None),
#         ('occ_lin', occ_lin, True, -0.0003, 0.0003, None, None),
#         ('const', const, True, -2, 10, None, None),
#         ('g', g, True, 0.2, 400, None, None),
#     )
#     # Note: Theta or Gamma MUST be fixed (and makes sense to fix theta usually)
#     fit = calculate_fit(x, data, params=params, func=NRG_func_generator(which='i_sense'), method='powell')


def testing_fit_methods():
    # Weakly coupled entropy dat
    # dat = get_dat(2164)
    # dat = get_dat(2167)
    dat = get_dat(2170)
    out = dat.SquareEntropy.get_Outputs(name='default')
    x = out.x
    data = np.nanmean(out.averaged[(0, 2,), :], axis=0)

    plotter = OneD(dat=dat)

    fig = plotter.figure(ylabel='Current /nA', title=f'Dat{dat.datnum}: Fitting Weakly coupled to NRG')

    fig.add_trace(plotter.trace(x=x, data=data, name='Data', mode='lines'))

    print(dat.SquareEntropy.get_fit(fit_name='default').best_values)
    params = lm.Parameters()
    params.add_many(
        # ('mid', 2.2, True, None, None, None, None),
        ('mid', 0, True, -200, 200, None, 0.001),
        # ('mid', 1, True, -100, 100, None, 0.001),
        ('theta', 3.9, False, 1, 6, None, 0.001),
        ('amp', 0.94, True, 0, 3, None, 0.001),
        # ('lin', 0.0015, True, 0, 0.005, None, None),
        # ('lin', 0.0, True, 0, 0.005, None, 0.00001),
        ('lin', 0.01, True, 0, 0.005, None, 0.00001),
        ('occ_lin', 0, True, -0.0003, 0.0003, None, 0.000001),
        # ('const', 7.2, True, None, None, None, None),
        ('const', 7, True, -2, 10, None, 0.001),
        # ('g', 0.2371, True, 0.2, 200, None, 0.01),
        ('g', 1, True, 0.2, 200, None, 0.01),
    )

    dfs = []

    for method in [
        # 'leastsq',
        'least_squares',
        'differential_evolution',
        # 'brute',
        # 'basinhopping',
        # 'ampgo',
        'nelder',
        # 'lbfgsb',
        'powell',
        # 'cg',
        # 'newton',
        'cobyla',
        # 'bfgs',
        # 'tnc',
        # 'trust-ncg',
        # 'trust-exact',
        # 'trust-krylov',
        # 'trust-constr',
        # 'dogleg',
        # 'slsqp',
        # 'emcee',
        # 'shgo',
        'dual_annealing'
    ]:
        try:
            t1 = time.time()
            fit = calculate_fit(x, data, params=params, func=NRG_func_generator(which='i_sense'), method=method)
            total_time = time.time() - t1

            # fig.add_trace((plotter.trace(x=x, data=fit.eval_init(x=x), name='Initial Fit', mode='lines')))
            fig.add_trace((plotter.trace(x=x, data=fit.eval_fit(x=x), name=f'{method} Fit', mode='lines')))
            df = fit.to_df()
            df['name'] = method
            df['duration'] = total_time
            df['reduced chi sq'] = fit.fit_result.redchi
            dfs.append(df)
        except Exception as e:
            print(f'Failed for {method} with error: {e}')

    df = pd.concat(dfs)
    df.index = df.name
    df.pop('name')
    print(df.to_string())
    fig.show()


def plotting_center_shift():
    nrg_func = NRG_func_generator('occupation')
    params = lm.Parameters()
    params.add_many(
        ('mid', 0, True, -200, 200, None, 0.001),
        ('theta', 3.9, False, 1, 500, None, 0.001),
        ('amp', 1, True, 0, 3, None, 0.001),
        ('lin', 0, True, 0, 0.005, None, 0.00001),
        ('occ_lin', 0, True, -0.0003, 0.0003, None, 0.000001),
        ('const', 0, True, -2, 10, None, 0.001),
        ('g', 1, True, 0.2, 2000, None, 0.01),
    )
    model = lm.Model(nrg_func)

    x = np.linspace(-10, 5000, 10000)
    gs = np.linspace(0, 200, 201)
    thetas = np.logspace(0.1, 2, 20)
    # thetas = np.linspace(1, 500, 10)
    # thetas = [1, 2, 5, 10, 20]
    all_mids = []
    for theta in thetas:
        params['theta'].value = theta
        mids = []
        for g in gs:
            params['g'].value = g
            occs = model.eval(x=x, params=params)
            mids.append(x[U.get_data_index(occs, 0.5, is_sorted=True)])

        all_mids.append(mids)
    plotter = OneD(dat=None)
    fig = plotter.figure(xlabel='Gamma /mV', ylabel='Shift of 0.5 OCC', title='Shift of 0.5 Occupation vs Theta and G')
    fig.update_layout(legend=dict(title='Theta /mV'))
    for mids, theta in zip(all_mids, thetas):
        fig.add_trace(plotter.trace(data=mids, x=gs, name=f'{theta:.1f}', mode='lines'))
    fig.show()
    return fig


if __name__ == '__main__':
    nrg = NRGData.from_mat()
    # plotting_center_shift()

    all_dats = get_dats((5780, 5795 + 1))
    for dat in all_dats:
        print(f'Dat{dat.datnum}\n'
              f'CSbias: {(dat.Logs.bds["CSBIAS/100"]+1.3)*10:.0f}uV\n'
              f'Repeats: {len(dat.Data.get_data("y"))}\n'
              f'ESP: {dat.Logs.fds["ESP"]:.1f}mV\n'
              f'ACC-Center: {np.nanmean(dat.Data.get_data("x")):.0f}mV\n')



