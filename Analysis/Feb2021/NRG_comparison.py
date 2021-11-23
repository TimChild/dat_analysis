"""
Sep 21 -- Used to figure out how to best fit data using NRG (i.e. what fitting method of lmfit to use and to try and
figure out a way to have the "zero" of NRG data line up somewhere close to an occupation of 0.5 for convenience when
fitting.
Found that the "powell" method was the only reliable method of fitting to interpolated data (probably makes sense since
anything that used gradient descent could easily be thrown off by not fitting to an analytical function).
"powell" is basically just a clever implementation of brute force minimization

No functions to save from here, and this won't be used again.
"""
from __future__ import annotations
import numpy as np
import plotly.io as pio
import lmfit as lm
import pandas as pd
import time
import logging

from dat_analysis.analysis_tools.nrg import NRG_func_generator, NRGData
from dat_analysis.dat_object.make_dat import get_dat, get_dats
from dat_analysis.plotting.plotly.dat_plotting import OneD
from dat_analysis.analysis_tools.general_fitting import calculate_fit
import dat_analysis.useful_functions as U

pio.renderers.default = "browser"
logger = logging.getLogger(__name__)


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
    nrg = NRGData.from_old_mat()
    # plotting_center_shift()

    all_dats = get_dats((5780, 5795 + 1))
    for dat in all_dats:
        print(f'Dat{dat.datnum}\n'
              f'CSbias: {(dat.Logs.bds["CSBIAS/100"]+1.3)*10:.0f}uV\n'
              f'Repeats: {len(dat.Data.get_data("y"))}\n'
              f'ESP: {dat.Logs.fds["ESP"]:.1f}mV\n'
              f'ACC-Center: {np.nanmean(dat.Data.get_data("x")):.0f}mV\n')



