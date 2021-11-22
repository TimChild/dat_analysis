"""
Using varying fridge temperature at various Gamma to see if we can detect lever arm change.
I.e. go from gamma broadened at 50mK to thermally broadened at 500mK. Should be able to get a good measure of lever arm
when thermally broadened, and the idea is that the lever arm won't change with temperature since gamma will be staying
fixed.

Sep 21 -- We had initially thought that the lever arm was changing significantly into the gamma broadened regime due to
the change of shape of the QD (as a means of explaining why we didn't see any Kondo supression of spin), however this
turned out not to be the case. Did not find any evidence for more than a linear change of lever arm vs coupling gate
but irrespective of gamma.

Probably not worth salvaging any functions from here.
"""
from dat_analysis.dat_object.make_dat import get_dats, DatHDF
from dat_analysis.dat_analysis.characters import DELTA, THETA, PM
from dat_analysis.plotting.plotly.dat_plotting import OneD
import dat_analysis.useful_functions as U
from Analysis.Feb2021.common import sort_by_temps, sort_by_coupling
from dat_analysis.analysis_tools.transition import do_transition_only_calc
from dat_analysis.analysis_tools.general_fitting import calculate_fit

import numpy as np
import lmfit as lm
import plotly.graph_objects as go
import plotly.io as pio
from typing import List, Dict
from functools import partial
from progressbar import progressbar

from concurrent.futures import ProcessPoolExecutor

pio.renderers.default = 'browser'


def check_min_max_temps():
    """prints min/max temp of all dats at each target temperature"""
    global dats_by_temp
    for k in dats_by_temp:
        ds = dats_by_temp[k]
        if len(ds) > 0:
            print(f'Target {k}mK:\n'
                  f'Max T = {np.max([dat.Logs.temps.mc * 1000 for dat in ds]):.1f}mK\n'
                  f'Min T = {np.min([dat.Logs.temps.mc * 1000 for dat in ds]):.1f}mK\n'
                  f'')


def get_specific_dat(temp: int, coupling: int) -> DatHDF:
    """Returns a single dat based on temp and coupling gate"""
    global dats_by_temp, dats_by_coupling_gate
    if temp not in dats_by_temp.keys():
        raise KeyError(f'{temp} not in {dats_by_temp.keys()}')
    if coupling not in dats_by_coupling_gate.keys():
        raise KeyError(f'{coupling} not in {dats_by_coupling_gate.keys()}')

    ds = dats_by_temp[temp]
    dat = [d for d in ds if d in dats_by_coupling_gate[coupling]][0]
    return dat


def check_centers():
    """Assuming 'simple' exists as a Transition fit name, will just plot centers and print any that deviate far from
    0 """
    global all_dats
    for dat in all_dats:
        fit = dat.Transition.get_fit(name='simple')
        if fit.best_values.mid > 5 or fit.best_values.mid < -5:
            print(f'Dat{dat.datnum}: mid = {fit.best_values.mid:.1f}mV')

    plotter = OneD(dats=all_dats)
    fig = plotter.figure(xlabel='Datnum', ylabel='Center /mV',
                         title=f'Dats{all_dats[0].datnum}-{all_dats[-1].datnum}: Centers')
    fig.add_trace(plotter.trace(data=[dat.Transition.get_fit(name='simple').best_values.mid for dat in all_dats],
                                x=[dat.datnum for dat in all_dats], mode='markers'))
    fig.show()


def run_simple_fits(dats: List[DatHDF]):
    """Runs 'simple' i_sense fits on dats"""
    datnums = [dat.datnum for dat in dats]
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(partial(do_transition_only_calc, save_name='simple', theta=None, gamma=None, width=None,
                                        t_func_name='i_sense', center_func='i_sense', overwrite=False), datnums))


def run_varying_width_fits(dat_temp_dict: Dict[float, List[DatHDF]], temp_width_dict: dict):
    """Runs 'varying_width' i_sense fits on dats (note: Only on relatively weakly coupled dats!)"""
    with ProcessPoolExecutor() as pool:
        for temp, ds in dat_temp_dict.items():
            datnums = [dat.datnum for dat in ds if dat.Logs.fds['ESC'] < -195]
            width = temp_width_dict[temp]
            results = list(pool.map(partial(do_transition_only_calc, save_name='varying_width', theta=None, gamma=None,
                                            width=width,
                                            t_func_name='i_sense', center_func='i_sense', overwrite=False), datnums))


def check_rough_broadening_vs_temp() -> go.Figure:
    global all_dats, dats_by_temp
    dats = all_dats
    plotter = OneD(dats=dats)
    fig = plotter.figure(xlabel='ESC /mV', ylabel='Theta /mV', title=f'Dats{dats[0].datnum}-{dats[-1].datnum}:'
                                                                     f' Rough idea of broadening')
    for temp, dats in dats_by_temp.items():
        if len(dats) != 0:
            dat = dats[0]
            dats = U.order_list(dats, [dat.Logs.fds['ESC'] for dat in dats])
            x = [dat.Logs.fds['ESC'] for dat in dats]
            thetas = [dat.Transition.get_fit(name='simple').best_values.theta for dat in dats]
            fig.add_trace(plotter.trace(x=x, data=thetas, mode='markers+lines', name=f'{temp}mK'))

    return fig


def theta_slope_in_weakly_coupled(dats: List[DatHDF], show_intermediate=False, fit_name: str = 'simple') -> go.Figure:
    """Calculates and plots slope of theta in weakly coupled for all temperatures (option to show the linear fits for
    each temp) """
    plotter = OneD(dats=dats)
    fig = plotter.figure(xlabel='Fridge Temp /mK', ylabel=f'Slope ({DELTA}{THETA}/{DELTA}ESC)',
                         title=f'Dats{dats[0].datnum}-{dats[-1].datnum}:'
                               f' Slope of thetas in weakly coupled')

    dats_sorted_by_temp = sort_by_temps(dats)
    line = lm.models.LinearModel()
    slopes = []
    temps = []
    for temp, dats in sorted(dats_sorted_by_temp.items()):
        if len(dats) > 0:
            dats = [dat for dat in dats if dat.Logs.fds['ESC'] < -235]
            escs = np.array([dat.Logs.fds['ESC'] for dat in dats])
            thetas = np.array([dat.Transition.get_fit(name=fit_name).best_values.theta for dat in dats])
            pars = line.guess(thetas, x=escs)
            fit = calculate_fit(x=escs, data=thetas, params=pars, func=line.func)
            slopes.append(fit.best_values.slope)
            temps.append(temp)
            p = OneD(dats=dats)
            f = p.figure(xlabel='ESC /mV', ylabel='Theta /mV',
                         title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: Temp = {temp}mK, Fit of theta slope')
            f.add_trace(p.trace(x=escs, data=thetas, name='Data'))
            f.add_trace(p.trace(x=escs, data=fit.eval_fit(x=escs), name='Fit', mode='lines'))
            if show_intermediate:
                f.show()
            print(f'{temp}mK:\nSlope = {fit.best_values.slope:.3g}{PM}{U.sig_fig(fit.params["slope"].stderr, 2):.2g}\n'
                  f'Reduced Chi Square = {fit.reduced_chi_sq:.3g}\n')

    fig.add_trace(plotter.trace(data=slopes, x=temps, mode='markers+lines'))
    return fig


def fit_single_esc_varying_width(show_figs=True) -> List[go.Figure]:
    """Fits transition with varying width based on temperature at a single ESC"""
    global dats_by_coupling_gate
    dats = dats_by_coupling_gate[-260]
    dats = U.order_list(dats, [dat.Logs.temps.mc for dat in dats])
    figs = []
    for dat, w in progressbar(zip(dats, [20, 30, 50, 100, 200])):
        save_name = 'varying_width'
        do_transition_only_calc(dat.datnum, save_name=save_name, theta=None, gamma=0, width=w,
                                t_func_name='i_sense')
        plotter = OneD(dat=dat)
        fig = plotter.figure(title=f'Dat{dat.datnum}: I_sense at {dat.Logs.temps.mc * 1000:.0f}mK',
                             ylabel='Current /nA')
        fig.add_trace(plotter.trace(data=dat.Transition.avg_data, x=dat.Transition.avg_x, mode='lines', name='Data'))
        fig.add_trace(plotter.trace(data=dat.Transition.get_fit(name=save_name).eval_fit(x=dat.Transition.avg_x),
                                    x=dat.Transition.avg_x, mode='lines', name='Fit'))
        [plotter.add_line(fig, value=xx, mode='vertical', color='black', linetype='dash') for xx in [w, -w]]
        if show_figs:
            fig.show()
        figs.append(fig)
        w_theta = dat.Transition.get_fit(name="simple").best_values.theta
        n_theta = dat.Transition.get_fit(name=save_name).best_values.theta
        print(f'Temp {dat.Logs.temps.mc * 1000:.0f}mK:\n'
              f'Wide Theta: {w_theta:.2f}mV\n'
              f'Narrow Theta: {n_theta:.2f}mV\n'
              f'Change: {(n_theta - w_theta) / w_theta * 100:.2f}%\n'
              )

    return figs


# Series of dats taken at 50, 500, 400, 300, 200, 100, 50mK, at each temp data taken at 6 places from weakly coupled
# to gamma broadened, At each position, there are three scans. 1 with no Heating bias, then with + - ~100% heating
# bias.
# NO_HEAT_DATS = list(range(5371, 5946+1, 3))
# POS_HEAT_DATS = list(range(5372, 5946+1, 3))
# NEG_HEAT_DATS = list(range(5373, 5946+1, 3))

# ALL = list(range(6811, 6910 + 1))# + list(range(6964, 7083+1)) # 20 positions along transition at 100, 500, 300, 50, ~10mK (all no heating)
# ALL = list(range(6964, 7083 + 1))
ALL = list(range(7084, 7093+1))
# ALL = list(range(7094, 7112+1))
# ALL = list(range(7129, 7302+1))


for num in [7074, 7217, 7226, 7225, 7232, 7231, 7233, 7289, 7283, 7277, 7279, 7275, 7282, 7284, 7281, 7280]:
    if num in ALL:
        ALL.remove(num)
# ALL.remove(7074)

if __name__ == '__main__':
    # Loading
    all_dats = get_dats(ALL)
    dats_by_temp = sort_by_temps(all_dats)
    dats_by_coupling_gate = sort_by_coupling(all_dats)

    # Fitting
    run_simple_fits(all_dats)
    run_varying_width_fits(dats_by_temp, {500: 200, 400: 150, 300: 100, 200: 75, 100: 50, 50: 30, 10: 20})

    # Checking
    # check_min_max_temps()
    # print(get_specific_dat(100, -230).datnum)
    # check_centers()
    # check_rough_broadening_vs_temp().show()

    # plotting
    fig1 = theta_slope_in_weakly_coupled(all_dats, show_intermediate=False)
    fig2 = theta_slope_in_weakly_coupled(all_dats, show_intermediate=False, fit_name='varying_width')
    fig1.show()
    fig2.show()

    #
    # trace = fig2.data[0]
    # trace.name = 'Varying Fit Widths'
    # fig1.add_trace(trace)
    # fig1.data[0].name = 'Full Width Fits'
    # line = lm.models.LinearModel()
    # pars = line.make_params(verbose=True)
    # pars['slope'].value = 0.0046 / 100
    # trace = go.Scatter(x=np.linspace(0, 500, 10), y=line.eval(x=np.linspace(0, 500, 10), params=pars), name='Expected',
    #                    line=dict(color='black', dash='dash'), mode='lines')
    # fig1.add_trace(trace)
    # fig1.show()


    ####################################################################################

    # # dats = get_dats((7113, 7121+1))
    # dats = get_dats((7122, 7128+1))
    #
    # run_simple_fits(dats)
    # plotter = OneD(dats=dats)
    # x = [dat.Logs.sweeprate for dat in dats]
    # data = [dat.Transition.get_fit(name='simple').best_values.theta for dat in dats]
    # fig = plotter.plot(x=x, data=data, title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: Theta vs Sweeprate',
    #                    xlabel='Sweeprate /mV/s', ylabel='Theta /mV')
    # fig.show()
    #
    # fit_name = 'i_sense:i_sense'
    # columns = ['Datnum', 'Sweeprate /mV/s', 'Avg Data Fit', 'Avg Data Fit uncertainty', 'Average of all Rows', 'Standard Error of all Rows', 'Standard Deviation of all Rows']
    # param = 'theta'
    # data = []
    # for dat in dats:
    #     all_fits = dat.Transition.get_row_fits(name=fit_name)
    #     all_vals = [fit.best_values.get(param) for fit in all_fits]
    #     avg_fit = dat.Transition.get_fit(name='simple')
    #     data.append([dat.datnum,
    #                  dat.Logs.sweeprate,
    #                  avg_fit.best_values.get(param),
    #                  avg_fit.params[param].stderr,
    #                  np.nanmean(all_vals),
    #                  np.nanstd(all_vals)/np.sqrt(len(all_vals)),
    #                  np.nanstd(all_vals)
    #                  ])
    # df = pd.DataFrame(data=data, columns=columns)
    # print(df.to_markdown())
    #
    # fig = plotter.figure(ylabel='Current /nA', title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: Transition at varying sweeprate')
    # for dat in dats:
    #     fig.add_trace(plotter.trace(x=dat.Transition.avg_x, data=dat.Transition.avg_data, name=f'Dat{dat.datnum}: {dat.Logs.sweeprate:.0f}mV/s', mode='lines'))
    #     fig.add_trace(plotter.trace(x=dat.Transition.avg_x, data=dat.Transition.get_fit(name='simple').eval_fit(x=dat.Transition.avg_x), name=f'Dat{dat.datnum}: Fit', mode='lines'))
    # fig.show()

