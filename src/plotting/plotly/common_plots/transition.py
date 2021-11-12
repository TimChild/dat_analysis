import re
from typing import List, Callable, Optional, Iterable

import numpy as np
from plotly import graph_objs as go

import src.characters
from Analysis.Feb2021.entropy_gamma_final import AnalysisGeneral
from src import useful_functions as U
from src.dat_object.dat_hdf import DatHDF
from src.dat_object.make_dat import get_dats
from src.plotting.plotly import OneD


def transition_trace(dats: List[DatHDF], x_func: Callable,
                     from_square_entropy: bool = True, fit_name: str = 'default',
                     param: str = 'amp', label: str = '',
                     **kwargs) -> go.Scatter:
    divide_const, divide_acc_divider = False, False
    if param == 'amp/const':
        divide_const = True
        param = 'amp'
    elif param == 'theta real':
        divide_acc_divider = True
        param = 'theta'

    plotter = OneD(dats=dats)
    if from_square_entropy:
        vals = [dat.SquareEntropy.get_fit(which_fit='transition', fit_name=fit_name, check_exists=True).best_values.get(
            param) for dat in dats]
    else:
        vals = [dat.Transition.get_fit(name=fit_name).best_values.get(param) for dat in dats]
        if divide_const:
            vals = [amp / dat.Transition.get_fit(name=fit_name).best_values.const for amp, dat in zip(vals, dats)]
        elif divide_acc_divider:
            divider_vals = [int(re.search(r'\d+', dat.Logs.xlabel)[0]) for dat in dats]  # e.g. get the 1000 part of ACC*1000 /mV
            vals = [val/divider for val, divider in zip(vals, divider_vals)]
    x = [x_func(dat) for dat in dats]
    trace = plotter.trace(x=x, data=vals, name=label, text=[dat.datnum for dat in dats], **kwargs)
    return trace


def single_transition_trace(dat: DatHDF, label: Optional[str] = None, subtract_fit=False,
                            fit_only=False, fit_name: str = 'narrow', transition_only=True,
                            se_output_name: str = 'SPS.005',
                            csq_mapped=False) -> go.Scatter():
    plotter = OneD(dat=dat)
    if transition_only:
        if csq_mapped:
            x = dat.Data.get_data('csq_x_avg')
        else:
            x = dat.Transition.avg_x
    else:
        if csq_mapped:
            raise NotImplementedError
        x = dat.SquareEntropy.avg_x

    if not fit_only:
        if transition_only:
            if csq_mapped:
                data = dat.Data.get_data('csq_mapped_avg')
            else:
                data = dat.Transition.avg_data
        else:
            data = dat.SquareEntropy.get_transition_part(name=se_output_name, part='cold')
    else:
        data = None  # Set below

    if fit_only or subtract_fit:
        if transition_only:
            fit = dat.Transition.get_fit(name=fit_name)
        else:
            fit = dat.SquareEntropy.get_fit(which_fit='transition', fit_name=fit_name)
        if fit_only:
            data = fit.eval_fit(x=x)
        elif subtract_fit:
            data = data - fit.eval_fit(x=x)

    trace = plotter.trace(x=x, data=data, name=label, mode='lines')
    return trace


def transition_fig(dats: Optional[List[DatHDF]] = None, xlabel: str = '/mV', title_append: str = '',
                   param: str = 'amp') -> go.Figure:
    plotter = OneD(dats=dats)
    titles = {
        'amp': 'Amplitude',
        'theta': 'Theta',
        'theta real': 'Theta',
        'g': 'Gamma',
        'amp/const': 'Amplitude/Const (sensitivity)',
    }
    ylabels = {
        'amp': 'Amplitude /nA',
        'theta': 'Theta /mV',
        'theta real': 'Theta /mV (real)',
        'g': 'Gamma /mV',
        'amp/const': 'Amplitude/Const'
    }

    fig = plotter.figure(xlabel=xlabel, ylabel=ylabels[param],
                         title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: {titles[param]}{title_append}')
    return fig


def plot_transition_values(datnums: List[int], save_name: str, general: AnalysisGeneral, param_name: str = 'theta',
                           transition_only: bool = True, show=True):
    param = param_name
    if transition_only:
        all_dats = get_dats(datnums)
        fig = transition_fig(dats=all_dats, xlabel='ESC /mV', title_append=' vs ESC for Transition Only scans',
                             param=param)
        dats = get_dats(datnums)
        fig.add_trace(transition_trace(dats, x_func=general.x_func, from_square_entropy=False,
                                       fit_name=save_name, param=param, label='Data'))
        # print(
        #     f'Avg weakly coupled cold theta = '
        #     f'{np.mean([dat.Transition.get_fit(name=fit_name).best_values.theta for dat in dats if dat.Logs.fds["ESC"] <= -330])}')
    else:
        all_dats = get_dats(datnums)
        fig = transition_fig(dats=all_dats, xlabel='ESC /mV', title_append=' vs ESC for Entropy scans', param=param)
        dats = get_dats(datnums)
        fig.add_trace(transition_trace(dats, x_func=general.x_func, from_square_entropy=True,
                                       fit_name=save_name, param=param, label='Data'))
        # print(
        #     f'Avg weakly coupled cold theta = {np.mean([dat.SquareEntropy.get_fit(which_fit="transition", fit_name=fit_name).best_values.theta for dat in dats if dat.Logs.fds["ESC"] <= -330])}')
    if show:
        fig.show()
    return fig


class TransitionGraphText:
    @classmethod
    def get_ylabel_from_param(cls, par_name: str) -> str:
        t = cls.get_full_name_from_param(par_name)
        u = cls.get_units_from_param(par_name)
        return f'{t}/{u}'

    @staticmethod
    def get_full_name_from_param(par_name: str) -> str:
        if par_name == 'mid':
            t = 'Center'
        elif par_name == 'theta':
            t = 'Theta'
        elif par_name == 'amp':
            t = 'Amplitude'
        elif par_name == 'lin':
            t = 'Linear Component'
        elif par_name == 'const':
            t = 'Constant Offset'
        elif par_name == 'g':
            t = 'Gamma'
        else:
            raise KeyError(f'{par_name} not a recognized key')
        return t

    @staticmethod
    def get_units_from_param(par_name: str) -> str:
        if par_name == 'mid':
            u = 'mV'
        elif par_name == 'theta':
            u = 'mV'
        elif par_name == 'amp':
            u = 'nA'
        elif par_name == 'lin':
            u = 'nA/mV'
        elif par_name == 'const':
            u = 'nA'
        elif par_name == 'g':
            u = 'mV'
        else:
            raise KeyError(f'{par_name} not a recognized key')
        return u


def plot_transition_row_fit_values(dat: DatHDF, param_name: str, fit_name: str = 'default') -> go.Figure:
    """
    Display the fit values for each row of data for a single dat.
    Args:
        dat ():
        param_name (): Which parameter to display from (mid, theta, amp, lin, const, g)
        fit_name ():

    Returns:
        Figure

    """
    title = f'Dat{dat.datnum}: {TransitionGraphText.get_full_name_from_param(param_name)} for each row'
    ylabel = TransitionGraphText.get_ylabel_from_param(par_name=param_name)

    x = dat.Data.y
    xlabel = dat.Logs.ylabel

    if param_name not in (keys := dat.Transition.get_fit(which='row', name=fit_name, row=0).best_values.keys):
        raise KeyError(f'{param_name} not in {keys}')

    params = [fit.params.get(param_name) for fit in dat.Transition.get_row_fits(name=fit_name)]
    fit_values = [param.value for param in params]
    fit_errs = [param.stderr for param in params]

    plotter = OneD(dat=dat)
    fig = plotter.plot(data=fit_values, data_err=fit_errs, x=x,
                       xlabel=xlabel, ylabel=ylabel,
                       title=title,
                       mode='markers')
    return fig


def plot_multiple_transition_row_fit_with_stdev(dats: Iterable[DatHDF],
                                                param_name: str,
                                                fit_name: str = 'default',
                                                x: U.ARRAY_LIKE = None,
                                                xlabel: str = 'Datnum',
                                                stdev_only=False) -> go.Figure:
    """
    Display avg of row fit values with stdev or just stdev for Transition fits of multiple dats (i.e. for looking
    at how varied single scan fits are for multiple dats)

    Args:
        dats ():
        param_name (): Which parameter to display from (mid, theta, amp, lin, const, g)
        fit_name (str):
        x (): Array for x axis (e.g. datnums)
        xlabel (): Label for x axis (e.g. "datnum")
        stdev_only (): Whether to plot the stdev of fits only rather than fit values

    Returns:

    """
    if x is None:
        x = [dat.datnum for dat in dats]

    def get_ylabel(name: str) -> str:
        if stdev_only:
            return f'{src.characters.SIG} {TransitionGraphText.get_ylabel_from_param(name)}'
        else:
            return TransitionGraphText.get_ylabel_from_param(name)

    def title():
        a = f'Dats{dats[0].datnum}-{dats[-1].datnum}: '
        b = f'Standard Deviation of '
        c = f'{TransitionGraphText.get_full_name_from_param(param_name)}'
        if stdev_only:
            return a + b + c
        else:
            return a + c

    dats = list(dats)
    if param_name not in (keys := dats[0].Transition.get_fit(which='row', name=fit_name, row=0).best_values.keys):
        raise KeyError(f'{param_name} not in {keys}')

    fit_vals = [[fit.best_values.get(param_name, default=np.nan) for fit in dat.Transition.get_row_fits(name=fit_name)]
                for dat in dats]
    errs = [np.nanstd(row) for row in fit_vals]
    if stdev_only:
        fit_vals = errs
        errs = None
    else:
        fit_vals = [np.mean(row) for row in fit_vals]

    plotter = OneD(dats=dats)
    fig = plotter.plot(data=fit_vals, data_err=errs, x=x, text=[f'{dat.datnum}' for dat in dats],
                       xlabel=xlabel, ylabel=get_ylabel(param_name),
                       title=title(),
                       mode='markers')
    return fig