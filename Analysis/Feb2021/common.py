from typing import List, Callable, Optional, Union

import numpy as np
from plotly import graph_objects as go
from progressbar import progressbar

from src.UsefulFunctions import edit_params
from Dash.DatPlotting import OneD
from DatObject.Attributes.Transition import i_sense, i_sense_digamma, i_sense_digamma_amplin
from DatObject.DatHDF import DatHDF

from DatObject.Make_Dat import get_dats, get_dat
from Plotting.Plotly.PlotlyUtil import HoverInfo, additional_data_dict_converter


def get_deltaT(dat):
    """Returns deltaT of a given dat in mV"""
    ho1 = dat.AWG.max(0)  # 'HO1/10M' gives nA * 10
    t = dat.Logs.temps.mc

    # Datnums to search through (only thing that should be changed)
    # datnums = set(range(1312, 1451 + 1)) - set(range(1312, 1451 + 1, 4))
    datnums = list(range(2143, 2156))

    dats = get_dats(datnums)

    dats = [d for d in dats if
            np.isclose(d.Logs.temps.mc, dat.Logs.temps.mc, rtol=0.1)]  # Get all dats where MC temp is within 10%
    bias_lookup = np.array([d.Logs.fds['HO1/10M'] for d in dats])

    indp = int(np.argmin(abs(bias_lookup - ho1)))
    indm = int(np.argmin(abs(bias_lookup + ho1)))
    theta_z = np.nanmean([d.Transition.avg_fit.best_values.theta for d in dats if d.Logs.fds['HO1/10M'] == 0])

    theta_p = dats[indp].Transition.avg_fit.best_values.theta
    theta_m = dats[indm].Transition.avg_fit.best_values.theta
    # theta_z = dats[indz].Transition.avg_fit.best_values.theta
    return (theta_p + theta_m) / 2 - theta_z


def plot_fit_integrated_comparison(dats: List[DatHDF], x_func: Callable, x_label: str, title_append: Optional[str] = '',
                                   int_info_name: Optional[str] = None, fit_name: str = 'SPS.0045',
                                   plot=True) -> go.Figure():
    if int_info_name is None:
        int_info_name = 'default'
    plotter = OneD(dats=dats)
    fig = plotter.figure(
        title=f'Dats{dats[0].datnum}-{dats[-1].datnum}: Fit and Integrated (\'{int_info_name}\') Entropy{title_append}',
        xlabel=x_label,
        ylabel='Entropy /kB')

    hover_infos = [
        HoverInfo(name='Dat', func=lambda dat: dat.datnum, precision='.d', units=''),
        HoverInfo(name='Temperature', func=lambda dat: dat.Logs.temps.mc * 1000, precision='.1f', units='mK'),
        HoverInfo(name='Bias', func=lambda dat: dat.AWG.max(0) / 10, precision='.1f', units='nA'),
        # HoverInfo(name='Fit Entropy', func=lambda dat: dat.SquareEntropy.get_fit(which_fit='entropy', fit_name=fit_name).best_values.dS,
        #           precision='.2f', units='kB'),
        HoverInfo(name='Integrated Entropy',
                  func=lambda dat: np.nanmean(dat.Entropy.get_integrated_entropy(name=int_info_name,
                                                                                 data=dat.SquareEntropy.get_Outputs(
                                                                                     name=fit_name).average_entropy_signal)[
                                              -10:]),
                  # TODO: Change to using proper output (with setpoints)
                  precision='.2f', units='kB'),
    ]

    funcs, template = additional_data_dict_converter(hover_infos)
    x = [x_func(dat) for dat in dats]
    hover_data = [[func(dat) for func in funcs] for dat in dats]

    entropies = [dat.Entropy.get_fit(name=fit_name).best_values.dS for dat in dats]
    entropy_errs = [np.nanstd([
        f.best_values.dS if f.best_values.dS is not None else np.nan
        for f in dat.Entropy.get_row_fits(name=fit_name) for dat in dats
    ]) / np.sqrt(dat.Data.y_array.shape[0]) for dat in dats]
    fig.add_trace(plotter.trace(
        data=entropies, data_err=entropy_errs, x=x, name=f'Fit Entropy',
        mode='markers+lines',
        trace_kwargs={'customdata': hover_data, 'hovertemplate': template})
    )
    integrated_entropies = [np.nanmean(
        dat.Entropy.get_integrated_entropy(name=int_info_name,
                                           data=dat.SquareEntropy.get_Outputs(name=fit_name).average_entropy_signal)[
        -10:]) for dat in dats]
    fig.add_trace(plotter.trace(
        data=integrated_entropies, x=x, name=f'Integrated',
        mode='markers+lines',
        trace_kwargs={'customdata': hover_data, 'hovertemplate': template})
    )

    if plot:
        fig.show(renderer='browser')
    return fig


def entropy_vs_time_trace(dats: List[DatHDF], trace_name=None, integrated=False, fit_name: str = 'SPS.005',
                          integrated_name: str = 'first'):
    plotter = OneD(dats=dats)
    if integrated is False:
        entropies = [dat.Entropy.get_fit(which='avg', name=fit_name).best_values.dS for dat in dats]
        entropies_err = [np.nanstd(
            [fit.best_values.dS if fit.best_values.dS is not None else np.nan for fit in
             dat.Entropy.get_row_fits(name=fit_name)]
        ) / np.sqrt(len(dats)) for dat in dats]
    else:
        entropies = [np.nanmean(
            dat.Entropy.get_integrated_entropy(
                name=integrated_name,
                data=dat.SquareEntropy.get_Outputs(name=fit_name).average_entropy_signal)[-10:]) for dat in dats]
        entropies_err = [0.0] * len(dats)

    times = [str(dat.Logs.time_completed) for dat in dats]

    trace = plotter.trace(data=entropies, data_err=entropies_err, x=times, text=[dat.datnum for dat in dats],
                          mode='lines', name=trace_name)
    return trace


def entropy_vs_time_fig(title: Optional[str] = None):
    plotter = OneD()
    if title is None:
        title = 'Entropy vs Time'
    fig = plotter.figure(xlabel='Time', ylabel='Entropy /kB', title=title)
    fig.update_xaxes(tickformat="%H:%M\n%a")
    return fig


def narrow_fit(dat: DatHDF, width, initial_params, fit_func=i_sense, check_exists=False, save_name='narrow',
               output_name: str = 'default', transition_only: bool = False,
               overwrite=False, csq_map: bool = False):
    """
    Get a fit only including +/- width in dat.x around center of transition
    kwargs is the stuff to pass to get_fit
    Return a fit
    """
    if transition_only is False:
        x = np.copy(dat.SquareEntropy.avg_x)
        y = np.copy(dat.SquareEntropy.get_Outputs(name=output_name, existing_only=True).averaged)
        y = np.mean(y[(0, 2), :], axis=0)  # Average Cold parts
    else:
        x = np.copy(dat.Transition.avg_x)
        if csq_map:
            y = np.copy(dat.Data.get('csq_mapped_avg'))
        else:
            y = np.copy(dat.Transition.avg_data)

    start_ind = np.nanargmin(np.abs(np.add(x, width)))
    end_ind = np.nanargmin(np.abs(np.subtract(x, width)))

    x[:start_ind] = [np.nan] * start_ind
    x[end_ind:] = [np.nan] * (len(x) - end_ind)

    y[:start_ind] = [np.nan] * start_ind
    y[end_ind:] = [np.nan] * (len(y) - end_ind)

    if transition_only is False:
        fit = dat.SquareEntropy.get_fit(
            x=x,
            data=y,
            initial_params=initial_params,
            fit_func=fit_func,
            check_exists=check_exists, fit_name=save_name,
            overwrite=overwrite)
    else:
        fit = dat.Transition.get_fit(
            x=x, data=y,
            initial_params=initial_params,
            fit_func=fit_func,
            check_exists=check_exists, name=save_name,
            overwrite=overwrite
        )
    return fit


def do_narrow_fits(dats: Union[List[DatHDF], int],
                   theta=None, gamma=None, width=500,
                   output_name: str = 'default', overwrite=False,
                   transition_only=False,
                   fit_func: str = 'i_sense_digamma',
                   fit_name: str = 'narrow',
                  ):
    if isinstance(dats, int):  # To allow multiprocessing
        dats = [get_dat(dats)]

    if transition_only is False:
        fit = dats[0].SquareEntropy.get_fit(which='avg', which_fit='transition', transition_part='cold',
                                            check_exists=False)
    else:
        fit = dats[0].Transition.avg_fit
    params = fit.params
    if gamma is None:
        params.add('g', value=0, vary=True, min=-50, max=1000)
    else:
        params.add('g', value=gamma, vary=False, min=-50, max=1000)
    if theta is None:
        theta = fit.best_values.theta
        theta_vary = True
    else:
        theta_vary = False

    if fit_func == 'i_sense_digamma_amplin':
        params.add('amplin', value=0, vary=True)
        func = i_sense_digamma_amplin
    else:
        func = i_sense_digamma

    new_pars = edit_params(params, param_name='theta', value=theta, vary=theta_vary)

    amp_fits = [narrow_fit(
        dat,
        width,
        initial_params=new_pars,
        fit_func=func,
        output_name=output_name,
        check_exists=False, save_name=fit_name,
        transition_only=transition_only,
        overwrite=overwrite,
       )
        for dat in progressbar(dats)]

    return amp_fits
