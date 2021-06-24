import plotly.io as pio
import plotly.graph_objects as go
import copy
import lmfit as lm
import numpy as np
from typing import List, Optional, Union


from src.dat_object.make_dat import get_dat, DatHDF
from src.analysis_tools import NrgUtil, NRGParams, setup_csq_dat, calculate_csq_map, calculate_csq_mapped_avg
from src.analysis_tools.general_fitting import calculate_fit, FitInfo
from src.plotting.plotly import OneD, TwoD, Data1D, Data2D
from src.characters import DELTA
from src.useful_functions import mean_data


pio.renderers.default = 'browser'

p1d = OneD(dat=None)
p2d = TwoD(dat=None)


def figure_1_add_NRG_fit_to_gamma_dndt() -> go.Figure:
    fit_name = 'forced_theta_linear_non_csq'
    dat = get_dat(2170)

    init_params = NRGParams(
        gamma=23.4352,
        theta=4.396,
        center=78.4,
        amp=0.675,
        lin=0.00121,
        const=7.367,
        lin_occ=0.0001453,
    )

    out = dat.SquareEntropy.get_Outputs(name=fit_name)
    x = out.x
    z = out.average_entropy_signal

    fig = p1d.figure(xlabel='V_P', ylabel=f'{DELTA}I (nA)')
    fig.add_trace(p1d.trace(x=x, data=z, name='Data', mode='lines'))

    dndt_init_params = copy.copy(init_params)
    dndt_init_params.amp = 0.0001  # To rescale the arbitrary NRG dndt scale
    dndt_init_params.theta = dat.SquareEntropy.get_fit(fit_name=fit_name).best_values.theta
    nrg_fitter = NrgUtil(inital_params=dndt_init_params)
    fit = nrg_fitter.get_fit(x=x, data=z, which_data='dndt')

    fig.add_trace(p1d.trace(x=x, data=fit.eval_fit(x=x), mode='lines', name='Fit'))
    fig.add_trace(p1d.trace(x=x, data=fit.eval_init(x=x), mode='lines', name='Init'))

    fig.show()
    return fig


def check_nrg_fit(datnum, exisiting_fit='forced_theta_linear'):
    """

    Args:
        datnum ():
        exisiting_fit (): To get the Theta value from

    Returns:

    """
    dat = get_dat(datnum)
    x = dat.Data.x
    data = dat.Data.get_data('csq_mapped_avg')
    theta = dat.Transition.get_fit(name=exisiting_fit).best_values.theta
    nrg_fitter = NrgUtil(NRGParams(gamma=1, theta=theta, center=0, amp=1.5, lin=0.003, const=0, lin_occ=0))
    fit = nrg_fitter.get_fit(x=x, data=data)
    fig = p1d.plot(data=data, x=x, trace_name='data')
    fig.add_trace(p1d.trace(x=x, data=fit.eval_fit(x=x), name='fit'))
    fig.add_trace(p1d.trace(x=x, data=fit.eval_init(x=x), name='init'))
    fig.show()
    print(f'Dat{dat.datnum}: G/T = {fit.best_values.g / fit.best_values.theta:.2f}')


def get_2d_i_sense(dat: DatHDF) -> Data2D:
    """
    Get the 2D data from the dat which is directly saved from measurement (in nA)

    Args:
        dat (): Dat object which interacts with HDF file

    Returns:

    """
    x = dat.Data.get_data('x')
    y = dat.Data.get_data('y')
    i_sense = dat.Data.get_data('i_sense')
    return Data2D(x=x, y=y, data=i_sense)


def get_2d_i_sense_csq_mapped(dat: DatHDF, csq_dat: DatHDF) -> Data2D:
    """
    Get the 2D data from the dat, and then map to csq using csq_dat
    Args:
        dat ():
        csq_dat():

    Returns:

    """
    setup_csq_dat(csq_dat.datnum, experiment_name=None, overwrite=False)
    calculate_csq_map(dat.datnum, experiment_name=None, csq_datnum=csq_dat.datnum, overwrite=False)
    data = Data2D(dat.Data.get_data('x'), dat.Data.get_data('y'), dat.Data.get_data('csq_mapped'))
    return data


def plot_2d_i_sense(data: Data2D, title_prepend: str = '', trace_type='heatmap', using_csq: bool = False) -> go.Figure:
    """
    Plot 2D i_sense data
    Args:
        data ():
        trace_type: heatmap or waterfall
        using_csq: If using csq mapping then change units etc

    Returns:

    """
    units = 'nA' if not using_csq else 'mV'
    fig = p2d.figure(xlabel='Sweepgate (mV)', ylabel=f'Current ({units})',
                     title=f'{title_prepend}2D I_sense in {units}')
    fig.add_trace(p2d.trace(data=data.data, x=data.x, y=data.y, trace_type=trace_type))
    return fig


def get_initial_params(data: Data1D, which='i_sense'):
    from src.dat_object.Attributes.Transition import get_param_estimates
    initial_params = get_param_estimates(x=data.x, data=data.data)
    if which == 'nrg':
        theta = initial_params['theta'].value
        initial_params.add_many(
            ('g', 1, True, theta / 1000, theta * 50),
            ('occ_lin', 0, True, -0.001, 0.001),
        )
    return initial_params


def fit_single_transition(data: Data1D, fit_with: str = 'i_sense',
                initial_params: Optional[lm.Parameters] = None) -> FitInfo:
    """
    Fit 1d_transition using either 'i_sense' function or 'nrg'
    Args:
        data (): 1D data to fit centers for
        fit_with (): Which function to use for fitting (i_sense or nrg)
        initial_params (): Optional initial params for fitting

    Returns:
        The fit results of fitting
    """
    if fit_with == 'i_sense':
        from src.dat_object.Attributes.Transition import i_sense
        func = i_sense
        method = 'leastsq'
    elif fit_with == 'nrg':
        from src.analysis_tools.nrg import NRG_func_generator
        func = NRG_func_generator(which='i_sense')
        method = 'powell'
    else:
        raise NotImplementedError

    if initial_params is None:
        initial_params = get_initial_params(data, which=fit_with)

    fit = calculate_fit(x=data.x, data=data.data, params=initial_params, func=func, method=method)
    return fit


def fit_2d_transition_data(data: Data2D, fit_with: str = 'i_sense',
                initial_params: Optional[lm.Parameters] = None) -> List[FitInfo]:
    """
    Fit 2d_transition using either 'i_sense' function or 'nrg'
    Args:
        data (): 2D data to fit centers for
        fit_with (): Which function to use for fitting (i_sense or nrg)
        initial_params (): Optional initial params for fitting

    Returns:
        The fit results of fitting
    """

    fits = [fit_single_transition(data=Data1D(x=data.x, data=d), fit_with=fit_with, initial_params=initial_params)
            for d in data.data]
    # fits = [calculate_fit(x=data.x, data=d, params=initial_params, func=func, method=method) for d in data.data]
    return fits


def add_centers_to_plot(fig: go.Figure, centers: Union[list, np.ndarray], ys: np.ndarray, color: str = 'white') -> go.Figure:
    fig.add_trace(p1d.trace(x=centers, data=ys, mode='markers', name='Centers',
                            trace_kwargs=dict(marker=dict(
                                color=color, size=3, symbol='circle',
                            ))))
    return fig


def run_and_plot_center_comparsion(fig: go.Figure, data: Data2D) -> go.Figure:
    """Runs both I_sense and NRG fits to each row of data and adds the center values of each to the figure provided
    Expects a 2D heatmap fig to be passed in"""
    i_sense_fits = fit_2d_transition_data(data, fit_with='i_sense', initial_params=None)
    nrg_params = get_initial_params(Data1D(data.x, data.data[0]), 'nrg')
    nrg_params['g'].value = 0.001
    nrg_params['g'].vary = False
    nrg_fits = fit_2d_transition_data(data, fit_with='nrg', initial_params=nrg_params)
    for fits, color in zip([i_sense_fits, nrg_fits], ['white', 'green']):
        centers = [f.best_values.mid for f in fits]
        fig = add_centers_to_plot(fig, centers, data.y, color=color)
    return fig


def run_and_plot_single_fit_comparison(fig: go.Figure, data: Data1D) -> go.Figure:
    """
    Runs both
    Args:
        fig ():
        data ():

    Returns:

    """
    i_sense_fit = fit_single_transition(data, fit_with='i_sense', initial_params=None)
    nrg_params = get_initial_params(data, 'nrg')
    nrg_params['g'].value = 0.001
    nrg_params['g'].vary = False
    nrg_params['occ_lin'].vary = False
    nrg_fit = fit_single_transition(data, fit_with='nrg', initial_params=nrg_params)
    for fit, name in zip([i_sense_fit, nrg_fit], ['i_sense', 'NRG']):
        fig.add_trace(p1d.trace(x=data.x, data=fit.eval_fit(x=data.x), name=name, mode='lines'))
    print(i_sense_fit)
    print(nrg_fit)
    return fig


def average_data(data: Data2D, centers) -> Data1D:
    """Averages 2D data using centers and returns Data1D"""
    avg_data, avg_x = mean_data(data.x, data.data, centers=centers, return_x=True)
    return Data1D(x=avg_x, data=avg_data)


def plot_single_transition(data: Data1D, title_prepend: str = '', using_csq: bool = False) -> go.Figure:
    """
    Plots single trace of transition
    Args:
        data ():
        title_prepend ():
        using_csq: Change units if using csq mapped data

    Returns:

    """
    units = 'nA' if not using_csq else 'mV'
    fig = p1d.figure(xlabel='Sweepgate (mV)', ylabel=f'Current ({units})',
                     title=f'{title_prepend}1D I_sense ({units})')
    fig.add_trace(p1d.trace(x=data.x, data=data.data, name='Data', mode='lines'))
    return fig


def compare_nrg_with_i_sense_for_single_dat(datnum: int,
                                            csq_map_datnum: Optional[int] = None,
                                            show_2d_centering_comparsion = False,
                                            show_1d_fit_comparison = True):
    dat = get_dat(datnum)
    if csq_map_datnum is not None:
        csq_dat = get_dat(csq_map_datnum)
        data = get_2d_i_sense_csq_mapped(dat, csq_dat)
        using_csq = True
    else:
        data = get_2d_i_sense(dat)
        using_csq = False

    if show_2d_centering_comparsion:
        fig = plot_2d_i_sense(data, title_prepend=f'Dat{dat.datnum}: ', trace_type='heatmap', using_csq=using_csq)
        run_and_plot_center_comparsion(fig, data).show()

    fits = fit_2d_transition_data(data, fit_with='i_sense', initial_params=None)
    centers = [f.best_values.mid for f in fits]
    avg_data = average_data(data, centers)

    if show_1d_fit_comparison:
        fig = plot_single_transition(avg_data, title_prepend=f'Dat{dat.datnum}: ', using_csq=using_csq)
        run_and_plot_single_fit_comparison(fig, avg_data).show()


def run_weakly_coupled_csq_mapped_nrg_fit(datnum: int, csq_datnum: int, overwrite: bool=False) -> FitInfo:
    dat = get_dat(datnum)
    csq_dat = get_dat(csq_datnum)

    # Use the NrgOcc Dat attribute to do the same calculations but with everything saved
    data = get_2d_i_sense_csq_mapped(dat=dat, csq_dat=csq_dat)
    fits = dat.Transition.get_row_fits(name='csq_i_sense', data=data.data, x=data.x, check_exists=False,
                                       overwrite=False)
    centers = [f.best_values.mid for f in fits]
    avg_data, avg_x = dat.NrgOcc.get_avg_data(x=data.x, data=data.data, centers=centers, return_x=True,
                                              name='csq_mapped',
                                              overwrite=False)
    avg_data = Data1D(avg_x, avg_data)
    pars = get_initial_params(avg_data, which='nrg')
    pars['g'].value = 0.005
    pars['g'].vary = False
    pars['occ_lin'].vary = False
    fit = dat.NrgOcc.get_fit(which='avg', name='csq_gamma_small',
                             initial_params=pars,
                             data=avg_data.data, x=avg_data.x,
                             calculate_only=False, check_exists=False)
    return fit


if __name__ == '__main__':
    # compare_nrg_with_i_sense_for_single_dat(datnum=2164, csq_map_datnum=2166,
    #                                         show_2d_centering_comparsion=False,
    #                                         show_1d_fit_comparison=True)

    dat = get_dat(2164)
    csq_dat = get_dat(2166)













