from plotly import graph_objs as go

import src.characters
from src.dat_object.dat_hdf import DatHDF
from src.plotting.plotly import OneD


def plot_stdev_of_avg_data(dat: DatHDF, data_type: str = 'transition', data_name: str = 'default') -> go.Figure:
    """Plot the stdev of averaging the 2D data (i.e. looking for whether more uncertainty near transition)"""
    plotter = OneD(dat=dat)

    fig = plotter.figure(
        ylabel=f'{src.characters.SIG}I_sense /nA',
        title=f'Dat{dat.datnum}: Standard deviation of averaged I_sense data after centering',
    )
    fig.add_trace(trace_stdev_of_avg_data(dat, data_type, data_name))
    return fig


def trace_stdev_of_avg_data(dat: DatHDF, data_type: str = 'transition', data_name: str = 'default') -> go.Scatter:
    """Trace for stdev of averaged 2D data"""
    if data_type.lower() == 'transition':
        _, stdev, x = dat.Transition.get_avg_data(name=data_name, return_x=True, return_std=True, check_exists=True)
    elif data_type.lower() == 'entropy':
        _, stdev, x = dat.Entropy.get_avg_data(name=data_name, return_x=True, return_std=True, check_exists=True)
    else:
        raise NotImplementedError(f'{data_type} not implemented')

    plotter = OneD(dat=dat)
    trace = plotter.trace(data=stdev, x=x,
                          name=f'Dat{dat.datnum}',
                          mode='lines')
    return trace