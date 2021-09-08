"""
Sep 21 -- Analysis from early on in measurement when dot tuning.

"""
from src.dat_object.make_dat import get_dat, get_dats
from src.analysis_tools.transition import calculate_transition_only_fit
from Analysis.Feb2021.common_plotting import transition_trace, transition_fig

from typing import TYPE_CHECKING, List, Union
from progressbar import progressbar
import plotly.io as pio
from itertools import chain

if TYPE_CHECKING:
    from src.dat_object.make_dat import DatHDF
    import plotly.graph_objects as go

pio.renderers.default = 'browser'


def process_dats(dats: List[DatHDF],
                 transition_fit_save_name: str = 'default',
                 transition_fit_func: str = 'i_sense') -> None:
    """
    Basically just does a transition fit on each dat but catches errors (which are common where scans might miss transition)

    Args:
        dats ():
        transition_fit_save_name ():
        transition_fit_func ():

    Returns:

    """
    for dat in progressbar(dats):
        try:
            calculate_transition_only_fit(dat.datnum, save_name=transition_fit_save_name,
                                          t_func_name=transition_fit_func,
                                          theta=None, gamma=0, width=None)
        except (TypeError, ValueError):  # Often rough scans which miss transition so fits fail
            print(f'Dat{dat.datnum}: Failed to calculate')


def plot_transition_data(dats: Union[List[List[DatHDF]], List[DatHDF]],
                         param_to_plot: str = 'amp/const',
                         fit_name: str = 'default',
                         x_axis_gate: str = "ESC",
                         other_gates_to_display: List[str] = ('ESS', 'CSS'),
                         ) -> go.Figure:
    """
    For plotting information from transition fits to data (i.e. good for finding best sensitivity of regime etc)

    Args:
        dats (): A single list of dats, or a list of list of dats (i.e. varying more than one axis)
        param_to_plot ():
        fit_name ():
        x_axis_gate ():
        other_gates_to_display ():

    Returns:

    """
    if not isinstance(dats[0], list):
        dats = [dats]  # Ensure it is chunks of dats (i.e. multiple sets of dats where another axis is also varied)
    dats: List[List[DatHDF]]  # For type checking only

    fig = transition_fig(dats=[dats[0][0], dats[-1][-1]],  # from first dat of first group to last dat of last group
                         xlabel=f'{x_axis_gate} /mV',
                         title_append=f' vs {x_axis_gate} for Transition Only scans',
                         param=param_to_plot)
    for all_dats in dats:
        dat = all_dats[0]  # Just use first dat of each set to get extra info from (so user much chunk dats accordingly)
        label = f'Dats{all_dats[0].datnum}-{all_dats[-1].datnum}: '
        for k in other_gates_to_display:
            label += f'{k}={dat.Logs.dacs[k]:.1f}mV, '

        all_dats = [dat for dat in all_dats if fit_name in dat.Transition.fit_names]  # Only those with fits saved
        fig.add_trace(transition_trace(all_dats, x_func=lambda dat: dat.Logs.fds['ESC'], from_square_entropy=False,
                                       fit_name=fit_name, param=param, label=label, mode='markers+lines'))
        return fig


if __name__ == '__main__':
    # datnum_chunks = [list(range(start, start+6)) for start in range(6104, 6165+1, 7)]  # 6 because there are also CSQ scans
    # datnum_chunks = [list(range(start, start+6)) for start in range(6167, 6223+1, 7)]  # 6 because there are also CSQ scans
    # datnum_chunks = [list(range(start, start+6)) for start in range(6257, 6345+1, 11)]  # 6 because there are also CSQ scans
    # datnum_chunks = [list(range(start, start+6)) for start in range(6356, 6389+1, 11)]  # 6 because there are also CSQ scans
    all_dats = [get_dats(list(range(start, start + 6))) for start in
                     range(6414, 6424 + 1, 10)]  # 6 because there are also CSQ scans

    fit_name = 'default'
    param = 'amp/const'

    all_dats_flat = list(chain(*all_dats))
    process_dats(all_dats_flat)

    fig = plot_transition_data(all_dats, param_to_plot=param, fit_name=fit_name,
                               x_axis_gate='ESC',
                               other_gates_to_display=['ESS', 'CSS'])
    fig.show()
