from __future__ import annotations
from src.DatObject.Attributes import Transition as T
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from singleton_decorator import singleton
import dash_html_components as html
from typing import List, Tuple, TYPE_CHECKING
import plotly.graph_objects as go
import numpy as np
from src.Dash.DatSpecificDash import DatDashPageLayout, DatDashMain, DatDashSideBar, DashOneD, DashTwoD, DashThreeD
from src.Plotting.Plotly.PlotlyUtil import add_horizontal
from src.DatObject.Make_Dat import DatHandler
import src.UsefulFunctions as U
from dash.exceptions import PreventUpdate
import logging
from functools import partial

if TYPE_CHECKING:
    from src.DatObject.DatHDF import DatHDF
get_dat = DatHandler().get_dat

logger = logging.getLogger(__name__)


class TransitionLayout(DatDashPageLayout):
    def get_mains(self) -> List[Tuple[str, DatDashMain]]:
        return [('Avg Fit', TransitionMainAvg()), ('Row Fits', TransitionMainRows())]

    def get_sidebar(self) -> DatDashSideBar:
        return TransitionSidebar()

    @property
    def id_prefix(self):
        return 'T'


class TransitionMainAvg(DatDashMain):

    def get_sidebar(self):
        return TransitionSidebar()

    @property
    def id_prefix(self):
        return 'Tmain'

    def layout(self):
        layout = html.Div([
            self.graph_area('graph-avg', 'Avg Fit'),
            [self.graph_area('graph-twoD', 'Data')],
        ],
            id=self.id('div-avg-graphs'))
        return layout

    def set_callbacks(self):
        self.sidebar.layout()  # Make sure layout has been generated
        inps = self.sidebar.inputs

        # Show main graph
        self.graph_callback('graph-avg', partial(get_figure, mode='avg'),
                            inputs=[
                                (inps['inp-datnum'].id, 'value'),
                            ],
                            states=[])

        # Show 2D data
        self.graph_callback('graph-twoD', partial(get_figure, mode='twoD'),
                            inputs=[
                                (inps['inp-datnum'].id, 'value'),
                            ])


class TransitionMainRows(TransitionMainAvg):

    def layout(self):
        layout = html.Div([
            self.graph_area('graph-row', 'Row Fit'),
            self.graph_area('graph-waterfall', 'All Rows'),
        ],
            id=self.id('div-row-graphs'))
        return layout

    def set_callbacks(self):
        self.sidebar.layout()
        inps = self.sidebar.inputs

        # Single row graph
        self.graph_callback('graph-row', partial(get_figure, mode='single_row'),
                            inputs=[
                                (inps['inp-datnum'].id, 'value'),
                                (inps['sl-slicer'].id, 'value'),
                            ],
                            states=[])

        # Waterfall graph
        self.graph_callback('graph-waterfall', partial(get_figure, mode='waterfall'),
                            inputs=[
                                (inps['inp-datnum'].id, 'value')
                            ])




@singleton
class TransitionSidebar(DatDashSideBar):

    @property
    def id_prefix(self):
        return 'Tsidebar'

    def layout(self):
        layout = html.Div([
            self.main_dropdown(),  # Choice between Avg view and Row view
            self.input_box(name='Dat', id_name='inp-datnum', placeholder='Choose Datnum', autoFocus=True, min=0),
            self.dropdown(name='Saved Fits', id_name='dd-saved-fits'),
            self.dropdown(name='Fit Func', id_name='dd-fit-func'),
            self.checklist(name='Param Vary', id_name='check-param-vary'),
            self._param_inputs(),

            html.Div(self.slider(name='Slicer', id_name='sl-slicer', updatemode='drag'), id=self.id('div-slicer')),
        ])

        # Set options here so it isn't so cluttered in layout above
        self.dropdown(id_name='dd-fit-func').options = [
            {'label': 'i_sense', 'value': 'i_sense'},
            {'label': 'i_sense_digamma', 'value': 'i_sense_digamma'},
            {'label': 'i_sense_digamma_quad', 'value': 'i_sense_digamma_quad'},
        ]
        self.checklist(id_name='check-param-vary').options = [
            {'label': 'theta', 'value': 'theta'},
            {'label': 'amp', 'value': 'amp'},
            {'label': 'gamma', 'value': 'gamma'},
            {'label': 'lin', 'value': 'lin'},
            {'label': 'const', 'value': 'const'},
            {'label': 'mid', 'value': 'mid'},
            {'label': 'quad', 'value': 'quad'},
        ]
        return layout

    def _param_inputs(self):
        par_input = dbc.Row([
            dbc.Col(
                dbc.FormGroup(
                    [
                        dbc.Label('theta', html_for=self.id('inp-theta')),
                        dbc.Input(type='value', id=self.id('inp-theta'))
                    ]
                )
            ),
            dbc.Col(
                dbc.FormGroup(
                    [
                        dbc.Label('amp', html_for=self.id('inp-amp')),
                        dbc.Input(type='value', id=self.id('inp-amp'))
                    ]
                )
            ),
            dbc.Col(
                dbc.FormGroup(
                    [
                        dbc.Label('gamma', html_for=self.id('inp-gamma')),
                        dbc.Input(type='value', id=self.id('inp-gamma'))
                    ]
                )
            ),
            dbc.Col(
                dbc.FormGroup(
                    [
                        dbc.Label('lin', html_for=self.id('inp-lin')),
                        dbc.Input(type='value', id=self.id('inp-lin'))
                    ]
                )
            ),
            dbc.Col(
                dbc.FormGroup(
                    [
                        dbc.Label('const', html_for=self.id('inp-const')),
                        dbc.Input(type='value', id=self.id('inp-const'))
                    ]
                )
            ),
            dbc.Col(
                dbc.FormGroup(
                    [
                        dbc.Label('mid', html_for=self.id('inp-mid')),
                        dbc.Input(type='value', id=self.id('inp-mid'))
                    ]
                )
            ),
            dbc.Col(
                dbc.FormGroup(
                    [
                        dbc.Label('quad', html_for=self.id('inp-quad')),
                        dbc.Input(type='value', id=self.id('inp-quad'))
                    ]
                )
            ),
        ])
        return par_input

    def set_callbacks(self):
        inps = self.inputs
        main = (self.main_dropdown().id, 'value')

        # Set slider bar for linecut
        self.make_callback(
            inputs=[
                (inps['inp-datnum'].id, 'value')],
            outputs=[
                (inps['sl-slicer'].id, 'min'),
                (inps['sl-slicer'].id, 'max'),
                (inps['sl-slicer'].id, 'step'),
                (inps['sl-slicer'].id, 'value'),
                (inps['sl-slicer'].id, 'marks'),
            ],
            func=set_slider_vals)


def get_figure(*args, mode='avg'):
    return go.Figure()


def set_slider_vals(datnum):
    if datnum:
        dat = get_dat(datnum)
        y = dat.Data.get_data('y')
        start, stop, step, value = 0, len(y)-1, 1, round(len(y)/2)
        marks = {int(v): str(v) for v in np.arange(start, stop, 10)}
        return start, stop, step, value, marks
    return 0, 1, 0.1, 0.5, {0: '0', 0.5: '0.5', 1: '1'}


def toggle_div(value):
    if value == [True]:
        return False
    else:
        return True


def get_saved_fit_names(datnum) -> List[dict]:
    if datnum:
        dat = get_dat(datnum)
        fit_paths = dat.Transition.fit_paths
        # TODO: get names from fit_paths
        fit_names = ''
        return [{'label': k, 'value': k} for k in fit_names]


def run_fits(fit_func, param_vary):
    if fit_func == 'i_sense':
        func = T.i_sense
        pars_names = ['const', 'mid', 'amp', 'lin', 'theta']
    elif fit_func == 'i_sense_digamma':
        func = T.i_sense_digamma
        pars_names = ['const', 'mid', 'amp', 'lin', 'theta', 'gamma']
    elif fit_func == 'i_sense_digamma_quad':
        func = T.i_sense_digamma_quad
        pars_names = ['const', 'mid', 'amp', 'lin', 'theta', 'gamma', 'quad']
    else:
        raise ValueError(f'{fit_func} is not recognized as a fit_function for Transition')


# Generate layout for to be used in App
layout = TransitionLayout().layout()

if __name__ == '__main__':
    dat = get_dat(9111)
