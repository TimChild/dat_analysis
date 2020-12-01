from __future__ import annotations
from src.DatObject.Attributes import Transition as T
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
        return [('Avg Fit', TransitionMainAvg()), 'Row Fits', TransitionMainRows()]

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
            self.graph_area('graph-main'),
            html.Div([self.graph_area('graph-secondary', 'Fit Graph')], id=self.id('div-secondary-graph'))
        ])
        return layout

    def set_callbacks(self):
        self.sidebar.layout()  # Make sure layout has been generated
        inps = self.sidebar.inputs

        # Show main graph
        self.graph_callback('graph-main', get_figure,
                            inputs=[(inps['inp-datnum'].id, 'value'),
                                    (inps['dd-data'].id, 'value'),
                                    (inps['sl-slicer'].id, 'value'),
                                    (inps['tog-slice'].id, 'value')],
                            states=[])

        # tog hide second graph
        self.make_callback((inps['tog-slice'].id, 'value'), (self.id('div-secondary-graph'), 'hidden'),
                           toggle_div)

        # Show linecut in second graph
        self.graph_callback('graph-secondary', plot_slice,
                            [(inps['inp-datnum'].id, 'value'),
                             (inps['sl-slicer'].id, 'value'),
                             (inps['tog-slice'].id, 'value')]
                            )


class TransitionMainRows(TransitionMainAvg):
    pass


@singleton
class TransitionSidebar(DatDashSideBar):

    @property
    def id_prefix(self):
        return 'Tsidebar'

    def layout(self):
        layout = html.Div([
            self.main_dropdown(),
            self.input_box(name='Dat', id_name='inp-datnum', placeholder='Choose Datnum', autoFocus=True, min=0),
            self.dropdown(name='Saved Fits', id_name='dd-saved-fits'),
            self.dropdown(name='Fit Func', id_name='dd-fit-func'),
            self.checklist(name='Param Vary', id_name='check-param-vary'),


            html.Div(self.dropdown(name='Figs', id_name='dd-figs', multi=True), id=self.id('div-figs-dd'), hidden=True),
            html.Div(self.dropdown(name='Data', id_name='dd-data'), id=self.id('div-data-dd'), hidden=False),
            self.toggle(name='Slice', id_name='tog-slice'),
            self.slider(name='Slicer', id_name='sl-slicer', updatemode='drag'),
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

    def set_callbacks(self):
        inps = self.inputs
        main = (self.main_dropdown().id, 'value')
        # Set Data options
        self.make_callback(
            inputs=[
                main,
                (inps['inp-datnum'].id, 'value')],
            outputs=[
                (self.id('div-data-dd'), 'hidden'),
                (inps['dd-data'].id, 'options')],
            func=get_data_options,
        )

        # Set Fig options
        self.make_callback(
            inputs=[
                main,
                (inps['inp-datnum'].id, 'value')],
            outputs=[
                (self.id('div-figs-dd'), 'hidden'),
                (inps['dd-figs'].id, 'options'),
            ],
            func=fig_name_dd_callback
        )

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