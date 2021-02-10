from __future__ import annotations
import dash_core_components as dcc
from singleton_decorator import singleton
import dash_html_components as html
import dash_bootstrap_components as dbc
from typing import List, Tuple, TYPE_CHECKING, Dict, Any
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


class SingleDatLayout(DatDashPageLayout):
    def get_mains(self) -> List[Tuple[str, DatDashMain]]:
        return [('Default View', SingleDatMain()), ('Existing Figures', SDMainExistingFigs())]

    def get_sidebar(self) -> DatDashSideBar:
        return SingleDatSidebar()

    @property
    def id_prefix(self):
        return 'SD'

    def layout(self):
        layout = dbc.Container(
            [
                dbc.Row(dbc.Col(self.top_bar_layout())),
                dbc.Row([
                    dbc.Col(self.main_area_layout(), width=7), dbc.Col(self.side_bar_layout())
                ])
            ], fluid=True
        )
        self.run_all_callbacks()

        return layout


class SingleDatMain(DatDashMain):

    def get_sidebar(self):
        return SingleDatSidebar()

    @property
    def id_prefix(self):
        return 'SDmain'

    def layout(self):
        layout = html.Div([
            self.graph_area('graph-main', datnum_id=self.sidebar.id('inp-datnum')),
            html.Div([self.graph_area('graph-secondary', 'Slice Graph',
                                      datnum_id=self.sidebar.id('inp-datnum'))],
                     id=self.id('div-secondary-graph'))
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


class SDMainExistingFigs(SingleDatMain):

    def layout(self):
        layout = html.Div([
            self.graph_area('graph-figs1', datnum_id=self.sidebar.id('inp-datnum')),
            self.graph_area('graph-figs2', datnum_id=self.sidebar.id('inp-datnum')),
            self.graph_area('graph-figs3', datnum_id=self.sidebar.id('inp-datnum'))
        ])
        return layout

    def set_callbacks(self):
        self.sidebar.layout()
        inps = self.sidebar.inputs

        # Show main graph
        self.graph_callback('graph-figs1', partial(get_fig_dd_callback, fig_index=0),
                            inputs=[
                                (self.sidebar.main_dropdown().id, 'value'),
                                (inps['inp-datnum'].id, 'value'),
                                (inps['dd-figs'].id, 'value')
                            ])

        # Show second graph
        self.graph_callback('graph-figs2', partial(get_fig_dd_callback, fig_index=1),
                            inputs=[
                                (self.sidebar.main_dropdown().id, 'value'),
                                (inps['inp-datnum'].id, 'value'),
                                (inps['dd-figs'].id, 'value')
                            ])

        # Show third graph
        self.graph_callback('graph-figs3', partial(get_fig_dd_callback, fig_index=2),
                            inputs=[
                                (self.sidebar.main_dropdown().id, 'value'),
                                (inps['inp-datnum'].id, 'value'),
                                (inps['dd-figs'].id, 'value')
                            ])


@singleton
class SingleDatSidebar(DatDashSideBar):

    @property
    def id_prefix(self):
        return 'SDsidebar'

    def layout(self):
        layout = html.Div([
            self.main_dropdown(),
            self.input_box(name='Dat', id_name='inp-datnum', placeholder='Choose Datnum', autoFocus=True, min=0),
            html.Div(self.dropdown(name='Figs', id_name='dd-figs', multi=True), id=self.id('div-figs-dd'), hidden=True),
            html.Div(self.dropdown(name='Data', id_name='dd-data'), id=self.id('div-data-dd'), hidden=False),
            self.toggle(name='Slice', id_name='tog-slice'),
            self.slider(name='Slicer', id_name='sl-slicer', updatemode='drag'),
            html.Hr(),
            html.H4('Logs'),
            self.logs_layout(),
        ])
        return layout

    def logs_layout(self) -> html.Div:
        def entry(title: str, info: Dict[str, Any]) -> html.Div:
            ent = dbc.Col(
                dbc.Card([
                    dbc.CardHeader(title),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(html.B(k)), dbc.Col(html.P(v)),
                        ]) for k, v in info.items()
                    ])
                ], ), width=4)
            return ent

        layout = html.Div(id=self.id('div-logs'),
                          children=[
                              entry(title='Temperatures', info={'50k': 58.23123, '4k': 4.9823, 'still': 1.09931})
                          ],
                          )
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


def fig_name_dd_callback(main, datnum) -> Tuple[bool, List[dict]]:
    """
    Args:
        main (): which page
        datnum (): datnum

    Returns:
        bool: visible
        List[dict]: options
    """
    if main != 'SD_Existing Figures':
        return True, []
    elif datnum:
        dat = get_dat(datnum)
        fig_names = dat.Figures.all_fig_names
        return False, [{'label': k, 'value': k} for k in fig_names]
    return False, []


def get_fig_dd_callback(main, datnum, fig_names, fig_index=0) -> go.Figure:
    if main != 'SD_Existing Figures':
        raise PreventUpdate
    elif datnum and fig_names:
        dat = get_dat(datnum)
        if fig_index < len(fig_names):
            target_fig = fig_names[fig_index]
            fig = dat.Figures.get_fig(name=target_fig)
            if fig is not None:
                return fig
    return go.Figure()


def get_data_options(main, datnum: int) -> Tuple[bool, List[dict]]:
    """Returns available data_keys as dcc options"""
    if main != 'SD_Default View':
        return True, []
    else:
        if datnum:
            dat: DatHDF = get_dat(datnum)
            data_names = set(dat.Data.keys)  # Set because it may have duplicates
            return False, [{'label': k, 'value': k} for k in sorted(data_names)]
    return False, []


def get_figure(datnum, data_name, y_line, slice_tog):
    print(datnum, data_name, y_line, slice_tog)
    if datnum and data_name:
        dat = get_dat(datnum)
        data = dat.Data.get_data(data_name)
        if data.ndim == 1:
            plotter = DashOneD(dat=dat)
        elif data.ndim == 2:
            plotter = DashTwoD(dat=dat)
        elif data.ndim == 3:
            plotter = DashThreeD(dat=dat)
        else:
            logger.warning(f'{data.shape} is not supported for plotting')
            return go.Figure()

        fig = plotter.plot(data, title=data_name)
        if slice_tog == [True]:
            add_horizontal(fig, y_line)
        return fig
    return go.Figure()


def set_slider_vals(datnum):
    if datnum:
        dat = get_dat(datnum)
        y = dat.Data.get_data('y')
        start, stop, step, value = 0, len(y) - 1, 1, round(len(y) / 2)
        marks = {int(v): str(v) for v in np.arange(start, stop, 10)}
        return start, stop, step, value, marks
    return 0, 1, 0.1, 0.5, {0: '0', 0.5: '0.5', 1: '1'}


def plot_slice(datnum, slice_val, slice_tog):
    if slice_tog == [True]:
        slice_val = int(slice_val)
        dat = get_dat(datnum)

        x = dat.Data.get_data('x')
        y = dat.Data.get_data('y')
        z = dat.Data.get_data('i_sense')

        x, z = [U.bin_data_new(arr, int(round(z.shape[-1] / 500))) for arr in (x, z)]

        data = z[slice_val]

        fig = go.Figure()
        fig.add_trace(go.Scatter(mode='lines', x=x, y=data))
        fig.update_layout(title=f'Slice at y = {y[slice_val]:.1f}')
        return fig
    raise PreventUpdate


def toggle_div(value):
    if value == [True]:
        return False
    else:
        return True


# Generate layout for to be used in App
layout = SingleDatLayout().layout()

if __name__ == '__main__':
    dat = get_dat(9111)
