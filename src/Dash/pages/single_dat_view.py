from __future__ import annotations
import dash_core_components as dcc
from singleton_decorator import singleton
import dash_html_components as html
import dash_bootstrap_components as dbc
from typing import List, Tuple, TYPE_CHECKING, Dict, Any, Union
import plotly.graph_objects as go
import numpy as np
from src.Dash.DatSpecificDash import DatDashPageLayout, DatDashMain, DatDashSideBar, DashOneD, DashTwoD, DashThreeD
from src.plotting.plotly.plotly_util import add_horizontal
from src.dat_object.make_dat import DatHandler
import src.useful_functions as U
from dash.exceptions import PreventUpdate
import logging
from functools import partial

if TYPE_CHECKING:
    from src.dat_object.dat_hdf import DatHDF
get_dat = DatHandler().get_dat

logger = logging.getLogger(__name__)


class SingleDatLayout(DatDashPageLayout):
    id_prefix = 'SD'
    def get_mains(self) -> List[Tuple[str, DatDashMain]]:
        return [('Default View', SingleDatMain()), ('Existing Figures', SDMainExistingFigs())]

    def get_sidebar(self) -> DatDashSideBar:
        return SingleDatSidebar()

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
    name = 'SD'

    def get_sidebar(self):
        return SingleDatSidebar()

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
        datnum = (inps['inp-datnum'].id, 'value')
        slice_val = (inps['sl-slicer'].id, 'value')
        data_name = (inps['dd-data'].id, 'value')

        # Show main graph
        self.graph_callback('graph-main', func=get_figure,
                            inputs=[
                                datnum,
                                data_name,
                                slice_val,
                            ],
                            states=[])

        # tog hide second graph
        self.make_callback(
            inputs=[
                datnum,
                data_name,
            ],
            outputs=(self.id('div-secondary-graph'), 'hidden'),
            func=show_slice,
        )

        # Show linecut in second graph
        self.graph_callback('graph-secondary', func=plot_slice,
                            inputs=[
                                datnum,
                                data_name,
                                slice_val,
                            ])


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
            html.Div(self.slider(name='Slicer', id_name='sl-slicer', updatemode='mouseup'),
                     id=self.id('div-slicer'), hidden=False),
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

        # layout = html.Div(id=self.id('div-logs'),
        #                   children=[
        #                       entry(title='Temperatures', info={'50k': 58.23123, '4k': 4.9823, 'still': 1.09931})
        #                   ],
        #                   )
        layout = html.Div()
        return layout

    def set_callbacks(self):
        inps = self.inputs
        main = (self.main_dropdown().id, 'value')
        datnum = (inps['inp-datnum'].id, 'value')
        data_name = (inps['dd-data'].id, 'value')

        # Set Data options
        self.make_callback(
            inputs=[
                main,
                datnum,
                ],
            outputs=[
                (self.id('div-data-dd'), 'hidden'),
                (inps['dd-data'].id, 'options')],
            func=get_data_options,
        )

        # Set Fig options
        self.make_callback(
            inputs=[
                main,
                datnum,
            ],
            outputs=[
                (self.id('div-figs-dd'), 'hidden'),
                (inps['dd-figs'].id, 'options'),
            ],
            func=fig_name_dd_callback
        )

        # Set slider bar for linecut
        self.make_callback(
            inputs=[
                datnum,
                ],
            outputs=[
                (inps['sl-slicer'].id, 'min'),
                (inps['sl-slicer'].id, 'max'),
                (inps['sl-slicer'].id, 'step'),
                (inps['sl-slicer'].id, 'value'),
                (inps['sl-slicer'].id, 'marks'),
            ],
            func=set_slider_vals)

        # Show slider
        self.make_callback(
            inputs=[
                datnum,
                data_name
            ],
            outputs=(self.id('div-slicer'), 'hidden'),
            func=show_slice,

        )


def show_slice(datnum, data_name) -> bool:
    if datnum:
        dat = get_dat(datnum)
        data = dat.Data.get_data(data_name)
        if data.ndim > 1:
            return False
    return True


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


def get_figure(datnum, data_name, y_line):
    print(datnum, data_name, y_line)
    if datnum and data_name:
        dat = get_dat(datnum)
        data = dat.Data.get_data(data_name)
        x = dat.Data.get_data('x')
        if data.ndim == 1:
            plotter = DashOneD(dat=dat)
        elif data.ndim == 2:
            plotter = DashTwoD(dat=dat)
        elif data.ndim == 3:
            plotter = DashThreeD(dat=dat)
        else:
            logger.warning(f'{data.shape} is not supported for plotting')
            return go.Figure()

        if data.shape[-1] != x.shape[0]:
            x = np.linspace(0, 1, data.shape[-1])

        fig = plotter.plot(data, x=x, title=f'Dat{datnum}: {data_name}')
        if data.ndim > 1:
            add_horizontal(fig, y_line)
        return fig
    return go.Figure()


def set_slider_vals(datnum):
    def val_to_str(val: float) -> str:
        if abs(val - int(val)) < 0.001:
            return str(int(val))
        else:
            return f'{U.sig_fig(val, sf=3):g}'

    def dash_bug_fix(val: float) -> Union[float, int]:
        """If slider key is divisible by 1, it must be an integer, otherwise can be aa float"""
        if abs(val - int(val)) < 0.0001:
            return int(val)
        else:
            return val

    if datnum:
        dat = get_dat(datnum)
        y = dat.Data.get_data('y')
        start, stop, step, value = y[0], y[-1], (y[-1]-y[0])/len(y), np.mean(y)
        marks = {dash_bug_fix(v): val_to_str(v) for v in np.linspace(start, stop, 10)}
        return start, stop, step, value, marks
    return 0, 1, 0.1, 0.5, {0: '0', 0.5: '0.5', 1: '1'}


def plot_slice(datnum, data_name, slice_val) -> go.Figure:
    if datnum:
        dat = get_dat(datnum)
        data = dat.Data.get_data(data_name)
        if data.ndim > 1:
            x = dat.Data.get_data('x')
            y = dat.Data.get_data('y')
            if slice_val is None:
                slice_val = y[0]
            slice_val = U.get_data_index(y, slice_val)
            z = dat.Data.get_data('i_sense')[slice_val]

            x, z = [U.bin_data_new(arr, int(round(z.shape[-1] / 500))) for arr in (x, z)]

            plotter = DashOneD(dat=dat)
            fig = plotter.plot(data=z, x=x, mode='lines', title=f'Dat{datnum}: Slice at y = {y[slice_val]:.1f}')
            return fig
    return go.Figure()


def toggle_div(value):
    if value == [True]:
        return False
    else:
        return True


# Generate layout for to be used in App
layout = SingleDatLayout().layout()

if __name__ == '__main__':
    dat = get_dat(9111)
