from __future__ import annotations
from src.Dash.DatSpecificDash import SharedFigs
from dictor import dictor
from singleton_decorator import singleton
import dash_html_components as html
from typing import List, Tuple, TYPE_CHECKING
import plotly.graph_objects as go
import numpy as np

from src.Dash.DatSpecificDash import SharedFigs
from src.Dash.DatSpecificDash import DatDashPageLayout, DatDashMain, DatDashSideBar
from src.Plotting.Plotly.PlotlyUtil import add_horizontal
from src.DatObject.Make_Dat import DatHandler
import src.UsefulFunctions as U
from dash.exceptions import PreventUpdate
import logging

if TYPE_CHECKING:
    from src.DatObject.DatHDF import DatHDF
get_dat = DatHandler().get_dat

logger = logging.getLogger(__name__)


class SharedLayout(DatDashPageLayout):
    def get_mains(self) -> List[Tuple[str, DatDashMain]]:
        return [('Shared Figure', SharedMain())]

    def get_sidebar(self) -> DatDashSideBar:
        return SharedSidebar()

    @property
    def id_prefix(self):
        return 'Shared'


class SharedMain(DatDashMain):
    name = 'Shared'

    def get_sidebar(self):
        return SharedSidebar()

    def layout(self):
        layout = html.Div([
            self.graph_area('graph-main', title='Shared Graph', datnum_id=None),
            html.Div([self.graph_area('graph-secondary', 'Slice Graph',
                                      datnum_id=None)],
                     id=self.id('div-secondary-graph'))
        ])
        return layout

    def set_callbacks(self):
        self.sidebar.layout()  # Make sure layout has been generated
        inps = self.sidebar.inputs
        # Show main graph
        self.graph_callback('graph-main', get_figure,
                            inputs=[
                                (inps['dd-figs'].id, 'value'),
                                (inps['sl-slicer'].id, 'value'),
                                (inps['tog-slice'].id, 'value'),
                            ],
                            states=[])

        # tog hide second graph
        self.make_callback((inps['tog-slice'].id, 'value'), (self.id('div-secondary-graph'), 'hidden'),
                           toggle_div)

        # Show linecut in second graph
        self.graph_callback('graph-secondary', plot_slice,
                            inputs=[
                                (self.id('graph-main'), 'figure'),
                                (inps['sl-slicer'].id, 'value'),
                                (inps['tog-slice'].id, 'value'),
                            ]),


# class SharedMainExistingFigs(SharedMain):
#
#     def layout(self):
#         layout = html.Div([
#             self.graph_area('graph-figs1', datnum_id=self.sidebar.id('inp-datnum')),
#             self.graph_area('graph-figs2', datnum_id=self.sidebar.id('inp-datnum')),
#             self.graph_area('graph-figs3', datnum_id=self.sidebar.id('inp-datnum'))
#         ])
#         return layout
#
#     def set_callbacks(self):
#         self.sidebar.layout()
#         inps = self.sidebar.inputs
#
#         # Show main graph
#         self.graph_callback('graph-figs1', partial(get_fig_dd_callback, fig_index=0),
#                             inputs=[
#                                 (self.sidebar.main_dropdown().id, 'value'),
#                                 (inps['inp-datnum'].id, 'value'),
#                                 (inps['dd-figs'].id, 'value')
#                             ])
#
#         # Show second graph
#         self.graph_callback('graph-figs2', partial(get_fig_dd_callback, fig_index=1),
#                             inputs=[
#                                 (self.sidebar.main_dropdown().id, 'value'),
#                                 (inps['inp-datnum'].id, 'value'),
#                                 (inps['dd-figs'].id, 'value')
#                             ])
#
#         # Show third graph
#         self.graph_callback('graph-figs3', partial(get_fig_dd_callback, fig_index=2),
#                             inputs=[
#                                 (self.sidebar.main_dropdown().id, 'value'),
#                                 (inps['inp-datnum'].id, 'value'),
#                                 (inps['dd-figs'].id, 'value')
#                             ])


@singleton
class SharedSidebar(DatDashSideBar):

    @property
    def id_prefix(self):
        return 'Ssidebar'

    def layout(self):
        layout = html.Div([
            # self.main_dropdown(),
            self.button(name='Refresh', id_name='but-refresh', color='success'),
            self.button(name='Reset', id_name='but-reset', color='danger'),
            html.Div(id=self.id('div-fake-output-reset'), style={'display': 'none'}),
            self.dropdown(name='Figs', id_name='dd-figs', multi=False),
            self.toggle(name='Slice', id_name='tog-slice'),
            self.slider(name='Slicer', id_name='sl-slicer', updatemode='drag'),
        ])
        return layout

    def set_callbacks(self):
        inps = self.inputs
        # main = (self.main_dropdown().id, 'value')
        refresh = (inps['but-refresh'].id, 'n_clicks')

        # Set Fig options
        self.make_callback(
            inputs=[
                # main,
                refresh,
            ],
            outputs=[
                (inps['dd-figs'].id, 'options'),
            ],
            func=fig_name_dd_callback
        )

        # Set slider bar for linecut
        self.make_callback(
            inputs=[
                (SharedMain().id('graph-main'), 'figure'),
            ],
            outputs=[
                (inps['sl-slicer'].id, 'min'),
                (inps['sl-slicer'].id, 'max'),
                (inps['sl-slicer'].id, 'step'),
                (inps['sl-slicer'].id, 'marks'),
            ],
            func=set_slider_vals
        )

        # Reset button
        self.make_callback(
            inputs=[
                (inps['but-reset'].id, 'n_clicks')
            ],
            outputs=[(self.id('div-fake-output-reset'), 'hidden')],
            func=reset_callback
        )


def fig_name_dd_callback(clicks) -> List[dict]:
    """
    Returns:
        List[dict]: options
    """
    if clicks:
        fig_names = SharedFigs().get_names()
        return [{'label': k, 'value': k} for k in fig_names]
    return []


def get_fig_dd_callback(main, datnum, fig_names, fig_index=0) -> go.Figure:
    if main != 'Shared_Existing Figures':
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
    if main != 'Shared_Default View':
        return True, []
    else:
        if datnum:
            dat: DatHDF = get_dat(datnum)
            data_names = set(dat.Data.keys)  # Set because it may have duplicates
            return False, [{'label': k, 'value': k} for k in sorted(data_names)]
    return False, []


def get_figure(fig_name, y_line, slice_tog):
    if fig_name:
        try:
            fig_dict = SharedFigs().get_fig(fig_name)
        except KeyError:
            fig_dict = None
        if fig_dict:
            fig = go.Figure(fig_dict)
            if slice_tog == [True] and fig_dict.get('data')[0].get('z', None) is not None:
                add_horizontal(fig, y_line)
            return fig
    return go.Figure()


# def get_figure(datnum, y_line, slice_tog):
#     if datnum:
#         dat = get_dat(datnum)
#
#         x = dat.Data.get_data('x')
#         y = dat.Data.get_data('y')
#         z = dat.Data.get_data('i_sense')
#
#         x, z = [U.bin_data_new(arr, int(round(z.shape[-1]/1250))) for arr in (x, z)]
#
#         fig = go.Figure()
#         fig.add_trace(go.Heatmap(x=x, y=y, z=z))
#         if slice_tog == [True]:
#             add_horizontal(fig, y_line)
#         return fig
#     else:
#         return go.Figure()


def set_slider_vals(fig: dict):
    if fig:
        # y = dat.Data.get_data('y')
        d = dictor(fig, 'data', [])
        if d:
            d = d[0]
            y = dictor(d, 'y', None)
            z = dictor(d, 'z', None)  # Only exists for 2D or something that can be sliced
            if y is not None and z is not None:
                start, stop = y[0], y[-1]
                step = abs((stop-start)/len(y))
                marks = {float(v): f'{v:.3g}' for v in np.linspace(start, stop, 10)}
                return [start, stop, step, marks]
    return [0, 1, 0.1, {0: '0', 0.5: '0.5', 1: '1'}]


def plot_slice(fig, slice_val, slice_tog):
    if slice_tog == [True]:
        if not slice_val:
            slice_val = 0
        d = fig.get('data', None)
        if d:
            d = d[0]
            x = d.get('x', None)
            y = d.get('y', None)
            z = d.get('z', None)

            if all([a is not None for a in [x, y, z]]):
                # x, z = [U.bin_data_new(arr, int(round(z.shape[-1] / 500))) for arr in (x, z)]
                slice_index = U.get_data_index(y, slice_val)
                data = z[slice_index]

                fig = go.Figure()
                fig.add_trace(go.Scatter(mode='lines', x=x, y=data))
                fig.update_layout(title=f'Slice at y = {y[slice_index]:.1f}')
                return fig
    return go.Figure()


def toggle_div(value):
    if value == [True]:
        return False
    else:
        return True


def reset_callback(clicks):
    if clicks:
        SharedFigs().del_figs()
    return True


# Generate layout for to be used in App
layout = SharedLayout().layout()

if __name__ == '__main__':
    dat = get_dat(9111)
