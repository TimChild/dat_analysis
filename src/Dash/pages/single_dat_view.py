import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, MATCH, ALL, ALLSMALLER
from typing import List, Union, Optional, Tuple
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio
from src.Dash.app import app  # To access callbacks
from src.Dash.BaseClasses import BasePageLayout, BaseMainArea, BaseSideBar




class SingleDatLayout(BasePageLayout):
    @property
    def id_prefix(self):
        return 'SD'

    def side_bar_layout(self):
        return SingleDatSidebar().layout()

    def main_area_layout(self):
        return SingleDatMain().layout()


class SingleDatMain(BaseMainArea):

    @property
    def id_prefix(self):
        return 'SDmain'

    def layout(self):
        self.graph_area_callback(self.id('graph-main'), get_figure,
                                 [Input(SingleDatSidebar().id('inp-datnum'), 'value')])
        return html.Div([
            self.graph_area(id=self.id('graph-main'))
        ])


class SingleDatSidebar(BaseSideBar):

    @property
    def id_prefix(self):
        return 'SDsidebar'

    def layout(self):
        layout = html.Div([
            self.input_box(name='Dat', id=self.id('inp-datnum'), placeholder='Choose Datnum', autoFocus=True, min=0)
        ])
        return layout


def get_figure(scan_num):
    # Get figure here
    if scan_num:
        fig = go.Figure()
        x = np.linspace(0, 10, 100)
        y = np.linspace(0, 15, 100)
        xx, yy = np.meshgrid(x, y)
        data = np.cos(xx)*scan_num + np.sin(yy)
        fig.add_trace(go.Heatmap(z=data))
        return fig
    else:
        return go.Figure()




layout = SingleDatLayout().layout()


