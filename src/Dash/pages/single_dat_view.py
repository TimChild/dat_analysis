from singleton_decorator import singleton
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
from src.Dash.BaseClasses import BasePageLayout, BaseMain, BaseSideBar


class SingleDatLayout(BasePageLayout):
    def get_mains(self) -> List[Tuple[str, BaseMain]]:
        return [('Page1', SingleDatMain())]

    def get_sidebar(self) -> BaseSideBar:
        return SingleDatSidebar()

    @property
    def id_prefix(self):
        return 'SD'


class SingleDatMain(BaseMain):

    def get_sidebar(self):
        return SingleDatSidebar()

    @property
    def id_prefix(self):
        return 'SDmain'

    def layout(self):
        layout = html.Div([
            self.graph_area(name=self.id('graph-main'))
        ])
        self.init_callbacks()
        return layout

    def init_callbacks(self):
        self.graph_callback('graph-main', get_figure,
                            [(self.sidebar.id('inp-datnum'), 'value')])

    def set_callbacks(self):
        pass

@singleton
class SingleDatSidebar(BaseSideBar):

    def get_main_callback_outputs(self) -> List[Tuple[str, str]]:
        pass

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


# Generate layout for to be used in App
layout = SingleDatLayout().layout()


