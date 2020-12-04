from typing import List, Tuple
from singleton_decorator import singleton
from src.Dash.BaseClasses import BasePageLayout, BaseMain, BaseSideBar
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output
from src.Dash.app import app

import plotly.graph_objects as go
import numpy as np


class TestLayout(BasePageLayout):

    def get_mains(self):
        return [('Page1', TestMain1()), ('Page2', TestMain2())]

    def get_sidebar(self):
        return TestSidebar()

    @property
    def id_prefix(self):
        return 'Test'


class TestMain1(BaseMain):
    def get_sidebar(self) -> BaseSideBar:
        return TestSidebar()

    @property
    def id_prefix(self):
        return 'TestMain1'

    def layout(self):
        layout = html.Div([
            self.graph_area(name='graph-1a', title='Graph 1a', default_fig=figs[0]),
            self.graph_area(name='graph-1b', title='Graph 1b', default_fig=figs[1])
        ])

        return layout

    def set_callbacks(self):
        pass


class TestMain2(BaseMain):
    def get_sidebar(self) -> BaseSideBar:
        return TestSidebar()

    @property
    def id_prefix(self):
        return 'TestMain2'

    def layout(self):
        layout = html.Div([
            self.graph_area(name='graph-2a', title='Graph 2a', default_fig=figs[2]),
            self.graph_area(name='graph-2b', title='Graph 2b', default_fig=figs[3])
        ])
        return layout

    def set_callbacks(self):

        pass


@singleton
class TestSidebar(BaseSideBar):

    def set_callbacks(self):
        pass

    @property
    def id_prefix(self):
        return 'TestSidebar'

    def layout(self):
        layout = html.Div([
            self.main_dropdown(),
            self.input_box(name='Dat', id_name=self.id('inp-datnum'), placeholder='Choose Datnum', autoFocus=True, min=0)
        ])
        return layout


x = np.linspace(0, 10, 100)
y = x
xx, yy = np.meshgrid(x, y)
datas = [np.cos(xx)*i+np.sin(yy)*j for i, j in zip([1, 2, 3, 4], [4, 3, 2, 1])]
figs = [go.Figure(go.Heatmap(x=x, y=y, z=data)) for data in datas]

layout = TestLayout().layout()
