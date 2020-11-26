from typing import List, Tuple
from src.Dash.BaseClasses import BasePageLayout, BaseMainArea, BaseSideBar
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output
from src.Dash.app import app

import plotly.graph_objects as go
import numpy as np

class TestLayout(BasePageLayout):
    @property
    def id_prefix(self):
        return 'Test'

    def main_area_layout(self):
        layout = html.Div([
            html.Div(TestMain1().layout(), id=self.id('div-main1')),
            html.Div(TestMain2().layout(), id=self.id('div-main2')),
        ])
        return layout

    def side_bar_layout(self):
        return TestSidebar().layout()


class TestMain1(BaseMainArea):
    @property
    def id_prefix(self):
        return 'TestMain1'

    def layout(self):
        layout = html.Div([
            self.graph_area(id=self.id('graph-1a'), name='Graph 1a', default_fig=figs[0]),
            self.graph_area(id=self.id('graph-1b'), name='Graph 1b', default_fig=figs[1])
        ])

        return layout


class TestMain2(BaseMainArea):
    @property
    def id_prefix(self):
        return 'TestMain2'

    def layout(self):
        layout = html.Div([
            self.graph_area(id=self.id('graph-2a'), name='Graph 2a', default_fig=figs[2]),
            self.graph_area(id=self.id('graph-2b'), name='Graph 2b', default_fig=figs[3])
        ])
        return layout


class TestSidebar(BaseSideBar):

    @property
    def id_prefix(self):
        return 'TestSidebar'

    def layout(self):
        layout = html.Div([
            self.main_dropdown(id=self.id('dd-main')),
            self.input_box(name='Dat', id=self.id('inp-datnum'), placeholder='Choose Datnum', autoFocus=True, min=0)
        ])
        return layout

    def get_main_options(self) -> List[dict]:
        return [{'label': 'First', 'value': 0}, {'label': 'Second', 'value': 1}]

    def get_main_callback_outputs(self) -> List[Tuple[str, str]]:
        return [(TestLayout().id('div-main1'), 'hidden'), (TestLayout().id('div-main2'), 'hidden')]


x = np.linspace(0, 10, 500)
y = x
xx, yy = np.meshgrid(x, y)
datas = [np.cos(xx)*i+np.sin(yy)*j for i, j in zip([1, 2, 3, 4], [4, 3, 2, 1])]
figs = [go.Figure(go.Heatmap(x=x, y=y, z=data)) for data in datas]

layout = TestLayout().layout()
