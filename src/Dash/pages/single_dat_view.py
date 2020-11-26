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
from src.Dash.BaseClasses import BasePageLayout


class SingleDatLayout(BasePageLayout):
    # def top_bar_layout(self):
    #     return html.Div('Test')

    def side_bar_layout(self):
        stuff = super().side_bar_layout()
        stuff.append(dbc.Button('Click me too!', id='button2'))
        return stuff

    # def main_area_layout(self):
    #     return html.Div('MainArea')
    pass


layout = SingleDatLayout().main_area_layout()


# Make callbacks
@app.callback(
    Output('graph-main', 'figure'),
    Input('inp-num', 'value')
)
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
