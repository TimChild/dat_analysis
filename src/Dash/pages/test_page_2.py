from src.Dash.BaseClasses import BasePageLayout
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from src.Dash.app import app


class TestLayout(BasePageLayout):
    def main_area_layout(self):
        layout = html.Div([dbc.Alert('Testing', id='alert'),
                           dbc.Button('Click Me!', id='button'),
                           dbc.Button('toggle', id='but-toggle'),
                           html.Div([dbc.Alert(color='warning', id='alert2')], id='div', hidden=False),
                           html.Div([dbc.Alert(color='info', id='alert3')], id='div2', hidden=False),
                           ], id=self.get_page_id())

        return layout

    def side_bar_layout(self):
        return html.Div('testpage sidebar')

    def get_page_id(self):
        return 'div-page2'


layout = TestLayout().main_area_layout()
global_list = []

@app.callback(
    Output('alert', 'children'),
    Input('button', 'n_clicks'),
)
def add_num(clicks):
    return str(clicks)


@app.callback(
    Output('div', 'hidden'),
    Input('but-toggle', 'n_clicks')
)
def appear(toggle):
    if toggle:
        if toggle % 2:
            return True
    return False


@app.callback(
    Output('alert2', 'children'),
    Input('button', 'n_clicks')
)
def alert2(val):
    global_list.append(val)
    return str(global_list)

@app.callback(
    Output('div2', 'hidden'),
    Input('but-toggle', 'n_clicks')
)
def appear(toggle):
    if toggle:
        if toggle % 3:
            return True
    return False

@app.callback(
    Output('alert3', 'children'),
    Input('button', 'n_clicks')
)
def alert2(val):
    global_list.append(val)
    return str(global_list)
