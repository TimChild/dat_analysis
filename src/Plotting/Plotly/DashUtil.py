from src.Plotting.Plotly import PlotlyUtil as PU
import src.CoreUtil as CU
from src.DatObject.Make_Dat import DatHandler
get_dat = DatHandler.get_dat
get_dats = DatHandler.get_dats

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from typing import List, Union, Optional, Tuple
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import logging

logger = logging.getLogger(__name__)

from src.Plotting.Plotly.PlotlyGraphs import all_entropy, all_ct, avg_entropy, avg_ct, entropy_values, transition_values
PLOT_DICT = {
    'default': all_ct,
    '2D Entropy': all_entropy,
    '2D Transition': all_ct,
    'Avg Entropy': avg_entropy,
    'Avg Transition': avg_ct,
    'Fit Values Entropy': entropy_values,
    'Fit Values Transition': transition_values
}


allowed_datnums = range(6000, 10000)



app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Row(
    [
        dbc.Col(
            'Plotting Area',
            id='graph-area',
            width=10, className='bg-danger'),
        dbc.Col(
            html.Div([
                dbc.Row(
                    id='graph-options',
                    className='bg-success'
                ),
                dbc.Row(
                    html.Div(
                        html.B('Info Area'),
                    ),
                    id='info-area',
                    style={'height': '100%'},
                    className='bg-info'
                )], style={'height': '100%'}),
            width=2, style={'height': '100%'})
    ], style={'height': '100vh'})

OPT_LAYOUTS = {
    'default': [
        html.Label([
            "Choose Dat:  ",
            dcc.Input(id="inp-datnum", value=None)
        ]),
        html.Label([
            'Graph Layout',
            dcc.Dropdown(id='dcc-fig-layout', value='one', options=
                            [
                                {'label': 'One', 'value': 'one'},
                                {'label': 'Two', 'value': 'two'},
                                {'label': 'Three', 'value': 'three'},
                                {'label': 'Four', 'value': 'four'},
                            ],
                         searchable=False,
                         clearable=False
                         )
        ]),
        html.Label([
            "Choose Graph Type",
            dcc.Dropdown(id="dcc-graph-types", value='default', options=[{'label': k, 'value': k} for k in PLOT_DICT], multi=True)
        ]),
    ],
}

app.layout['graph-options'].children = OPT_LAYOUTS['default']


@app.callback(
    Output('graph-area', 'children'),
    Input('inp-datnum', 'value'),
    Input('dcc-fig-layout', 'value'),
    Input('dcc-graph-types', 'value')
)
def update_graphs(datnum, layout, plot_types):
    if datnum is None:
        return get_figures(None)
    datnum = int(datnum)
    if datnum in allowed_datnums:
        plot_types = CU.ensure_list(plot_types)
        dat = get_dat(datnum)
        figs = list()
        for p in plot_types[:4]:
            figs.append(PLOT_DICT[p](dat))
        if len(plot_types) > 4:
            logger.error('Not implemented using more than 4 graphs yet')
            # TODO: get rest of figures and just add graphs which don't have callbacks

        layout = get_figures(figs)
        return layout
    else:
        return get_figures(None)


def set_figs(figs: List[go.Figure]):
    figs = CU.ensure_list(figs)
    if len(figs) == 1:
        n = 'one'
    elif len(figs) == 2:
        n = 'two'
    elif len(figs) == 3:
        n = 'three'
    elif len(figs) == 4:
        n = 'four'
    else:
        raise NotImplemented('Not implemented more than 4 graphs yet')
    for i in range(len(figs)):
        app.layout[f'{n}-fig-{i+1}'].figure = figs[i]


def get_figures(figs):
    if figs is None or figs == 'default':
        return dcc.Graph(id='one-fig-1')
    num = len(figs)
    if 0 < num <= 4:
        if num == 1:
            return dcc.Graph(id='one-fig-1', figure=figs[0])
        elif num == 2:
            return [dbc.Row(dcc.Graph(id='two-fig-1', figure=figs[0], style=dict(width='100%')), no_gutters=True),
                    dbc.Row(dcc.Graph(id='two-fig-2', figure=figs[1], style=dict(width='100%')), no_gutters=True)]
        elif num == 3:
            return [dbc.Row(dcc.Graph(id='three-fig-1', figure=figs[0], style=dict(width='100%')), no_gutters=True),
                    dbc.Row([dbc.Col(dcc.Graph(id='three-fig-2', figure=figs[1], style=dict(width='100%'))),
                            dbc.Col(dcc.Graph(id='three-fig-3', figure=figs[2], style=dict(width='100%')))], no_gutters=True)]
        elif num == 4:
            return [dbc.Row([dbc.Col(dcc.Graph(id='four-fig-1', figure=figs[0], style=dict(width='100%'))),
                            dbc.Col(dcc.Graph(id='four-fig-2', figure=figs[1], style=dict(width='100%')))], no_gutters=True),
                    dbc.Row([dbc.Col(dcc.Graph(id='four-fig-3', figure=figs[2], style=dict(width='100%'))),
                            dbc.Col(dcc.Graph(id='four-fig-4', figure=figs[3], style=dict(width='100%')))], no_gutters=True)]
    else:
        raise NotImplemented('Not implemented this yet')

# PLOT_LAYOUTS = {
#     'default': dcc.Graph(id='one-fig-1'),
#     'one': dcc.Graph(id='one-fig-1'),
#     'two': [dbc.Row(dcc.Graph(id='two-fig-1', style=dict(width='100%')), no_gutters=True),
#             dbc.Row(dcc.Graph(id='two-fig-2', style=dict(width='100%')), no_gutters=True)],
#     'three': [dbc.Row(dcc.Graph(id='three-fig-1', style=dict(width='100%')), no_gutters=True),
#               dbc.Row([dbc.Col(dcc.Graph(id='three-fig-2', style=dict(width='100%'))),
#                        dbc.Col(dcc.Graph(id='three-fig-3', style=dict(width='100%')))], no_gutters=True)],
#     'four': [dbc.Row([dbc.Col(dcc.Graph(id='four-fig-1', style=dict(width='100%'))),
#                       dbc.Col(dcc.Graph(id='four-fig-2', style=dict(width='100%')))], no_gutters=True),
#              dbc.Row([dbc.Col(dcc.Graph(id='four-fig-3', style=dict(width='100%'))),
#                       dbc.Col(dcc.Graph(id='four-fig-4', style=dict(width='100%')))], no_gutters=True)]
# }


x = np.linspace(0, 100, 1000)
y = np.sin(x)
fig = px.scatter(x=x, y=y)

# for i in range(1,5):
#     app.layout[f'four-fig-{i}'].figure = fig


if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload=True, port=8000)
    pass
