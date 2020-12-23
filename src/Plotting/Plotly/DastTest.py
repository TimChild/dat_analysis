""" Aim of this is to a single dat Viewer... pass in a single dat, and have access to plotting lots of the data and
displaying metadata"""

import src.CoreUtil as CU
from src.DatObject.Make_Dat import DatHandler

get_dat = DatHandler.get_dat
get_dats = DatHandler.get_dats

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, MATCH, ALL, ALLSMALLER
from typing import List, Union, Optional, Tuple
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import plotly.io as pio
import numbers
import logging

logger = logging.getLogger(__name__)

from src.Plotting.Plotly.PlotlyGraphs import all_entropy, all_ct, avg_entropy, avg_ct, entropy_values, transition_values

THEME = ''

if THEME == 'dark':
    dash_theme = dbc.themes.DARKLY
    plotly_theme = 'plotly_dark'
else:
    dash_theme = dbc.themes.BOOTSTRAP
    plotly_theme = 'none'

pio.templates.default = plotly_theme

PLOT_DICT = {
    '2D Entropy': all_entropy,
    '2D Transition': all_ct,
    'Avg Entropy': avg_entropy,
    'Avg Transition': avg_ct,
    'Fit Values Entropy': entropy_values,
    'Fit Values Transition': transition_values
}

# Things for interacting with Single Dat view
datnum_input = dbc.FormGroup(
    [
        dbc.Label('Datnum', html_for='inp-datnum'),
        dbc.Col(
            dbc.Input(type='number', bs_size='sm', autoFocus=True, id='inp-datnum', placeholder='Choose Datnum', min=0,
                      debounce=True)
        )
    ],
    row=True
)

graph_type_input = dbc.FormGroup(
    [
        dbc.Label('Graphs', html_for='dcc-graph-types'),
        dbc.Col(
            dcc.Dropdown(
                id='dcc-graph-types',
                placeholder='Choose Graphs', options=[{'label': k, 'value': k} for k in PLOT_DICT], multi=True
            )
        )
    ],
    row=True
)

options_area = dbc.Form([datnum_input, graph_type_input])

app = dash.Dash(__name__, external_stylesheets=[dash_theme])

app.layout = html.Div(
    style={'height': '100vh', 'overflow': 'hidden'},
    children=[
        dbc.Container(
            children=[
                dbc.Row(html.H1(id='title', children='Default Title'), style={'maxHeight': '10vh'}, className='mx-1'),
                dbc.Row(
                    children=[
                        dbc.Col(
                            width=10,
                            children=html.Div(
                                id='graph-area',
                                style={'maxHeight': '90vh', 'overflowY': 'scroll'}, className='no-scrollbar',
                                children=dcc.Graph()
                            )
                        ),
                        dbc.Col(
                            width=2,
                            children=[
                                dbc.Row(dbc.Col([
                                    html.H5('Options'), options_area
                                ], id='opts-area'),
                                    style={'height': '40vh', 'maxHeight': '40vh', 'overflowY': 'scroll'},
                                    className='no-scrollbar'),
                                html.Hr(),
                                html.H5('Dat Info'),
                                dbc.Row([dbc.Col(id='info-area')],
                                        style={'height': '60vh', 'maxHeight': '60vh', 'overflowY': 'scroll'},
                                        className='no-scrollbar')
                            ])
                    ]),
                dbc.Row('text at the bottom of the app to see where the bottom is', className='bg-success',
                        style={'height': '300px'})
            ], fluid=True)
    ])

# Cards with 1 figure in each, CardColumn to sort them into columns
graph_area = dbc.CardColumns(
    id='cc-graphs',
    style={'columnCount': 2},
    children=[dbc.Card(dcc.Graph())] * 6)


def _get_dat(datnum):
    # So that I can change this later to be more careful
    if datnum is None:
        return None
    datnum = int(datnum)
    return get_dat(datnum)


def format_graphs(figs: List[Union[go.Figure, dbc.Card]]) -> dbc.CardColumns:
    """
    Takes a list of either go.Figure's or dbc.Card's (which should usually contain figures) and returns a dbc.CardColumns
    to fill the graph area.

    Args:
        figs ():

    Returns:

    """
    def card(fig: go.Figure, index):
        return dbc.Card(dcc.Graph(id={'name': 'fig', 'index': index}, figure=fig))

    figs = CU.ensure_list(figs)
    if len(figs) == 1:
        cols = 1
    elif len(figs) <= 8:
        cols = 2
    else:
        cols = 3

    children = [f if isinstance(f, dbc.Card) else card(f, i) for i, f in enumerate(figs)]  # TODO: Need to be careful about indexes of cards... they need unique identifiers.

    cc = dbc.CardColumns(
        id='cc-graphs',
        style={'columnCount': cols},
        children=[c for c in children]
    )
    return cc


@app.callback(
    Output('graph-area', 'children'),
    Input('inp-datnum', 'value'),
    Input('dcc-graph-types', 'value')
)
def update_graphs(datnum, graph_types):
    dat = _get_dat(datnum)
    if np.any([x is None for x in [dat, graph_types]]):
        return dcc.Graph()

    figs = list()
    for t in graph_types:
        figs.append(PLOT_DICT[t](dat))
    return format_graphs(figs)


@app.callback(
    Output('info-area', 'children'),
    Input('inp-datnum', 'value'),
)
def update_info(datnum):
    dat = _get_dat(datnum)
    if dat is None:
        return html.P('Choose a Dat to see info')

    info = get_info(dat)
    ps = list()
    for k, v in info.items():
        if v is not None:
            if isinstance(v, numbers.Number):
                string = f'{k}: {v:.3g}'
            else:
                string = f'{k}: {v}'
            ps.append(html.P(string, style={'padding': '0px', 'margin': '0px'}))
    return html.Div(ps)


from src.DataStandardize.ExpSpecific.Sep20 import get_real_lct


@app.callback(
    Output('title', 'children'),
    Input('inp-datnum', 'value')
)
def update_title(datnum):
    if datnum is None:
        return 'Single Dat Viewer'
    return f'Single Dat view of Dat{datnum}'


def get_info(dat):
    info = {'Two Part': dat.Logs.part_of[1] == 2,
            'Part': dat.Logs.part_of[0],
            'Time elapsed': dat.Logs.time_elapsed,
            'Width': dat.Data.x_array[-1] - dat.Data.x_array[0],
            'Repeats': dat.Data.y_array.shape[-1],
            'LCSS': dat.Logs.fds['LCSS'],
            'LCSQ': dat.Logs.fds['LCSQ'],
            'LP*2': dat.Logs.fds['LP*2'],
            'LCT': dat.Logs.fds.get('LCT', None),
            'LCT/0.196': dat.Logs.fds.get('LCT/0.196', None),
            'Any_LCT': get_real_lct(dat),
            'LCB': dat.Logs.fds['LCB'],
            'dS': CU.get_nested_attr_default(dat, 'Other.EA_values.dS', None),
            'dS uncertainty': CU.get_nested_attr_default(dat, 'Other.EA_uncertainties.fit_dS', None),
            'Uncertainty batch size': CU.get_nested_attr_default(dat,
                                                                 'Other.EA_analysis_params.batch_uncertainty',
                                                                 None),
            'Entropy fit range': CU.get_nested_attr_default(dat, 'Other.EA_analysis_params.E_fit_range',
                                                            None),
            'Transition amp': CU.get_nested_attr_default(dat, 'Other.EA_values.amp', None),
            'Amp uncertainty': CU.get_nested_attr_default(dat, 'Other.EA_uncertainties.amp', None),
            'Theta': CU.get_nested_attr_default(dat, 'Other.EA_values.tc', None),
            'Theta uncertainty': CU.get_nested_attr_default(dat, 'Other.EA_uncertainties.tc', None),
            'Transition fit range': CU.get_nested_attr_default(dat,
                                                               'Other.EA_analysis_params.CT_fit_range',
                                                               None),
            'RCSS': dat.Logs.bds['RCSS'],
            'RCSQ': dat.Logs.bds['RCSQ'],
            'RCT': dat.Logs.bds['RCT'],
            'RCB': dat.Logs.bds['RCB'],
            'R2T(10M)': dat.Logs.fds.get('R2T(10M)', None),
            'R2T/0.001': dat.Logs.fds.get('R2T/0.001', None),
            'HQPC bias mV': dat.OldSquareEntropy.SquareAWG.AWs[0][0][1]
            }
    return info





if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload=True, port=8001)
    pass
