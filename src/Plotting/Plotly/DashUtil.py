from src.Plotting.Plotly import PlotlyUtil as PU

import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objects as go
import numpy as np

fig = go.Figure()
line = go.Scatter(mode='lines+markers', x=np.linspace(0,10,100), y=np.sin(np.linspace(0,20,100)))
fig.add_trace(line)


app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

class Dasher:
    def __init__(self, fig):
        self.app = dash.Dash()
        self.fig = fig

    def update(self):
        self.app.layout = html.Div([dcc.Graph(figure=fig)])


import threading

def run_dash(app):
    app.run_server(debug=False)

if __name__ == '__main__':
    d = Dasher(fig)
    t = threading.Thread(target=run_dash, args=[d.app])
    t.start()
    print('hi')
