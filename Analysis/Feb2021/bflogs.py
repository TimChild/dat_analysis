import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.Dash.DatPlotting import OneD



df = pd.read_csv('../../Status_21-03-25.csv')
# df = pd.read_csv('Status_21-03-01.csv')


for i, r in enumerate(df.columns):
    print(f'{i}: {r}')

arr = np.array(df)
oil = arr[:, 53]
win = arr[:, 49]
wo = arr[:, 51]
x = arr[:, 1]


plotter = OneD()
plotter.MAX_POINTS = 1000000000
fig = plotter.figure(xlabel='Time', ylabel='Temperature /C', title='PT Compressor Temperatures 25th Mar 21')
# fig = plotter.figure(xlabel='Time', ylabel='Temperature /C', title='PT Compressor Temperatures 1st Mar 21')
for d, label in zip([oil, win, wo], ['Oil', 'Water In', 'Water Out']):
    fig.add_trace(plotter.trace(data=d, x=x, name=label, mode='lines'))
fig.show(renderer='browser')
