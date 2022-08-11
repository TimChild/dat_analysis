from __future__ import annotations
from typing import List, Union, Optional, TYPE_CHECKING
import plotly.graph_objects as go
import numpy as np

if TYPE_CHECKING:
    pass


def show_named_plotly_colours():
    """
    function to display to user the colours to match plotly's named
    css colours.

    Reference:
        #https://community.plotly.com/t/plotly-colours-list/11730/3

    Returns:
        plotly dataframe with cell colour to match named colour name

    """
    s='''
        aliceblue, antiquewhite, aqua, aquamarine, azure,
        beige, bisque, black, blanchedalmond, blue,
        blueviolet, brown, burlywood, cadetblue,
        chartreuse, chocolate, coral, cornflowerblue,
        cornsilk, crimson, cyan, darkblue, darkcyan,
        darkgoldenrod, darkgray, darkgrey, darkgreen,
        darkkhaki, darkmagenta, darkolivegreen, darkorange,teal
        darkorchid, darkred, darksalmon, darkseagreen,
        darkslateblue, darkslategray, darkslategrey,
        darkturquoise, darkviolet, deeppink, deepskyblue,
        dimgray, dimgrey, dodgerblue, firebrick,
        floralwhite, forestgreen, fuchsia, gainsboro,
        ghostwhite, gold, goldenrod, gray, grey, green,
        greenyellow, honeydew, hotpink, indianred, indigo,
        ivory, khaki, lavender, lavenderblush, lawngreen,
        lemonchiffon, lightblue, lightcoral, lightcyan,
        lightgoldenrodyellow, lightgray, lightgrey,
        lightgreen, lightpink, lightsalmon, lightseagreen,
        lightskyblue, lightslategray, lightslategrey,
        lightsteelblue, lightyellow, lime, limegreen,
        linen, magenta, maroon, mediumaquamarine,
        mediumblue, mediumorchid, mediumpurple,
        mediumseagreen, mediumslateblue, mediumspringgreen,
        mediumturquoise, mediumvioletred, midnightblue,
        mintcream, mistyrose, moccasin, navajowhite, navy,
        oldlace, olive, olivedrab, orange, orangered,
        orchid, palegoldenrod, palegreen, paleturquoise,
        palevioletred, papayawhip, peachpuff, peru, pink,
        plum, powderblue, purple, red, rosybrown,
        royalblue, saddlebrown, salmon, sandybrown,
        seagreen, seashell, sienna, silver, skyblue,
        slateblue, slategray, slategrey, snow, springgreen,
        steelblue, tan, teal, thistle, tomato, turquoise,
        violet, wheat, white, whitesmoke, yellow,
        yellowgreen
        '''
    li=s.split(',')
    li=[l.replace('\n','') for l in li]
    li=[l.replace(' ','') for l in li]

    import pandas as pd
    import plotly.graph_objects as go

    df=pd.DataFrame.from_dict({'colour': li})
    fig = go.Figure(data=[go.Table(
      header=dict(
        values=["plotly Named CSS colours"],
        line_color='black', fill_color='white',
        align='center', font=dict(color='black', size=14)
      ),
      cells=dict(
        values=[df.colour],
        line_color=[df.colour], fill_color=[df.colour],
        align='center', font=dict(color='black', size=11)
      ))
    ])

    fig.show(renderer='browser')


def make_slider_figure(datas: List[Union[List[np.ndarray], np.ndarray]],
                       xs: Union[np.ndarray, List[np.ndarray]],
                       ys: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                       ids=None, titles=None, labels=None, xlabel='', ylabel='',
                       plot_kwargs=None):
    """
    Get plotly figure with data in layers and a slider to change between them
    Args:
        ylabel (str): Y label
        xlabel (str): X label
        plot_kwargs (dict): Args to pass into either go.Heatmap or go.Scatter depending on 2D or 1D
        xs (Union(np.ndarray, list)): x_array for each slider position or single array to use for all
        ys (Union(np.ndarray, list)): y_array for each slider position or single array to use for all
        datas (list): list of datas for each slider position (i.e. can be list of list of arrays or just list of arrays)
        ids (list): ID for slider
        titles (list): Title at slider position
        labels (List[str]): Label for each trace per step (i.e. four lines per step, four labels)

    Returns:
        (go.Figure): plotly figure
    """
    if plot_kwargs is None:
        plot_kwargs = {}
    assert type(datas) == list

    if ys is None:
        datas_per_step = np.atleast_2d(datas[0]).shape[0]
    else:
        datas_per_step = 1

    ids = ids if ids else range(len(datas))
    titles = titles if titles else range(len(datas))
    labels = labels if labels is not None else [None] * datas_per_step

    if isinstance(xs, np.ndarray):
        xs = [xs] * len(datas)

    if ys is not None:
        if isinstance(ys, np.ndarray):
            ys = [ys] * len(datas)

    fig = go.Figure()
    if ys is not None:
        for data, x, y in zip(datas, xs, ys):
            fig.add_trace(
                go.Heatmap(
                    visible=False,
                    x=x,
                    y=y,
                    z=data,
                    **plot_kwargs
                ))
            fig.data[0].visible = True
    else:
        for data, x in zip(datas, xs):
            x = np.atleast_2d(x)
            data = np.atleast_2d(data)
            if x.shape[0] == 1 and data.shape[0] != 1:
                x = np.tile(x, (data.shape[0], 1))
            for x, d, label in zip(x, data, labels):
                plot_kwargs['mode'] = plot_kwargs.pop('mode', 'lines')
                fig.add_trace(go.Scatter(x=x, y=d, visible=False, name=label, **plot_kwargs))

        for i in range(datas_per_step):
            fig.data[i].visible = True

    steps = []
    for i, (id, title) in enumerate(zip(ids, titles)):
        step = dict(
            method='update',
            args=[{'visible': [False] * len(fig.data)},
                  {'title': f'{title}'},
                  plot_kwargs],
            label=f'{id}'
        )
        for j in range(datas_per_step):
            step['args'][0]['visible'][i * datas_per_step + j] = True  # Toggle i'th trace to visible
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={'prefix': ''},
        pad={'t': 50},
        steps=steps
    )]
    fig.update_layout(sliders=sliders, title=titles[0],
                      xaxis_title=xlabel, yaxis_title=ylabel)
    return fig


def add_vertical(fig, x):
    fig.update_layout(shapes=[dict(type='line', yref='paper', y0=0, y1=1, xref='x', x0=x, x1=x)])


def add_horizontal(fig, y):
    fig.update_layout(shapes=[dict(type='line', yref='y', y0=y, y1=y, xref='paper', x0=0, x1=1)])


if __name__ == '__main__':
    pass
