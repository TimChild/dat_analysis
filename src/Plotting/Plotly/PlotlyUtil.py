import os
from typing import Callable
from dataclasses import dataclass
import sys
import plotly.offline
from typing import List, Union, Optional, Tuple
import plotly.graph_objects as go
import numpy as np

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
        values=["Plotly Named CSS colours"],
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


@dataclass
class HoverInfo:
    name: str
    func: Callable
    precision: str = '.2f'
    units: str = 'mV'
    position: Optional[int] = None


def additional_data_dict_converter(info: List[HoverInfo], customdata_start: int = 0) -> (list, str):
    """
    Converts a list of HoverInfos into a list of functions and a hover template string
    Args:
        info (List[HoverInfo]): List of HoverInfos containing ['name', 'func', 'precision', 'units', 'position']
            'name' and 'func' are necessary, the others are optional. 'func' should take DatHDF as an argument and return
            a value. 'precision' is the format specifier (e.g. '.2f'), and units is added afterwards
        customdata_start (int): Where to start customdata[i] from. I.e. start at 1 if plot function already adds datnum
            as customdata[0].
    Returns:
        Tuple[list, str]: List of functions which get data from dats, template string to use in hovertemplate
    """
    items = list()
    for d in info:
        name = d.name
        func = d.func
        precision = d.precision
        units = d.units
        position = d.position if d.position is not None else len(items)

        items.insert(position, (func, (name, precision, units)))  # Makes list of (func, (template info))

    funcs = [f for f, _ in items]
    # Make template for each func in order.. (i+custom_data_start) to reserve customdata[0] for datnum
    template = '<br>'.join(
        [f'{name}=%{{customdata[{i + customdata_start}]:{precision}}}{units}' for i, (_, (name, precision, units)) in
         enumerate(items)])
    return funcs, template


# # qt5 backend for Ipython AFTER importing QtWebEngineWidgets which has to be imported first
# try:
#     from IPython import get_ipython
#     ip = get_ipython()  # This function exists at runtime if in Ipython kernel
#     ip.enable_gui('qt5')
# except:
#     print('\n\n\nERROR when trying to enable qt5 backend support of IPython\n\n\n')
#     pass
#
#
# class PlotlyViewer(QtWebEngineWidgets.QWebEngineView, QMainWindow):
#     def __init__(self, fig, exec=False):
#         # Create a QApplication instance or use the existing one if it exists
#         self.fig = fig
#
#         self.app = QApplication.instance() if QApplication.instance() else QApplication(sys.argv)
#         super().__init__()
#
#         self.file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp.html"))
#         plotly.offline.plot(self.fig, filename=self.file_path, auto_open=False)
#         self.load(QUrl.fromLocalFile(self.file_path))
#         self.setWindowTitle("Plotly Viewer")
#         self.show()
#
#         if exec:
#             self.appc_()
#
#     def closeEvent(self, event):
#         os.remove(self.file_path)
#
#     def draw(self):
#         plotly.offline.plot(self.fig, filename=self.file_path, auto_open=False)
#         self.load(QUrl.fromLocalFile(self.file_path))
#


def get_figure(datas, xs, ys=None, ids=None, titles=None, labels=None, xlabel='', ylabel='',
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
        (go.Figure): Plotly figure
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


def fig_setup(fig: go.Figure, title=None, x_label=None, y_label=None, legend_title=None):
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, legend_title=legend_title)


def add_vertical(fig, x):
    fig.update_layout(shapes=[dict(type='line', yref='paper', y0=0, y1=1, xref='x', x0=x, x1=x)])


def add_horizontal(fig, y):
    fig.update_layout(shapes=[dict(type='line', yref='y', y0=y, y1=y, xref='paper', x0=0, x1=1)])


# TODO: Make classes and functions which return a plotly figure with numbered subplots
# TODO: in a way that I can use the 'axs' to address the plots.
# TODO: Will that actually be easier in the future? Or should I just hold onto traces more carefully?
# def get_fig(num=None, rows=None, cols=None):
#     if all([v is None for v in [num, rows, cols]]):
#         fig = go.Figure()
#         ax = None
#         return


# class Ax:
#     def __init__(self, fig, num):
#         self.fig = fig
#         self.num = num
#
#     @property
#     def row(self):
#         return
#
#
# def add_line(fig, x, z, x_label=None, y_label=None, label=None, mode='lines', **kwargs) -> go.Scatter:
#     trace = go.Scatter(mode=mode, x=x, y=z, name=label)
#
#
# def plot_1d(x, z, x_label=None, y_label=None, title=None, fig=None):
#     if fig is None:
#         fig = go.Figure()
#     trace = go.Scatter(x=x, y=z, labels={'x': x_label, 'y': y_label})


if __name__ == '__main__':
    # num = 10
    # xs = [np.tile(np.linspace(0, 10, 100), (5, 1)) for i in range(num)]
    # datas = [np.sin(x) for x in xs]
    #
    # fig = get_figure(datas, xs, ids=None, titles=None, labels=['1', '2', '3', '4', '5'], xlabel='xlabel',
    #                  ylabel='ylabel')
    # v = PlotlyViewer(fig)
    pass
