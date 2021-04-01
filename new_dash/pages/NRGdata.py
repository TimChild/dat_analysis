from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Any, Callable
from dataclasses import dataclass
import logging

import numpy as np
from plotly import graph_objects as go

from dash_dashboard.base_classes import BasePageLayout, BaseMain, BaseSideBar, PageInteractiveComponents, \
    CommonInputCallbacks
import dash_dashboard.component_defaults as c
import dash_html_components as html
from dash_extensions.enrich import MultiplexerTransform  # Dash Extensions has some super useful things!

from Analysis.Feb2021.NRG_comparison import NRGData
from src.Dash.DatPlotting import TwoD

logger = logging.getLogger(__name__)

NAME = 'NRG'
URL_ID = 'NRG'
page_collection = None  # Gets set when running in multipage mode


class Components(PageInteractiveComponents):
    def __init__(self):
        super().__init__()

        # Graphs
        self.graph_1 = c.graph_area(id_name='graph-1', graph_header='',  # self.header_id -> children to update header
                                    pending_callbacks=self.pending_callbacks)
        self.graph_2 = c.graph_area(id_name='graph-2', graph_header='',
                                    pending_callbacks=self.pending_callbacks)

        # Input
        self.dd_which_nrg = c.dropdown(id_name='dd-which-nrg', multi=False, persistence=True)

        self.setup_initial_state()

    def setup_initial_state(self):
        self.dd_which_nrg.options = [{'label': k, 'value': k}
                                     for k in NRGData.__annotations__ if k not in ['ens', 'ts']]
        self.dd_which_nrg.value = 'occupation'


# A reminder that this is helpful for making many callbacks which have similar inputs
class CommonCallback(CommonInputCallbacks):
    components = Components()  # Only use this for accessing IDs only... DON'T MODIFY

    def __init__(self, example):
        super().__init__()  # Just here to shut up PyCharm
        self.example_value = example
        pass

    def callback_names_funcs(self):
        """
        Return a dict of {<name>: <callback_func>}
        """
        return {
            "example": self.example_func,
        }

    def example_func(self):
        """Part of example, can be deleted"""
        return self.example_value

    @classmethod
    def get_inputs(cls) -> List[Tuple[str, str]]:
        return [
            (cls.components.inp_example.id, 'value'),
        ]

    @classmethod
    def get_states(cls) -> List[Tuple[str, str]]:
        return []


class NRGLayout(BasePageLayout):
    top_bar_title = 'Title -- May want to override in PageLayout override'

    # Defining __init__ only for typing purposes (i.e. to specify page specific Components as type for self.components)
    def __init__(self, components: Components):
        super().__init__(page_components=components)
        self.components = components

    def get_mains(self) -> List[NRGMain]:
        return [NRGMain(self.components), ]

    def get_sidebar(self) -> BaseSideBar:
        return NRGSidebar(self.components)


class NRGMain(BaseMain):
    name = 'NRG'

    # Defining __init__ only for typing purposes (i.e. to specify page specific Components as type for self.components)
    def __init__(self, components: Components):
        super().__init__(page_components=components)
        self.components = components

    def layout(self):
        lyt = html.Div([
            self.components.graph_1,
            self.components.graph_2,
        ])
        return lyt

    def set_callbacks(self):
        self.make_callback(outputs=(self.components.graph_1.graph_id, 'figure'),
                           inputs=BasicNRGGraphCallback.get_inputs(),
                           states=BasicNRGGraphCallback.get_states(),
                           func=BasicNRGGraphCallback.get_callback_func('2d'))

        self.make_callback(outputs=(self.components.graph_2.graph_id, 'figure'),
                           inputs=BasicNRGGraphCallback.get_inputs(),
                           states=BasicNRGGraphCallback.get_states(),
                           func=BasicNRGGraphCallback.get_callback_func('1d'))


class NRGSidebar(BaseSideBar):
    id_prefix = 'NRGSidebar'

    # Defining __init__ only for typing purposes (i.e. to specify page specific Components as type for self.components)
    def __init__(self, components: Components):
        super().__init__(page_components=components)
        self.components = components

    def layout(self):
        lyt = html.Div([
            self.components.dd_main,
            self.components.dd_which_nrg,
        ])
        return lyt

    def set_callbacks(self):
        pass


class BasicNRGGraphCallback(CommonInputCallbacks):
    components = Components()  # Only use this for accessing IDs only... DON'T MODIFY

    def __init__(self, which):
        super().__init__()  # Just here to shut up PyCharm
        self.which = which if which else 'occupation'

    @classmethod
    def get_inputs(cls) -> List[Tuple[str, str]]:
        return [
            (cls.components.dd_which_nrg.id, 'value'),
        ]

    @classmethod
    def get_states(cls) -> List[Tuple[str, str]]:
        return []

    def callback_names_funcs(self):
        """
        Return a dict of {<name>: <callback_func>}
        """
        return {
            "2d": self.two_d,
            "1d": self.one_d,
        }

    def two_d(self) -> go.Figure:
        return plot_nrg(which=self.which, plot=False)

    def one_d(self) -> go.Figure:
        return go.Figure()
        pass


def plot_nrg(which: str,
             nrg: Optional[NRGData] = None, plot=True) -> go.Figure:
    @dataclass
    class PlotInfo:
        data: np.ndarray
        title: str

    if nrg is None:
        nrg = NRGData.from_mat()

    x = nrg.ens
    xlabel = 'Energy'
    y = nrg.ts
    ylabel = 'Temperature /K'

    if which == 'conductance':
        pi = PlotInfo(data=nrg.conductance,
                      title='NRG Conductance')
    elif which == 'dndt':
        pi = PlotInfo(data=nrg.dndt,
                      title='NRG dN/dT')
    elif which == 'entropy':
        pi = PlotInfo(data=nrg.entropy,
                      title='NRG Entropy')
    elif which == 'occupation':
        pi = PlotInfo(data=nrg.occupation,
                      title='NRG Occupation')
    elif which == 'int_dndt':
        pi = PlotInfo(data=nrg.int_dndt,
                      title='NRG Integrated dN/dT')
    else:
        raise KeyError(f'{which} not recognized')

    plotter = TwoD()
    fig = plotter.figure(xlabel=xlabel, ylabel=ylabel, title=pi.title)
    fig.add_trace(plotter.trace(data=pi.data, x=x, y=y))
    if plot:
        fig.show()
    return fig


def layout(*args):  # *args only because dash_extensions passes in the page name for some reason
    inst = NRGLayout(Components())
    inst.page_collection = page_collection
    return inst.layout()


def callbacks(app):
    inst = NRGLayout(Components())
    inst.page_collection = page_collection
    inst.layout()  # Most callbacks are generated while running layout
    return inst.run_all_callbacks(app)


if __name__ == '__main__':
    from dash_dashboard.app import test_page

    test_page(layout=layout, callbacks=callbacks, port=8051)


