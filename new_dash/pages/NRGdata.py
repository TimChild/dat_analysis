from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Any, Callable
from dataclasses import dataclass
import logging

import numpy as np
from plotly import graph_objects as go

from dash_dashboard.base_classes import BasePageLayout, BaseMain, BaseSideBar, PageInteractiveComponents, \
    CommonInputCallbacks
from dash_dashboard.util import triggered_by
import dash_dashboard.component_defaults as c
import dash_html_components as html
from dash_extensions.enrich import MultiplexerTransform  # Dash Extensions has some super useful things!
from dash import no_update

from Analysis.Feb2021.NRG_comparison import NRGData, NRG_func_generator
from src.Dash.DatPlotting import TwoD
from src.DatObject.Make_Dat import get_dat
from src.Dash.DatPlotting import OneD

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

        self.inp_datnum = c.input_box(id_name='inp-datnum', persistence=True)
        self.slider_gamma = c.slider(id_name='sl-gamma', updatemode='drag', persistence=True)
        self.slider_theta = c.slider(id_name='sl-theta', updatemode='drag', persistence=True)
        self.slider_mid = c.slider(id_name='sl-mid', updatemode='drag', persistence=True)
        self.slider_amp = c.slider(id_name='sl-amp', updatemode='drag', persistence=True)
        self.slider_lin = c.slider(id_name='sl-lin', updatemode='drag', persistence=True)
        self.slider_const = c.slider(id_name='sl-const', updatemode='drag', persistence=True)
        self.slider_occ_lin = c.slider(id_name='sl-occ-lin', updatemode='drag', persistence=True)

        self.setup_initial_state()

    def setup_initial_state(self):
        self.dd_which_nrg.options = [{'label': k, 'value': k}
                                     for k in list(NRGData.__annotations__) +['i_sense'] if k not in ['ens', 'ts']]
        self.dd_which_nrg.value = 'occupation'

        for component, setup in {self.slider_gamma: [-2.0, 2.5, 0.1,
                                                     {int(x) if x % 1 == 0 else x: f'{10 ** x:.2f}' for x in
                                                      np.linspace(-2, 2.5, 5)}, 1],
                                 self.slider_theta: [0.01, 10, 0.1, None, 3.8],
                                 self.slider_mid: [-100, 100, 1, None, 0],
                                 self.slider_amp: [0.1, 2.5, 0.05, None, 1],
                                 self.slider_lin: [0, 0.02, 0.0001, None, 0],
                                 self.slider_const: [0, 10, 0.01, None, 7],
                                 self.slider_occ_lin: [-0.003, 0.003, 0.0001, None, 0],
                                 }.items():
            component.min = setup[0]
            component.max = setup[1]
            component.step = setup[2]
            if setup[3] is None:
                marks = {int(x) if x % 1 == 0 else x: f'{x:.3f}' for x in np.linspace(setup[0], setup[1], 5)}
            else:
                marks = setup[3]
            component.marks = marks
            component.value = setup[4]


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
            html.Hr(),
            self.input_wrapper('Dat', self.components.inp_datnum),
            self.input_wrapper('gamma', self.components.slider_gamma, mode='label'),
            self.input_wrapper('theta', self.components.slider_theta, mode='label'),
            self.input_wrapper('mid', self.components.slider_mid, mode='label'),
            self.input_wrapper('amp', self.components.slider_amp, mode='label'),
            self.input_wrapper('lin', self.components.slider_lin, mode='label'),
            self.input_wrapper('const', self.components.slider_const, mode='label'),
            self.input_wrapper('lin_occ', self.components.slider_occ_lin, mode='label'),
        ])
        return lyt

    def set_callbacks(self):
        pass


class BasicNRGGraphCallback(CommonInputCallbacks):
    components = Components()  # Only use this for accessing IDs only... DON'T MODIFY

    def __init__(self, which,
                 datnum,
                 mid, g, theta, amp, lin, const, occ_lin,
                 ):
        super().__init__()  # Just here to shut up PyCharm
        self.which = which if which else 'occupation'
        self.which_triggered = triggered_by(self.components.dd_which_nrg.id)

        self.datnum = datnum
        self.mid = mid
        self.g = 10**g
        self.theta = theta
        self.amp = amp
        self.lin = lin
        self.const = const
        self.occ_lin = occ_lin

    @classmethod
    def get_inputs(cls) -> List[Tuple[str, str]]:
        return [
            (cls.components.dd_which_nrg.id, 'value'),

            (cls.components.inp_datnum.id ,'value'),
            (cls.components.slider_mid.id, 'value'),
            (cls.components.slider_gamma.id, 'value'),
            (cls.components.slider_theta.id, 'value'),
            (cls.components.slider_amp.id, 'value'),
            (cls.components.slider_lin.id, 'value'),
            (cls.components.slider_const.id, 'value'),
            (cls.components.slider_occ_lin.id, 'value'),
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
        if not self.which_triggered:
            return no_update
        return plot_nrg(which=self.which, plot=False)

    def one_d(self) -> go.Figure:
        nrg_func = NRG_func_generator(which=self.which)
        plotter = OneD(dat=None)
        fig = plotter.figure(xlabel='Sweepgate /mV', ylabel='Current /nA', title=f'NRG I_sense')
        if self.datnum:
            dat = get_dat(self.datnum)
            out = dat.SquareEntropy.get_Outputs(name='default')
            x = out.x
            data = np.nanmean(out.averaged[(0, 2,), :], axis=0)
            fig.add_trace(plotter.trace(x=x, data=data, name='Data', mode='lines'))
        else:
            x = np.linspace(-200, 200, 101)

        nrg_data = nrg_func(x, self.mid, self.g, self.theta, self.amp, self.lin, self.const, self.occ_lin)
        fig.add_trace(plotter.trace(x=x, data=nrg_data, name='NRG', mode='lines'))
        return fig


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
    y = nrg.ts/0.001
    ylabel = 'Temperature/Gamma'

    if which == 'conductance':
        pi = PlotInfo(data=nrg.conductance,
                      title='NRG Conductance')
    elif which == 'dndt':
        pi = PlotInfo(data=nrg.dndt,
                      title='NRG dN/dT')
    elif which == 'entropy':
        pi = PlotInfo(data=nrg.entropy,
                      title='NRG Entropy')
    elif which == 'occupation' or which == 'i_sense':
        pi = PlotInfo(data=nrg.occupation,
                      title='NRG Occupation')
    elif which == 'int_dndt':
        pi = PlotInfo(data=nrg.int_dndt,
                      title='NRG Integrated dN/dT')
    else:
        raise KeyError(f'{which} not recognized')

    plotter = TwoD(dat=None)
    fig = plotter.figure(xlabel=xlabel, ylabel=ylabel, title=pi.title)
    fig.add_trace(plotter.trace(data=pi.data, x=x, y=y))
    fig.update_yaxes(type='log')
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
