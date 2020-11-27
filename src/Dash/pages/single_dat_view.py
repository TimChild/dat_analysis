from singleton_decorator import singleton
import dash_html_components as html
from typing import List, Tuple
import plotly.graph_objects as go
import numpy as np
from src.Dash.BaseClasses import BasePageLayout, BaseMain, BaseSideBar

from src.DatObject.Make_Dat import DatHandler
get_dat = DatHandler().get_dat


class SingleDatLayout(BasePageLayout):
    def get_mains(self) -> List[Tuple[str, BaseMain]]:
        return [('Page1', SingleDatMain())]

    def get_sidebar(self) -> BaseSideBar:
        return SingleDatSidebar()

    @property
    def id_prefix(self):
        return 'SD'


class SingleDatMain(BaseMain):

    def get_sidebar(self):
        return SingleDatSidebar()

    @property
    def id_prefix(self):
        return 'SDmain'

    def layout(self):
        layout = html.Div([
            self.graph_area(name=self.id('graph-main'))
        ])
        self.init_callbacks()
        return layout

    def init_callbacks(self):
        self.graph_callback('graph-main', get_figure,
                            [(self.sidebar.id('inp-datnum'), 'value')])

    def set_callbacks(self):
        pass


@singleton
class SingleDatSidebar(BaseSideBar):

    @property
    def id_prefix(self):
        return 'SDsidebar'

    def layout(self):
        layout = html.Div([
            self.input_box(name='Dat', id_name='inp-datnum', placeholder='Choose Datnum', autoFocus=True, min=0),
            self.dropdown(name='Data', id_name='dd-data'),
            self.toggle(name='Slice', id_name='tog-slice'),
            self.slider(name='Slicer', id_name='sl-slicer')
        ])
        return layout



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


# Generate layout for to be used in App
layout = SingleDatLayout().layout()


