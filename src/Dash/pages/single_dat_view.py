from singleton_decorator import singleton
import dash
import dash_html_components as html
from typing import List, Tuple
import plotly.graph_objects as go
import numpy as np
from src.Dash.BaseClasses import BasePageLayout, BaseMain, BaseSideBar
from src.Plotting.Plotly.PlotlyUtil import add_horizontal
from src.DatObject.Make_Dat import DatHandler
import src.UsefulFunctions as U
from dash.exceptions import PreventUpdate
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
            self.graph_area('graph-main'),
            html.Div([self.graph_area('graph-secondary', 'Slice Graph')], id=self.id('div-secondary-graph'))
        ])
        self.set_callbacks()
        return layout

    def set_callbacks(self):
        self.sidebar.layout()  # Make sure layout has been generated
        inps = self.sidebar.inputs

        # Show main graph
        self.graph_callback('graph-main', get_figure,
                            inputs=[(inps['inp-datnum'].id, 'value'),
                                    (inps['sl-slicer'].id, 'value'),
                                    (inps['tog-slice'].id, 'value')],
                            states=[])

        # tog hide second graph
        self.make_callback((inps['tog-slice'].id, 'value'), (self.id('div-secondary-graph'), 'hidden'),
                           toggle_div)

        # Show linecut in second graph
        self.graph_callback('graph-secondary', plot_slice,
                            [(inps['inp-datnum'].id, 'value'),
                             (inps['sl-slicer'].id, 'value'),
                             (inps['tog-slice'].id, 'value')]
                            )


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
            self.slider(name='Slicer', id_name='sl-slicer', updatemode='drag')
        ])
        self.set_callbacks()
        return layout

    def set_callbacks(self):
        inps = self.inputs

        # Set slider bar for linecut
        self.make_callback(
            inputs=[(inps['inp-datnum'].id, 'value')],
            outputs=[
                (inps['sl-slicer'].id, 'min'),
                (inps['sl-slicer'].id, 'max'),
                (inps['sl-slicer'].id, 'step'),
                (inps['sl-slicer'].id, 'value'),
                (inps['sl-slicer'].id, 'marks'),
            ],
            func=set_slider_vals)

def get_figure(datnum, y_line, slice_tog):
    if datnum:
        dat = get_dat(datnum)

        x = dat.Data.get_data('x')
        y = dat.Data.get_data('y')
        z = dat.Data.get_data('i_sense')

        x, z = [U.bin_data_new(arr, int(round(z.shape[-1]/1250))) for arr in (x, z)]

        fig = go.Figure()
        fig.add_trace(go.Heatmap(x=x, y=y, z=z))
        if slice_tog == [True]:
            add_horizontal(fig, y_line)
        return fig
    else:
        return go.Figure()


def set_slider_vals(datnum):
    if datnum:
        dat = get_dat(datnum)
        y = dat.Data.get_data('y')
        start, stop, step, value = 0, len(y)-1, 1, round(len(y)/2)
        marks = {int(v): str(v) for v in np.arange(start, stop, 10)}
        return start, stop, step, value, marks
    return 0, 1, 0.1, 0.5, {0: '0', 0.5: '0.5', 1: '1'}


def plot_slice(datnum, slice_val, slice_tog):
    if slice_tog == [True]:
        slice_val = int(slice_val)
        dat = get_dat(datnum)

        x = dat.Data.get_data('x')
        y = dat.Data.get_data('y')
        z = dat.Data.get_data('i_sense')

        x, z = [U.bin_data_new(arr, int(round(z.shape[-1]/500))) for arr in (x, z)]

        data = z[slice_val]

        fig = go.Figure()
        fig.add_trace(go.Scatter(mode='lines', x=x, y=data))
        fig.update_layout(title=f'Slice at y = {y[slice_val]:.1f}')
        return fig
    raise PreventUpdate


def toggle_div(value):
    if value == [True]:
        return False
    else:
        return True


# Generate layout for to be used in App
layout = SingleDatLayout().layout()


if __name__ == '__main__':
    dat = get_dat(9111)