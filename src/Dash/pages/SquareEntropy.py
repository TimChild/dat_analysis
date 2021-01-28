from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
import abc
import dash
import pandas as pd
from src.DatObject.Attributes import Transition as T
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from singleton_decorator import singleton
import dash_html_components as html
from typing import List, Tuple, TYPE_CHECKING
import plotly.graph_objects as go
import numpy as np
from src.Dash.DatSpecificDash import DatDashPageLayout, DatDashMain, DatDashSideBar, DashOneD, DashTwoD, DashThreeD
from src.Plotting.Plotly.AttrSpecificPlotting import SquareEntropyPlotter
from src.Dash.BaseClasses import get_trig_id
from src.Plotting.Plotly.PlotlyUtil import add_horizontal
from src.DatObject.Make_Dat import DatHandler
import src.UsefulFunctions as U
from dash.exceptions import PreventUpdate
import logging
from functools import partial

if TYPE_CHECKING:
    from src.DatObject.DatHDF import DatHDF
get_dat = DatHandler().get_dat

logger = logging.getLogger(__name__)


class SquareEntropyLayout(DatDashPageLayout):
    def get_mains(self) -> List[Tuple[str, DatDashMain]]:
        return [
            ('Avg Entropy Fit', SquareEntropyMainAvg()),
            ('Row Fits', SquareEntropyMainRows()),
            ('Avg Data', SquareEntropyMainAvgData()),
            ('Cycled Data', SquareEntropyMainCycled()),
            ('Raw Data', SquareEntropyMainRaw()),
        ]

    def get_sidebar(self) -> DatDashSideBar:
        return SquareEntropySidebar()

    @property
    def id_prefix(self):
        return 'SE'


@dataclass
class GraphInfo:
    title: str
    which_fig: str


class SquareEntropyMain(DatDashMain, abc.ABC):
    @abc.abstractmethod
    def main_only_id(self):
        """Override to return a unique main_only_id. (e.g. 'Avg')"""
        pass

    @abc.abstractmethod
    def graph_list(self) -> List[GraphInfo]:
        """Override to return a List of GraphInfos"""
        pass

    @property
    def id_prefix(self):
        return f'SEmain{self.main_only_id()}'

    def get_sidebar(self):
        return SquareEntropySidebar()

    def layout(self):
        layout = html.Div(
            [self.graph_area(name=f'graph-{i}', title=graph_info.title, datnum_id=self.sidebar.id('inp-datnum'))
             for i, graph_info in enumerate(self.graph_list())])
        return layout

    def set_common_graph_callback(self, graph_id: int, graph_info: GraphInfo):
        """
        Sets callbacks for graph_one (can be used by all subclasses of this). Easy way to make sure they all
        implement the same inputs in the same order
        Args:
            graph_id (): Which graph (i.e. 0, 1, 2...)
            graph_info (): Dataclass containing all info required to specify graph conditions

        Returns:

        """
        inps = self.sidebar.inputs
        main = (self.sidebar.main_dropdown().id, 'value')
        self.graph_callback(f'graph-{graph_id}',  # Call on self matches with whichever subclass calls
                            func=partial(get_figure, which_fig=graph_info.which_fig),
                            inputs=[
                                (inps['inp-datnum'].id, 'value'),
                                (inps['sl-slicer'].id, 'value'),
                            ],
                            states=[])

    def set_callbacks(self):
        self.sidebar.layout()  # Make sure layout has been generated
        for i, graph_info in enumerate(self.graph_list()):
            self.set_common_graph_callback(graph_id=i, graph_info=graph_info)


class SquareEntropyMainAvg(SquareEntropyMain):
    def main_only_id(self):
        return 'Avg'

    def graph_list(self) -> List[GraphInfo]:
        return [
            GraphInfo(title='Avg Fit', which_fig='avg'),
            GraphInfo(title='Data', which_fig='TwoD'),
        ]


class SquareEntropyMainRows(SquareEntropyMain):
    def main_only_id(self):
        return 'Rows'

    def graph_list(self) -> List[GraphInfo]:
        return [
            GraphInfo(title='Row Fit', which_fig='rows'),
            GraphInfo(title='All Rows', which_fig='waterfall'),
        ]


class SquareEntropyMainAvgData(SquareEntropyMain):
    def main_only_id(self):
        return 'AvgData'

    def graph_list(self) -> List[GraphInfo]:
        return [
            GraphInfo(title='Avg Data', which_fig='avg_data'),
        ]


class SquareEntropyMainCycled(SquareEntropyMain):
    def main_only_id(self):
        return 'Cycled'

    def graph_list(self) -> List[GraphInfo]:
        return [
            GraphInfo(title='Setpoints and Cycles averaged', which_fig='cycled'),
        ]


class SquareEntropyMainRaw(SquareEntropyMain):
    def main_only_id(self):
        return 'Raw'

    def graph_list(self) -> List[GraphInfo]:
        return [
            GraphInfo(title='Raw Data', which_fig='rows'),
            GraphInfo(title='Add Data', which_fig='TwoD'),
        ]


@singleton
class SquareEntropySidebar(DatDashSideBar):

    @property
    def id_prefix(self):
        return 'SEsidebar'

    def layout(self):
        layout = html.Div([
            self.main_dropdown(),  # Choice between Avg view and Row view
            self.input_box(name='Dat', id_name='inp-datnum', placeholder='Choose Datnum', autoFocus=True, min=0),
            # self.dropdown(name='Saved Fits', id_name='dd-saved-fits', multi=True),
            # self.dropdown(name='Fit Func', id_name='dd-fit-func'),
            # self.checklist(name='Param Vary', id_name='check-param-vary'),
            # self._param_inputs(),
            # self.button(name='Run Fit', id_name='but-run-fit'),
            #
            # self.div(id_name='div-button-output', style={'display': 'none'}),
            # # ^^ A blank thing I can use to update other things AFTER fits run
            #
            html.Div(self.slider(name='Slicer', id_name='sl-slicer', updatemode='mouseup'), id=self.id('div-slicer')),
            # html.Hr(),  # Separate inputs from info
            # self.table(name='Fit Values', id_name='table-fit-values'),
        ])

        # # Set options here so it isn't so cluttered in layout above
        # self.dropdown(id_name='dd-fit-func').options = [
        #     {'label': 'i_sense', 'value': 'i_sense'},
        #     {'label': 'i_sense_digamma', 'value': 'i_sense_digamma'},
        #     {'label': 'i_sense_digamma_quad', 'value': 'i_sense_digamma_quad'},
        # ]
        # cl = self.checklist(id_name='check-param-vary')
        # cl.options = [
        #     {'label': 'theta', 'value': 'theta'},
        #     {'label': 'amp', 'value': 'amp'},
        #     {'label': 'gamma', 'value': 'gamma'},
        #     {'label': 'lin', 'value': 'lin'},
        #     {'label': 'const', 'value': 'const'},
        #     {'label': 'mid', 'value': 'mid'},
        #     {'label': 'quad', 'value': 'quad'},
        # ]
        # cl.value = [d['value'] for d in cl.options]  # Set default to all vary

        return layout

    def set_callbacks(self):
        inps = self.inputs

        # Make some common inputs quicker to use
        main = (self.main_dropdown().id, 'value')
        datnum = (inps['inp-datnum'].id, 'value')
        slice_val = (inps['sl-slicer'].id, 'value')

        # # Set Saved Fits options
        # self.make_callback(
        #     inputs=[
        #         datnum,
        #         (inps['div-button-output'].id, 'children')
        #     ],
        #     outputs=[
        #         (inps['dd-saved-fits'].id, 'options')
        #     ],
        #     func=get_saved_fit_names
        # )

        # Set slider bar for linecut
        self.make_callback(
            inputs=[
                datnum],
            outputs=[
                (inps['sl-slicer'].id, 'min'),
                (inps['sl-slicer'].id, 'max'),
                (inps['sl-slicer'].id, 'step'),
                (inps['sl-slicer'].id, 'value'),
                (inps['sl-slicer'].id, 'marks'),
            ],
            func=set_slider_vals)

        # # Set table info
        # self.make_callback(
        #     inputs=[
        #         main,
        #         datnum,
        #         slice_val,
        #         (inps['dd-saved-fits'].id, 'value'),
        #         (inps['div-button-output'].id, 'children'),  # Just to trigger update
        #     ],
        #     outputs=[
        #         (inps['table-fit-values'].id, 'columns'),
        #         (inps['table-fit-values'].id, 'data'),
        #     ],
        #     func=update_tab_fit_values
        # )
        #
        # # Run Fits
        # self.make_callback(
        #     inputs=[
        #         (inps['but-run-fit'].id, 'n_clicks'),
        #     ],
        #     outputs=[
        #         (inps['div-button-output'].id, 'children')
        #     ],
        #     func=run_fits,
        #     states=[
        #         main,
        #         datnum,
        #         (inps['dd-fit-func'].id, 'value'),
        #
        #         (inps['check-param-vary'].id, 'value'),
        #
        #         (inps['inp-theta'].id, 'value'),
        #         (inps['inp-amp'].id, 'value'),
        #         (inps['inp-gamma'].id, 'value'),
        #         (inps['inp-lin'].id, 'value'),
        #         (inps['inp-const'].id, 'value'),
        #         (inps['inp-mid'].id, 'value'),
        #         (inps['inp-quad'].id, 'value'),
        #     ]
        # )

        # Set Slicer visible
        self.make_callback(
            inputs=[main],
            outputs=[(self.id('div-slicer'), 'hidden')],
            func=partial(toggle_div, div_id='slicer')
        )

    # def _param_inputs(self):
    #     par_input = dbc.Row([
    #         dbc.Col(
    #             dbc.FormGroup(
    #                 [
    #                     dbc.Label('theta', html_for=self.id('inp-theta')),
    #                     self.input_box(val_type='number', id_name='inp-theta', className='px-0', bs_size='sm')
    #                 ],
    #             ), className='p-1'
    #         ),
    #         dbc.Col(
    #             dbc.FormGroup(
    #                 [
    #                     dbc.Label('amp', html_for=self.id('inp-amp')),
    #                     self.input_box(val_type='number', id_name='inp-amp', className='px-0', bs_size='sm')
    #                 ],
    #             ), className='p-1'
    #         ),
    #         dbc.Col(
    #             dbc.FormGroup(
    #                 [
    #                     dbc.Label('gamma', html_for=self.id('inp-gamma')),
    #                     self.input_box(val_type='number', id_name='inp-gamma', className='px-0', bs_size='sm')
    #                 ],
    #             ), className='p-1'
    #         ),
    #         dbc.Col(
    #             dbc.FormGroup(
    #                 [
    #                     dbc.Label('lin', html_for=self.id('inp-lin')),
    #                     self.input_box(val_type='number', id_name='inp-lin', className='px-0', bs_size='sm')
    #                 ],
    #             ), className='p-1'
    #         ),
    #         dbc.Col(
    #             dbc.FormGroup(
    #                 [
    #                     dbc.Label('const', html_for=self.id('inp-const')),
    #                     self.input_box(val_type='number', id_name='inp-const', className='px-0', bs_size='sm')
    #                 ],
    #             ), className='p-1'
    #         ),
    #         dbc.Col(
    #             dbc.FormGroup(
    #                 [
    #                     dbc.Label('mid', html_for=self.id('inp-mid')),
    #                     self.input_box(val_type='number', id_name='inp-mid', className='px-0', bs_size='sm')
    #                 ],
    #             ), className='p-1'
    #         ),
    #         dbc.Col(
    #             dbc.FormGroup(
    #                 [
    #                     dbc.Label('quad', html_for=self.id('inp-quad')),
    #                     self.input_box(val_type='number', id_name='inp-quad', className='px-0', bs_size='sm')
    #                 ],
    #             ), className='p-1'
    #         ),
    #     ])
    #     return par_input


def get_figure(datnum, fit_names, slice_val=0, which_fig='avg'):
    """
    Returns figure
    Args:
        datnum (): datnum
        fit_names (): name of fit to show
        slice_val (): y-val to slice if asking for slice
        which_fig (): Which figure to show

    Returns:

    """
    # If button_done is the trigger, should fit_name stored there (which is currently just 'Dash' every time)
    # ctx = dash.callback_context
    # if not ctx.triggered:
    #     return go.Figure()
    if datnum is not None:
        datnum = int(datnum)
        dat = get_dat(datnum, datname='base', overwrite=False, exp2hdf=None)
        plotter = SquareEntropyPlotter(dat)

        # if fit_names is None or fit_names == []:
        #     fit_names = ['default']
        #
        # checks = [False if n == 'default' else True for n in fit_names]

        if which_fig == 'avg':
            fig = plotter.plot_entropy_signal()
            return fig

        elif which_fig == 'rows':
            if not slice_val:
                slice_val = 0
            fig = plotter.plot_row_entropy(row=slice_val)
            return fig

        elif which_fig == 'avg_data':
            fig = plotter.plot_avg()
            return fig

        elif which_fig == 'cycled':
            if not slice_val:
                slice_val = 0
            fig = plotter.plot_cycled(row=slice_val)
            return fig

        elif which_fig == 'raw':
            if not slice_val:
                slice_val = 0
            fig = plotter.plot_cycled(row=slice_val)
            return fig
    raise PreventUpdate


def toggle_div(value, div_id: str = None, default_state: bool = False) -> bool:
    """
    Whether div should be visible or not based on main
    Args:
        value (): Input value from callback
        div_id (): Which div is being toggled (probably set in a partial(toggle_div, div_id = <val>))
        default_state (): Which state it should return by default

    Returns:
        (bool): Bool of hidden (i.e. False is NOT hidden)
    """
    hidden = default_state
    if value is not None:
        if div_id == 'slicer':
            if value in ['Avg Entropy Fit', 'Avg Data']:
                hidden = True
            else:
                hidden = False

    return hidden


def set_slider_vals(datnum):
    if datnum:
        dat = get_dat(datnum)
        y = dat.Data.get_data('y')
        start, stop, step, value = 0, len(y) - 1, 1, round(len(y) / 2)
        marks = {int(v): str(v) for v in np.arange(start, stop, 10)}
        return start, stop, step, value, marks
    return 0, 1, 0.1, 0.5, {0: '0', 0.5: '0.5', 1: '1'}


##############################

# def update_tab_fit_values(main, datnum, slice_val, fit_names, button_done) -> Tuple[List[dict], dict]:
#     """see ((https://dash.plotly.com/datatable) for info on returns"""
#     df = pd.DataFrame()
#     if datnum:
#         dat = get_dat(datnum)
#         t: T.Transition = dat.Transition
#
#         if slice_val is None:
#             slice_val = 0
#
#         if fit_names is None or fit_names == []:
#             fit_names = ['default']
#
#         checks = [False if n == 'default' else True for n in fit_names]
#
#         if main == 'T_Avg Fit':
#             fit_values = [t.get_fit(which='avg', name=n, check_exists=check).best_values for n, check in
#                           zip(fit_names, checks)]
#         elif main == 'T_Row Fits':
#             fit_values = [t.get_fit(which='row', row=slice_val, name=n, check_exists=check).best_values for n, check in
#                           zip(fit_names, checks)]
#         else:
#             raise ValueError(f'{main} not an expected value')
#         if fit_values:
#             df = pd.DataFrame()
#             for fvs in fit_values:
#                 df = df.append(fvs.to_df())
#         else:
#             raise ValueError(f'No fit values found')
#         df.index = [n for n in fit_names]
#     df = df.applymap(lambda x: f'{x:.3g}')
#     df = df.reset_index()  # Make index into a normal Column
#     # ret = dbc.Table.from_dataframe(df).children  # convert to something that can be passed to dbc.Table.children
#     cols = [{'name': n, 'id': n} for n in df.columns]
#     data = df.to_dict('records')
#     return cols, data







# def get_saved_fit_names(datnum) -> List[dict]:
#     if datnum:
#         dat = get_dat(datnum)
#         fit_names = dat.Entropy.fit_names
#         return [{'label': k, 'value': k} for k in fit_names]
#     raise PreventUpdate
#
#
# def run_fits(button_click,
#              main,
#              datnum,
#              fit_func,
#              params_vary,
#              theta_value, amp_value, gamma_value, lin_value, const_value, mid_value, quad_value):
#     if button_click and datnum:
#         dat = get_dat(datnum)
#         par_values = {
#             'theta': theta_value,
#             'amp': amp_value,
#             'g': gamma_value,
#             'lin': lin_value,
#             'const': const_value,
#             'mid': mid_value,
#             'quad': quad_value,
#         }
#         if 'gamma' in params_vary:
#             params_vary.append('g')  # I use 'g' instead of 'gamma' in fitting funcs etc...
#         par_varies = {k: True if k in params_vary else False for k in par_values}
#         print(par_varies)
#         original_pars = dat.Transition.avg_fit.params
#         if fit_func == 'i_sense' or fit_func is None:
#             func = T.i_sense
#             pars_names = ['const', 'mid', 'amp', 'lin', 'theta']
#         elif fit_func == 'i_sense_digamma':
#             func = T.i_sense_digamma
#             pars_names = ['const', 'mid', 'amp', 'lin', 'theta', 'g']
#             T._append_param_estimate_1d(original_pars, 'g')
#         elif fit_func == 'i_sense_digamma_quad':
#             func = T.i_sense_digamma_quad
#             pars_names = ['const', 'mid', 'amp', 'lin', 'theta', 'g', 'quad']
#             T._append_param_estimate_1d(original_pars, ['g', 'quad'])
#         else:
#             raise ValueError(f'{fit_func} is not recognized as a fit_function for Transition')
#
#         new_pars = U.edit_params(original_pars, param_name=pars_names, value=[par_values[k] for k in pars_names],
#                                  vary=[par_varies[k] for k in pars_names])
#
#         if main == 'T_Row Fits':
#             [dat.Transition.get_fit(which='row', row=i, name='Dash', initial_params=new_pars, fit_func=func,
#                                     check_exists=False) for i in range(dat.Transition.data.shape[0])]
#
#         # Always run avg fit since it will be MUCH faster anyway
#         dat.Transition.get_fit(which='avg', name='Dash', initial_params=new_pars, fit_func=func,
#                                check_exists=False)
#         return 'Dash'
#     else:
#         raise PreventUpdate


# Generate layout for to be used in App
layout = SquareEntropyLayout().layout()

if __name__ == '__main__':
    dat = get_dat(9111)
