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
from typing import List, Tuple, TYPE_CHECKING, Optional
import plotly.graph_objects as go
import numpy as np
from src.Dash.DatSpecificDash import DatDashPageLayout, DatDashMain, DatDashSideBar, DashOneD, DashTwoD, DashThreeD
from src.Plotting.Plotly.AttrSpecificPlotting import SquareEntropyPlotter
from src.Characters import DELTA
from src.Dash.BaseClasses import get_trig_id
from src.Plotting.Plotly.PlotlyUtil import add_horizontal
from src.DatObject.Make_Dat import DatHandler
import src.UsefulFunctions as U
from dash.exceptions import PreventUpdate
import logging
from functools import partial
from Dash.DatPlotting import OneD, TwoD

if TYPE_CHECKING:
    from src.DatObject.DatHDF import DatHDF
    from src.DatObject.Attributes import SquareEntropy as SE
get_dat = DatHandler().get_dat

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SquareEntropyLayout(DatDashPageLayout):
    def get_mains(self) -> List[Tuple[str, DatDashMain]]:
        return [
            ('Averaged Data', SquareEntropyMainAvg()),
            ('Per Row', SquareEntropyMainRows()),
            ('2D', SquareEntropyMainTwoD()),
            ('Heating Cycle', SquareEntropyMainHeatingCycle()),
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
        self.graph_callback(f'graph-{graph_id}',  # Call on self matches with whichever subclass calls
                            func=partial(get_figure, which_fig=graph_info.which_fig),
                            inputs=[
                                (inps['inp-datnum'].id, 'value'),
                                (inps['sl-slicer'].id, 'value'),
                                (inps['sl-setpoints'].id, 'value'),
                                (inps['dd-ent-saved-fits'].id, 'value'),
                                (inps['dd-trans-saved-fits'].id, 'value'),
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
            GraphInfo(title='Avg Entropy', which_fig='entropy_avg'),
            GraphInfo(title='Avg Transition', which_fig='transition_avg'),
        ]


class SquareEntropyMainRows(SquareEntropyMain):
    def main_only_id(self):
        return 'Rows'

    def graph_list(self) -> List[GraphInfo]:
        return [
            GraphInfo(title='Entropy by Row', which_fig='entropy_rows'),
            GraphInfo(title='Transition by Row', which_fig='transition_rows'),
        ]


class SquareEntropyMainTwoD(SquareEntropyMain):
    def main_only_id(self):
        return 'AvgData'

    def graph_list(self) -> List[GraphInfo]:
        return [
            GraphInfo(title='Entropy 2D', which_fig='entropy_2d'),
            GraphInfo(title='Transition 2D', which_fig='transition_2d'),
        ]


class SquareEntropyMainHeatingCycle(SquareEntropyMain):
    def main_only_id(self):
        return 'Cycled'

    def graph_list(self) -> List[GraphInfo]:
        return [
            GraphInfo(title='Heating Cycle', which_fig='heating_cycle'),
        ]


class SquareEntropyMainRaw(SquareEntropyMain):
    def main_only_id(self):
        return 'Raw'

    def graph_list(self) -> List[GraphInfo]:
        return [
            GraphInfo(title='Raw Data by Row', which_fig='raw_rows'),
            GraphInfo(title='Raw Data 2D', which_fig='raw_2d'),
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

            # Setting where to start and finish averaging setpoints (dcc.RangeSlider)
            html.Div(self.slider(name='Setpoint Avg', id_name='sl-setpoint', updatemode='mouseup', range_type='range',
                                 persistence=True), id=self.id('div-setpoint')),
            html.Div(self.slider(name='Slicer', id_name='sl-slicer', updatemode='mouseup'), id=self.id('div-slicer')),

            # Fit tables
            html.Hr(),  # Separate inputs from info
            self.table(name='Entropy Fit Values', id_name='table-ent-fit-values'),
            html.Hr(),  # Separate inputs from info
            self.table(name='Transition Fit Values', id_name='table-trans-fit-values'),

            # Entropy Fit params
            html.Hr(),  # Separate Fit parts
            html.H4('Entropy Fit Params'),
            self.dropdown(name='Saved Fits', id_name='dd-ent-saved-fits', multi=True),
            self.checklist(name='Param Vary', id_name='check-ent-param-vary'),
            self._param_inputs(which='entropy'),
            self.button(name='Run Fit', id_name='but-ent-run-fit'),
            # A blank thing I can use to update other things AFTER fits run
            self.div(id_name='div-ent-button-output', style={'display': 'none'}),

            # Transition Fit params
            html.Hr(),  # Separate Fit parts
            html.H4('Transition Fit Params'),
            self.dropdown(name='Saved Fits', id_name='dd-trans-saved-fits', multi=True),
            self.checklist(name='Param Vary', id_name='check-trans-param-vary'),
            self._param_inputs(which='transition'),
            self.dropdown(name='Fit Func', id_name='dd-trans-fit-func'),
            self.button(name='Run Fit', id_name='but-trans-run-fit'),
            # A blank thing I can use to update other things AFTER fits run
            self.div(id_name='div-trans-button-output', style={'display': 'none'}),

        ])

        entropy_pars_checklist = self.inputs['check-ent-param-vary']
        trans_pars_checklist = self.inputs['check-trans-param-vary']
        trans_func_dd = self.inputs['dd-trans-fit-func']

        # Set options here so it isn't so cluttered in layout above
        entropy_pars_checklist.options = [
            {'label': 'theta', 'value': 'theta'},
            {'label': 'dT', 'value': 'amp'},
            {'label': 'dS', 'value': 'gamma'},
            {'label': 'lin', 'value': 'lin'},
            {'label': 'const', 'value': 'const'},
            {'label': 'mid', 'value': 'mid'},
        ]
        entropy_pars_checklist.value = [d['value'] for d in entropy_pars_checklist.options]  # Set default to all vary

        trans_func_dd.options = [
            {'label': 'i_sense', 'value': 'i_sense'},
            {'label': 'i_sense_digamma', 'value': 'i_sense_digamma'},
            {'label': 'i_sense_digamma_quad', 'value': 'i_sense_digamma_quad'},
        ]

        trans_pars_checklist.options = [
            {'label': 'theta', 'value': 'theta'},
            {'label': 'amp', 'value': 'amp'},
            {'label': 'gamma', 'value': 'gamma'},
            {'label': 'lin', 'value': 'lin'},
            {'label': 'const', 'value': 'const'},
            {'label': 'mid', 'value': 'mid'},
            {'label': 'quad', 'value': 'quad'},
        ]
        trans_pars_checklist.value = [d['value'] for d in trans_pars_checklist.options]  # Set default to all vary

        return layout

    def set_callbacks(self):
        self.layout()  # Ensure layout run already
        inps = self.inputs

        # Make some common inputs quicker to use
        main = (self.main_dropdown().id, 'value')
        datnum = (inps['inp-datnum'].id, 'value')
        slice_val = (inps['sl-slicer'].id, 'value')

        # Set slider bar for linecut
        self.make_callback(
            inputs=[
                datnum
            ],
            outputs=[
                (inps['sl-slicer'].id, 'min'),
                (inps['sl-slicer'].id, 'max'),
                (inps['sl-slicer'].id, 'step'),
                (inps['sl-slicer'].id, 'value'),
                (inps['sl-slicer'].id, 'marks'),
            ],
            func=set_slider_vals)

        for pre, which in zip(['ent', 'trans'], ['entropy', 'transition']):
            # Set Saved Fits options
            self.make_callback(
                inputs=[
                    datnum,
                    (inps[f'div-{pre}-button-output'].id, 'children')
                ],
                outputs=[
                    (inps[f'dd-{pre}-saved-fits'].id, 'options')
                ],
                func=partial(get_saved_fit_names, which=which)
            )

            # Set table info
            self.make_callback(
                inputs=[
                    main,
                    datnum,
                    slice_val,
                    (inps[f'dd-{pre}-saved-fits'].id, 'value'),
                    (inps[f'div-{pre}-button-output'].id, 'children'),  # Just to trigger update
                ],
                outputs=[
                    (inps[f'table-{pre}-fit-values'].id, 'columns'),
                    (inps[f'table-{pre}-fit-values'].id, 'data'),
                ],
                func=partial(update_tab_fit_values, which=which)
            )

        # Run Fits for Entropy
        self.make_callback(
            inputs=[
                (inps[f'but-ent-run-fit'].id, 'n_clicks'),
            ],
            outputs=[
                (inps['div-ent-button-output'].id, 'children')
            ],
            func=partial(run_entropy_fits),
            states=[
                main,
                datnum,
                (inps['check-ent-param-vary'].id, 'value'),
                (inps['inp-ent-theta'].id, 'value'),
                (inps['inp-ent-dT'].id, 'value'),
                (inps['inp-ent-dS'].id, 'value'),
                (inps['inp-ent-lin'].id, 'value'),
                (inps['inp-ent-const'].id, 'value'),
                (inps['inp-ent-mid'].id, 'value'),
            ]
        )

        # Run Fits for Transition
        self.make_callback(
            inputs=[
                (inps[f'but-trans-run-fit'].id, 'n_clicks'),
            ],
            outputs=[
                (inps['div-trans-button-output'].id, 'children')
            ],
            func=partial(run_transition_fits),
            states=[
                datnum,
                (inps['dd-trans-fit-func'].id, 'value'),

                (inps['check-trans-param-vary'].id, 'value'),

                (inps['inp-trans-theta'].id, 'value'),
                (inps['inp-trans-amp'].id, 'value'),
                (inps['inp-trans-gamma'].id, 'value'),
                (inps['inp-trans-lin'].id, 'value'),
                (inps['inp-trans-const'].id, 'value'),
                (inps['inp-trans-mid'].id, 'value'),
                (inps['inp-trans-quad'].id, 'value'),
                (inps['sl-setpoint'].id, 'value'),
            ]
        )

        # Set Slicer visible
        self.make_callback(
            inputs=[main],
            outputs=[(self.id('div-slicer'), 'hidden')],
            func=partial(toggle_div, div_id='slicer')
        )

    def _param_inputs(self, which: str):
        """
        Makes param inputs for fit function
        Args:
            which (): For 'entropy' or 'transition'

        Returns:
            (dbc.Row): Returns layout of param inputs, all inputs are accessible through self.inputs[<id>]
        """

        def single_input(name: str, prefix: str) -> dbc.Col:
            inp_item = dbc.Col(
                dbc.FormGroup(
                    [
                        dbc.Label(name, html_for=self.id(f'inp-{prefix}-{name}')),
                        self.input_box(val_type='number', id_name=f'inp-{prefix}-{name}', className='px-0',
                                       bs_size='sm')
                    ],
                ), className='p-1'
            )
            return inp_item

        def all_inputs(names: List[str], prefix: str) -> dbc.Row:
            par_inputs = dbc.Row([single_input(name, prefix=prefix) for name in names])
            return par_inputs

        if which == 'entropy':
            names = ['theta', 'dT', 'dS', 'lin', 'const', 'mid']
            pre = 'ent'
        elif which == 'transition':
            names = ['theta', 'amp', 'gamma', 'lin', 'const', 'mid', 'quad']
            pre = 'trans'
        else:
            raise ValueError(f'{which} not recognized. Should be in ["entropy", "transition"]')

        param_inp_layout = all_inputs(names, prefix=pre)
        return param_inp_layout


def get_figure(datnum, slice_val=0, which_fig='entropy_avg') -> dict:
    if datnum is not None:
        plotter = Plotter(datnum=datnum, slice_val=slice_val)
        return plotter.get_figure(which_fig=which_fig)
    raise PreventUpdate


def get_figure(datnum, slice_val=0, which_fig='avg'):
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

        if which_fig == 'entropy_avg':
            fig = plotter.plot_entropy_avg()
            return fig

        elif which_fig == 'entropy_rows':
            if not slice_val:
                slice_val = 0
            fig = plotter.plot_entropy_row(row=slice_val)
            return fig

        if which_fig == 'integrated_entropy_avg':
            fig = plotter.plot_integrated_entropy_avg()
            return fig

        elif which_fig == 'integrated_entropy_rows':
            if not slice_val:
                slice_val = 0
            fig = plotter.plot_integrated_entropy_row(row=slice_val)
            return fig

        elif which_fig == 'transition_avg':
            fig = plotter.plot_transition_avg()
            return fig

        elif which_fig == 'transition_rows':
            if not slice_val:
                slice_val = 0
            fig = plotter.plot_transition_row(row=slice_val)
            return fig

        elif which_fig == 'raw_rows':
            if not slice_val:
                slice_val = 0
            fig = plotter.plot_raw_row(row=slice_val)
            return fig

        elif which_fig == 'entropy_2d':
            pass

        elif which_fig == 'transition_2d':
            pass

        elif which_fig == 'raw_2d':
            pass

        elif which_fig == 'heating_cycle':
            pass

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
            if value in ['Averaged Data', '2D']:
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


def update_tab_fit_values(main, datnum, slice_val, fit_names, button_done, which: str = None) -> Tuple[
    List[dict], dict]:
    """
    Updates Table with fit information for <which> fit
    Args:
        datnum ():
        slice_val ():
        fit_names ():
        button_done ():
        which (): which fit type to return table for. This arg should be passed in using partial(func, which=<which>)

    Returns:
        Columns and Data to make a datatable
    """
    """see ((https://dash.plotly.com/datatable) for info on returns"""
    df = pd.DataFrame()
    if datnum:
        dat = get_dat(datnum)

        if slice_val is None:
            slice_val = 0

        if which == 'transition' and fit_names:
            if main in ['SE_Averaged Data', 'SE_2D']:  # Avg types
                logger.debug(fit_names)
                fit_values = [dat.SquareEntropy.get_fit(which='avg', name=n, check_exists=True).best_values for n in
                              fit_names]
            elif main in ['SE_Per Row', 'SE_Cycled Data', 'SE_Raw Data']:  # Row types
                fit_values = [
                    dat.SquareEntropy.get_fit(which='row', row=slice_val, name=n, check_exists=True).best_values for n
                    in
                    fit_names]
            else:
                raise ValueError(f'{main} not an expected value')
        elif which == 'entropy':
            if fit_names is None or fit_names == []:
                fit_names = ['default']
            checks = [False if n == 'default' else True for n in fit_names]

            if main in ['SE_Averaged Data', 'SE_2D']:  # Avg types
                fit_values = [dat.Entropy.get_fit(which='avg', name=n, check_exists=check).best_values for n, check in
                              zip(fit_names, checks)]
            elif main in ['SE_Per Row', 'SE_Cycled Data', 'SE_Raw Data']:  # Row types
                fit_values = [dat.Entropy.get_fit(which='row', row=slice_val, name=n, check_exists=check).best_values
                              for n, check
                              in
                              zip(fit_names, checks)]
            else:
                raise ValueError(f'{main} not an expected value')
        else:
            raise PreventUpdate
        if fit_values:
            df = pd.DataFrame()
            for fvs in fit_values:
                df = df.append(fvs.to_df())
        else:
            raise ValueError(f'No fit values found')
        df.index = [n for n in fit_names]
    df = df.applymap(lambda x: f'{x:.3g}')
    df = df.reset_index()  # Make index into a normal Column

    # ret = dbc.Table.from_dataframe(df).children  # convert to something that can be passed to dbc.Table.children
    cols = [{'name': n, 'id': n} for n in df.columns]
    data = df.to_dict('records')
    return cols, data


def get_saved_fit_names(datnum, button, which: str = None) -> List[dict]:
    """Get names of available fits for <which> (e.g. 'entropy' or 'transition' where transition is V0 part only
    which is saved in dat.SquareEntropy"""
    if which is None or which not in ['entropy', 'transition']:
        raise ValueError(f'{which} not recognized. Should be in ["entropy", "transition"]')
    if datnum:
        dat = get_dat(datnum)
        if which == 'entropy':
            fit_names = dat.Entropy.fit_names
        elif which == 'transition':
            fit_names = dat.SquareEntropy.fit_names
        else:
            raise NotImplementedError
        return [{'label': k, 'value': k} for k in fit_names]
    raise PreventUpdate


def run_transition_fits(button_click,
                        datnum,
                        fit_func,
                        params_vary,
                        theta_value, amp_value, gamma_value, lin_value, const_value, mid_value, quad_value,
                        setpoints,
                        ):
    logger.debug(f'button_click = {button_click}, datnum = {datnum}')  # Seeing what button click actually returns

    if button_click and datnum:
        dat = get_dat(datnum)

        # Make params and fit_func from inputs
        par_values = {
            'theta': theta_value,
            'amp': amp_value,
            'g': gamma_value,
            'lin': lin_value,
            'const': const_value,
            'mid': mid_value,
            'quad': quad_value,
        }
        if 'gamma' in params_vary:
            params_vary.append('g')  # I use 'g' instead of 'gamma' in fitting funcs etc...
        par_varies = {k: True if k in params_vary else False for k in par_values}

        original_pars = dat.Transition.avg_fit.params
        if fit_func == 'i_sense' or fit_func is None:
            func = T.i_sense
            pars_names = ['const', 'mid', 'amp', 'lin', 'theta']
        elif fit_func == 'i_sense_digamma':
            func = T.i_sense_digamma
            pars_names = ['const', 'mid', 'amp', 'lin', 'theta', 'g']
            T._append_param_estimate_1d(original_pars, 'g')
        elif fit_func == 'i_sense_digamma_quad':
            func = T.i_sense_digamma_quad
            pars_names = ['const', 'mid', 'amp', 'lin', 'theta', 'g', 'quad']
            T._append_param_estimate_1d(original_pars, ['g', 'quad'])
        else:
            raise ValueError(f'{fit_func} is not recognized as a fit_function for Transition')

        new_pars = U.edit_params(original_pars, param_name=pars_names, value=[par_values[k] for k in pars_names],
                                 vary=[par_varies[k] for k in pars_names])

        # Get other inputs
        sp_start, sp_fin = setpoints

        # Run Fits
        pp = dat.SquareEntropy.get_ProcessParams(name='Dash',
                                                 setpoint_start=sp_start, setpoint_fin=sp_fin,
                                                 transition_fit_func=func, transition_fit_params=new_pars,
                                                 save_name='Dash')
        dat.SquareEntropy.get_Outputs(name='Dash', inputs=None, process_params=pp, overwrite=False)
        return 'Dash'
    else:
        raise PreventUpdate


def run_entropy_fits(button_click,
                     main,
                     datnum,
                     params_vary,
                     theta_value, dT_value, dS_value, lin_value, const_value, mid_value,
                     ):
    logger.debug(f'button_click = {button_click}, datnum = {datnum}')  # Seeing what button click actually returns

    if button_click and datnum:
        dat = get_dat(datnum)

        # Make params and fit_func from inputs
        par_values = {
            'theta': theta_value,
            'dT': dT_value,
            'dS': dS_value,
            'lin': lin_value,
            'const': const_value,
            'mid': mid_value,
        }
        par_varies = {k: True if k in params_vary else False for k in par_values}

        original_pars = dat.Entropy.avg_fit.params

        new_pars = U.edit_params(original_pars, param_name=list(par_values.keys()), value=list(par_values.values()),
                                 vary=list(par_varies.values()))

        # Run Fits
        dash_output: SE.Output = dat.SquareEntropy.get_Outputs(name='Dash')
        if main in ['SE_Averaged Datas']:
            dat.Entropy.get_fit(which='avg', name='Dash', initial_params=new_pars,
                                data=dash_output.average_entropy_signal, x=dash_output.x, check_exists=False)
        elif main in ['SE_Per Row']:
            [dat.Entropy.get_fit(which='row', row=i, name='Dash', initial_params=new_pars,
                                 data=row, x=dash_output.x) for i, row in enumerate(dash_output.entropy_signal)]
        return 'Dash'
    else:
        raise PreventUpdate


class Plotter:
    def __init__(
            self, datnum: int,
            slice_val: int, setpoints: Tuple[int, int],
            entropy_fit_names: List[str], transition_fit_names: List[str],
    ):
        """Should be initialized with all information which 'get_figure' receives from callback"""
        dat = get_dat(datnum)
        self.slice_val = slice_val
        self.setpoints = setpoints
        self.entropy_fit_names = entropy_fit_names
        self.transition_fit_names = transition_fit_names

        # Some almost always useful things initialized here
        self.dat: DatHDF = dat
        self.one_plotter: OneD = OneD(dat)
        self.two_plotter: TwoD = TwoD(dat)

    def get_figure(self, which_fig: str) -> dict:
        if which_fig == 'entropy_avg':
            return self.entropy_avg()
        elif which_fig == 'entropy_row':
            return self.entropy_row()
        elif which_fig == 'entropy_2d':
            return self.entropy_2d()
        elif which_fig == 'integrated_entropy_avg':
            return self.integrated_entropy_avg()
        elif which_fig == 'integrated_entropy_row':
            return self.integrated_entropy_row()
        elif which_fig == 'integrated_entropy_2d':
            return self.integrated_entropy_2d()
        elif which_fig == 'transition_avg':
            return self.transition_avg()
        elif which_fig == 'transition_row':
            return self.transition_row()
        elif which_fig == 'transition_2d':
            return self.transition_2d()
        elif which_fig == 'raw_row':
            return self.raw_row()
        elif which_fig == 'heating_cycle':
            return self.heating_cycle()
        raise NotImplementedError(f'{which_fig} not recognized')

    def _avg(self, x: np.ndarray, data: np.ndarray,
             name: str,
             trace_name: Optional[str] = None,
             xlabel: Optional[str] = None, ylabel: Optional[str] = None) -> go.Figure:
        """
        Helpful starting point for avg figures
        Args:
            x ():
            data (): 1D data only here
            name (): Name to put in Title
            trace_name (): Optional name of trace (defaults to None)
            xlabel (): Optional xlabel (defaults to dat.Logs.xlabel)
            ylabel (): Optional ylabel (defaults to "arbitrary")

        Returns:
            (go.Figure): Figure instance which can be modified further before return dict
        """
        fig = self.one_plotter.figure(xlabel=xlabel, ylabel=ylabel, title=f'Dat{self.dat.datnum}: {name} Avg')
        fig.add_trace(self.one_plotter.trace(data=data, x=x, mode='lines', name=trace_name))
        return fig

    def _row(self, x: np.ndarray, data: np.ndarray,
             name: str,
             row_num: int,
             trace_name: Optional[str] = None,
             xlabel: Optional[str] = None, ylabel: Optional[str] = None) -> go.Figure:
        """
        Helpful starting point for row figures
        Args:
            x ():
            data (): 1D data only here
            name (): Name to put in Title
            trace_name (): Optional name of trace (defaults to None)
            xlabel (): Optional xlabel (defaults to dat.Logs.xlabel)
            ylabel (): Optional ylabel (defaults to "arbitrary")

        Returns:
            (go.Figure): Figure instance which can be modified further before return dict
        """
        fig = self.one_plotter.figure(xlabel=xlabel, ylabel=ylabel, title=f'Dat{self.dat.datnum}: {name} Row {row_num}')
        fig.add_trace(self.one_plotter.trace(data=data, x=x, mode='markers', name=trace_name))
        return fig

    def _2d(self, x: np.ndarray, y: np.ndarray, data: np.ndarray,
            name: str,
            xlabel: Optional[str] = None, ylabel: Optional[str] = None) -> go.Figure:
        """
        Helpful starting point for 2d figures
        Args:
            x ():
            y ():
            data (): 2D data only here
            name (): Name to put in Title
            xlabel (): Optional xlabel (defaults to dat.Logs.xlabel)
            ylabel (): Optional ylabel (defaults to "arbitrary")

        Returns:
            (go.Figure): Figure instance which can be modified further before return dict
        """
        fig = self.two_plotter.figure(xlabel=xlabel, ylabel=ylabel, title=f'Dat{self.dat.datnum}: {name} 2D')
        fig.add_trace(self.two_plotter.trace(data=data, x=x, y=y, trace_type='heatmap'))
        return fig

    def entropy_avg(self) -> dict:
        x = self.dat.SquareEntropy.x
        fig = self._avg(x=x, data=data, name='Entropy', ylabel=f'{DELTA}Current /nA')
        return fig

    def entropy_row(self):
        pass

    def entropy_2d(self):
        pass

    def integrated_entropy_avg(self):
        pass

    def integrated_entropy_row(self):
        pass

    def integrated_2d(self):
        pass

    def transition_avg(self):
        pass

    def transition_row(self):
        pass

    def transition_2d(self):
        pass

    def raw_row(self):
        pass

    def heating_cycle(self):
        pass


# Generate layout for to be used in App
layout = SquareEntropyLayout().layout()

if __name__ == '__main__':
    d = get_dat(9111)
