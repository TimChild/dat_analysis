from dataclasses import dataclass
import abc
import pandas as pd

from dat_analysis.dat_object.attributes.square_entropy import square_wave_time_array

from dat_analysis.dat_object.attributes import transition as T
import dash_bootstrap_components as dbc
from singleton_decorator import singleton
import dash_html_components as html
from typing import List, Tuple, TYPE_CHECKING, Optional
import plotly.graph_objects as go
import numpy as np
from OLD.Dash.DatSpecificDash import DatDashPageLayout, DatDashMain, DatDashSideBar
from dat_analysis.dat_analysis.characters import DELTA
from dat_analysis.hdf_util import NotFoundInHdfError
from dat_analysis.dat_object.make_dat import DatHandler
import dat_analysis.useful_functions as U
from dash.exceptions import PreventUpdate
import logging
from functools import partial
from dat_analysis.plotting.plotly.dat_plotting import OneD, TwoD

if TYPE_CHECKING:
    from dat_analysis.dat_object.dat_hdf import DatHDF
    from dat_analysis.dat_object.attributes import square_entropy as SE
    from dat_analysis.analysis_tools.general_fitting import FitInfo
get_dat = DatHandler().get_dat

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@singleton
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
    def name(self):
        return f'SE_{self.main_only_id()}'

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
                                (inps['sl-setpoint'].id, 'value'),
                                (inps['sl-heating-start-end'].id, 'value'),
                                (inps['dd-output'].id, 'value'),
                                (inps['dd-ent-saved-fits'].id, 'value'),
                                (inps['dd-trans-saved-fits'].id, 'value'),
                                (inps['div-ent-button-output'].id, 'children'),
                                (inps['div-trans-button-output'].id, 'children'),
                                (inps['dd-int-info'].id, 'value'),
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
            GraphInfo(title='Avg Integrated Entropy', which_fig='integrated_entropy_avg'),
        ]


class SquareEntropyMainRows(SquareEntropyMain):
    def main_only_id(self):
        return 'Rows'

    def graph_list(self) -> List[GraphInfo]:
        return [
            GraphInfo(title='Entropy by Row', which_fig='entropy_row'),
            GraphInfo(title='Transition by Row', which_fig='transition_row'),
            GraphInfo(title='Integrated Entropy by Row', which_fig='integrated_entropy_row'),
        ]


class SquareEntropyMainTwoD(SquareEntropyMain):
    def main_only_id(self):
        return 'TwoD'

    def graph_list(self) -> List[GraphInfo]:
        return [
            GraphInfo(title='Entropy 2D', which_fig='entropy_2d'),
            GraphInfo(title='Transition 2D', which_fig='transition_2d'),
            GraphInfo(title='Integrated Entropy 2D', which_fig='integrated_entropy_2d'),
        ]


class SquareEntropyMainHeatingCycle(SquareEntropyMain):
    def main_only_id(self):
        return 'HeatingCycle'

    def graph_list(self) -> List[GraphInfo]:
        return [
            GraphInfo(title='Heating Cycle', which_fig='heating_cycle'),
        ]


class SquareEntropyMainRaw(SquareEntropyMain):
    def main_only_id(self):
        return 'Raw'

    def graph_list(self) -> List[GraphInfo]:
        return [
            GraphInfo(title='Raw Data by Row', which_fig='raw_row'),
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
            html.Div(self.slider(name='HC Start-End', id_name='sl-heating-start-end', updatemode='mouseup',
                                 range_type='range', persistence=True), id=self.id('div-heating-start-end')),
            html.Div(self.slider(name='Slicer', id_name='sl-slicer', updatemode='mouseup'), id=self.id('div-slicer')),

            # Data options
            self.dropdown(name='Output', id_name='dd-output', multi=False, persistence=True),
            self.dropdown(name='Integration Info', id_name='dd-int-info', multi=True, persistence=True),

            # Fit tables
            html.Hr(),  # Separate inputs from info
            self.table(name='Entropy Fit Values', id_name='table-ent-fit-values'),
            html.Hr(),  # Separate inputs from info
            self.table(name='Transition Fit Values', id_name='table-trans-fit-values'),

            # Entropy Fit params
            html.Hr(),  # Separate Fit parts
            html.H4('Entropy Fit Params'),
            self.dropdown(name='Saved Fits', id_name='dd-ent-saved-fits', multi=True, persistence=True),
            self.checklist(name='Param Vary', id_name='check-ent-param-vary'),
            self._param_inputs(which='entropy'),
            self.button(name='Run Fit', id_name='but-ent-run-fit'),
            # A blank thing I can use to update other things AFTER fits run
            self.div(id_name='div-ent-button-output', style={'display': 'none'}),

            # Transition Fit params
            html.Hr(),  # Separate Fit parts
            html.H4('Transition Fit Params'),
            self.dropdown(name='Saved Fits', id_name='dd-trans-saved-fits', multi=True, persistence=True),
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
        output_dd = self.inputs['dd-output']
        sl_slicer = self.inputs['sl-slicer']

        # Set options here so it isn't so cluttered in layout above
        entropy_pars_checklist.options = [
            {'label': 'theta', 'value': 'theta'},
            {'label': 'dT', 'value': 'dT'},
            {'label': 'dS', 'value': 'dS'},
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

        output_dd.options = [{'label': 'default', 'value': 'default'}]
        output_dd.value = 'default'

        sl_slicer.value = 0

        return layout

    def set_callbacks(self):
        self.layout()  # Ensure layout run already
        inps = self.inputs

        # Make some common inputs quicker to use
        main = (self.main_dropdown().id, 'value')
        datnum = (inps['inp-datnum'].id, 'value')
        slice_val = (inps['sl-slicer'].id, 'value')


        # Output dropdown
        self.make_callback(
            inputs=[
                datnum,
                (inps['div-trans-button-output'].id, 'children'),
            ],
            outputs=[
                (inps['dd-output'].id, 'options')
            ],
            func=partial(get_saved_names, which='output')
        )

        # Integration Info dropdown
        self.make_callback(
            inputs=[
                datnum,
                (inps['div-trans-button-output'].id, 'children'),
            ],
            outputs=[
                (inps['dd-int-info'].id, 'options')
            ],
            func=partial(get_saved_names, which='int-info')
        )

        # Set setpoint range bar
        self.make_callback(
            inputs=[
                datnum
            ],
            outputs=[
                (inps['sl-setpoint'].id, 'min'),
                (inps['sl-setpoint'].id, 'max'),
                (inps['sl-setpoint'].id, 'step'),
                (inps['sl-setpoint'].id, 'marks'),
            ],
            func=partial(set_slider_vals, which='setpoint')
        )

        # Set Heating start-end range bar
        self.make_callback(
            inputs=[
                datnum
            ],
            outputs=[
                (inps['sl-heating-start-end'].id, 'min'),
                (inps['sl-heating-start-end'].id, 'max'),
                (inps['sl-heating-start-end'].id, 'step'),
                (inps['sl-heating-start-end'].id, 'marks'),
            ],
            func=partial(set_slider_vals, which='heating-start-end')
        )

        # Heating-start-end hide for anything but Heating Cycle
        self.make_callback(
            inputs=[main],
            outputs=[(self.id('div-heating-start-end'), 'hidden')],
            func=partial(show_div, which_mains=[SquareEntropyLayout().id('Heating Cycle')])
        )

        self.make_callback(
            inputs=[
                datnum,
                (inps['dd-output'].id, 'value'),
            ],
            outputs=(inps['sl-setpoint'].id, 'value'),
            func=set_setpoint_slider_value
        )

        # Set slider bar for linecut
        self.make_callback(
            inputs=[
                datnum
            ],
            outputs=[
                (inps['sl-slicer'].id, 'min'),
                (inps['sl-slicer'].id, 'max'),
                (inps['sl-slicer'].id, 'step'),
                (inps['sl-slicer'].id, 'marks'),
            ],
            func=partial(set_slider_vals, which='slicer'))

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
                func=partial(get_saved_names, which=which)
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
            func=run_entropy_fits,
            states=[
                main,
                datnum,
                (inps['dd-output'].id, 'value'),
                (inps['check-ent-param-vary'].id, 'value'),
                (inps['inp-ent-theta'].id, 'value'),
                (inps['inp-ent-dT'].id, 'value'),
                (inps['inp-ent-dS'].id, 'value'),
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
                dbc.Form(
                    [
                        dbc.Label(name, html_for=self.id(f'inp-{prefix}-{name}')),
                        self.input_box(val_type='number', id_name=f'inp-{prefix}-{name}', className='px-0'
                                       )
                    ],
                ), className='p-1'
            )
            return inp_item

        def all_inputs(names: List[str], prefix: str) -> dbc.Row:
            par_inputs = dbc.Row([single_input(name, prefix=prefix) for name in names])
            return par_inputs

        if which == 'entropy':
            names = ['theta', 'dT', 'dS', 'const', 'mid']
            pre = 'ent'
        elif which == 'transition':
            names = ['theta', 'amp', 'gamma', 'lin', 'const', 'mid', 'quad']
            pre = 'trans'
        else:
            raise ValueError(f'{which} not recognized. Should be in ["entropy", "transition"]')

        param_inp_layout = all_inputs(names, prefix=pre)
        return param_inp_layout


def show_div(main, which_mains: List[str]) -> bool:
    """
    Returns True or False to Div s.t. an input is only shown on Main pages listed in 'which_mains'
    Args:
        main (): Callback input of which main is currently selected
        which_mains (): List of which main pages should have this div shown on

    Returns:
        (bool): True for hidden, False for shown
    """
    if main:
        if main in which_mains:
            return False  # If in which main, then False to hidden
        logger.debug(f'Hiding on {main}')
        return True  # Otherwise hide
    return False  # If no info on main, show everything


def get_figure(datnum, slice_val, setpoints, heating_start_end, output_name, entropy_names, transition_names, entropy_div, transition_div, int_info_names,
               which_fig='entropy_avg') -> dict:
    plotter = Plotter(datnum=datnum, slice_val=slice_val, setpoints=setpoints,
                      heating_start_end=heating_start_end,
                      output_name=output_name,
                      entropy_fit_names=entropy_names,
                      transition_fit_names=transition_names,
                      entropy_update=entropy_div,
                      transition_update=transition_div,
                      int_info_names=int_info_names)
    return plotter.get_figure(which_fig=which_fig)


# def get_figure(datnum, slice_val=0, which_fig='avg'):
#     """
#     Returns figure
#     Args:
#         datnum (): datnum
#         fit_names (): name of fit to show
#         slice_val (): y-val to slice if asking for slice
#         which_fig (): Which figure to show
#
#     Returns:
#
#     """
#     # If button_done is the trigger, should fit_name stored there (which is currently just 'Dash' every time)
#     # ctx = dash.callback_context
#     # if not ctx.triggered:
#     #     return go.Figure()
#     if datnum is not None:
#         datnum = int(datnum)
#         dat = get_dat(datnum, datname='base', overwrite=False, exp2hdf=None)
#         plotter = SquareEntropyPlotter(dat)
#
#         # if fit_names is None or fit_names == []:
#         #     fit_names = ['default']
#         #
#         # checks = [False if n == 'default' else True for n in fit_names]
#
#         if which_fig == 'entropy_avg':
#             fig = plotter.plot_entropy_avg()
#             return fig
#
#         elif which_fig == 'entropy_rows':
#             if not slice_val:
#                 slice_val = 0
#             fig = plotter.plot_entropy_row(row=slice_val)
#             return fig
#
#         if which_fig == 'integrated_entropy_avg':
#             fig = plotter.plot_integrated_entropy_avg()
#             return fig
#
#         elif which_fig == 'integrated_entropy_rows':
#             if not slice_val:
#                 slice_val = 0
#             fig = plotter.plot_integrated_entropy_row(row=slice_val)
#             return fig
#
#         elif which_fig == 'transition_avg':
#             fig = plotter.plot_transition_avg()
#             return fig
#
#         elif which_fig == 'transition_rows':
#             if not slice_val:
#                 slice_val = 0
#             fig = plotter.plot_transition_row(row=slice_val)
#             return fig
#
#         elif which_fig == 'raw_rows':
#             if not slice_val:
#                 slice_val = 0
#             fig = plotter.plot_raw_row(row=slice_val)
#             return fig
#
#         elif which_fig == 'entropy_2d':
#             pass
#
#         elif which_fig == 'transition_2d':
#             pass
#
#         elif which_fig == 'raw_2d':
#             pass
#
#         elif which_fig == 'heating_cycle':
#             pass
#
#     raise PreventUpdate


def toggle_div(main, div_id: str = None, default_state: bool = False) -> bool:
    """
    Whether div should be visible or not based on main
    Args:
        main (): Input value from callback
        div_id (): Which div is being toggled (probably set in a partial(toggle_div, div_id = <val>))
        default_state (): Which state it should return by default

    Returns:
        (bool): Bool of hidden (i.e. False is NOT hidden)
    """
    hidden = default_state
    if main is not None:
        if div_id == 'slicer':
            if main in ['SE_Averaged Data', 'SE_Heating Cycle', 'SE_2D']:
                hidden = True
            elif main in ['SE_Per Row', 'SE_Raw Data']:
                hidden = False
            else:
                raise NotImplementedError(f'{main} not recognized')
    return hidden


def set_slider_vals(datnum,
                    which='slicer'):
    if datnum:
        dat = get_dat(datnum)
        if which == 'slicer':
            y = dat.Data.get_data('y')
            start, stop, step = 0, len(y) - 1, 1
            marks = {int(v): str(v) for v in np.linspace(start, stop, 5)}
        elif which == 'setpoint':
            awg = dat.SquareEntropy.square_awg
            start = 0
            stop = awg.info.wave_len/awg.measure_freq/4  # 4 parts
            step = stop/50
            marks = {v: f'{v:.2g}' for v in np.linspace(start, stop, 10)}
        elif which == 'heating-start-end':
            x = dat.SquareEntropy.avg_x
            awg = dat.SquareEntropy.square_awg
            start = np.nanmin(x)
            stop = np.nanmax(x)
            step = (stop-start)/awg.info.num_steps
            marks = {v: f'{v:.1f}' for v in np.linspace(start, stop, 5)}
        else:
            raise NotImplementedError(f'{which} not recognized')
        return start, stop, step, marks
    return 0, 1, 0.1, 0.5, {0: '0', 0.5: '0.5', 1: '1'}


def set_setpoint_slider_value(datnum, output_name):
    """Sets the starting values to show in setpoint slider (i.e. what the selected output used)"""
    if datnum:
        dat = get_dat(datnum)
        out = dat.SquareEntropy.get_Outputs(name=output_name)
        pp = out.process_params
        awg = dat.SquareEntropy.square_awg
        start = pp.setpoint_start/awg.measure_freq if pp.setpoint_start is not None else 0
        fin = pp.setpoint_fin/awg.measure_freq if pp.setpoint_fin is not None else awg.info.wave_len/4/awg.measure_freq
        return start, fin
    raise PreventUpdate


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
                logger.debug(f'update_tab_fit_values: which={which}, fit_names" {fit_names}')
                fit_values = [dat.SquareEntropy.get_fit(which='avg', fit_name=n, check_exists=True).best_values for n in
                              fit_names]
            elif main in ['SE_Per Row', 'SE_Raw Data']:  # Row types
                fit_values = [
                    dat.SquareEntropy.get_fit(which='row', row=slice_val, fit_name=n, check_exists=True).best_values for n
                    in
                    fit_names]
            elif main in ['SE_Heating Cycle']:  # Non relevant types
                raise PreventUpdate
            else:
                raise ValueError(f'{main} not an expected value')
        elif which == 'entropy':
            if fit_names is None or fit_names == []:
                fit_names = ['default']
            checks = [False if n == 'default' else True for n in fit_names]

            if main in ['SE_Averaged Data', 'SE_2D']:  # Avg types
                fit_values = [dat.Entropy.get_fit(which='avg', name=n, check_exists=check).best_values for n, check in
                              zip(fit_names, checks)]
            elif main in ['SE_Per Row', 'SE_Raw Data']:  # Row types
                fit_values = [dat.Entropy.get_fit(which='row', row=slice_val, name=n, check_exists=check).best_values
                              for n, check in zip(fit_names, checks)]
            elif main in ['SE_Heating Cycle']:  # Non relevant types
                raise PreventUpdate
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


def get_saved_names(datnum, button, which: str = None) -> List[dict]:
    """Get names of available fits or outputs for <which> (e.g. 'entropy'/'transition'/'output' where transition
        is saved in dat.SquareEntropy"""
    if which is None or which not in ['entropy', 'transition', 'output', 'int-info']:
        raise ValueError(f'{which} not recognized. Should be in ["entropy", "transition", "output", "int-info"]')
    if datnum:
        dat = get_dat(datnum)
        if which == 'entropy':
            names = dat.Entropy.fit_names
        elif which == 'transition':
            names = dat.SquareEntropy.fit_names  # Transition fits for SE are saved in here because for multiple parts
        elif which == 'output':
            names = dat.SquareEntropy.Output_names()
        elif which == 'int-info':
            names = dat.Entropy.get_integration_info_names()
        else:
            raise NotImplementedError
        return [{'label': k, 'value': k} for k in names]
    raise PreventUpdate


def run_transition_fits(button_click,
                        datnum,
                        fit_func,
                        params_vary,
                        theta_value, amp_value, gamma_value, lin_value, const_value, mid_value, quad_value,
                        setpoints,
                        ):

    if button_click and datnum:  # Only button_click is an input, everything else is a State
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
        setpoint_times = square_wave_time_array(dat.SquareEntropy.square_awg)
        sp_start, sp_fin = [U.get_data_index(setpoint_times, sp) for sp in setpoints]
        logger.debug(f'Setpoint times: {setpoints}, Setpoint indexs: {sp_start, sp_fin}')

        # Run Fits
        pp = dat.SquareEntropy.get_ProcessParams(name=None,  # Start from default and modify from there
                                                 setpoint_start=sp_start, setpoint_fin=sp_fin,
                                                 transition_fit_func=func, transition_fit_params=new_pars,
                                                 save_name='Dash')
        dat.SquareEntropy.get_Outputs(name='Dash', inputs=None, process_params=pp, overwrite=True)
        return 'Dash'
    else:
        raise PreventUpdate


def run_entropy_fits(button_click,
                     main,
                     datnum,
                     output_name,
                     params_vary,
                     theta_value, dT_value, dS_value, const_value, mid_value,
                     ):

    logger.debug(f'Starting entropy_fits')
    if button_click and datnum:  # Only button_click is an input, everything else is a State:
        dat = get_dat(datnum)

        # Make params and fit_func from inputs
        par_values = {
            'theta': theta_value,
            'dT': dT_value,
            'dS': dS_value,
            'const': const_value,
            'mid': mid_value,
        }
        par_varies = {k: True if k in params_vary else False for k in par_values}

        original_pars = dat.Entropy.avg_fit.params

        new_pars = U.edit_params(original_pars, param_name=list(par_values.keys()), value=list(par_values.values()),
                                 vary=list(par_varies.values()))
        logger.debug(f'New_pars: {new_pars}\npar_varies:{par_varies}\nparams_vary: {params_vary}')
        # Run Fits
        out: SE.Output = dat.SquareEntropy.get_Outputs(name=output_name)

        if main in ['SE_Averaged Data']:
            dat.Entropy.get_fit(which='avg', name='Dash', initial_params=new_pars,
                                data=out.average_entropy_signal, x=out.x, check_exists=False)
        elif main in ['SE_Per Row']:
            [dat.Entropy.get_fit(which='row', row=i, name='Dash', initial_params=new_pars,
                                 data=row, x=out.x) for i, row in enumerate(out.entropy_signal)]
        elif main in ['SE_Heating Cycle']:
            dat.Entropy.get_fit(which='avg', name='Dash', initial_params=new_pars,
                                data=out.average_entropy_signal, x=out.x, check_exists=False)
            [dat.Entropy.get_fit(which='row', row=i, name='Dash', initial_params=new_pars,
                                 data=row, x=out.x) for i, row in enumerate(out.entropy_signal)]
        elif main in ['SE_Raw Data']:
            logger.debug(f'Bad Page Finished entropy_fits')
            raise PreventUpdate
        else:
            logger.debug(f'Finished entropy_fits')
            raise NotImplementedError(f'{main} not recongnized')
        logger.debug(f'Properly Finished entropy_fits')
        return 'Dash'
    else:
        logger.debug(f'No Datnum Finished entropy_fits')
        raise PreventUpdate


class Plotter:
    def __init__(
            self, datnum: int,
            slice_val: int, setpoints: Tuple[float, float],
            heating_start_end: Tuple[float, float],
            output_name: str,
            entropy_fit_names: List[str], transition_fit_names: List[str],
            transition_update: str,  # Hidden div which gets updated when transition fit runs
            entropy_update: str,  # Hidden div which gets updated when entropy fit runs
            int_info_names: List[str]  # Names of integration_infos
    ):
        """Should be initialized with all information which 'get_figure' receives from callback"""
        if datnum is None:
            raise PreventUpdate
        dat: DatHDF = get_dat(datnum)
        self.slice_val = slice_val
        self.setpoints = setpoints
        self.heating_start_end = heating_start_end
        self.output_name = output_name
        self.entropy_fit_names = entropy_fit_names if entropy_fit_names is not None else []
        self.transition_fit_names = transition_fit_names if transition_fit_names is not None else []
        self.int_info_names = int_info_names if int_info_names is not None else []

        # Some almost always useful things initialized here
        self.dat: DatHDF = dat
        logger.debug(f'Plotter init - output_name: {output_name}')
        self.named_output: SE.Output = dat.SquareEntropy.get_Outputs(name=output_name, check_exists=True)
        self.y_array = dat.Data.y_array
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
        elif which_fig == 'raw_2d':
            return self.raw_2d()
        elif which_fig == 'heating_cycle':
            return self.heating_cycle()
        raise NotImplementedError(f'{which_fig} not recognized')

    def _avg(self, x: Optional[np.ndarray], data: Optional[np.ndarray],
             name: str,
             trace_name: Optional[str] = None,
             xlabel: Optional[str] = None, ylabel: Optional[str] = None,
             fig_only: bool = False) -> go.Figure:
        """
        Helpful starting point for avg figures
        Args:
            x ():
            data (): 1D data only here
            name (): Name to put in Title
            trace_name (): Optional name of trace (defaults to None)
            xlabel (): Optional xlabel (defaults to dat.Logs.xlabel)
            ylabel (): Optional ylabel (defaults to "arbitrary")
            fig_only (): If True, adds no trace (i.e. ignores x and data)

        Returns:
            (go.Figure): Figure instance which can be modified further before return dict
        """
        fig = self.one_plotter.figure(xlabel=xlabel, ylabel=ylabel, title=f'Dat{self.dat.datnum}: {name} Avg')
        if fig_only is False:
            if x is None or data is None:
                raise ValueError(f'fig_only: {fig_only}, x is None: {x is None}, data is None: {data is None}. '
                                 f'If fig_only is not True, both x and data must be provided')
            fig.add_trace(self.one_plotter.trace(data=data, x=x, mode='lines', name=trace_name))
        return fig

    def _row(self, x: Optional[np.ndarray], data: Optional[np.ndarray],
             name: str,
             row_num: Optional[int] = None,
             trace_name: Optional[str] = None,
             xlabel: Optional[str] = None, ylabel: Optional[str] = None,
             fig_only: bool = False) -> go.Figure:
        """
        Helpful starting point for row figures
        Args:
            x ():
            data (): 1D data only here
            name (): Name to put in Title
            row_num (): Optional row num for title (defaults to self.slice_val)
            trace_name (): Optional name of trace (defaults to None)
            xlabel (): Optional xlabel (defaults to dat.Logs.xlabel)
            ylabel (): Optional ylabel (defaults to "arbitrary")
            fig_only (): If True, adds no trace (i.e. ignores x and data)

        Returns:
            (go.Figure): Figure instance which can be modified further before return dict
        """
        if row_num is None:
            row_num = self.slice_val
        fig = self.one_plotter.figure(xlabel=xlabel, ylabel=ylabel, title=f'Dat{self.dat.datnum}: {name} Row {row_num}')
        if fig_only is False:
            if x is None or data is None:
                raise ValueError(f'fig_only: {fig_only}, x is None: {x is None}, data is None: {data is None}. '
                                 f'If fig_only is not True, both x and data must be provided')
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

    def _add_fit(self, fig: go.Figure, x: np.ndarray, fit: FitInfo, name: Optional[str] = None) -> go.Figure:
        if name is not None:
            name = f'{name}_fit'
        fig.add_trace(self.one_plotter.trace(data=fit.eval_fit(x=x), x=x, mode='lines', name=name))
        return fig

    def entropy_avg(self) -> dict:
        out = self.named_output
        x = out.x
        data = out.average_entropy_signal
        fig = self._avg(x=x, data=data, name='Entropy', ylabel=f'{DELTA}Current /nA')

        for name in self.entropy_fit_names:
            fit = self.dat.Entropy.get_fit(which='avg', name=name)
            self._add_fit(fig, x=x, fit=fit, name=name)
        return fig.to_dict()

    def entropy_row(self):
        out = self.named_output
        x = out.x
        data = out.entropy_signal[self.slice_val]
        fig = self._row(x=x, data=data, name='Entropy', ylabel=f'{DELTA}Current /nA', trace_name='Avg Data')

        for name in self.entropy_fit_names:
            fit = self.dat.Entropy.get_fit(which='row', row=self.slice_val, name=name)
            self._add_fit(fig, x=x, fit=fit, name=name)
        return fig.to_dict()

    def entropy_2d(self):
        out = self.named_output
        x = out.x
        y = self.y_array
        data = out.entropy_signal
        fig = self._2d(x=x, y=y, data=data, name='Entropy', ylabel=self.dat.Logs.ylabel)
        return fig.to_dict()

    def integrated_entropy_avg(self):
        out = self.named_output
        x = out.x
        fig = self._avg(x=None, data=None, name='Integrated Entropy', ylabel=f'{DELTA}S/kB', fig_only=True)
        for name in self.int_info_names:
            try:
                data = self.dat.Entropy.get_integrated_entropy(name=name, data=out.average_entropy_signal)
                fig.add_trace(self.one_plotter.trace(x=x, data=data, name=name, mode='lines'))
            except NotFoundInHdfError:
                logger.debug(f'No integrated entropy found for dat{self.dat.datnum}')

        return fig.to_dict()

    def integrated_entropy_row(self):
        out = self.named_output
        x = out.x
        fig = self._avg(x=None, data=None, name='Integrated Entropy', ylabel=f'{DELTA}S/kB', fig_only=True)
        for name in self.int_info_names:
            try:
                data = self.dat.Entropy.get_integrated_entropy(name=name, data=out.entropy_signal[self.slice_val])
                fig.add_trace(self.one_plotter.trace(x=x, data=data, name=name))
            except NotFoundInHdfError:
                logger.debug(f'No integrated entropy found for dat{self.dat.datnum}')
        return fig.to_dict()

    def integrated_entropy_2d(self):
        out = self.named_output
        x = out.x
        y = self.y_array
        fig = self._avg(x=None, data=None, name='Integrated Entropy', ylabel=f'{DELTA}S/kB', fig_only=True)
        name = self.int_info_names[0] if len(self.int_info_names) > 0 else 'default'
        try:
            data = self.dat.Entropy.get_integrated_entropy(name=name, data=out.entropy_signal)
            fig = self._2d(x=x, y=y, data=data, name='Integrated Entropy', ylabel=self.dat.Logs.ylabel)
            return fig.to_dict()
        except NotFoundInHdfError:
            logger.debug(f'No integrated entropy found for dat{self.dat.datnum}')
            return go.Figure()

    def transition_avg(self):
        out = self.named_output
        x = out.x
        data = out.averaged
        fig = self._avg(x=None, data=None, name='Transition', ylabel=f'{DELTA}Current /nA', fig_only=True)
        for row, label in zip(data, ['v0_0', 'vP', 'v0_1', 'vM']):
            fig.add_trace(self.one_plotter.trace(row, name=label, x=x, mode='lines'))

        for name in self.transition_fit_names:
            fit = self.dat.SquareEntropy.get_fit(which='avg', fit_name=name)
            self._add_fit(fig, x=x, fit=fit, name=name)
        return fig.to_dict()

    def transition_row(self):
        out = self.named_output
        x = out.x
        data = out.cycled[self.slice_val]
        fig = self._row(x=None, data=None, name='Transition', ylabel=f'{DELTA}Current /nA', fig_only=True)
        for row, label in zip(data, ['v0_0', 'vP', 'v0_1', 'vM']):
            fig.add_trace(self.one_plotter.trace(row, name=label, x=x, mode='lines'))

        for name in self.transition_fit_names:
            fit = self.dat.SquareEntropy.get_fit(which='row', row=self.slice_val, fit_name=name, which_fit='transition',
                                                 check_exists=True)  # Because only using existing, don't need to
            # worry about which transition_part it is
            self._add_fit(fig, x=x, fit=fit, name=name)
        return fig.to_dict()

    def transition_2d(self):
        out = self.named_output
        x = out.x
        y = self.y_array
        data = out.cycled
        cold_parts = (0, 2)
        data = np.mean(data[:, cold_parts, :], axis=1)
        fig = self._2d(x=x, y=y, data=data, name='Cold Transition', ylabel=self.dat.Logs.ylabel)
        return fig.to_dict()

    def raw_row(self):
        out = self.named_output
        x = self.dat.Data.x
        data = self.dat.SquareEntropy.data[self.slice_val]

        orig_rs_method = self.one_plotter.RESAMPLE_METHOD
        self.one_plotter.RESAMPLE_METHOD = 'downsample'
        fig = self._row(x=x, data=data, name='Raw', ylabel=f'{DELTA}Current /nA')
        self.one_plotter.RESAMPLE_METHOD = orig_rs_method
        return fig.to_dict()

    def raw_2d(self):
        out = self.named_output
        x = out.x
        y = self.y_array
        data = self.dat.SquareEntropy.data

        orig_rs_method = self.one_plotter.RESAMPLE_METHOD
        self.two_plotter.RESAMPLE_METHOD = 'downsample'
        fig = self._2d(x=x, y=y, data=data, name='Raw', ylabel=self.dat.Logs.ylabel)
        self.two_plotter.RESAMPLE_METHOD = orig_rs_method
        return fig.to_dict()

    def heating_cycle(self):
        square_awg: SE.AWG.AWG = self.dat.SquareEntropy.square_awg
        num_pts = square_awg.info.wave_len
        duration = num_pts/square_awg.measure_freq
        x = square_wave_time_array(square_awg)
        idx_start, idx_end = U.get_data_index(self.dat.SquareEntropy.x, self.heating_start_end, is_sorted=True)

        data = self.dat.SquareEntropy.data

        avg = np.mean(data, axis=0)
        avg = np.reshape(avg, (-1, num_pts))  # Line up all cycles on top of each other
        avg = avg[idx_start: idx_end]  # Only keep those which are within start-end
        avg = np.mean(avg, axis=0)
        avg = avg - np.mean(avg)

        masks = square_awg.get_single_wave_masks(num=0)  # Always use SW 0 for heating atm

        # Apply start-end indexs to masks as well
        # x = x[idx_start:idx_end]
        # masks = masks[:, idx_start:idx_end]
        fig = self.one_plotter.figure(xlabel='Time through Square Wave /s', ylabel=f'{DELTA}Current /nA',
                                      title=f'Dat{self.dat.datnum}: All data averaged to one Square Wave')
        for mask, label in zip(masks, ['v0_0', 'vP', 'v0_1', 'vM']):
            fig.add_trace(self.one_plotter.trace(data=avg*mask, x=x, mode='lines', name=label))

        for sect_start in np.linspace(0, duration, 4, endpoint=False):
            for sp, color in zip(self.setpoints, ['teal', 'tomato']):
                val = sect_start+sp
                self.one_plotter.add_line(fig, value=val, mode='vertical', color=color)
        return fig.to_dict()


# Generate layout for to be used in App
layout = SquareEntropyLayout().layout()


if __name__ == '__main__':
    d = get_dat(9111)

