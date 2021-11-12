from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Callable, Union
import abc
from dataclasses import dataclass
import logging

from dash_dashboard.base_classes import PageInteractiveComponents, \
    CommonInputCallbacks, PendingCallbacks
from dash_dashboard.util import triggered_by
from OLD.new_dash.base_class_overrides import DatDashPageLayout, DatDashMain, DatDashSidebar
import dash_dashboard.component_defaults as c
import dash_dashboard.util as ddu

import dash_html_components as html
import dash_bootstrap_components as dbc
from dash import no_update
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from src.dat_object.make_dat import get_dat, DatHDF
from src.plotting.plotly.dat_plotting import OneD, TwoD
import src.useful_functions as U
from src.analysis_tools.general_fitting import CalculatedTransitionFit, CalculatedEntropyFit, \
    calculate_se_output, calculate_tonly_data, TransitionCalcParams, set_centers, get_data_in_range, \
    _get_transition_fit_func_params

from src.analysis_tools.entropy import GammaAnalysisParams
from src.dat_object.attributes.SquareEntropy import Output
from src.dat_object.attributes.Entropy import IntegrationInfo, scaling

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NAME = 'Single Entropy'
URL_ID = 'SingleEntropy'
page_collection = None  # Gets set when running in multipage mode


class Components(PageInteractiveComponents):
    def __init__(self, pending_callbacks: Optional[PendingCallbacks] = None):
        super().__init__(pending_callbacks)
        self.inp_datnum = c.input_box(id_name='inp-datnum', val_type='number', debounce=True,
                                      placeholder='Enter Datnum', persistence=True)

        # Options for viewing saved info
        self.dd_t_fit_names = c.dropdown(id_name='dd-t-fit-names', multi=True)
        self.slider_single_row = c.slider(id_name='sl-single-row', persistence=False)

        # ##############################
        # Options when calculating fits
        self.tog_calculate = c.toggle(id_name='tog-calculate', label='Calculate Fit', persistence=True)
        self.collapse_calculate_options = c.collapse(id_name='collapse-calculate-options')
        self.div_calc_done = c.div(id_name='div-calc-done', style={'display': 'none'})  # Blank div for loading spinner
        self.but_run = c.button(id_name='but-run', text='Run Fits', color='success',
                                spinner=dbc.Spinner(self.div_calc_done))
        self.div_center_calc_done = c.div(id_name='div-center-calc-done', style={'display': 'none'})
        self.but_run_generate_centers = c.button(id_name='but-gen-centers', text='Run ALL center fits', color='warning',
                                                 spinner=dbc.Spinner(self.div_center_calc_done))
        self.tog_overwrite_centers = c.toggle(id_name='tog-overwrite-centers')

        # Centering ONLY
        self.dd_center_func = c.dropdown(id_name='dd-center-func', persistence=True)

        # Transition fitting params
        self.slider_rows = c.range_slider(id_name='sl-rows', persistence=False)
        self.dd_transition_func = c.dropdown(id_name='dd-transition-func', persistence=True)
        self.inp_fit_width = c.input_box(id_name='inp-fit-width', persistence=True)
        self.inp_force_theta = c.input_box(id_name='inp-force-theta', persistence=True)
        self.inp_force_gamma = c.input_box(id_name='inp-force-gamma', persistence=True)

        # CSQ mapping
        self.tog_csq_mapped = c.toggle(id_name='tog-csq-mapped', persistence=True)
        self.inp_csq_datnum = c.input_box(id_name='inp-csq-datnum', persistence=False)

        # Stores result of calculation so that all things depending on calculated don't have to recaculate
        self.store_calculated = c.store(id_name='store-calculated', storage_type='memory')
        # ###############################

        # Graphs
        self.graph_1 = c.graph_area(id_name='graph-1', graph_header='Avg Transition',
                                    pending_callbacks=self.pending_callbacks)
        self.graph_2 = c.graph_area(id_name='graph-2', graph_header='Single Row',
                                    pending_callbacks=self.pending_callbacks)
        self.graph_3 = c.graph_area(id_name='graph-3', graph_header='Data minus Fit',
                                    pending_callbacks=self.pending_callbacks)

        self.graph_4 = c.graph_area(id_name='graph-4', graph_header='2D Transition',
                                    pending_callbacks=self.pending_callbacks)
        self.graph_5 = c.graph_area(id_name='graph-5', graph_header='2D Transition Waterfall',
                                    pending_callbacks=self.pending_callbacks)

        # Info Area
        self.div_info_title = c.div(id_name='div-info-title')
        self.table_1 = c.table(id_name='tab-fit', dataframe=None)

        # ###### Further init of components ##########
        for dd in [self.dd_center_func, self.dd_transition_func]:
            dd.options = [{'label': n, 'value': n} for n in ['i_sense', 'i_sense_digamma', 'i_sense_digamma_amplin']]

        self.dd_center_func.options.append({'label': 'No Centering', 'value': 'No Centering'})

    def transition_param_inputs(self) -> List[Tuple[str, str]]:
        """use_tonly, tonly_datnum, tonly_func, tonly_width, tonly_rows"""
        return [
            (self.inp_datnum.id, 'value'),
            (self.dd_transition_func.id, 'value'),
            (self.inp_fit_width.id, 'value'),
            (self.slider_rows.id, 'value'),
            (self.dd_center_func.id, 'value'),
            (self.inp_force_theta.id, 'value'),
            (self.inp_force_gamma.id, 'value'),
        ]


class SingleTransitionLayout(DatDashPageLayout):

    # Defining __init__ only for typing purposes (i.e. to specify page specific Components as type for self.components)
    def __init__(self, components: Components):
        super().__init__(page_components=components)
        self.components = components

    def get_mains(self) -> List[SingleTransitionMain]:
        return [SingleTransitionMain(self.components), ]

    def get_sidebar(self) -> DatDashSidebar:
        return SingleTransitionSidebar(self.components)


class SingleTransitionMain(DatDashMain, abc.ABC):
    name = "SingleTransitionMain"

    # Defining __init__ only for typing purposes (i.e. to specify page specific Components as type for self.components)
    def __init__(self, components: Components):
        super().__init__(page_components=components)
        self.components = components

    def layout(self):
        lyt = html.Div([
            dbc.Row([
                dbc.Col([
                    self.components.graph_1,
                    self.components.graph_2,
                    self.components.graph_3,
                ], width=8),
                dbc.Col([
                    self.components.div_info_title,
                    html.H6('Transition Fit'),
                    self.components.table_1,
                    html.Hr(),
                    self.components.graph_4,
                    self.components.graph_5,
                    html.Hr(),
                ], width=4)
            ])
        ])
        return lyt

    def set_callbacks(self):
        components = self.components
        # Graph Callbacks
        for graph, cb_func in {components.graph_1: 'avg_data',
                               components.graph_2: 'single_row',
                               components.graph_3: 'data_minus_fit',
                               components.graph_4: '2d_transition',
                               components.graph_5: '2d_transition_waterfall',
                               }.items():
            self.make_callback(outputs=(graph.graph_id, 'figure'),
                               inputs=GraphCallbacks.get_inputs(),
                               func=GraphCallbacks.get_callback_func(cb_func),
                               states=GraphCallbacks.get_states())

        # Table Callbacks
        self.make_callback(outputs=(self.components.div_info_title.id, 'children'),
                           inputs=(self.components.inp_datnum.id, 'value'),
                           func=lambda datnum: html.H5(
                               f'Dat{datnum}: Fit Info') if datnum is not None else 'Invalid Datnum')

        self.make_callback(outputs=TableCallbacks.get_outputs(self.components.table_1),
                           inputs=TableCallbacks.get_inputs(),
                           states=TableCallbacks.get_states(),
                           func=TableCallbacks.get_callback_func('transition table'))


class SingleTransitionSidebar(DatDashSidebar):
    id_prefix = 'SingleTransitionSidebar'

    # Defining __init__ only for typing purposes (i.e. to specify page specific Components as type for self.components)
    def __init__(self, components: Components):
        super().__init__(page_components=components)
        self.components = components

    def layout(self):
        comps = self.components
        self.components.collapse_calculate_options.children = [
            # Options when calculating fits
            comps.store_calculated,  # Storage for calculations
            comps.but_run, comps.but_run_generate_centers,
            self.input_wrapper('Overwrite Center Fits', comps.tog_overwrite_centers),

            c.space(height='10px'),

            # Centering
            html.Hr(),
            html.H6('Centering'),
            self.input_wrapper('Center Func', comps.dd_center_func),

            # Transition fitting params
            html.Hr(),
            self.input_wrapper('Fit Func', comps.dd_transition_func),
            self.input_wrapper('Width', comps.inp_fit_width),
            self.input_wrapper('Rows', comps.slider_rows, mode='label'),
            self.input_wrapper('Force Theta', comps.inp_force_theta),
            self.input_wrapper('Force Gamma', comps.inp_force_gamma),

            # CSQ mapping
            html.Hr(),
            html.H6('CSQ Mapping Params'),
            self.input_wrapper('Use CSQ mapping', comps.tog_csq_mapped),
            self.input_wrapper('Dat', comps.inp_csq_datnum),
        ]

        lyt = html.Div([
            self.components.dd_main,
            self.input_wrapper('Datnum', self.components.inp_datnum),
            self.input_wrapper('T fits', self.components.dd_t_fit_names),
            html.Hr(),
            self.input_wrapper('Calculate New Fit', comps.tog_calculate),
            c.space(height='10px'),
            self.components.collapse_calculate_options,
        ])
        return lyt

    def set_callbacks(self):
        cmps = self.components

        # Set Options specific to Dat
        self.make_callback(outputs=[(cmps.dd_t_fit_names.id, 'options'), (cmps.dd_t_fit_names.id, 'value')],
                           inputs=DatOptionsCallbacks.get_inputs(),
                           states=DatOptionsCallbacks.get_states(),
                           func=DatOptionsCallbacks.get_callback_func('transition fits'))

        # Collapse Calculate only options
        self.make_callback(outputs=(cmps.collapse_calculate_options.id, 'is_open'),
                           inputs=(cmps.tog_calculate.id, 'value'),
                           func=lambda val: True if val else False)

        # Setup rows sliders
        self.make_callback(outputs=RowRangeSliderSetupCallback.get_outputs(cmps.slider_rows.id),
                           inputs=RowRangeSliderSetupCallback.get_inputs(cmps.inp_datnum.id),
                           states=RowRangeSliderSetupCallback.get_states(cmps.slider_rows.id),
                           func=RowRangeSliderSetupCallback.get_callback_func()
                           )

        # Run calculation
        self.make_callback(serverside_outputs=(cmps.store_calculated.id, 'data'),
                           outputs=(cmps.div_calc_done.id, 'children'),
                           inputs=CalculateCallback.get_inputs(),
                           func=CalculateCallback.get_callback_func('calculate'),
                           states=CalculateCallback.get_states())
        # For the spinner of Run Calculation
        # self.make_callback(
        #     outputs=(cmps.div_calc_done.id, 'children'),
        #     inputs=(cmps.store_calculated.id, 'data'),
        #     func=lambda d: 'done'  # Just return anything basically
        # )

        # Run Center Calculation
        self.make_callback(
            outputs=(cmps.div_center_calc_done.id, 'children'),
            inputs=CalculateCallback.get_inputs(),
            func=CalculateCallback.get_callback_func('calculate_centers'),
            states=CalculateCallback.get_states())


class RowRangeSliderSetupCallback(c.RangeSliderSetupCallback):
    components = Components()

    def __init__(self, datnum: int, current_value):
        dat = get_dat(datnum) if datnum is not None else None

        min_ = 0
        max_ = 1
        step = 1
        marks = {}
        value = (0, 1)
        if dat is not None:
            yshape = dat.Data.get_data('y').shape[0]
            max_ = yshape-1
            marks = {int(v): str(int(v)) for v in np.linspace(min_, max_, 5)}
            if current_value and all([min_ < v < max_ for v in current_value]):
                value = current_value
            else:
                value = (min_, max_)
        super().__init__(min=min_, max=max_, step=step, marks=marks, value=value)

    @classmethod
    def get_inputs(cls, datnum_id: str):
        return [(datnum_id, 'value')]

    @classmethod
    def get_states(cls, slider_id_name: str):
        """Use current state of slider to decide whether to reset or keep"""
        return [(slider_id_name, 'value')]


class GraphCallbacks(CommonInputCallbacks):
    components = Components()  # Only use this for accessing IDs only... DON'T MODIFY

    # noinspection PyMissingConstructor
    def __init__(self, datnum, t_fit_names,  # plotting existing
                 calculated,
                 ):
        self.datnum: int = datnum
        # plotting existing
        self.t_fit_names: List[str] = listify_dash_input(t_fit_names)

        self.calculated: StoreData = calculated
        self.calculated_triggered = triggered_by(self.components.store_calculated.id)

        # ################# Post calculations
        self.dat = get_dat(self.datnum) if self.datnum is not None else None

    @classmethod
    def get_inputs(cls) -> List[Tuple[str, str]]:
        cmps = cls.components
        return [
            (cmps.inp_datnum.id, 'value'),
            (cmps.dd_t_fit_names.id, 'value'),
            (cmps.store_calculated.id, 'data'),
        ]

    @classmethod
    def get_states(cls) -> List[Tuple[str, str]]:
        cmps = cls.components
        return [
        ]

    def callback_names_funcs(self):
        """
        Return a dict of {<name>: <callback_func>}
        """
        return {
            "avg_data": self.avg_data,
            "single_row": self.single_row,
            "transition_2d": self.transition_2d,
            "transition_2d_waterfall": self.transition_2d_waterfall,

        }

    def _correct_call_args(self) -> bool:
        """Common check for bad call args which shouldn't be used for plotting"""
        if any([self.dat is None, not is_square_entropy_dat(self.dat)]):
            return False
        return True

    def transition_2d(self) -> go.Figure:
        if not self._correct_call_args():
            logger.warning(f'Bad call args to GraphCallback')
            return go.Figure()
        if not self.calculated_triggered or self.calculated.analysis_params.transition_only_datnum is None:
            dat = self.dat
            plotter = TwoD(dat=dat)
            out = dat.SquareEntropy.get_row_only_output(name='default', calculate_only=True)
            x = out.x
            y = dat.Data.get_data('y')
            data = np.nanmean(out.cycled[:, (0, 2,), :], axis=1)
            fig = plotter.plot(data=data, x=x, y=y, title=f'Dat{dat.datnum}: 2D Cold Transition')
            if self.calculated_triggered:
                out = self.calculated.calculated_entropy_fit.output
                x = out.x
                data = np.nanmean(out.cycled[:, (0, 2,), :], axis=1)
                ys = y_from_rows(self.calculated.analysis_params.entropy_data_rows, y, mode='values')
                fig.add_trace(plotter.trace(data=data, x=x, y=np.linspace(ys[0], ys[1], out.entropy_signal.shape[0])))
                for h in ys:
                    plotter.add_line(fig, h, mode='horizontal', color='black')
        else:
            dat = get_dat(self.calculated.analysis_params.transition_only_datnum)
            plotter = TwoD(dat=dat)
            x = dat.Transition.x
            y = dat.Data.get_data('y')
            data = dat.Transition.data
            ys = y_from_rows(self.calculated.analysis_params.transition_data_rows, y, mode='values')
            fig = plotter.plot(data=data, x=x, y=y, title=f'Dat{dat.datnum}: 2D Transition only')
            for h in ys:
                plotter.add_line(fig, h, mode='horizontal', color='black')
        return fig

    def transition_2d_waterfall(self) -> go.Figure:
        return go.Figure()

    def single_row(self) -> go.Figure:
        return go.Figure()

    def avg_data(self) -> go.Figure:
        """Transition figure"""
        if not self._correct_call_args():
            logger.warning(f'Bad call args to GraphCallback')
            return go.Figure()
        dat = self.dat
        plotter = OneD(dat=dat)
        fig = plotter.figure(title=f'Dat{dat.datnum}: Transition')

        return fig
        # TODO
        # if self.calculated_triggered:
        #     if (datnum := self.calculated.analysis_params.transition_only_datnum) is not None:
        #         x = self.calculated.calculated_transition_fit.x
        #         data = self.calculated.calculated_transition_fit.data
        #         fig.add_trace(plotter.trace(data=data, x=x, mode='lines', name=f'T only Dat{datnum}'))
        #     else:
        #         out = self.calculated.calculated_entropy_fit.output
        #         fig.add_trace(
        #             plotter.trace(data=np.nanmean(out.averaged[(0, 2), :], axis=0), x=out.x,
        #                           mode='lines', name=f'Selected Rows'))
        # existing_names = dat.SquareEntropy.get_fit_names(which='transition')
        # for n in self.t_fit_names:
        #     if n in existing_names:
        #         fit = dat.SquareEntropy.get_fit(fit_name=n, which_fit='transition')
        #         fig.add_trace(plotter.trace(data=fit.eval_fit(x=x), x=x, name=f'{n}_fit', mode='lines'))
        #
        # if self.calculated_triggered:
        #     tfit_stuff = self.calculated.calculated_transition_fit
        #     fig.add_trace(plotter.trace(data=tfit_stuff.fit.eval_fit(x), x=x, name='Calculated_fit', mode='lines',
        #                                 trace_kwargs=dict(line=dict(color='black', dash='dash'))))
        # return fig
        #


class DatOptionsCallbacks(CommonInputCallbacks):
    """Common callback to fill in options for dats"""
    components = Components()

    # noinspection PyMissingConstructor
    def __init__(self, datnum: int, se_name, e_names, t_names, int_names):
        self.datnum: Optional[int] = datnum
        self.se_name: str = se_name
        self.e_names: List[str] = listify_dash_input(e_names)
        self.t_names: List[str] = listify_dash_input(t_names)
        self.int_names: List[str] = listify_dash_input(int_names)

        # Generated
        self.dat = get_dat(datnum) if self.datnum is not None else None

    @classmethod
    def get_inputs(cls) -> List[Tuple[str, str]]:
        return [
            (cls.components.inp_datnum.id, 'value'),
        ]

    @classmethod
    def get_states(cls) -> List[Tuple[str, str]]:
        cmps = cls.components
        return [
            # Saved fits info

        ]

    def callback_names_funcs(self) -> dict:
        return {
            'transition fits': self.transition,
        }

    @classmethod
    def output_for_id(cls, id_name: str) -> List[Tuple[str, str]]:
        return [(id_name, 'options'), (id_name, 'value')]

    def _valid_call(self) -> bool:
        if any([self.dat is None, not is_square_entropy_dat(self.dat)]):
            return False
        return True

    def transition(self) -> Tuple[List[Dict[str, str]], str]:
        """Options for T fits dropdown"""
        if not self._valid_call():
            return [], no_update
        return ddu.new_options_keep_old_selection(self.dat.Transition.fit_names, self.t_names)



################################################################################################################
# TODO: Need to carry on from here (i.e. changing from single_entropy to single_transition)




class TableCallbacks(CommonInputCallbacks):
    components = Components()  # For ID's only

    def __init__(self, se_name, e_names, t_names, int_names,
                 calculated,
                 datnum):
        super().__init__()  # Shutting up PyCharm
        self.se_name: str = se_name
        self.e_names: List[str] = listify_dash_input(e_names)
        self.t_names: List[str] = listify_dash_input(t_names)
        self.int_names: List[str] = listify_dash_input(int_names)
        self.datnum: Optional[int] = datnum

        self.calculated: StoreData = calculated
        self.calculated_triggered = triggered_by(self.components.store_calculated.id)

        # Generated
        self.dat = get_dat(datnum) if self.datnum is not None else None

    @staticmethod
    def get_outputs(table: dbc.Table) -> List[Tuple[str, str]]:
        """Columns and Data for Table callbacks"""
        return [(table.id, 'columns'), (table.id, 'data')]

    @classmethod
    def get_inputs(cls) -> List[Tuple[str, str]]:
        return [
            *cls.components.saved_fits_inputs(),
            (cls.components.store_calculated.id, 'data'),
        ]

    @classmethod
    def get_states(cls) -> List[Tuple[str, str]]:
        return [
            (cls.components.inp_datnum.id, 'value'),
        ]

    def callback_names_funcs(self):
        return {
            'entropy_table': self.entropy_table,
            'transition_table': self.transition_table,
            'integrated_table': self.integration_table,
        }

    def _valid_call(self) -> bool:
        """Common check for bad call args which shouldn't be used for plotting"""
        if any([self.dat is None, not is_square_entropy_dat(self.dat)]):
            return False
        return True

    @staticmethod
    def _df_to_table_props(df: pd.DataFrame) -> Tuple[List[Dict[str, str]], List[dict]]:
        df.insert(0, 'Name', df.pop('name'))
        if 'success' in df.columns:
            df.insert(len(df.columns) - 1, 'success', df.pop('success'))
        df = df.applymap(lambda x: f'{x:.3g}' if isinstance(x, (float, np.float)) else x)
        return [{'name': col, 'id': col} for col in df.columns], df.to_dict('records')

    def _get_fit_table(self, existing_names: List[str], requested_names: List[str], fit_getter: Callable,
                       which_calculated: str):
        dfs = []
        for name in requested_names:
            if name in existing_names:
                fit = fit_getter(name)
                df = fit.to_df()
                df['name'] = name
                dfs.append(df)
        if self.calculated_triggered:
            if which_calculated == 'entropy':
                fit = self.calculated.calculated_entropy_fit.fit
            elif which_calculated == 'transition':
                fit = self.calculated.calculated_transition_fit.fit
            elif which_calculated == 'integrated':
                fit = self.calculated.calculated_int_info
            else:
                raise NotImplementedError(f'{which_calculated} not a valid choice')
            df = fit.to_df()
            df['name'] = 'calculated'
            dfs.append(df)
        if len(dfs) == 0:
            return [], []
        df = pd.concat(dfs)
        return self._df_to_table_props(df)

    def entropy_table(self):
        """Table of fit values for Entropy fits"""
        if not self._valid_call():
            return [], []
        return self._get_fit_table(existing_names=self.dat.Entropy.fit_names,
                                   requested_names=self.e_names,
                                   fit_getter=lambda fit_name: self.dat.Entropy.get_fit(name=fit_name),
                                   which_calculated='entropy')

    def transition_table(self):
        """Table of fit values for Transition fits"""
        if not self._valid_call():
            return [], []
        return self._get_fit_table(existing_names=self.dat.SquareEntropy.get_fit_names(which='transition'),
                                   requested_names=self.t_names,
                                   fit_getter=lambda fit_name: self.dat.SquareEntropy.get_fit(fit_name=fit_name,
                                                                                              which_fit='transition'),
                                   which_calculated='transition')

    def integration_table(self):
        """Table of fit values for Integrated Infos"""
        if not self._valid_call():
            return [], []
        return self._get_fit_table(existing_names=self.dat.Entropy.get_integration_info_names(),
                                   requested_names=self.int_names,
                                   fit_getter=lambda fit_name: self.dat.Entropy.get_integration_info(name=fit_name),
                                   which_calculated='integrated')


class CalculateCallback(CommonInputCallbacks):
    components = Components()

    # noinspection PyMissingConstructor,PyUnusedLocal
    def __init__(self, run, run_centers,
                 datnum,
                 overwrite_centers,
                 sp_start, se_transition_func, se_fit_width, se_rows,
                 use_tonly, tonly_datnum, tonly_func, tonly_width, tonly_rows,
                 center_func, force_theta, force_gamma,
                 force_dt, force_amp, int_from_se,
                 use_csq, csq_datnum,
                 ):
        self.run = triggered_by(self.components.but_run.id)  # i.e. True if run was the trigger
        self.run_centers = triggered_by(self.components.but_run_generate_centers.id)
        self.overwrite_centers = True if overwrite_centers else False
        self.datnum = datnum

        # SE fitting
        self.sp_start = sp_start if sp_start else 0.0
        self.ent_transition_func = se_transition_func
        self.ent_width = se_fit_width
        self.ent_rows = se_rows if se_rows else (None, None)

        # Tonly fitting
        self.use_tonly = use_tonly
        self.tonly_datnum = tonly_datnum
        self.tonly_func = tonly_func
        self.tonly_width = tonly_width
        self.tonly_rows = tonly_rows

        self.center_func = center_func
        self.force_theta = force_theta
        self.force_gamma = force_gamma

        # Integration info
        self.force_dt = force_dt
        self.force_amp = force_amp
        self.int_from_se = int_from_se

        # CSQ mapping
        self.csq_map = use_csq
        self.csq_datnum = csq_datnum

        # ## Post init
        self.dat = get_dat(self.datnum) if self.datnum else None

    @classmethod
    def get_inputs(cls) -> List[Tuple[str, str]]:
        return [
            (cls.components.but_run.id, 'n_clicks'),
            (cls.components.but_run_generate_centers.id, 'n_clicks'),
        ]

    @classmethod
    def get_states(cls) -> List[Tuple[str, str]]:
        return [
            (cls.components.inp_datnum.id, 'value'),
            (cls.components.tog_overwrite_centers.id, 'value'),
            *cls.components.se_params_inputs(),
            *cls.components.transition_param_inputs(),
            *cls.components.e_and_t_params_inputs(),
            *cls.components.int_params_inputs(),
            (cls.components.tog_csq_mapped.id, 'value'),
            (cls.components.inp_csq_datnum.id, 'value'),
        ]

    @classmethod
    def get_outputs(cls, id_name: str) -> Tuple[str, str]:
        return id_name, 'data'

    def callback_names_funcs(self):
        return {
            'calculate': self.calculate,
            'calculate_centers': self.calculate_centers,
        }

    def calculate_centers(self) -> str:
        """Calculates center fits for dats with option to overwrite and DOES write to HDF. Can be extremely slow!

        Note: Currently only uses csq_mapped/force_theta/force_gamma... Does not use setpoint or width.
        Note: Saves center fits under same name regardless of whether from CSQ mapping or not.. (should be very similar)
        """

        def get_x_data(dat: DatHDF) -> Tuple[np.ndarray, np.ndarray]:
            x = dat.Data.get_data('x')
            data = dat.Data.get_data('i_sense') if not self.csq_map else e_dat.Data.get_data('csq_mapped')
            if data.ndim == 2:
                data = data[0]
            return x, data

        if not self.run_centers:
            raise PreventUpdate

        e_dat = self.dat
        if e_dat is None:
            raise PreventUpdate

        init_x, init_data = get_x_data(e_dat)
        calc_params = TransitionCalcParams(initial_x=init_x, initial_data=init_data,
                                           force_theta=self.force_theta, force_gamma=self.force_gamma,
                                           csq_mapped=self.csq_map)
        # Do Center calculations
        set_centers(e_dat, self.center_func, calc_params=calc_params, se_data=True, csq_mapped=self.csq_map)

        if self.use_tonly:
            t_dat = get_dat(self.tonly_datnum)
            init_x, init_data = get_x_data(t_dat)
            calc_params = TransitionCalcParams(initial_x=init_x, initial_data=init_data,
                                               force_theta=self.force_theta, force_gamma=self.force_gamma,
                                               csq_mapped=self.csq_map)
            set_centers(t_dat, self.center_func, se_data=False, calc_params=calc_params, csq_mapped=self.csq_map)

        return f'Done'

    def calculate(self) -> Tuple[StoreData, str]:
        """Calculates new Entropy/Transition/Integrated fits with EXISTING center fits only, and does NOT save
            anything to the HDF"""
        if not self.run:
            raise PreventUpdate
        # Put all params into a class for easy access later when displaying things in Dash
        params = GammaAnalysisParams(
            csq_mapped=self.csq_map,
            save_name='NOT SAVED',
            entropy_datnum=self.datnum,
            setpoint_start=self.sp_start,
            entropy_transition_func_name=self.ent_transition_func, entropy_fit_width=self.ent_width,
            entropy_data_rows=self.ent_rows,
            force_dt=self.force_dt, force_amp=self.force_amp,
            sf_from_square_transition=self.int_from_se,
            force_theta=self.force_theta, force_gamma=self.force_gamma,  # Applies to entropy and transition only
            transition_center_func_name=self.center_func,
            transition_only_datnum=self.tonly_datnum,
            transition_func_name=self.tonly_func,
            transition_fit_width=self.tonly_width,
            transition_data_rows=self.tonly_rows,
            # Tonly stuff set below if used
        )

        # Calculate SE output for requested rows only (no fitting in here)
        out = calculate_se_output(dat=self.dat, rows=params.entropy_data_rows, csq_mapped=params.csq_mapped,
                                  center_func_name=params.transition_center_func_name,
                                  setpoint_start=params.setpoint_start)

        # Calculate Entropy fit
        efit = self.dat.Entropy.get_fit(x=out.x, data=out.average_entropy_signal, calculate_only=True)
        calculated_e = CalculatedEntropyFit(x=out.x, data=out.average_entropy_signal, fit=efit, output=out)

        # Calculate Transition fit (from either SE or Tonly)
        if params.transition_only_datnum is None:  # Then need transition fit from SE
            x, data = out.x, np.mean(out.averaged[(0, 2), :], axis=0)
            t_func_name = params.entropy_transition_func_name
            width = params.entropy_fit_width
        else:
            tdat = get_dat(params.transition_only_datnum)
            x, data = calculate_tonly_data(tdat, rows=params.transition_data_rows, csq_mapped=params.csq_mapped,
                                           center_func_name=params.transition_center_func_name)
            t_func_name = params.transition_func_name
            width = params.transition_fit_width

        x, data = get_data_in_range(x, data, width)
        t_func, t_params = _get_transition_fit_func_params(x=x, data=data,
                                                           t_func_name=t_func_name,
                                                           theta=params.force_theta, gamma=params.force_gamma)
        tfit = self.dat.Transition.get_fit(x=x, data=data,
                                           initial_params=t_params, fit_func=t_func,
                                           calculate_only=True)
        calculated_t = CalculatedTransitionFit(x=x, data=data, fit=tfit)

        # Calculate Integrated
        amp = None
        dt = None
        if params.sf_from_square_transition:
            # Need to calc hot and cold
            fs = []
            for t in ['cold', 'hot']:
                # Note: Using initial params potentially from Tonly, but should be extremely similar
                fs.append(self.dat.SquareEntropy.get_fit(initial_params=t_params, fit_func=t_func,
                                                         x=out.x, data=out.averaged, calculate_only=True,
                                                         transition_part=t))
            amp = fs[0].best_values.amp
            dt = fs[1].best_values.theta - fs[0].best_values.theta
        if self.use_tonly:
            amp = tfit.best_values.amp
        if params.force_amp:
            amp = params.force_amp
        if params.force_dt:
            dt = params.force_dt
        int_info = IntegrationInfo(dT=dt, amp=amp, dx=(dx := float(np.nanmean(np.diff(out.x)))),
                                   sf=scaling(dt, amp, dx))

        return StoreData(analysis_params=params,
                         SE_output=out,
                         calculated_entropy_fit=calculated_e,
                         calculated_transition_fit=calculated_t,
                         calculated_int_info=int_info), 'done'


@dataclass
class StoreData:
    analysis_params: GammaAnalysisParams
    SE_output: Output
    calculated_entropy_fit: CalculatedEntropyFit
    calculated_transition_fit: CalculatedTransitionFit
    calculated_int_info: IntegrationInfo


def listify_dash_input(val: Optional[str, List[str]]) -> List[str]:
    """Makes dash inputs into a list of strings instead of any of (None, '', 'value' or ['value1', 'value2'])"""
    if isinstance(val, list):
        return val
    elif val is None or val == '':
        return []
    elif isinstance(val, str):
        return [val]
    else:
        raise RuntimeError(f"Don't know how to listify {val}")


def is_square_entropy_dat(dat: Union[None, DatHDF]) -> bool:
    if dat is None:
        return False
    try:
        # noinspection PyPropertyAccess
        dat.Logs.awg
    except U.NotFoundInHdfError:
        return False
    return True


def y_from_rows(rows: Tuple[Optional[int], Optional[int]], y_data: np.ndarray, mode: str = 'values'):
    rows = (rows[0] if rows[0] else 0, rows[1] if rows[1] else y_data.shape[0])
    if mode == 'values':
        ret = y_data[rows[0]], y_data[rows[1]]
    elif mode == 'indexes':
        ret = rows
    else:
        raise KeyError(f'{mode} not recognized')
    return ret


# Required for multipage
# noinspection PyUnusedLocal
def layout(*args):  # *args only because dash_extensions passes in the page name for some reason
    inst = SingleTransitionLayout(Components())
    inst.page_collection = page_collection
    return inst.layout()


def callbacks(app):
    inst = SingleTransitionLayout(Components(pending_callbacks=PendingCallbacks()))
    inst.page_collection = page_collection
    inst.layout()  # Most callbacks are generated while running layout
    return inst.run_all_callbacks(app)


# ###########################################


if __name__ == '__main__':
    from dash_dashboard.app import test_page

    test_page(layout=layout, callbacks=callbacks, single_threaded=False, port=8050)
