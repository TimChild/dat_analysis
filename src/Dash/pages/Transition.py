from __future__ import annotations
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


class TransitionLayout(DatDashPageLayout):
    def get_mains(self) -> List[Tuple[str, DatDashMain]]:
        return [('Avg Fit', TransitionMainAvg()), ('Row Fits', TransitionMainRows())]

    def get_sidebar(self) -> DatDashSideBar:
        return TransitionSidebar()

    @property
    def id_prefix(self):
        return 'T'


class TransitionMainAvg(DatDashMain):

    def get_sidebar(self):
        return TransitionSidebar()

    @property
    def id_prefix(self):
        return 'Tmain'

    def layout(self):
        layout = html.Div([
            self.graph_area('graph-avg', 'Avg Fit'),
            self.graph_area('graph-twoD', 'Data'),
        ],
            id=self.id('div-avg-graphs'))
        return layout

    def set_callbacks(self):
        self.sidebar.layout()  # Make sure layout has been generated
        inps = self.sidebar.inputs
        main = (self.sidebar.main_dropdown().id, 'value')

        # Show main graph
        self.graph_callback('graph-avg', partial(get_figure, mode='avg'),
                            inputs=[
                                main,
                                (inps['inp-datnum'].id, 'value'),
                                (inps['dd-saved-fits'].id, 'value'),
                                (inps['div-button-output'].id, 'children'),  # Just to trigger update
                            ],
                            states=[])

        # Show 2D data
        self.graph_callback('graph-twoD', partial(get_figure, mode='twoD'),
                            inputs=[
                                main,
                                (inps['inp-datnum'].id, 'value'),
                                (inps['dd-saved-fits'].id, 'value'),
                                (inps['div-button-output'].id, 'children'),  # Just to trigger update
                            ])


class TransitionMainRows(TransitionMainAvg):

    def layout(self):
        layout = html.Div([
            self.graph_area('graph-row', 'Row Fit'),
            self.graph_area('graph-waterfall', 'All Rows'),
        ],
            id=self.id('div-row-graphs'))
        return layout

    def set_callbacks(self):
        self.sidebar.layout()
        inps = self.sidebar.inputs
        main = (self.sidebar.main_dropdown().id, 'value')

        # Single row graph
        self.graph_callback('graph-row', partial(get_figure, mode='single_row'),
                            inputs=[
                                main,
                                (inps['inp-datnum'].id, 'value'),
                                (inps['dd-saved-fits'].id, 'value'),
                                (inps['div-button-output'].id, 'children'),  # Just to trigger update
                                (inps['sl-slicer'].id, 'value'),
                            ],
                            states=[])

        # Waterfall graph
        self.graph_callback('graph-waterfall', partial(get_figure, mode='waterfall'),
                            inputs=[
                                main,
                                (inps['inp-datnum'].id, 'value'),
                                (inps['dd-saved-fits'].id, 'value'),
                                (inps['div-button-output'].id, 'children'),  # Just to trigger update
                            ])


@singleton
class TransitionSidebar(DatDashSideBar):

    @property
    def id_prefix(self):
        return 'Tsidebar'

    def layout(self):
        layout = html.Div([
            self.main_dropdown(),  # Choice between Avg view and Row view
            self.input_box(name='Dat', id_name='inp-datnum', placeholder='Choose Datnum', autoFocus=True, min=0),
            self.dropdown(name='Saved Fits', id_name='dd-saved-fits', multi=True),
            self.dropdown(name='Fit Func', id_name='dd-fit-func'),
            self.checklist(name='Param Vary', id_name='check-param-vary'),
            self._param_inputs(),
            self.button(name='Run Fit', id_name='but-run-fit'),

            self.div(id_name='div-button-output', style={'display': 'none'}),
            # ^^ A blank thing I can use to update other things AFTER fits run

            html.Div(self.slider(name='Slicer', id_name='sl-slicer', updatemode='mouseup'), id=self.id('div-slicer')),
            html.Hr(),  # Separate inputs from info
            self.table(name='Fit Values', id_name='table-fit-values'),
        ])

        # Set options here so it isn't so cluttered in layout above
        self.dropdown(id_name='dd-fit-func').options = [
            {'label': 'i_sense', 'value': 'i_sense'},
            {'label': 'i_sense_digamma', 'value': 'i_sense_digamma'},
            {'label': 'i_sense_digamma_quad', 'value': 'i_sense_digamma_quad'},
        ]
        cl = self.checklist(id_name='check-param-vary')
        cl.options = [
            {'label': 'theta', 'value': 'theta'},
            {'label': 'amp', 'value': 'amp'},
            {'label': 'gamma', 'value': 'gamma'},
            {'label': 'lin', 'value': 'lin'},
            {'label': 'const', 'value': 'const'},
            {'label': 'mid', 'value': 'mid'},
            {'label': 'quad', 'value': 'quad'},
        ]
        cl.value = [d['value'] for d in cl.options]  # Set default to all vary

        return layout

    def set_callbacks(self):
        inps = self.inputs

        # Make some common inputs quicker to use
        main = (self.main_dropdown().id, 'value')
        datnum = (inps['inp-datnum'].id, 'value')
        slice_val = (inps['sl-slicer'].id, 'value')

        # Set Saved Fits options
        self.make_callback(
            inputs=[
                datnum,
                (inps['div-button-output'].id, 'children')
            ],
            outputs=[
                (inps['dd-saved-fits'].id, 'options')
            ],
            func=get_saved_fit_names
        )

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

        # Set table info
        self.make_callback(
            inputs=[
                main,
                datnum,
                slice_val,
                (inps['dd-saved-fits'].id, 'value'),
                (inps['div-button-output'].id, 'children'),  # Just to trigger update
            ],
            outputs=[
                (inps['table-fit-values'].id, 'children')
            ],
            func=update_tab_fit_values
        )

        # Run Fits
        self.make_callback(
            inputs=[
                (inps['but-run-fit'].id, 'n_clicks'),
            ],
            outputs=[
                (inps['div-button-output'].id, 'children')
            ],
            func=run_fits,
            states=[
                main,
                datnum,
                (inps['dd-fit-func'].id, 'value'),

                (inps['check-param-vary'].id, 'value'),

                (inps['inp-theta'].id, 'value'),
                (inps['inp-amp'].id, 'value'),
                (inps['inp-gamma'].id, 'value'),
                (inps['inp-lin'].id, 'value'),
                (inps['inp-const'].id, 'value'),
                (inps['inp-mid'].id, 'value'),
                (inps['inp-quad'].id, 'value'),
            ]
        )

    def _param_inputs(self):
        par_input = dbc.Row([
            dbc.Col(
                dbc.FormGroup(
                    [
                        dbc.Label('theta', html_for=self.id('inp-theta')),
                        self.input_box(val_type='number', id_name='inp-theta', className='px-0', bs_size='sm')
                    ],
                ), className='p-1'
            ),
            dbc.Col(
                dbc.FormGroup(
                    [
                        dbc.Label('amp', html_for=self.id('inp-amp')),
                        self.input_box(val_type='number', id_name='inp-amp', className='px-0', bs_size='sm')
                    ],
                ), className='p-1'
            ),
            dbc.Col(
                dbc.FormGroup(
                    [
                        dbc.Label('gamma', html_for=self.id('inp-gamma')),
                        self.input_box(val_type='number', id_name='inp-gamma', className='px-0', bs_size='sm')
                    ],
                ), className='p-1'
            ),
            dbc.Col(
                dbc.FormGroup(
                    [
                        dbc.Label('lin', html_for=self.id('inp-lin')),
                        self.input_box(val_type='number', id_name='inp-lin', className='px-0', bs_size='sm')
                    ],
                ), className='p-1'
            ),
            dbc.Col(
                dbc.FormGroup(
                    [
                        dbc.Label('const', html_for=self.id('inp-const')),
                        self.input_box(val_type='number', id_name='inp-const', className='px-0', bs_size='sm')
                    ],
                ), className='p-1'
            ),
            dbc.Col(
                dbc.FormGroup(
                    [
                        dbc.Label('mid', html_for=self.id('inp-mid')),
                        self.input_box(val_type='number', id_name='inp-mid', className='px-0', bs_size='sm')
                    ],
                ), className='p-1'
            ),
            dbc.Col(
                dbc.FormGroup(
                    [
                        dbc.Label('quad', html_for=self.id('inp-quad')),
                        self.input_box(val_type='number', id_name='inp-quad', className='px-0', bs_size='sm')
                    ],
                ), className='p-1'
            ),
        ])
        return par_input


def update_tab_fit_values(main, datnum, slice_val, fit_names, button_done) -> Tuple[List[str], dict]:
    """dash_table.DataTable takes """
    df = pd.DataFrame()
    if datnum:
        dat = get_dat(datnum)
        t: T.Transition = dat.Transition

        if slice_val is None:
            slice_val = 0

        if fit_names is None or fit_names == []:
            fit_names = ['default']

        checks = [False if n == 'default' else True for n in fit_names]

        if main == 'T_Avg Fit':
            fit_values = [t.get_fit(which='avg', name=n, check_exists=check).best_values for n, check in zip(fit_names, checks)]
        elif main == 'T_Row Fits':
            fit_values = [t.get_fit(which='row', row=slice_val, name=n, check_exists=check).best_values for n, check in zip(fit_names, checks)]
        else:
            raise ValueError(f'{main} not an expected value')
        if fit_values:
            df = pd.DataFrame()
            for fvs in fit_values:
                df = df.append(fvs.to_df())
        else:
            raise ValueError(f'No fit values found')
        df.index = [n for n in fit_names]
    df = df.applymap(lambda x: f'{x:.3g}')
    df = df.reset_index()  # Make index into a normal Column
    ret = dbc.Table.from_dataframe(df).children  # convert to something that can be passed to dbc.Table.children
    return ret


def get_figure(main, datnum, fit_names, fit_done, slice_val=0, mode='avg'):
    # If button_done is the trigger, should fit_name stored there (which is currently just 'Dash' every time)
    # ctx = dash.callback_context
    # if not ctx.triggered:
    #     return go.Figure()
    if False:
        pass
    else:
        if datnum:
            dat = get_dat(datnum)
            t: T.Transition = dat.Transition
            # trig_id = ctx.triggered[0]['prop_id'].split('.')[0]
            # if trig_id == 'Tsidebar_div-button-output':
            #     if fit_done not in fit_names:
            #         fit_names.append(fit_done)
            if fit_names is None or fit_names == []:
                fit_names = ['default']

            checks = [False if n == 'default' else True for n in fit_names]

            if main == 'T_Avg Fit':
                if mode == 'avg':
                    x = t.avg_x
                    x = U.get_matching_x(x, shape_to_match=1000)
                    fits = [t.get_fit(which='avg', name=n, check_exists=check) for n, check in zip(fit_names, checks)]
                    plotter = DashOneD(dat)
                    fig = plotter.plot(t.avg_data, x=t.avg_x, ylabel='Current /nA', title=f'Dat{dat.datnum}: Avg Transition Fit',
                                       trace_name='Avg Data')
                    for fit, n in zip(fits, fit_names):
                        fig.add_trace(plotter.trace(fit.eval_fit(x), x=x, name=f'Fit_{n}', mode='lines'))
                    return fig
                elif mode == 'twoD':
                    plotter = DashTwoD(dat)
                    fig = plotter.plot(dat.Transition.data, dat.Transition.x, dat.Data.y, title=f'Dat{dat.datnum}: Full 2D Data')
                    return fig

            elif main == 'T_Row Fits':
                if mode == 'single_row':
                    if not slice_val:
                        slice_val = 0
                    x = t.x
                    x = U.get_matching_x(x, shape_to_match=1000)
                    fits = [t.get_fit(which='row', row=slice_val, name=n, check_exists=check) for n, check in zip(fit_names, checks)]
                    plotter = DashOneD(dat)
                    fig = plotter.plot(dat.Transition.data[slice_val], x=dat.Transition.x, ylabel='Current /nA',
                                       title=f'Dat{dat.datnum}: Single Row Fit: {slice_val}', trace_name=f'Row {slice_val} data')
                    for fit, n in zip(fits, fit_names):
                        fig.add_trace(plotter.trace(fit.eval_fit(x), x=x, name=f'Fit_{n}', mode='lines'))
                    return fig
                elif mode == 'waterfall':
                    plotter = DashTwoD(dat)
                    fig = plotter.plot(t.data, t.x, dat.Data.y, ylabel='Current /nA', title=f'Dat{dat.datnum}: Waterfall plot of Data',
                                       plot_type='waterfall')
                    return fig
    raise PreventUpdate


def set_slider_vals(datnum):
    if datnum:
        dat = get_dat(datnum)
        y = dat.Data.get_data('y')
        start, stop, step, value = 0, len(y) - 1, 1, round(len(y) / 2)
        marks = {int(v): str(v) for v in np.arange(start, stop, 10)}
        return start, stop, step, value, marks
    return 0, 1, 0.1, 0.5, {0: '0', 0.5: '0.5', 1: '1'}


def toggle_div(value):
    if value == [True]:
        return False
    else:
        return True


def get_saved_fit_names(datnum, fits_done) -> List[dict]:
    if datnum:
        dat = get_dat(datnum)
        t: T.Transition = dat.Transition
        fit_names = t.fit_names
        print(fit_names)
        return [{'label': k, 'value': k} for k in fit_names]
    raise PreventUpdate


def run_fits(button_click,
             main,
             datnum,
             fit_func,
             params_vary,
             theta_value, amp_value, gamma_value, lin_value, const_value, mid_value, quad_value):

    if button_click and datnum:
        dat = get_dat(datnum)
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
        print(par_varies)
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

        if main == 'T_Row Fits':
            [dat.Transition.get_fit(which='row', row=i, name='Dash', initial_params=new_pars, fit_func=func,
                                    check_exists=False) for i in range(dat.Transition.data.shape[0])]

        # Always run avg fit since it will be MUCH faster anyway
        dat.Transition.get_fit(which='avg', name='Dash', initial_params=new_pars, fit_func=func,
                               check_exists=False)
        return 'Dash'
    else:
        raise PreventUpdate


# Generate layout for to be used in App
layout = TransitionLayout().layout()

if __name__ == '__main__':
    dat = get_dat(9111)
