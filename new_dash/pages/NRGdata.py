from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass
import logging
import lmfit as lm
import threading
from itertools import chain
from collections import OrderedDict

import numpy as np
from scipy.interpolate import interp1d
from plotly import graph_objects as go
import dash_bootstrap_components as dbc

from dash_dashboard.base_classes import BaseSideBar, PageInteractiveComponents, \
    CommonInputCallbacks, PendingCallbacks
from new_dash.base_class_overrides import DatDashPageLayout, DatDashMain, DatDashSidebar

from dash_dashboard.util import triggered_by
import dash_dashboard.util as du
import dash_dashboard.component_defaults as ccs
import dash_html_components as html
from dash import no_update
from dash.exceptions import PreventUpdate
import dash_core_components as dcc

from src.analysis_tools.nrg import NRG_func_generator, NRGData
from Analysis.Feb2021.common import data_from_output
from src.analysis_tools.entropy import do_entropy_calc
from src.plotting.plotly.dat_plotting import TwoD
from src.dat_object.make_dat import get_dat
from src.plotting.plotly.dat_plotting import OneD
from src.characters import THETA
from src.useful_functions import ensure_list, NotFoundInHdfError
from src.analysis_tools.general_fitting import calculate_fit, get_data_in_range, Values
from src.useful_functions import get_data_index

if TYPE_CHECKING:
    from src.dat_object.Attributes.SquareEntropy import Output
    from src.dat_object.make_dat import DatHDF

logger = logging.getLogger(__name__)
thread_lock = threading.Lock()

NAME = 'NRG'
URL_ID = 'NRG'
page_collection = None  # Gets set when running in multipage mode

# Other page constants
ESC_GAMMA_LIMIT = -240  # Where gamma broadening begins


class Components(PageInteractiveComponents):
    def __init__(self, pending_callbacks=None):
        super().__init__(pending_callbacks)
        # Graphs
        self.graph_1 = ccs.graph_area(id_name='graph-1', graph_header='',  # self.header_id -> children to update header
                                      pending_callbacks=self.pending_callbacks)
        self.graph_2 = ccs.graph_area(id_name='graph-2', graph_header='',
                                      pending_callbacks=self.pending_callbacks)
        self.graph_3 = ccs.graph_area(id_name='graph-3', graph_header='',
                                      pending_callbacks=self.pending_callbacks)
        self.graph_4 = ccs.graph_area(id_name='graph-4', graph_header='', pending_callbacks=self.pending_callbacks)
        self.graph_5 = ccs.graph_area(id_name='graph-5', graph_header='',
                                      pending_callbacks=self.pending_callbacks)

        # Input
        self.experiment_name = ccs.dropdown(id_name='dd-experiment-name', multi=False, persistence=True)
        self.dd_which_nrg = ccs.dropdown(id_name='dd-which-nrg', multi=True, persistence=True)
        self.dd_which_x_type = ccs.dropdown(id_name='dd-which-x-type', multi=False, persistence=True)

        self.inp_datnum = ccs.input_box(id_name='inp-datnum', persistence=True)
        self.dd_hot_or_cold = ccs.dropdown(id_name='dd-hot-or-cold', multi=False, persistence=True)
        self.but_fit = ccs.button(id_name='but-fit', text='Fit to Data')

        self.tog_vary_theta = ccs.toggle(id_name='tog-vary-theta', persistence=True)
        self.tog_vary_gamma = ccs.toggle(id_name='tog-vary-gamma', persistence=True)

        self.slider_gamma = ccs.slider(id_name='sl-gamma', updatemode='drag', persistence=False)
        self.slider_theta = ccs.slider(id_name='sl-theta', updatemode='drag', persistence=False)
        self.slider_mid = ccs.slider(id_name='sl-mid', updatemode='drag', persistence=False)
        self.slider_amp = ccs.slider(id_name='sl-amp', updatemode='drag', persistence=False)
        self.slider_lin = ccs.slider(id_name='sl-lin', updatemode='drag', persistence=False)
        self.slider_const = ccs.slider(id_name='sl-const', updatemode='drag', persistence=False)
        self.slider_occ_lin = ccs.slider(id_name='sl-occ-lin', updatemode='drag', persistence=False)

        # Output
        self.store_fit = ccs.store(id_name='ss-store-fit', storage_type='memory')
        self.text_redchi = dcc.Textarea(id='text-redchi', style={'width': '100%', 'height': '50px'})
        self.text_params = dcc.Textarea(id='text-params', style={'width': '100%', 'height': '200px'})

        self.setup_initial_state()

    def setup_initial_state(self):
        self.experiment_name.options = du.list_to_options(['May21', 'FebMar21'])

        self.dd_which_nrg.options = [{'label': k, 'value': k}
                                     for k in list(NRGData.__annotations__) + ['i_sense_cold', 'i_sense_hot'] if
                                     k not in ['ens', 'ts']]
        self.dd_which_nrg.value = 'occupation'
        self.dd_which_x_type.options = [{'label': k, 'value': k} for k in ['Energy', 'Gate']]
        self.dd_which_x_type.value = 'Gate'

        self.dd_hot_or_cold.options = [{'label': t, 'value': t} for t in ['cold', 'hot']]
        self.dd_hot_or_cold.value = 'cold'

        # Note: Some of this gets reset in UpdateSliderCallback based on datnum
        for component, setup in {self.slider_gamma: [-2.0, 2.5, 0.025,
                                                     {int(x) if x % 1 == 0 else x: f'{10 ** x:.2f}' for x in
                                                      np.linspace(-2, 2.5, 5)}, np.log10(10)],
                                 self.slider_theta: [0.01, 60, 0.1, None, 4],
                                 self.slider_mid: [-100, 100, 0.1, None, 0],
                                 self.slider_amp: [0.05, 1.5, 0.01, None, 0.75],
                                 self.slider_lin: [0, 0.01, 0.00001, None, 0.0012],
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

    def Inputs_sliders(self) -> List[Tuple[str, str]]:
        """
        mid, gamma, theta, amp, lin, const, occ_lin
        Returns:

        """
        return [
            (self.slider_gamma.id, 'value'),
            (self.slider_theta.id, 'value'),
            (self.slider_mid.id, 'value'),
            (self.slider_amp.id, 'value'),
            (self.slider_lin.id, 'value'),
            (self.slider_const.id, 'value'),
            (self.slider_occ_lin.id, 'value')
        ]


class NRGLayout(DatDashPageLayout):
    # Defining __init__ only for typinhtml.Div() (i.e. to specify page specific Components as type for self.components)
    def __init__(self, components: Components):
        super().__init__(page_components=components)
        self.components = components

    # def get_mains(self) -> List[NRGMain]:
    #     return [NRGMain(self.components), ]

    def get_main(self) -> NRGMain:
        return NRGMain(self.components)

    def get_sidebar(self) -> BaseSideBar:
        return NRGSidebar(self.components)


class NRGMain(DatDashMain):
    name = 'NRG'

    # Defining __init__ only for typing purposes (i.e. to specify page specific Components as type for self.components)
    def __init__(self, components: Components):
        super().__init__(page_components=components)
        self.components = components

    def layout(self):
        lyt = html.Div([
            self.components.graph_2,
            self.components.graph_3,
            self.components.graph_4,
            self.components.graph_5,
            self.components.graph_1,
        ])
        return lyt

    def set_callbacks(self):
        self.make_callback(outputs=(self.components.graph_2.graph_id, 'figure'),
                           inputs=SliderInputCallback.get_inputs(),
                           states=SliderInputCallback.get_states(),
                           func=SliderInputCallback.get_callback_func('1d'))

        self.make_callback(outputs=(self.components.graph_3.graph_id, 'figure'),
                           inputs=SliderInputCallback.get_inputs(),
                           states=SliderInputCallback.get_states(),
                           func=SliderInputCallback.get_callback_func('1d-data-changed'))

        self.make_callback(outputs=(self.components.graph_4.graph_id, 'figure'),
                           inputs=SliderInputCallback.get_inputs(),
                           states=SliderInputCallback.get_states(),
                           func=SliderInputCallback.get_callback_func('1d-data-subtract-fit'))

        self.make_callback(outputs=(self.components.graph_5.graph_id, 'figure'),
                           inputs=SliderInputCallback.get_inputs(),
                           states=SliderInputCallback.get_states(),
                           func=SliderInputCallback.get_callback_func('1d-data-vs-n'))

        self.make_callback(outputs=(self.components.graph_1.graph_id, 'figure'),
                           inputs=SliderInputCallback.get_inputs(),
                           states=SliderInputCallback.get_states(),
                           func=SliderInputCallback.get_callback_func('2d'))


class NRGSidebar(DatDashSidebar):
    id_prefix = 'NRGSidebar'

    # Defining __init__ only for typing purposes (i.e. to specify page specific Components as type for self.components)
    def __init__(self, components: Components):
        super().__init__(page_components=components)
        self.components = components

    def layout(self):
        lyt = html.Div([
            self.components.store_fit,  # Just a place to store fit as intermediate callback step

            self.input_wrapper('Experiment Name', self.components.experiment_name),
            self.input_wrapper('Data Type', self.components.dd_which_nrg),
            self.input_wrapper('X axis', self.components.dd_which_x_type),
            html.Hr(),
            self.input_wrapper('Dat', self.components.inp_datnum),
            self.input_wrapper('Fit to', self.components.dd_hot_or_cold),
            dbc.Row([
                dbc.Col([
                    self.components.but_fit,
                    self.input_wrapper('Vary Theta', self.components.tog_vary_theta),
                    self.input_wrapper('Vary Gamma', self.components.tog_vary_gamma),
                ]),
                dbc.Col([
                    self.input_wrapper('Reduced Chi Square', self.components.text_redchi, mode='label')
                ])
            ]),
            html.Hr(),
            self.input_wrapper('gamma', self.components.slider_gamma, mode='label'),
            self.input_wrapper('theta', self.components.slider_theta, mode='label'),
            self.input_wrapper('mid', self.components.slider_mid, mode='label'),
            self.input_wrapper('amp', self.components.slider_amp, mode='label'),
            self.input_wrapper('lin', self.components.slider_lin, mode='label'),
            self.input_wrapper('const', self.components.slider_const, mode='label'),
            self.input_wrapper('lin_occ', self.components.slider_occ_lin, mode='label'),
            html.Hr(),
            self.components.text_params,
        ])
        return lyt

    def set_callbacks(self):
        # Store Fit
        self.make_callback(serverside_outputs=(self.components.store_fit.id, 'data'),
                           inputs=SliderStateCallback.get_inputs(),
                           states=SliderStateCallback.get_states(),
                           func=SliderStateCallback.get_callback_func('run_fit'))

        # Text Box for Params
        self.make_callback(outputs=(self.components.text_params.id, 'value'),
                           inputs=SliderInputCallback.get_inputs(),
                           states=SliderInputCallback.get_states(),
                           func=SliderInputCallback.get_callback_func('params'))

        # Text Box Reduced Chi Square
        self.make_callback(outputs=(self.components.text_redchi.id, 'value'),
                           inputs=FitResultCallbacks.get_inputs(),
                           states=FitResultCallbacks.get_states(),
                           func=FitResultCallbacks.get_callback_func('update_redchi'))

        # Slider values when fitting
        self.make_callback(outputs=FitResultCallbacks.get_outputs_slider_values(),
                           inputs=FitResultCallbacks.get_inputs(),
                           states=FitResultCallbacks.get_states(),
                           func=FitResultCallbacks.get_callback_func('update_sliders'))

        # Slider options when changing datnum
        self.make_callback(outputs=SliderStateCallback.get_outputs_all_not_value(),
                           inputs=SliderStateCallback.get_inputs(),
                           states=SliderStateCallback.get_states(),
                           func=SliderStateCallback.get_callback_func('set_ranges'))


class SliderInputCallback(CommonInputCallbacks):
    """For components which are updated on every slider change"""
    components = Components()  # Only use this for accessing IDs only... DON'T MODIFY

    def __init__(self,
                 which, x_type,
                 datnum,
                 g, theta, mid, amp, lin, const, occ_lin,
                 experiment_name,
                 ):
        super().__init__()  # Just here to shut up PyCharm
        self.experiment_name = experiment_name if experiment_name else None
        self.which = ensure_list(which if which else ['occupation'])
        self.x_type = x_type
        self.which_triggered = triggered_by(self.components.dd_which_nrg.id)

        self.datnum = datnum
        self.mid = mid
        self.g = 10 ** g
        self.theta = theta
        self.amp = amp
        self.lin = lin
        self.const = const
        self.occ_lin = occ_lin

    @classmethod
    def get_inputs(cls) -> List[Tuple[str, str]]:
        return [
            (cls.components.dd_which_nrg.id, 'value'),
            (cls.components.dd_which_x_type.id, 'value'),
            (cls.components.inp_datnum.id, 'value'),

            *cls.components.Inputs_sliders(),
        ]

    @classmethod
    def get_states(cls) -> List[Tuple[str, str]]:
        return [
            (cls.components.experiment_name.id, 'value'),
        ]

    def callback_names_funcs(self):
        """
        Return a dict of {<name>: <callback_func>}
        """
        return {
            "2d": self.two_d,
            "1d": self.one_d,
            "1d-data-changed": self.one_d_data_changed,
            "1d-data-subtract-fit": self.one_d_data_subtract_fit,
            "params": self.text_params,
            "1d-data-vs-n": self.one_d_data_vs_n,
        }

    def two_d(self) -> go.Figure:
        if not self.which_triggered:
            return no_update
        which = self.which[0]
        if which in ['i_sense_cold', 'i_sense_hot']:
            which = 'i_sense'
        fig = plot_nrg(which=which, plot=False, x_axis_type=self.x_type.lower())
        return fig

    def one_d(self, invert_fit_on_data=False) -> go.Figure:
        """

        Args:
            invert_fit_on_data (): False to modify NRG to fit data, True to modify Data to fit NRG

        Returns:

        """
        plotter = OneD(dat=None)
        title_prepend = f'NRG fit to Data' if not invert_fit_on_data else 'Data fit to NRG'
        title_append = f' -- Dat{self.datnum}' if self.datnum else ''
        xlabel = 'Sweepgate /mV' if not invert_fit_on_data else 'Ens*1000'
        ylabel = 'Current /nA' if not invert_fit_on_data else '1-Occupation'
        fig = plotter.figure(xlabel=xlabel, ylabel=ylabel, title=f'{title_prepend}: G={self.g:.2f}mV, '
                                                                 f'{THETA}={self.theta:.2f}mV, '
                                                                 f'{THETA}/G={self.theta / self.g:.2f}'
                                                                 f'{title_append}')
        min_, max_ = 0, 1
        if self.datnum:
            x_for_nrg = None
            for i, which in enumerate(self.which):
                x, data = _get_x_and_data(self.datnum, self.experiment_name, which)
                x_for_nrg = x
                if invert_fit_on_data is True:
                    x, data = invert_nrg_fit_params(x, data, gamma=self.g, theta=self.theta, mid=self.mid, amp=self.amp,
                                                    lin=self.lin, const=self.const, occ_lin=self.occ_lin,
                                                    data_type=which)
                if i == 0 and data is not None:
                    min_, max_ = np.nanmin(data), np.nanmax(data)
                    fig.add_trace(plotter.trace(x=x, data=data, name=f'Data - {which}', mode='lines'))
                else:
                    if data is not None:
                        scaled = scale_data(data, min_, max_)
                        fig.add_trace(
                            plotter.trace(x=x, data=scaled.scaled_data, name=f'Scaled Data - {which}', mode='lines'))
                        if min_ - (max_ - min_) / 10 < scaled.new_zero < max_ + (max_ - min_) / 10:
                            plotter.add_line(fig, scaled.new_zero, mode='horizontal', color='black',
                                             linetype='dot', linewidth=1)
        else:
            x_for_nrg = np.linspace(-100, 100, 1001)

        for i, which in enumerate(self.which):
            if which == 'i_sense_cold':
                which = 'i_sense'
            elif which == 'i_sense_hot':
                if 'i_sense_cold' in self.which:
                    continue
                which = 'i_sense'
            nrg_func = NRG_func_generator(which=which)
            if invert_fit_on_data:
                # nrg_func(x, mid, gamma, theta, amp, lin, const, occ_lin)
                nrg_data = nrg_func(x_for_nrg, self.mid, self.g, self.theta, 1, 0, 0, 0)
                if which == 'i_sense':
                    nrg_data += 0.5  # 0.5 because that still gets subtracted otherwise
                # x = (x_for_nrg - self.mid - self.g*(-1.76567) - self.theta*(-1)) / self.g
                x = (x_for_nrg - self.mid) / self.g
            else:
                x = x_for_nrg
                nrg_data = nrg_func(x, self.mid, self.g, self.theta, self.amp, self.lin, self.const, self.occ_lin)
            cmin, cmax = np.nanmin(nrg_data), np.nanmax(nrg_data)
            if i == 0 and min_ == 0 and max_ == 1:
                fig.add_trace(plotter.trace(x=x, data=nrg_data, name=f'NRG {which}', mode='lines'))
                min_, max_ = cmin, cmax
            else:
                scaled = scale_data(nrg_data, min_, max_)
                fig.add_trace(plotter.trace(x=x, data=scaled.scaled_data, name=f'Scaled NRG {which}', mode='lines'))
                if min_ - (max_ - min_) / 10 < scaled.new_zero < max_ + (max_ - min_) / 10:
                    plotter.add_line(fig, scaled.new_zero, mode='horizontal', color='black',
                                     linetype='dot', linewidth=1)
        return fig

    def one_d_data_changed(self):
        return self.one_d(invert_fit_on_data=True)

    def one_d_data_subtract_fit(self):
        if self.datnum:
            dat = get_dat(self.datnum, exp2hdf=self.experiment_name)
            plotter = OneD(dat=dat)
            xlabel = 'Sweepgate /mV'
            fig = plotter.figure(xlabel=xlabel, ylabel='Current /nA', title=f'Data Subtract Fit: G={self.g:.2f}mV, '
                                                                            f'{THETA}={self.theta:.2f}mV, '
                                                                            f'{THETA}/G={self.theta / self.g:.2f}')
            for i, which in enumerate(self.which):
                if 'i_sense' in which:
                    x, data = _get_x_and_data(self.datnum, self.experiment_name, which)
                    nrg_func = NRG_func_generator(which='i_sense')
                    nrg_data = nrg_func(x, self.mid, self.g, self.theta, self.amp, self.lin, self.const, self.occ_lin)
                    data_sub_nrg = data - nrg_data
                    fig.add_trace(plotter.trace(x=x, data=data_sub_nrg, name=f'{which} subtract NRG', mode='lines'))
            return fig
        return go.Figure()

    def one_d_data_vs_n(self):
        if self.datnum:
            self.which = ['i_sense_cold', 'dndt', 'occupation']

            try:
                x, data_dndt = _get_x_and_data(self.datnum, self.experiment_name, 'dndt')
            except NotFoundInHdfError:
                logger.warning(f'Dat{self.datnum}: dndt data not found, probably a transition only dat')
                return go.Figure()

            nrg_func = NRG_func_generator(which='dndt')
            nrg_dndt = nrg_func(x, self.mid, self.g, self.theta, self.amp, self.lin, self.const, self.occ_lin)
            nrg_func = NRG_func_generator(which='occupation')
            occupation = nrg_func(x, self.mid, self.g, self.theta, self.amp, self.lin, self.const, self.occ_lin)

            # Rescale dN/dTs to have a peak at 1
            nrg_dndt = nrg_dndt * (1 / np.nanmax(nrg_dndt))
            x_max = x[get_data_index(data_dndt, np.nanmax(data_dndt))]
            x_range = abs(x[-1] - x[0])
            indexs = get_data_index(x, [x_max - x_range / 50, x_max + x_range / 50])
            avg_peak = np.nanmean(data_dndt[indexs[0]:indexs[1]])
            # avg_peak = np.nanmean(data_dndt[np.nanargmax(data_dndt) - round(x.shape[0] / 50):
            #                                 np.nanargmax(data_dndt) + round(x.shape[0] / 50)])
            data_dndt = data_dndt * (1 / avg_peak)
            if (new_max := np.nanmax(np.abs([np.nanmax(data_dndt), np.nanmin(data_dndt)]))) > 5:  # If very noisy
                data_dndt = data_dndt / (new_max / 5)  # Reduce to +-5ish

            interp_range = np.where(np.logical_and(occupation < 0.99, occupation > 0.01))
            if len(interp_range[0]) > 5:  # If enough data to actually plot something
                interp_data = occupation[interp_range]
                interp_x = x[interp_range]

                interper = interp1d(x=interp_x, y=interp_data, assume_sorted=True, bounds_error=False)

                occ_x = interper(x)

                plotter = OneD(dat=None)

                fig = plotter.figure(xlabel='Occupation', ylabel='Arbitrary',
                                     title=f'dN/dT vs N: G={self.g:.2f}mV, '
                                           f'{THETA}={self.theta:.2f}mV, '
                                           f'{THETA}/G={self.theta / self.g:.2f}'
                                           f' -- Dat{self.datnum}')
                fig.add_trace(plotter.trace(x=occ_x, data=data_dndt, name='Data dN/dT', mode='lines+markers'))
                fig.add_trace(plotter.trace(x=occ_x, data=nrg_dndt, name='NRG dN/dT', mode='lines'))
                return fig
        return go.Figure()

    def text_params(self) -> str:
        return f'Gamma: {self.g:.4f}mV\n' \
               f'Theta: {self.theta:.3f}mV\n' \
               f'Center: {self.mid:.3f}mV\n' \
               f'Amplitude: {self.amp:.3f}nA\n' \
               f'Linear: {self.lin:.5f}nA/mV\n' \
               f'Constant: {self.const:.3f}nA\n' \
               f'Linear(Occupation): {self.occ_lin:.7f}nA/mV\n' \
               f''


OUTPUT_NAME = 'SPS.01'
OUTPUT_SETPOINT = 0.01


def _get_output(datnum, experiment_name) -> Output:
    def calculate_se_output(dat: DatHDF):
        if dat.Logs.dacs['ESC'] >= ESC_GAMMA_LIMIT:  # Gamma broadened so no centering
            logger.info(f'Dat{dat.datnum}: Calculating {OUTPUT_NAME} without centering')
            do_entropy_calc(dat.datnum, save_name=OUTPUT_NAME, setpoint_start=OUTPUT_SETPOINT, csq_mapped=False,
                            experiment_name=experiment_name,
                            center_for_avg=False)
        else:  # Not gamma broadened so needs centering
            logger.info(f'Dat{dat.datnum}: Calculating {OUTPUT_NAME} with centering')
            do_entropy_calc(dat.datnum, save_name=OUTPUT_NAME, setpoint_start=OUTPUT_SETPOINT, csq_mapped=False,
                            center_for_avg=True,
                            experiment_name=experiment_name,
                            t_func_name='i_sense')

    if datnum:
        dat = get_dat(datnum, exp2hdf=experiment_name)
        if OUTPUT_NAME not in dat.SquareEntropy.Output_names():
            with thread_lock:
                if OUTPUT_NAME not in dat.SquareEntropy.Output_names():  # check again in case a previous thread did this
                    calculate_se_output(dat)
        out = dat.SquareEntropy.get_Outputs(name=OUTPUT_NAME)
    else:
        raise RuntimeError(f'No datnum found to load data from')
    return out


def _get_x_and_data(datnum, experiment_name, which: str) -> Tuple[np.ndarray, np.ndarray]:
    if datnum:
        dat = get_dat(datnum, exp2hdf=experiment_name)
        try:
            awg = dat.Logs.awg
            se = True
        except NotFoundInHdfError:
            se = False
        if se:
            out = _get_output(datnum, experiment_name)
            x = out.x
            data = data_from_output(out, which)
        else:
            if 'i_sense' not in which:
                raise NotFoundInHdfError(f'Dat{datnum} is a Transition only dat and has no {which} data.')
            x = dat.Transition.x
            if dat.Logs.dacs['ESC'] >= ESC_GAMMA_LIMIT:
                data = np.nanmean(dat.Transition.data, axis=0)
            else:
                data = dat.Transition.avg_data
        return x, data
    else:
        raise RuntimeError(f'No datnum found to load data from')


class SliderStateCallback(CommonInputCallbacks):
    """For updating the sliders"""
    components = Components()

    def __init__(self,
                 datnum,
                 run_fit,
                 experiment_name,
                 hot_or_cold,
                 vary_theta, vary_gamma,
                 g, theta, mid, amp, lin, const, occ_lin,
                 ):
        super().__init__()
        self.run_fit = triggered_by(self.components.but_fit.id)  # Don't actually care about n_clicks from run_fit
        self.datnum_triggered = triggered_by(self.components.inp_datnum.id)

        self.experiment_name = experiment_name if experiment_name else None
        self.datnum = datnum
        self.hot_or_cold = hot_or_cold

        self.vary_theta = True if vary_theta is not None and True in vary_theta else False
        self.vary_gamma = True if vary_gamma is not None and True in vary_gamma else False
        self.mid = mid
        self.g = 10 ** g
        self.theta = theta
        self.amp = amp
        self.lin = lin
        self.const = const
        self.occ_lin = occ_lin

    @classmethod
    def get_inputs(cls) -> List[Tuple[str, str]]:
        return [
            (cls.components.inp_datnum.id, 'value'),
            (cls.components.but_fit.id, 'n_clicks'),
        ]

    @classmethod
    def get_states(cls) -> List[Tuple[str, str]]:
        return [
            (cls.components.experiment_name.id, 'value'),
            (cls.components.dd_hot_or_cold.id, 'value'),
            (cls.components.tog_vary_theta.id, 'value'),
            (cls.components.tog_vary_gamma.id, 'value'),
            *cls.components.Inputs_sliders(),
        ]

    @classmethod
    def get_outputs_all_not_value(cls) -> List[Tuple[str, str]]:
        cs = cls.components
        return list(chain(*[SliderInfo.make_callback_outputs(id_, without_value=True)
                            for id_ in [cs.slider_gamma.id,
                                        cs.slider_theta.id,
                                        cs.slider_mid.id,
                                        cs.slider_amp.id,
                                        cs.slider_lin.id,
                                        cs.slider_const.id,
                                        cs.slider_occ_lin.id]]))

    def callback_names_funcs(self) -> Dict[str, Callable]:
        return {
            "set_ranges": self.set_ranges,
            "run_fit": self.make_fit_for_store,
        }

    def set_ranges(self):
        """
        For updating slider min-max based on dat
        Note: Does NOT update values at all
        """
        slider_infos = OrderedDict({'gamma': SliderInfo(min=-2.0, max=2.5, step=0.025,
                                                        marks={int(x) if x % 1 == 0 else x: f'{10 ** x:.2f}' for x in
                                                               np.linspace(-2, 2.5, 5)}, value=None),
                                    'theta': SliderInfo(min=0.01, max=60, step=0.1, marks=None, value=None),
                                    'mid': SliderInfo(min=-200, max=40, step=0.1, marks=None, value=None),
                                    'amp': SliderInfo(min=0.05, max=1.5, step=0.01, marks=None, value=None),
                                    'lin': SliderInfo(min=0, max=0.01, step=0.00001, marks=None, value=None),
                                    'const': SliderInfo(min=0, max=10, step=0.01, marks=None, value=None),
                                    'occ_lin': SliderInfo(min=-0.003, max=0.003, step=0.0001, marks=None,
                                                          value=None)})
        if self.datnum:
            x, data = _get_x_and_data(self.datnum, self.experiment_name, 'i_sense_cold')
            # Update mid options
            mid_slider = slider_infos['mid']
            mid_slider.min = x[0]
            mid_slider.max = x[-1]
            mid_slider.set_marks()

            # Update const options
            const_slider = slider_infos['const']
            const_slider.min = np.nanmin(data)
            const_slider.max = np.nanmax(data)
            const_slider.set_marks()

        ret_tuple = tuple(chain(*[slider.to_tuple(without_value=True) for slider in slider_infos.values()]))
        return ret_tuple

    def make_fit_for_store(self) -> Optional[FitResultInfo]:
        """Runs the fit and stores in serverside Store for any other callbacks to use from there"""
        if self.run_fit and self.datnum:
            if self.hot_or_cold == 'cold':
                # data = data_from_output(out, 'i_sense_cold')
                x, data = _get_x_and_data(self.datnum, self.experiment_name, 'i_sense_cold')
            elif self.hot_or_cold == 'hot':
                # data = data_from_output(out, 'i_sense_hot')
                x, data = _get_x_and_data(self.datnum, self.experiment_name, 'i_sense_hot')
            else:
                raise PreventUpdate
            x, data = get_data_in_range(x, data, width=6000, center=self.mid)
            params = lm.Parameters()
            params.add_many(
                ('mid', self.mid, True, np.nanmin(x), np.nanmax(x), None, None),
                ('theta', self.theta, self.vary_theta, 0.5, 200, None, None),
                ('amp', self.amp, True, 0.1, 3, None, None),
                ('lin', self.lin, True, 0, 0.005, None, None),
                ('occ_lin', self.occ_lin, True, -0.0003, 0.0003, None, None),
                ('const', self.const, True, np.nanmin(data), np.nanmax(data), None, None),
                ('g', self.g, self.vary_gamma, 1/1000*self.theta, max(200, 50*self.theta), None, None),
            )
            # Note: Theta or Gamma MUST be fixed (and makes sense to fix theta usually)
            fit = calculate_fit(x, data, params=params, func=NRG_func_generator(which='i_sense'), method='powell')
            fit_return = FitResultInfo(success=fit.success, best_values=fit.best_values,
                                       reduced_chi_sq=fit.reduced_chi_sq)
            return fit_return
        return None


class FitResultCallbacks(CommonInputCallbacks):
    """For updating any callbacks which rely on the fit result store"""
    components = Components()

    def __init__(self, fit):
        super().__init__()
        self.fit: FitResultInfo = fit if isinstance(fit, FitResultInfo) else None

    @classmethod
    def get_inputs(cls) -> List[Tuple[str, str]]:
        return [(cls.components.store_fit.id, 'data')]

    @classmethod
    def get_states(cls) -> List[Tuple[str, str]]:
        pass

    @classmethod
    def get_outputs_slider_values(cls) -> List[Tuple[str, str]]:
        return cls.components.Inputs_sliders()

    def callback_names_funcs(self) -> Dict[str, Callable]:
        return {
            'update_sliders': self.update_sliders_from_fit,
            'update_redchi': self.update_redchi,
        }

    def update_sliders_from_fit(self) -> Tuple[float, float, float, float, float, float, float]:
        """
        For updating slider options based on fit values
        Note: ONLY updates values
        """
        if self.fit:
            fit = self.fit
            if not fit.success:
                logger.warning('Fit failed, doing no update')
                raise PreventUpdate
            v = fit.best_values
            return np.log10(v.g), v.theta, v.mid, v.amp, v.lin, v.const, v.occ_lin
        else:
            raise PreventUpdate

    def update_redchi(self):
        if self.fit:
            return f'{self.fit.reduced_chi_sq:.4g}'
        else:
            return f'No Fit Result'


@dataclass
class FitResultInfo:
    success: bool
    best_values: Values
    reduced_chi_sq: float


@dataclass
class SliderInfo:
    min: float = 0
    max: float = 1
    step: float = 0.1
    marks: Optional[Dict[float, str]] = None
    value: Optional[float] = 0.5

    def __post_init__(self):
        if self.marks is None:
            self.set_marks()

    def set_marks(self):
        self.marks = {int(x) if x % 1 == 0 else x: f'{x:.3f}' for x in np.linspace(self.min, self.max, 5)}

    def to_tuple(self, without_value=False) -> Tuple:
        if without_value:
            return self.min, self.max, self.step, self.marks
        else:
            return self.min, self.max, self.step, self.marks, self.value

    @classmethod
    def make_callback_outputs(cls, id_name: str, without_value=False):
        """Generates the list of properties which match the way self.to_tuple() spits out data"""
        ks = ['min', 'max', 'step', 'marks', 'value']
        if without_value:
            ks.remove('value')
        return [(id_name, k) for k in ks]


@dataclass
class ScaledData:
    scaled_data: np.ndarray
    size_ratio: float
    new_zero: float


def invert_nrg_fit_params(x: np.ndarray, data: np.ndarray, gamma, theta, mid, amp, lin, const, occ_lin,
                          data_type: str = 'i_sense') -> Tuple[np.ndarray, np.ndarray]:

    if data_type in ['i_sense', 'i_sense_cold', 'i_sense_hot']:
        # new_data = 1/(amp * (1 + occ_lin * (x - mid))) * data - lin * (x-mid) - const # - 1/2
        # new_data = 1/(amp * (1 + 0 * (x - mid))) * data - lin * (x-mid) - const # - 1/2
        new_data = (data - lin * (x - mid) - const + amp / 2) / (amp * (1 + occ_lin * (x - mid)))
    else:
        new_data = data
    # new_x = (x - mid - gamma*(-1.76567) - theta*(-1)) / gamma
    new_x = (x - mid) / gamma
    # new_x = (x - mid)
    return new_x, new_data


def scale_data(data: np.ndarray, target_min: float, target_max: float) -> ScaledData:
    """Rescales data to fall between target_min and target_max"""
    data_min, data_max = np.nanmin(data), np.nanmax(data)
    size_ratio = (target_max - target_min) / (
            data_max - data_min)  # How much to stretch new data to be same magnitude as existing
    # if abs(size_ratio - 1) >= 0.2:  # Only rescale if more than 20% different
    if abs(size_ratio) <= 0.2 or abs(size_ratio) >= 5:  # Only rescale if more than 5x different
        new_data = target_min + ((data - data_min) * size_ratio)
        new_zero = -data_min * size_ratio + target_min
    elif abs(np.mean([data_min, data_max]) - np.mean([target_min, target_max])) > \
            abs(np.mean([target_min, target_max])) * 1.5:  # Only shift, don't rescale
        new_data = target_min + (data - data_min)
        new_zero = -data_min + target_min
    else:
        new_data = data
        new_zero = 0
    return ScaledData(scaled_data=new_data, size_ratio=size_ratio, new_zero=new_zero)


def plot_nrg(which: str,
             nrg: Optional[NRGData] = None, plot=True,
             x_axis_type: str = 'energy') -> go.Figure:
    @dataclass
    class PlotInfo:
        data: np.ndarray
        title: str

    if nrg is None:
        nrg = NRGData.from_old_mat()

    x = nrg.ens
    if x_axis_type == 'energy':
        xlabel = 'Energy'
    elif x_axis_type == 'gate':
        xlabel = 'Gate /Arbitrary mV'
        x = np.flip(x)
    else:
        raise NotImplementedError
    y = nrg.ts / 0.001
    ylabel = f'{THETA}/Gamma'

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
    elif which == 'i_sense':
        pi = PlotInfo(data=1 - nrg.occupation,
                      title='NRG I_sense (1-Occ)')
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
    inst = NRGLayout(Components(pending_callbacks=PendingCallbacks()))
    inst.page_collection = page_collection
    inst.layout()  # Most callbacks are generated while running layout
    return inst.run_all_callbacks(app)


if __name__ == '__main__':
    from dash_dashboard.app import test_page

    test_page(layout=layout, callbacks=callbacks, port=8051)
