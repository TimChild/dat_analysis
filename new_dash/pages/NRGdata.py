from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass
import logging
import lmfit as lm
import threading

import numpy as np
from plotly import graph_objects as go

from dash_dashboard.base_classes import BaseSideBar, PageInteractiveComponents, \
    CommonInputCallbacks, PendingCallbacks
from new_dash.base_class_overrides import DatDashPageLayout, DatDashMain, DatDashSidebar

from dash_dashboard.util import triggered_by
import dash_dashboard.component_defaults as c
import dash_html_components as html
from dash import no_update
from dash.exceptions import PreventUpdate
import dash_core_components as dcc

from Analysis.Feb2021.NRG_comparison import NRGData, NRG_func_generator
from Analysis.Feb2021.common import do_entropy_calc, data_from_output
from src.Dash.DatPlotting import TwoD
from src.DatObject.Make_Dat import get_dat
from src.Dash.DatPlotting import OneD
from src.Characters import THETA
from src.UsefulFunctions import ensure_list
from src.AnalysisTools.fitting import calculate_fit, get_data_in_range

if TYPE_CHECKING:
    from src.DatObject.Attributes.SquareEntropy import Output
    from src.DatObject.Make_Dat import DatHDF

logger = logging.getLogger(__name__)
thread_lock = threading.Lock()

NAME = 'NRG'
URL_ID = 'NRG'
page_collection = None  # Gets set when running in multipage mode


class Components(PageInteractiveComponents):
    def __init__(self, pending_callbacks=None):
        super().__init__(pending_callbacks)

        # Graphs
        self.graph_1 = c.graph_area(id_name='graph-1', graph_header='',  # self.header_id -> children to update header
                                    pending_callbacks=self.pending_callbacks)
        self.graph_2 = c.graph_area(id_name='graph-2', graph_header='',
                                    pending_callbacks=self.pending_callbacks)
        self.graph_3 = c.graph_area(id_name='graph-3', graph_header='',
                                    pending_callbacks=self.pending_callbacks)

        # Input
        self.dd_which_nrg = c.dropdown(id_name='dd-which-nrg', multi=True, persistence=True)
        self.dd_which_x_type = c.dropdown(id_name='dd-which-x-type', multi=False, persistence=True)

        self.inp_datnum = c.input_box(id_name='inp-datnum', persistence=True)
        self.dd_hot_or_cold = c.dropdown(id_name='dd-hot-or-cold', multi=False, persistence=True)
        self.but_fit = c.button(id_name='but-fit', text='Fit to Data')

        self.slider_gamma = c.slider(id_name='sl-gamma', updatemode='drag', persistence=False)
        self.slider_theta = c.slider(id_name='sl-theta', updatemode='drag', persistence=False)
        self.slider_mid = c.slider(id_name='sl-mid', updatemode='drag', persistence=False)
        self.slider_amp = c.slider(id_name='sl-amp', updatemode='drag', persistence=False)
        self.slider_lin = c.slider(id_name='sl-lin', updatemode='drag', persistence=False)
        self.slider_const = c.slider(id_name='sl-const', updatemode='drag', persistence=False)
        self.slider_occ_lin = c.slider(id_name='sl-occ-lin', updatemode='drag', persistence=False)

        # Output
        self.text_params = dcc.Textarea(id='text-params', style={'width': '100%', 'height': '200px'})

        self.setup_initial_state()

    def setup_initial_state(self):
        self.dd_which_nrg.options = [{'label': k, 'value': k}
                                     for k in list(NRGData.__annotations__) + ['i_sense_cold', 'i_sense_hot'] if
                                     k not in ['ens', 'ts']]
        self.dd_which_nrg.value = 'occupation'
        self.dd_which_x_type.options = [{'label': k, 'value': k} for k in ['Energy', 'Gate']]
        self.dd_which_x_type.value = 'Gate'

        self.dd_hot_or_cold.options = [{'label': t, 'value': t} for t in ['cold', 'hot']]
        self.dd_hot_or_cold.value = 'cold'

        for component, setup in {self.slider_gamma: [-2.0, 2.5, 0.025,
                                                     {int(x) if x % 1 == 0 else x: f'{10 ** x:.2f}' for x in
                                                      np.linspace(-2, 2.5, 5)}, np.log10(10)],
                                 self.slider_theta: [0.01, 20, 0.1, None, 4],
                                 self.slider_mid: [-200, 40, 0.1, None, 0],
                                 self.slider_amp: [0.3, 1.5, 0.01, None, 0.75],
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


class NRGLayout(DatDashPageLayout):
    # Defining __init__ only for typing purposes (i.e. to specify page specific Components as type for self.components)
    def __init__(self, components: Components):
        super().__init__(page_components=components)
        self.components = components

    def get_mains(self) -> List[NRGMain]:
        return [NRGMain(self.components), ]

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
            self.components.graph_1,
        ])
        return lyt

    def set_callbacks(self):
        self.make_callback(outputs=(self.components.graph_2.graph_id, 'figure'),
                           inputs=NRGSliderCallback.get_inputs(),
                           states=NRGSliderCallback.get_states(),
                           func=NRGSliderCallback.get_callback_func('1d'))

        self.make_callback(outputs=(self.components.graph_3.graph_id, 'figure'),
                           inputs=NRGSliderCallback.get_inputs(),
                           states=NRGSliderCallback.get_states(),
                           func=NRGSliderCallback.get_callback_func('1d-data-changed'))

        self.make_callback(outputs=(self.components.graph_1.graph_id, 'figure'),
                           inputs=NRGSliderCallback.get_inputs(),
                           states=NRGSliderCallback.get_states(),
                           func=NRGSliderCallback.get_callback_func('2d'))


class NRGSidebar(DatDashSidebar):
    id_prefix = 'NRGSidebar'

    # Defining __init__ only for typing purposes (i.e. to specify page specific Components as type for self.components)
    def __init__(self, components: Components):
        super().__init__(page_components=components)
        self.components = components

    def layout(self):
        lyt = html.Div([
            self.components.dd_main,
            self.input_wrapper('Data Type', self.components.dd_which_nrg),
            self.input_wrapper('X axis', self.components.dd_which_x_type),
            html.Hr(),
            self.input_wrapper('Dat', self.components.inp_datnum),
            self.input_wrapper('Fit to', self.components.dd_hot_or_cold),
            self.components.but_fit,
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
        self.make_callback(outputs=(self.components.text_params.id, 'value'),
                           inputs=NRGSliderCallback.get_inputs(),
                           states=NRGSliderCallback.get_states(),
                           func=NRGSliderCallback.get_callback_func('params'))

        self.make_callback(outputs=UpdateSliderCallback.get_outputs(),
                           inputs=UpdateSliderCallback.get_inputs(),
                           states=UpdateSliderCallback.get_states(),
                           func=UpdateSliderCallback.get_callback_func('run_fit'))


class NRGSliderCallback(CommonInputCallbacks):
    components = Components()  # Only use this for accessing IDs only... DON'T MODIFY

    def __init__(self,
                 which, x_type,
                 datnum,
                 g, theta, mid, amp, lin, const, occ_lin,
                 ):
        super().__init__()  # Just here to shut up PyCharm

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
        return []

    def callback_names_funcs(self):
        """
        Return a dict of {<name>: <callback_func>}
        """
        return {
            "2d": self.two_d,
            "1d": self.one_d,
            "1d-data-changed": self.one_d_data_changed,
            "params": self.text_params,
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
        title_append = f' -- Dat{self.datnum}' if self.datnum else ''
        xlabel = 'Sweepgate /mV' if not invert_fit_on_data else 'Ens*1000'
        fig = plotter.figure(xlabel=xlabel, ylabel='Current /nA', title=f'NRG I_sense: G={self.g:.2f}mV, '
                                                                        f'{THETA}={self.theta:.2f}mV, '
                                                                        f'{THETA}/G={self.theta / self.g:.2f}'
                                                                        f'{title_append}')
        min_, max_ = 0, 1
        if self.datnum:
            out = _get_output(self.datnum)
            for i, which in enumerate(self.which):
                data = data_from_output(out, which)
                x = out.x
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
            x_for_nrg = out.x
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
                nrg_data = nrg_func(x_for_nrg, 0, self.g, self.theta, 1, 0, 0, 0)
                if which == 'i_sense':
                    nrg_data += 0.5  # 0.5 because that still gets subtracted otherwise
                x = x_for_nrg / self.g
            else:
                x = x_for_nrg
                nrg_data = nrg_func(x, self.mid, self.g, self.theta, self.amp, self.lin, self.const, self.occ_lin)
            cmin, cmax = np.nanmin(nrg_data), np.nanmax(nrg_data)
            if i == 0 and min_ == 0 and max_ == 1:
                fig.add_trace(plotter.trace(x=x, data=nrg_data, name=f'NRG {which}', mode='lines'))
                min_, max_ = cmin, cmax
            else:
                # avg_diff = ((max_+min_)-(cmax+cmin))/2  # How much to translate new data to be centered same as existing
                scaled = scale_data(nrg_data, min_, max_)
                fig.add_trace(plotter.trace(x=x, data=scaled.scaled_data, name=f'Scaled NRG {which}', mode='lines'))
                if min_ - (max_ - min_) / 10 < scaled.new_zero < max_ + (max_ - min_) / 10:
                    plotter.add_line(fig, scaled.new_zero, mode='horizontal', color='black',
                                     linetype='dot', linewidth=1)
        return fig

    def one_d_data_changed(self):
        return self.one_d(invert_fit_on_data=True)

    def text_params(self) -> str:
        return f'Gamma: {self.g:.4f}mV\n' \
               f'Theta: {self.theta:.3f}mV\n' \
               f'Center: {self.mid:.3f}mV\n' \
               f'Amplitude: {self.amp:.3f}nA\n' \
               f'Linear: {self.lin:.5f}nA/mV\n' \
               f'Constant: {self.const:.3f}nA\n' \
               f'Linear(Occupation): {self.occ_lin:.7f}nA/mV\n' \
               f''


def _get_output(datnum) -> Output:
    def calculate_se_output(dat: DatHDF):
        if dat.Logs.fds['ESC'] >= -240:  # Gamma broadened so no centering
            logger.info(f'Dat{dat.datnum}: Calculating SPS.005 without centering')
            do_entropy_calc(dat.datnum, save_name='SPS.005', setpoint_start=0.005, csq_mapped=False,
                            center_for_avg=False)
        else:  # Not gamma broadened so needs centering
            logger.info(f'Dat{dat.datnum}: Calculating SPS.005 with centering')
            do_entropy_calc(dat.datnum, save_name='SPS.005', setpoint_start=0.005, csq_mapped=False,
                            center_for_avg=True,
                            t_func_name='i_sense')

    if datnum:
        dat = get_dat(datnum)
        if 'SPS.005' not in dat.SquareEntropy.Output_names():
            with thread_lock:
                if 'SPS.005' not in dat.SquareEntropy.Output_names():  # check again in case a previous thread did this
                    calculate_se_output(dat)
        out = dat.SquareEntropy.get_Outputs(name='SPS.005')
    else:
        raise RuntimeError(f'No datnum found to load data from')
    return out


class UpdateSliderCallback(CommonInputCallbacks):
    components = Components()

    def __init__(self,
                 run_fit,
                 datnum,
                 hot_or_cold,
                 g, theta, mid, amp, lin, const, occ_lin,
                 ):
        super().__init__()
        self.run_fit = triggered_by(self.components.but_fit.id)  # Don't actually care about n_clicks from run_fit

        self.datnum = datnum
        self.hot_or_cold = hot_or_cold
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
            (cls.components.but_fit.id, 'n_clicks'),
        ]

    @classmethod
    def get_states(cls) -> List[Tuple[str, str]]:
        return [
            (cls.components.inp_datnum.id, 'value'),
            (cls.components.dd_hot_or_cold.id, 'value'),
            *cls.components.Inputs_sliders(),
        ]

    @classmethod
    def get_outputs(cls) -> List[Tuple[str, str]]:
        return cls.components.Inputs_sliders()

    def callback_names_funcs(self) -> Dict[str, Callable]:
        return {
            "run_fit": self.update_sliders_from_fit,
        }

    def update_sliders_from_fit(self) -> Tuple[float, float, float, float, float, float, float]:
        if self.run_fit and self.datnum:
            out = _get_output(self.datnum)
            if self.hot_or_cold == 'cold':
                data = data_from_output(out, 'i_sense_cold')
            elif self.hot_or_cold == 'hot':
                data = data_from_output(out, 'i_sense_hot')
            else:
                raise PreventUpdate
            x = out.x
            x, data = get_data_in_range(x, data, width=500)
            params = lm.Parameters()
            params.add_many(
                ('mid', self.mid, True, -500, 200, None, None),
                ('theta', self.theta, True, 0.5, 50, None, None),
                ('amp', self.amp, True, 0.1, 3, None, None),
                ('lin', self.lin, True, 0, 0.005, None, None),
                ('occ_lin', self.occ_lin, True, -0.0003, 0.0003, None, None),
                ('const', self.const, True, -2, 10, None, None),
                ('g', self.g, True, 0.2, 400, None, None),
            )
            # Note: Theta or Gamma MUST be fixed (and makes sense to fix theta usually)
            fit = calculate_fit(x, data, params=params, func=NRG_func_generator(which='i_sense'), method='powell')
            if not fit.success:
                logger.warning('Fit failed, doing no update')
                raise PreventUpdate
            v = fit.best_values
            return np.log10(v.g), v.theta, v.mid, v.amp, v.lin, v.const, v.occ_lin
        else:
            raise PreventUpdate


@dataclass
class ScaledData:
    scaled_data: np.ndarray
    size_ratio: float
    new_zero: float


def invert_nrg_fit_params(x: np.ndarray, data: np.ndarray, gamma, theta, mid, amp, lin, const, occ_lin,
                          data_type: str = 'i_sense'):
    if data_type in ['i_sense', 'i_sense_cold', 'i_sense_hot']:
        # new_data = 1/(amp * (1 + occ_lin * (x - mid))) * data - lin * (x-mid) - const # - 1/2
        # new_data = 1/(amp * (1 + 0 * (x - mid))) * data - lin * (x-mid) - const # - 1/2
        new_data = (data - lin * (x - mid) - const + amp / 2) / (amp * (1 + occ_lin * (x - mid)))
    else:
        new_data = data
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
        nrg = NRGData.from_mat()

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
