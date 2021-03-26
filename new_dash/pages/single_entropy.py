from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Any, Callable, Union
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)

from dash_dashboard.base_classes import BasePageLayout, BaseMain, BaseSideBar, PageInteractiveComponents, \
    CommonInputCallbacks, PendingCallbacks
from new_dash.base_class_overrides import DatDashPageLayout, DatDashMain, DatDashSidebar
import dash_dashboard.component_defaults as c

import dash_html_components as html
import dash_bootstrap_components as dbc
from dash_extensions.enrich import MultiplexerTransform  # Dash Extensions has some super useful things!
from dash import no_update
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from src.DatObject.Make_Dat import get_dat, get_dats
from src.Dash.DatPlotting import OneD, TwoD
import src.UsefulFunctions as U

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NAME = 'Single Entropy'
URL_ID = 'SingleEntropy'
page_collection = None  # Gets set when running in multipage mode


class Components(PageInteractiveComponents):
    def __init__(self, pending_callbacks: Optional[PendingCallbacks] = None):
        super().__init__(pending_callbacks)
        self.inp_datnum = c.input_box(id_name='inp-datnum', val_type='number', debounce=True,
                                      placeholder='Enter Datnum', persistence=True)
        self.dd_se_names = c.dropdown(id_name='dd-se-names', multi=False)
        self.dd_e_fit_names = c.dropdown(id_name='dd-e-fit-names', multi=True)
        self.dd_t_fit_names = c.dropdown(id_name='dd-t-fit-names', multi=True)
        self.dd_int_info_names = c.dropdown(id_name='dd-int-info-names', multi=True)

        # Graphs
        self.graph_1 = c.graph_area(id_name='graph-1', graph_header='dN/dT',
                                    pending_callbacks=self.pending_callbacks)
        self.graph_2 = c.graph_area(id_name='graph-2', graph_header='Transition',
                                    pending_callbacks=self.pending_callbacks)
        self.graph_3 = c.graph_area(id_name='graph-3', graph_header='Integrated',
                                    pending_callbacks=self.pending_callbacks)

        # Info Area
        self.div_info_title = c.div(id_name='div-info-title')
        self.table_1 = c.table(id_name='tab-efit', dataframe=None)
        self.table_2 = c.table(id_name='tab-tfit', dataframe=None)
        self.table_3 = c.table(id_name='tab-int_info', dataframe=None)



# A reminder that this is helpful for making many callbacks which have similar inputs
class CommonCallbackExample(CommonInputCallbacks):
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
            "example": self.example_func(),
        }

    def example_func(self):
        """Part of example, can be deleted"""
        return self.example_value

    @classmethod
    def get_inputs(cls) -> List[Tuple[str, str]]:
        return [
        ]

    @classmethod
    def get_states(cls) -> List[Tuple[str, str]]:
        return []


class SingleEntropyLayout(DatDashPageLayout):

    # Defining __init__ only for typing purposes (i.e. to specify page specific Components as type for self.components)
    def __init__(self, components: Components):
        super().__init__(page_components=components)
        self.components = components

    def get_mains(self) -> List[SingleEntropyMain]:
        return [SingleEntropyMain(self.components), ]

    def get_sidebar(self) -> DatDashSidebar:
        return SingleEntropySidebar(self.components)


class SingleEntropyMain(DatDashMain):
    name = 'SingleEntropy'

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
               ], width=9),
               dbc.Col([
                   self.components.div_info_title,
                   html.H6('Entropy Fit'),
                   self.components.table_1,
                   html.Hr(),
                   html.H6('Transition Fit'),
                   self.components.table_2,
                   html.Hr(),
                   html.H6('Integrated Info'),
                   self.components.table_3,
                   html.Hr(),

               ], width=3)
            ])
        ])
        return lyt

    def set_callbacks(self):
        components = self.components
        # Graph Callbacks
        self.make_callback(outputs=(self.components.graph_1.graph_id, 'figure'),
                           inputs=GraphCallbacks.get_inputs(),
                           func=GraphCallbacks.get_callback_func('entropy_signal'),
                           states=GraphCallbacks.get_states())

        # Table Callbacks
        self.make_callback(outputs=(self.components.div_info_title.id, 'children'),
                           inputs=(self.components.inp_datnum.id, 'value'),
                           func=lambda datnum: html.H5(f'Dat{datnum}: Fit Info') if datnum is not None else 'Invalid Datnum')

        for table, cb_func in {components.table_1: 'entropy_table',
                               components.table_2: 'transition_table',
                               components.table_3: 'integrated_table'}.items():
            self.make_callback(outputs=TableCallbacks.get_outputs(table),
                               inputs=TableCallbacks.get_inputs(),
                               states=TableCallbacks.get_states(),
                               func=TableCallbacks.get_callback_func(cb_func))


class SingleEntropySidebar(DatDashSidebar):
    id_prefix = 'SingleEntropySidebar'

    # Defining __init__ only for typing purposes (i.e. to specify page specific Components as type for self.components)
    def __init__(self, components: Components):
        super().__init__(page_components=components)
        self.components = components

    def layout(self):
        lyt = html.Div([
            self.components.dd_main,
            self.input_wrapper('Datnum', self.components.inp_datnum),
            self.input_wrapper('SE Output', self.components.dd_se_names),
            self.input_wrapper('E fits', self.components.dd_e_fit_names),
            self.input_wrapper('T fits', self.components.dd_t_fit_names),
            self.input_wrapper('Int sf', self.components.dd_int_info_names),
        ])
        return lyt

    def set_callbacks(self):
        components = self.components

        # Set Options specific to Dat
        for k, v in {components.dd_se_names: 'se outputs',
                     components.dd_e_fit_names: 'entropy fits',
                     components.dd_t_fit_names: 'transition fits',
                     components.dd_int_info_names: 'integrated fits'}.items():
            self.make_callback(outputs=[(k.id, 'options'), (k.id, 'value')],
                               inputs=DatOptionsCallbacks.get_inputs(),
                               states=DatOptionsCallbacks.get_states(),
                               func=DatOptionsCallbacks.get_callback_func(v))


# Callback functions
class GraphCallbacks(CommonInputCallbacks):
    components = Components()  # Only use this for accessing IDs only... DON'T MODIFY

    def __init__(self, datnum, se_name, e_fit_names, t_fit_names):
        super().__init__()  # Just here to shut up PyCharm
        self.datnum: int = datnum
        self.se_name: str = se_name  # SE output names
        self.e_fit_names: List[str] = listify_dash_input(e_fit_names)
        self.t_fit_names: List[str] = listify_dash_input(t_fit_names)
        self.dat = get_dat(self.datnum) if self.datnum is not None else None

    @classmethod
    def get_inputs(cls) -> List[Tuple[str, str]]:
        return [
            (cls.components.inp_datnum.id, 'value'),
            (cls.components.dd_se_names.id, 'value'),
            (cls.components.dd_e_fit_names.id, 'value'),
            (cls.components.dd_t_fit_names.id, 'value'),
        ]

    @classmethod
    def get_states(cls) -> List[Tuple[str, str]]:
        return []

    def callback_names_funcs(self):
        """
        Return a dict of {<name>: <callback_func>}
        """
        return {
            "entropy_signal": self.entropy_signal(),
        }

    def _correct_call_args(self) -> bool:
        """Common check for bad call args which shouldn't be used for plotting"""
        if any([self.dat is None]):
            return False
        return True

    def entropy_signal(self) -> go.Figure:
        """dN/dT figure"""
        if not self._correct_call_args():
            logger.warning(f'Bad call args to GraphCallback')
            return go.Figure()
        dat = self.dat
        plotter = OneD(dat=dat)
        fig = plotter.figure(title=f'Dat{dat.datnum}')
        out = dat.SquareEntropy.get_Outputs(name=self.se_name, check_exists=True)
        x = out.x
        data = out.average_entropy_signal
        fig.add_trace(plotter.trace(data=data, x=x, mode='lines', name='Data'))
        for n in self.e_fit_names:
            if n in dat.Entropy.fit_names:
                fit = dat.Entropy.get_fit(name=n)
                fig.add_trace(plotter.trace(data=fit.eval_fit(x=x), x=x, name=f'{n}_fit', mode='lines'))
        return fig


class DatOptionsCallbacks(CommonInputCallbacks):
    """Common callback to fill in options for dats"""
    components = Components()

    def __init__(self, datnum: int, se_name, e_names, t_names, int_names):
        super().__init__()  # Shutting up PyCharm
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
        return [
            (cls.components.dd_se_names.id, 'value'),
            (cls.components.dd_e_fit_names.id, 'value'),
            (cls.components.dd_t_fit_names.id, 'value'),
            (cls.components.dd_int_info_names.id, 'value'),
        ]

    def callback_names_funcs(self) -> dict:
        return {
            'se outputs': self.se_outputs(),
            'entropy fits': self.entropy(),
            'transition fits': self.transition(),
            'integrated fits': self.integrated(),
        }

    @staticmethod
    def _val(new_opts: List[str], current: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(current, str):
            current = [current]

        values = []
        if new_opts is not None and current is not None:
            for x in current:
                if x in new_opts:
                    values.append(x)
        if len(values) == 1:
            values = values[0]  # return str for only one value selected to keep in line with how dash does things
        elif len(values) == 0:
            if len(new_opts) > 0:
                values = new_opts[0]
            else:
                values = ''
        return values

    def se_outputs(self) -> Tuple[List[Dict[str, str]], str]:
        """Options for SE_output dropdown"""
        if self.dat is None:
            return [], ''
        opts = self.dat.SquareEntropy.Output_names()
        val = self._val(opts, self.se_name)
        return self._list_to_options(opts), val

    def entropy(self) -> Tuple[List[Dict[str, str]], str]:
        """Options for E fits dropdown"""
        if self.dat is None:
            return [], ''
        opts = self.dat.Entropy.fit_names
        val = self._val(opts, self.e_names)
        return self._list_to_options(opts), val

    def transition(self) -> Tuple[List[Dict[str, str]], str]:
        """Options for T fits dropdown"""
        if self.dat is None:
            return [], ''
        opts = self.dat.SquareEntropy.get_fit_names(which='transition')
        val = self._val(opts, self.t_names)
        return self._list_to_options(opts), val

    def integrated(self) -> Tuple[List[Dict[str, str]], str]:
        """Options for Int info dropdown"""
        if self.dat is None:
            return [], ''
        opts = self.dat.Entropy.get_integration_info_names()
        val = self._val(opts, self.int_names)
        return self._list_to_options(opts), val

    @staticmethod
    def _list_to_options(opts_list: List[str]) -> List[Dict[str, str]]:
        return [{'label': k, 'value': k} for k in opts_list]


class TableCallbacks(CommonInputCallbacks):
    components = Components()  # For ID's only

    def __init__(self, se_name, e_names, t_names, int_names, datnum):
        super().__init__()  # Shutting up PyCharm
        self.se_name: str = se_name
        self.e_names: List[str] = listify_dash_input(e_names)
        self.t_names: List[str] = listify_dash_input(t_names)
        self.int_names: List[str] = listify_dash_input(int_names)
        self.datnum: Optional[int] = datnum

        # Generated
        self.dat = get_dat(datnum) if self.datnum is not None else None

    @staticmethod
    def get_outputs(table: dbc.Table) -> List[Tuple[str, str]]:
        """Columns and Data for Table callbacks"""
        return [(table.id, 'columns'), (table.id, 'data')]

    @classmethod
    def get_inputs(cls) -> List[Tuple[str, str]]:
        return [
            (cls.components.dd_se_names.id, 'value'),
            (cls.components.dd_e_fit_names.id, 'value'),
            (cls.components.dd_t_fit_names.id, 'value'),
            (cls.components.dd_int_info_names.id, 'value'),
        ]

    @classmethod
    def get_states(cls) -> List[Tuple[str, str]]:
        return [
            (cls.components.inp_datnum.id, 'value'),
        ]

    def callback_names_funcs(self):
        return {
            'entropy_table': self.entropy_table(),
            'transition_table': self.transition_table(),
            'integrated_table': self.integration_table(),
        }

    def _valid_call(self) -> bool:
        """Common check for bad call args which shouldn't be used for plotting"""
        if any([self.dat is None]):
            return False
        return True

    @staticmethod
    def _df_to_table_props(df: pd.DataFrame) -> Tuple[List[Dict[str, str]], List[dict]]:
        df.insert(0, 'Name', df.pop('name'))
        df = df.applymap(lambda x: f'{x:.3g}' if isinstance(x, (float, np.float)) else x)
        return [{'name': col, 'id': col} for col in df.columns], df.to_dict('records')

    def _get_fit_table(self, existing_names: List[str], requested_names: List[str], fit_getter: Callable):
        dfs = []
        for name in requested_names:
            if name in existing_names:
                fit = fit_getter(name)
                df = fit.to_df()
                df['name'] = name
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
                                   fit_getter=lambda fit_name: self.dat.Entropy.get_fit(name=fit_name))

    def transition_table(self):
        """Table of fit values for Transition fits"""
        if not self._valid_call():
            return [], []
        return self._get_fit_table(existing_names=self.dat.SquareEntropy.get_fit_names(which='transition'),
                                   requested_names=self.t_names,
                                   fit_getter=lambda fit_name: self.dat.SquareEntropy.get_fit(fit_name=fit_name,
                                                                                              which_fit='transition'))

    def integration_table(self):
        """Table of fit values for Integrated Infos"""
        if not self._valid_call():
            return [], []
        return self._get_fit_table(existing_names=self.dat.Entropy.get_integration_info_names(),
                                   requested_names=self.int_names,
                                   fit_getter=lambda fit_name: self.dat.Entropy.get_integration_info(name=fit_name))


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


# Required for multipage
def layout(*args):  # *args only because dash_extensions passes in the page name for some reason
    inst = SingleEntropyLayout(Components())
    inst.page_collection = page_collection
    return inst.layout()


def callbacks(app):
    inst = SingleEntropyLayout(Components(pending_callbacks=PendingCallbacks()))
    inst.page_collection = page_collection
    inst.layout()  # Most callbacks are generated while running layout
    return inst.run_all_callbacks(app)


if __name__ == '__main__':
    from dash_dashboard.app import test_page

    test_page(layout=layout, callbacks=callbacks, single_threaded=False, port=8050)
