from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Any, Callable
from dataclasses import dataclass
import logging
logging.basicConfig(level=logging.DEBUG)

from dash_dashboard.base_classes import BasePageLayout, BaseMain, BaseSideBar, PageInteractiveComponents, \
    CommonInputCallbacks, PendingCallbacks
from new_dash.base_class_overrides import DatDashPageLayout, DatDashMain, DatDashSidebar
import dash_dashboard.component_defaults as c

import dash_html_components as html
from dash_extensions.enrich import MultiplexerTransform  # Dash Extensions has some super useful things!
from dash import no_update
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from src.DatObject.Make_Dat import get_dat, get_dats
from src.old_dash.DatPlotting import OneD, TwoD

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
        self.graph_1 = c.graph_area(id_name='graph-1', graph_header='Main Graph',
                                    pending_callbacks=self.pending_callbacks)



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


class GraphCallbacks(CommonInputCallbacks):
    components = Components()  # Only use this for accessing IDs only... DON'T MODIFY

    def __init__(self, datnum, se_names, e_fit_names, t_fit_names):
        super().__init__()  # Just here to shut up PyCharm
        self.datnum: int = datnum
        self.se_names: str = se_names  # SE output names
        self.e_fit_names: str = e_fit_names if e_fit_names is not None else []  #
        self.t_fit_names: str = t_fit_names if t_fit_names is not None else []  #

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

    # def _data_exists(self, which: str, avg: bool = True) -> bool:
    #     """Common check for whether the data already exists"""
    #     dat = self.dat
    #     if which == 'square_entropy':
    #         if self.se_names in dat.SquareEntropy.Output_names():
    #             return True
    #     elif which == 'entropy':
    #         if avg:
    #             if all(self.e_fit_names in dat.Entropy.fit_names:
    #                 return True
    #         else:
    #             raise NotImplementedError  # TODO: save_name plus row number or something to check
    #     else:
    #         raise KeyError(f'{which} not recognized')
    #     return False

    def entropy_signal(self) -> go.Figure:
        """dN/dT figure"""
        print('entropy_signal')
        if not self._correct_call_args():
            logger.warning(f'Bad call to entropy_signal')
            return go.Figure()
        raise ValueError
        dat = self.dat
        logger.info(f'1')
        plotter = OneD(dat=dat)
        logger.info(f'2')
        fig = plotter.figure(title=f'Dat{dat.datnum}')
        logger.info(f'3')
        out = dat.SquareEntropy.get_Outputs(name=self.se_names, existing_only=True)
        logger.info(f'4')
        x = out.x
        data = out.entropy_signal
        fig.add_trace(plotter.trace(data=data, x=x, mode='lines', name='Data'))
        logger.info(f'5')
        for n in self.e_fit_names:
            if n in dat.Entropy.fit_names:
                fit = dat.Entropy.get_fit(name=n)
                fig.add_trace(plotter.trace(data=fit.eval_fit(x=x), x=x, name=f'{n}_fit'))
        logger.info(f'Returning from entropy_signal')
        return fig


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
            self.components.graph_1,
        ])
        return lyt

    def set_callbacks(self):
        self.make_callback(outputs=(self.components.graph_1.graph_id, 'figure'),
                           inputs=GraphCallbacks.get_inputs(),
                           func=GraphCallbacks.get_callback_func('entropy_signal'),
                           states=GraphCallbacks.get_states())


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
        ])
        return lyt

    def set_callbacks(self):
        pass


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
    from new_dash.temp_copy_app import test_page
    # import dash_extensions
    # app = dash_extensions.enrich.DashProxy(__name__)
    # app.layout = layout
    # callbacks(app)
    # app.run_server(debug=True, port=8090, threaded=False)
    test_page(layout=layout, callbacks=callbacks, single_threaded=True)
    # g = GraphCallbacks(1919, None, None, None)
    # fig = g.entropy_signal()
    # fig.show(renderer='browser')