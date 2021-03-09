from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Any, Callable

from src.Dash.DatSpecificDash import DatDashPageLayout, DatDashMain, DatDashSideBar
from singleton_decorator import singleton
import dash_html_components as html

import src.UsefulFunctions as U
from src.DatObject.Make_Dat import get_dat, get_dats, DatHDF

import logging

logger = logging.getLogger(__name__)


class GammaLayout(DatDashPageLayout):
    id_prefix = 'Gamma'

    def get_mains(self) -> List[Tuple[str, DatDashMain]]:
        return [GammaMain().name_self(), ]

    def get_sidebar(self) -> DatDashSideBar:
        return GammaSidebar()


class GammaMain(DatDashMain):
    name = 'Gamma'

    def get_sidebar(self):
        return GammaSidebar()

    def layout(self):
        return html.Div(children=[
            self.collapse(button_text='Page Description', is_open=False, children=
                    html.P('Looking at entropy of 0->1 transition varying ESC (Entropy Sensor Coupling gate) and ESP '
                           '(Entropy Sensor Plunger gate). Hoping to see Ln2 entropy determined from both fitting and '
                           'integration in weakly coupled regime. In gamma broadened regime, the fitting technique '
                           'does not work, but the integrated entropy should continue to work.')
                          ),
            html.Hr(),
            html.H2('Transition Data'),
            self.graph_area(name='graph-trans-amplitude', title='Transition amplitude', )
        ])

    def set_callbacks(self):
        self.sidebar.layout()  # Make sure layout has been generated
        inps = self.sidebar.inputs


@singleton
class GammaSidebar(DatDashSideBar):
    id_prefix = 'GammaSidebar'

    def layout(self):
        layout = html.Div([
            self.main_dropdown(),
        ])
        return layout

    def set_callbacks(self):
        inps = self.inputs
        main = (self.main_dropdown().id, 'value')


# Generate layout for to be used in App
layout = GammaLayout().layout()

