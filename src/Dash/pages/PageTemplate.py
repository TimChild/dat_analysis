from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Any, Callable

from src.Dash.DatSpecificDash import DatDashPageLayout, DatDashMain, DatDashSideBar
from singleton_decorator import singleton
import dash_html_components as html
from dash_extensions.enrich import Trigger  # Dash Extensions has some super useful things!

import src.useful_functions as U
from src.dat_object.make_dat import get_dat, get_dats, DatHDF


import logging

logger = logging.getLogger(__name__)


class TemplateLayout(DatDashPageLayout):
    id_prefix = 'Template'

    def get_mains(self) -> List[Tuple[str, DatDashMain]]:
        return [TemplateMain().name_self(), ]

    def get_sidebar(self) -> DatDashSideBar:
        return TemplateSidebar()


class TemplateMain(DatDashMain):
    name = 'Template'

    def get_sidebar(self):
        return TemplateSidebar()

    def layout(self):
        return html.Div()

    def set_callbacks(self):
        self.sidebar.layout()  # Make sure layout has been generated
        inps = self.sidebar.inputs


@singleton
class TemplateSidebar(DatDashSideBar):
    id_prefix = 'TemplateSidebar'

    def layout(self):
        layout = html.Div([
            self.main_dropdown(),
        ])
        return layout

    def set_callbacks(self):
        inps = self.inputs
        main = (self.main_dropdown().id, 'value')


# Generate layout for to be used in App
layout = TemplateLayout().layout()

