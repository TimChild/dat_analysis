from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Any, Callable
from dataclasses import dataclass
import logging

from dash_dashboard.base_classes import BasePageLayout, BaseMain, BaseSideBar, PageInteractiveComponents, \
    CommonInputCallbacks
import dash_dashboard.component_defaults as c
import dash_html_components as html
from dash_extensions.enrich import MultiplexerTransform  # Dash Extensions has some super useful things!

logger = logging.getLogger(__name__)

NAME = 'Template'
URL_ID = 'Template'
page_collection = None  # Gets set when running in multipage mode


class Components(PageInteractiveComponents):
    def __init__(self):
        super().__init__()
        self.inp_example = c.input_box(id_name='inp-exmaple', val_type='text', debounce=False,
                                       placeholder='Example Input', persistence=False)
        self.button = c.button('but-test', 'raise exception')
        self.div_button = c.div(id_name='div-button')
        self.div_example = c.div(id_name='div-example')


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
        print('in test func about to raise error')
        x = func(0)
        return self.example_value

    @classmethod
    def get_inputs(cls) -> List[Tuple[str, str]]:
        return [
            (cls.components.button.id, 'n_clicks'),
        ]

    @classmethod
    def get_states(cls) -> List[Tuple[str, str]]:
        return []


class TemplateLayout(BasePageLayout):
    top_bar_title = 'Title -- May want to override in PageLayout override'

    # Defining __init__ only for typing purposes (i.e. to specify page specific Components as type for self.components)
    def __init__(self, components: Components):
        super().__init__(page_components=components)
        self.components = components

    def get_mains(self) -> List[TemplateMain]:
        return [TemplateMain(self.components), ]

    def get_sidebar(self) -> BaseSideBar:
        return TemplateSidebar(self.components)


class TemplateMain(BaseMain):
    name = 'Template'

    # Defining __init__ only for typing purposes (i.e. to specify page specific Components as type for self.components)
    def __init__(self, components: Components):
        super().__init__(page_components=components)
        self.components = components

    def layout(self):
        lyt = html.Div([
            self.components.div_example,
        ])
        return lyt

    def set_callbacks(self):
        self.make_callback(outputs=(self.components.div_example.id, 'children'), inputs=[
            (self.components.inp_example.id, 'value'),
        ], func=lambda text: text)


class TemplateSidebar(BaseSideBar):
    id_prefix = 'TemplateSidebar'

    # Defining __init__ only for typing purposes (i.e. to specify page specific Components as type for self.components)
    def __init__(self, components: Components):
        super().__init__(page_components=components)
        self.components = components

    def layout(self):
        lyt = html.Div([
            self.components.dd_main,
            self.components.inp_example,
            self.components.button,
            self.components.div_button,
        ])
        return lyt

    def set_callbacks(self):
        self.make_callback(
            outputs=(self.components.div_button.id, 'children'),
            inputs=CommonCallback.get_inputs(),
            func=CommonCallback.get_callback_func('example')
        )
        # self.make_callback(outputs=(self.components.div_button.id, 'children'),
        #                    inputs=(self.components.button.id, 'n_clicks'),
        #                    func=lambda clicks: func)


def func(clicks):
    raise ValueError



def layout(*args):  # *args only because dash_extensions passes in the page name for some reason
    inst = TemplateLayout(Components())
    inst.page_collection = page_collection
    return inst.layout()


def callbacks(app):
    inst = TemplateLayout(Components())
    inst.page_collection = page_collection
    inst.layout()  # Most callbacks are generated while running layout
    return inst.run_all_callbacks(app)


if __name__ == '__main__':
    from dash_dashboard.app import test_page

    test_page(layout=layout, callbacks=callbacks)
