"""
This provides some helpful classes for making layouts of pages easier.
"""
from typing import Optional, List, Dict, Union, Callable
import abc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from src.Dash.app import app


class BaseDashRequirements(abc.ABC):
    @property
    @abc.abstractmethod
    def id_prefix(self):
        """Something which returns a unique ID prefix
        Examples:
            Base
            BaseMain
            BaseSidebar
            SDMain (e.g. for SingleDat main area)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def layout(self):
        """Should return the full layout of whatever the relevant part is
        Examples:
            layout = html.Div([
                        child1,
                        child2,
                        etc,
                        ])
            return layout
        """

        raise NotImplementedError

    def id(self, id_name: str):
        """ALWAYS use this for creating any ID (i.e. this will make ID's unique between different parts of app)
        Examples:
            html.Div(id=self.id('div-picture'))  # Will actually get an id like 'SDMain_div-picture'
        """
        return f'{self.id_prefix}_{id_name}'


class BasePageLayout(BaseDashRequirements):
    """
    The overall page layout which should be used per major section of the app (i.e. looking at a single dat vs looking
    at multiple dats). Switching between whole pages will reset anything when going back to previous pages.

    For switching between similar sections where it is beneficial to move back and forth, the contents area should be
    hidden/unhidden to "switch" back and forth. This will not reset progress, and any callbacks which apply to several
    unhidden/hidden parts will be applied to all of them.
    """

    @property
    @abc.abstractmethod
    def id_prefix(self):
        """Something which returns a unique ID prefix for any id"""
        return 'Base'

    def layout(self):
        layout = dbc.Container(
            [
                dbc.Row(dbc.Col(self.top_bar_layout())),
                dbc.Row([
                    dbc.Col(self.main_area_layout(), width=10), dbc.Col(self.side_bar_layout())
                ])
            ], fluid=True
        )
        return layout

    def top_bar_layout(self):
        layout = dbc.NavbarSimple([
            dbc.NavItem(dbc.NavLink("Single Dat", href='/pages/single-dat-view')),
            dbc.NavItem(dbc.NavLink("Second Page", href='/pages/second-page')),
        ],
            brand="Tim's Dat Viewer",
        )
        return layout

    def main_area_layout(self):
        layout = html.Div(id=self.id('div-main'))
        return layout

    def side_bar_layout(self):
        return html.Div(id=self.id('div-sidebar'))


class BaseMainArea(BaseDashRequirements):
    """
    This is the area that should be hidden/unhidden for sections of app which are closely related, i.e. looking at
    different aspects of the same dat, or different ways to look at multiple dats. Everything shown in this main area
    should rely on the same sidebar

    There may be several different instances/subclasses of this for a single full page, but all of which share the same
    sidebar and hide/unhide in the same main area
    """

    @property
    @abc.abstractmethod
    def id_prefix(self):
        """Something which returns an ID prefix for any ID in main area"""
        return "BaseMain"

    def layout(self):
        return html.Div([
            self.graph_area(id=self.id('graph-main'))
        ])

    def graph_area(self, id: str, name: Optional[str] = None):
        g = dcc.Graph(id=id)
        if name:
            n = dbc.CardHeader(name)
            graph = dbc.Card([
                n, g
            ])
        else:
            graph = dbc.Card([g])
        return graph

    def graph_area_callback(self, graph_id: str, func: Callable,
                            inputs: List[Input],
                            states: Optional[List[State]] = None):
        app.callback(Output(graph_id, 'figure'), *inputs, *states)(func)


class BaseSideBar(BaseDashRequirements):
    """
    This should be subclassed for each full page to give relevant sidebar options for each main section of the app
    (i.e. working with single dats will require different options in general than comparing multiple dats)
    """

    @property
    @abc.abstractmethod
    def id_prefix(self):
        """Something which returns an ID prefix for any ID in the sidebar"""
        return "BaseSidebar"

    def layout(self):
        """Return the full layout of sidebar to be used"""
        layout = html.Div([
            self.input_box(name='Dat', id=self.id('inp-datnum'), placeholder='Choose Datnum', autoFocus=True, min=0)
        ])

    def input_box(self, name: str, id: Optional[str] = None, val_type='number', debounce=True, placeholder: str = '',
                  **kwargs):
        addon = dbc.InputGroupAddon(name, addon_type='prepend')
        inp = dbc.Input(id=id, type=val_type, bs_size='sm', placeholder=placeholder, debounce=debounce, **kwargs)
        return dbc.InputGroup([addon, inp])


