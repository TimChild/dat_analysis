"""
This provides some helpful classes for making layouts of pages easier.
"""
from __future__ import annotations
import threading
from typing import Optional, List, Dict, Union, Callable, Tuple
from collections import OrderedDict
import abc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from src.Dash.app import app
import plotly.graph_objects as go
from src.Dash.app import app, ALL_PAGES


class EnforceSingleton:
    singletonExists = False
    lock = threading.Lock()

    def __init__(self):
        with EnforceSingleton.lock:
            if self.__class__.singletonExists:
                raise Exception(f'Instance already exists for {self.__class__.__name__}. There should only be 1 '
                                f'instance. Consider adding @singleton decorator to the subclass')
            self.__class__.singletonExists = True


class BaseDashRequirements(abc.ABC):
    """
    Things that are useful or necessary for all of my Dash components
    """

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

    def make_callback(self, inputs: Union[List[Tuple[str, str]], Tuple[str, str]],
                      outputs: Union[List[Tuple[str, str]], Tuple[str, str]],
                      func: Callable,
                      states: Union[List[Tuple[str, str]], Tuple[str, str]] = None):
        """
        Helper function for attaching callbacks more easily

        Args:
            inputs (List[Tuple[str, str]]): The tuples that would go into dash.dependencies.Input() (i.e. (<id>, <property>)
            outputs (List[Tuple[str, str]]): Similar, (<id>, <property>)
            states (List[Tuple[str, str]]): Similar, (<id>, <property>)
            func (Callable): The function to wrap with the callback (make sure it takes the right number of inputs in order and returns the right number of outputs in order)

        Returns:

        """
        if isinstance(inputs, tuple):
            inputs = [inputs]
        if isinstance(outputs, tuple):
            outputs = [outputs]
        if states is None:
            states = []
        Inputs = [Input(*inp) for inp in inputs]
        Outputs = [Output(*out) for out in outputs]
        States = [State(*s) for s in states]
        app.callback(*Outputs, *Inputs, *States)(func)  # Makes callback here


class BasePageLayout(BaseDashRequirements, EnforceSingleton):
    """
    The overall page layout which should be used per major section of the app (i.e. looking at a single dat vs looking
    at multiple dats). Switching between whole pages will reset anything when going back to previous pages.

    For switching between similar sections where it is beneficial to move back and forth, the contents area should be
    hidden/unhidden to "switch" back and forth. This will not reset progress, and any callbacks which apply to several
    unhidden/hidden parts will be applied to all of them.
    """

    def __init__(self):
        super().__init__()
        self.mains = self.get_mains()
        self.sidebar = self.get_sidebar()

    @abc.abstractmethod
    def get_mains(self) -> List[Tuple[str, BaseMain]]:
        """
        Override to return list of BaseMain areas to use in Page
        Examples:
            return [('Page1', TestMain1()), ('Page2', TestMain2())]  # where TestMains are subclasses of BaseMain
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_sidebar(self) -> BaseSideBar:
        """
        Override to return the SideBar which will be used for the Page
        Examples:
            return TestSideBar()  # where TestSideBar is a subclass of SideBar
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def id_prefix(self):
        """Something which returns a unique ID prefix for any id"""
        return 'Base'

    def layout(self):
        """
        Overall layout of main Pages, generally this should not need to be changed, but more complex pages can be
        created by subclassing this. Just don't forget to still include a top_bar_layout which has the links to all
        pages.
        """
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
        """
        This generally should not be changed since this is what is used to switch between whole Pages, but in case
        something else should be added to the top bar for a certain page, this can be overridden (just remember to
        include a call to super().top_bar_layout() and incorporate that into the new top_bar_layout.
        """
        layout = dbc.NavbarSimple([dbc.NavItem(dbc.NavLink(k, href=v)) for k, v in ALL_PAGES.items()],
                                  brand="Tim's Dat Viewer",
                                  )
        return layout

    def main_area_layout(self):
        """
        Makes the main area layout based on self.get_mains()
        """

        layout = html.Div([html.Div(v.layout(), id=self.id(k)) for k, v in self.mains])

        return layout

    def side_bar_layout(self):
        """
        Override this to return a layout from BaseSideBar
        Examples:
            return BaseSideBar().layout()
        """
        # Have to run this callback here because info about Main pages is known by Layout but not SideBar
        self.sidebar._make_main_callbacks(labels=[k for k, _ in self.mains],
                                          main_ids=[self.id(k) for k, _ in self.mains])
        return self.sidebar.layout()


class BaseMain(BaseDashRequirements):
    """
    This is the area that should be hidden/unhidden for sections of app which are closely related, i.e. looking at
    different aspects of the same dat, or different ways to look at multiple dats. Everything shown in this main area
    should rely on the same sidebar

    There may be several different instances/subclasses of this for a single full page, but all of which share the same
    sidebar and hide/unhide in the same main area
    """

    def __init__(self):
        self.sidebar = self.get_sidebar()  # So that Objects in main can easily make callbacks with SideBar

    @abc.abstractmethod
    def get_sidebar(self) -> BaseSideBar:
        """Override to return THE instance of the sidebar being used to control this Main Area"""
        raise NotImplementedError

    @abc.abstractmethod
    def set_callbacks(self):
        """
        Override this to run all callbacks for main area (generally callbacks depending on things in self.sidebar)
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def id_prefix(self):
        """Something which returns an ID prefix for any ID in main area"""
        return "BaseMain"

    def layout(self):
        return html.Div([
            self.graph_area(name=self.id('graph-main'))
        ])

    def graph_area(self, name: str, title: Optional[str] = None, default_fig: go.Figure = None):
        if default_fig is None:
            default_fig = go.Figure()
        g = dcc.Graph(id=self.id(name), figure=default_fig)
        if title:
            n = dbc.CardHeader(title)
            graph = dbc.Card([
                n, g
            ])
        else:
            graph = dbc.Card([g])
        return graph

    def graph_callback(self, name: str, func: Callable,
                       inputs: List[Tuple[str, str]],
                       states: List[Tuple[str, str]] = None):
        self.make_callback(inputs=inputs, outputs=(self.id(name), 'figure'), func=func, states=states)


# Usually will want to use @singleton decorator for subclasses of this
class BaseSideBar(BaseDashRequirements, EnforceSingleton):
    """
    This should be subclassed for each full page to give relevant sidebar options for each main section of the app
    (i.e. working with single dats will require different options in general than comparing multiple dats)
    """

    def __init__(self):
        super().__init__()
        self._main_dd = None  # Hold instance of the dcc dropdown for selecting different main windows.

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
        return layout

    def input_box(self, name: str, id: Optional[str] = None, val_type='number', debounce=True, placeholder: str = '',
                  **kwargs):
        addon = dbc.InputGroupAddon(name, addon_type='prepend')
        inp = dbc.Input(id=id, type=val_type, placeholder=placeholder, debounce=debounce, **kwargs)
        return dbc.InputGroup([addon, inp])

    def main_dropdown(self):
        """Return the dcc.Dropdown which will be used to switch between pages (i.e. for placement in sidebar etc)
        Note: It will be initialized by the BaseLayout instance (which has the list of Main pages which exist through
        a call to _make_main_callbacks
        """
        if self._main_dd is None:
            dd_id = self.id('dd-main')
            dd = dcc.Dropdown(id=dd_id)
            self._main_dd = dd
        return self._main_dd

    def _make_main_callbacks(self, labels: List[str], main_ids: List[str]):
        """
        Called by BaseLayout in order to set up the main_dropdown callbacks. This should not be called by the user.
        Only place self.main_dropdown() wherever you want in the sidebar (or wherever), and the rest will be taken
        care of.
        Args:
            labels (): Names to show in dropdown (determined by the names given in BaseLayout
            main_ids (): id's of the Main layout Divs (made based on labels using BaseLayout.id
        Returns:
            None: just sets up the necessary callback and adds options to the main_dropdown()
        """
        self.layout()  # Make sure layout has been run at least once to initialize self._main_dd

        if self._main_dd is not None:
            self._main_dd.options = [{'label': label, 'value': m_id} for label, m_id in zip(labels, main_ids)]
            outputs = [(m_id, 'hidden') for m_id in main_ids]
            inp = (self._main_dd.id, 'value')
            func = self._main_dd_callback_func()
            self.make_callback(inputs=[inp], outputs=outputs, func=func)
        elif len(main_ids) > 1:
            raise RuntimeError(f'There is more than one Main in the Page layout, '
                               f'but no main_dropdown in the SideBar in order to switch between the Mains!')
        else:
            pass  # If only 1 main, no need for a sidebar or a callback, the one main will be shown by default

    def _main_dd_callback_func(self) -> Callable:
        """Should not be called by user. Should only be called after self._main_dd.options have been set"""
        opts = [d['value'] for d in self._main_dd.options]
        def func(inp):
            outs = {k: True for k in opts}
            if inp is not None:
                if inp in outs:
                    outs[inp] = False  # Set selected page to False (not hidden)
                    return list(outs.values())
            outs[next(iter(outs))] = False  # Otherwise set first page to False (not hidden)
            return list(outs.values())

        return func


"""Generate layout of page to be used in app

Examples:
layout = BasePageLayout().layout() 
"""
