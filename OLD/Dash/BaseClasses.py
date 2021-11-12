"""
This provides some helpful classes for making layouts of pages easier. Everything in here should be fully
general to ANY Dash app, not just Dat analysis. For Dat analysis specific, implement in DatSpecificDash.
"""
from __future__ import annotations
import pandas as pd
from dash_extensions import Download
from dash_extensions.snippets import send_file
from dash_extensions.enrich import Input, Output, State  # https://pypi.org/project/dash-extensions/

import threading
from typing import Optional, List, Union, Callable, Tuple, Any
import abc
import plotly.graph_objects as go

import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import dash_table
from dash.exceptions import PreventUpdate

from OLD.Dash.app import app, ALL_PAGES

import logging

logger = logging.getLogger(__name__)


CALLBACK_TYPE = Tuple[str, str]  # All Inputs/Outputs/States/Triggers require (<id>, <target>)
# e.g. ('button-test', 'n_clicks')


def get_trig_id(ctx) -> str:
    """Pass in dash.callback_context to get the id of the trigger returned"""
    return ctx.triggered[0]['prop_id'].split('.')[0]


class EnforceSingleton:
    """Simplified from https://www.reddit.com/r/Python/comments/2qkwgh/class_to_enforce_singleton_pattern_on_subclasses/

    Enforces that subclasses are singletons (or are only instantiated once).
    """
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

    def make_callback(self, inputs: Union[List[CALLBACK_TYPE], CALLBACK_TYPE] = None,
                      outputs: Union[List[CALLBACK_TYPE], CALLBACK_TYPE] = None,
                      func: Callable = None,
                      states: Union[List[CALLBACK_TYPE], CALLBACK_TYPE] = None,
                      triggers: Union[List[CALLBACK_TYPE], CALLBACK_TYPE] = None):

        """
        Helper function for attaching callbacks more easily

        Args:
            inputs (List[CALLBACK_TYPE]): The tuples that would go into dash.dependencies.Input() (i.e. (<id>, <property>)
            outputs (List[CALLBACK_TYPE]): Similar, (<id>, <property>)
            states (List[CALLBACK_TYPE]): Similar, (<id>, <property>)
            func (Callable): The function to wrap with the callback (make sure it takes the right number of inputs in order and returns the right number of outputs in order)
            triggers (): Triggers callback but is not passed to function

        Returns:

        """

        def ensure_list(val) -> List[CALLBACK_TYPE]:
            if isinstance(val, tuple):
                return [val]
            elif val is None:
                return []
            elif isinstance(val, list):
                return val
            else:
                raise TypeError(f'{val} is not valid')

        if inputs is None and triggers is None:
            raise ValueError(f"Can't have both inputs and triggers set as None... "
                             f"\n{inputs, triggers, outputs, states}")

        inputs, outputs, states, triggers = [ensure_list(v) for v in [inputs, outputs, states, triggers]]

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
            fluid=True, className='p-0',
            children=[
                dbc.Row(
                    className='header-bar',
                    children=dbc.Col(className='p-0', children=self.top_bar_layout())
                ),
                dbc.Container(
                    fluid=True,
                    className='below-header',
                    children=[
                        dbc.Container(fluid=True, className='sidebar',
                                      children=self.side_bar_layout()
                                      ),
                        dbc.Container(fluid=True, className='content-area',
                                      children=self.main_area_layout()
                                      )
                    ])
            ])
        self.run_all_callbacks()

        return layout

    def run_all_callbacks(self):
        """
        This should be the ONLY place that all of the callbacks are run, and it is done AFTER all the layouts
        are made.
        """
        for main in self.mains:
            main[1].set_callbacks()
        self.sidebar.set_callbacks()

    def top_bar_layout(self) -> dbc.NavbarSimple:
        """
        This generally should not be changed since this is what is used to switch between whole Pages, but in case
        something else should be added to the top bar for a certain page, this can be overridden (just remember to
        include a call to super().top_bar_layout() and incorporate that into the new top_bar_layout.
        """
        layout = dbc.NavbarSimple(children=[dbc.NavItem(dbc.NavLink(k, href=v)) for k, v in ALL_PAGES.items()],
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

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Override to provide the name which will show up in the main dropdown (it will also make the id's unique)
        Note: can just set a class attribute instead of a whole property"""
        pass

    @abc.abstractmethod
    def get_sidebar(self) -> BaseSideBar:
        """Override to return THE instance of the sidebar being used to control this Main Area"""
        raise NotImplementedError

    @abc.abstractmethod
    def set_callbacks(self):
        """
        Override this to run all callbacks for main area (generally callbacks depending on things in self.sidebar)

        This method is called by the BaseLayout
        """
        raise NotImplementedError

    @property
    def id_prefix(self):
        """Something which returns an ID prefix for any ID in main area"""
        return f"{self.name}_Main"

    def name_self(self) -> Tuple[str, Any]:
        """Used in PageLayout.get_mains() and so that I can know that the name in the main dropdown is known to the
        subclass (i.e. not only written in the PageLayout.get_mains() function directly so that I can use the info
        in the common callback to prevent updating hidden views)"""
        return self.name, self

    def layout(self):
        return html.Div([
            self.graph_area(name=self.id('graph-main'))
        ])

    def graph_area(self, name: str, title: Optional[str] = None, default_fig: go.Figure = None,
                   save_option_kwargs: Optional[dict] = None):
        if save_option_kwargs is None:
            save_option_kwargs = {}
        if default_fig is None:
            default_fig = go.Figure()
        if title is None:
            title = ''
        n = dbc.CardHeader(
            dbc.Row([
                dbc.Col(html.H3(title), width='auto'),
                self.graph_save_options(self.id(name), **save_option_kwargs),
            ], justify='between')
        )

        g = dcc.Graph(id=self.id(name), figure=default_fig)
        graph = dbc.Card([
            n, g
        ])
        return graph

    def graph_callback(self, name: str, func: Callable,
                       inputs: List[Tuple[str, str]],
                       states: List[Tuple[str, str]] = None):
        self.make_callback(inputs=inputs, outputs=(self.id(name), 'figure'), func=func, states=states)

    def graph_save_options(self, graph_id, **kwargs):
        layout = dbc.Row([
            dbc.Col(self._download_button(graph_id, 'html'), width='auto'),
            dbc.Col(self._download_button(graph_id, 'jpg'), width='auto'),
            dbc.Col(self._download_button(graph_id, 'svg'), width='auto'),
            dbc.Col(self._download_name(graph_id), width='auto'),
        ], no_gutters=True)
        self._run_graph_save_callbacks(graph_id)
        return layout

    def _run_graph_save_callbacks(self, graph_id):
        self._download_callback(graph_id, 'html')
        self._download_callback(graph_id, 'jpg')
        self._download_callback(graph_id, 'svg')

    def _download_callback(self, graph_id, file_type: str):
        """https://pypi.org/project/dash-extensions/"""

        def make_file(n_clicks, fig: dict, filename: str):
            if n_clicks:
                fig = go.Figure(fig)
                if not filename:
                    filename = fig.layout.title.text
                    if not filename:
                        filename = 'DashFigure'

                fname = filename + f'.{file_type}'
                bytes_ = False
                if file_type == 'html':
                    data = fig.to_html()
                    mtype = 'text/html'
                elif file_type == 'jpg':
                    fig.write_image('temp/dash_temp.jpg', format='jpg')
                    return send_file('temp/dash_temp.jpg', filename=fname, mime_type='image/jpg')
                elif file_type == 'svg':
                    fig.write_image('temp/dash_temp.svg', format='svg')
                    return send_file('temp/dash_temp.svg', fname, 'image/svg+xml')
                else:
                    raise ValueError(f'{file_type} not supported')

                return dict(content=data, filename=fname, mimetype=mtype, byte=bytes_)
            else:
                raise PreventUpdate

        if file_type not in ['html', 'jpg', 'svg']:
            raise ValueError(f'{file_type} not supported')

        dl_id = f'{graph_id}_download-{file_type}'
        but_id = f'{graph_id}_but-{file_type}-download'
        name_id = f'{graph_id}_inp-download-name'
        app.callback(
            Output(dl_id, 'data'), Input(but_id, 'n_clicks'), State(graph_id, 'figure'), State(name_id, 'value')
        )(make_file)

    def _download_button(self, graph_id, file_type: str):
        if file_type not in ['html', 'jpg', 'svg']:
            raise ValueError(f'{file_type} not supported')
        button = [dbc.Button(f'Download {file_type.upper()}', id=f'{graph_id}_but-{file_type}-download'),
                  Download(id=f'{graph_id}_download-{file_type}')]
        return button

    def _download_name(self, graph_id):
        name = dbc.Input(id=f'{graph_id}_inp-download-name', type='text', placeholder='Download Name')
        return name

    def collapse(self, button_text: str, is_open: bool, children, id_override: Optional[str] = None) -> dbc.Collapse:
        """Simple collapse component with button to go around content in main area and a callback set to make the
        button toggle the collapse"""
        def callback(is_open: bool) -> bool:
            return not is_open

        if id_override:
            id_ = id_override
        else:
            id_ = self.id(button_text)
        obj = html.Div(children=[
            dbc.Button(id=f'but-{id_}', children=button_text),
            dbc.Collapse(id=id_, children=children, is_open=is_open),
            ])

        self.make_callback(
            outputs=(id_, 'is_open'),
            func=callback,
            triggers=(f'but-{id_}', 'n_clicks'),
            states=(id_, 'is_open')
        )
        return obj



class SidebarInputs(dict):
    """For keeping a dictionary of all components in side bar in {name: component} format"""
    pass


def sidebar_input_wrapper(*args, add_addon=True, add_label=False):
    """wraps SideBar input definitions so that any call with a new name is created and added to the SideBar input dict,
    and any call for an existing named input returns the original one"""
    import functools

    def decorator(func):

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            assert isinstance(self, BaseSideBar)
            if len(args) > 0:
                logger.error(
                    f'{args} are being passed into {func.__name__} which is wrapped with sidebar_input_wrapper. '
                    f'sidebar_input_wrapper requires all arguments to be kwargs')
            exists = False
            if 'id_name' in kwargs:
                id_name = kwargs['id_name']
                exists = True
            else:
                raise ValueError(f'No id_name found for args, kwargs: {args}, {kwargs}')

            # Either get existing, or store a new input
            if id_name not in self.inputs:
                self.inputs[id_name] = func(self, *args, **kwargs)
            ret = self.inputs[id_name]

            # If add_addon is True, then use the 'name' argument to make a prefix
            if 'name' in kwargs:
                name = kwargs['name']
                if add_addon:
                    addon = input_prefix(name)
                    ret = dbc.InputGroup([addon, ret])
                if add_label:
                    label = dbc.Label(name)
                    ret = dbc.Form([label, ret])
            elif exists is False and any([b is True for b in [add_addon, add_label]]) and 'name' not in kwargs:
                logger.error(f'add_... selected for {func.__name__} but no "name" found in kwargs: {kwargs}\n'
                             f'set "add_addon=False" for the sidebar_input_wrapper to avoid this error message')
            return ret

        return wrapper

    if len(args) == 1 and callable(args[0]):  # If not using the additional arguments
        return decorator(args[0])
    else:
        return decorator


def input_prefix(name: str):
    """For getting a nice label before inputs"""
    return dbc.InputGroupText(name)


# Usually will want to use @singleton decorator for subclasses of this
class BaseSideBar(BaseDashRequirements, EnforceSingleton):
    """
    This should be subclassed for each full page to give relevant sidebar options for each main section of the app
    (i.e. working with single dats will require different options in general than comparing multiple dats)
    """

    def __init__(self):
        super().__init__()
        self._main_dd = None  # Hold instance of the dcc dropdown for selecting different main windows. Separate to other inputs
        self.inputs = SidebarInputs()  # For storing a dictionary of all Input components in sidebar

    @property
    @abc.abstractmethod
    def id_prefix(self):
        """Something which returns an ID prefix for any ID in the sidebar"""
        return "BaseSidebar"

    def layout(self):
        """Return the full layout of sidebar to be used"""
        layout = html.Div([
            self.input_box(name='Dat', id_name=self.id('inp-datnum'), placeholder='Choose Datnum', autoFocus=True,
                           min=0)
        ])
        return layout

    @abc.abstractmethod
    def set_callbacks(self):
        """Override this to set any callbacks which update items in the sidebar"""
        pass

    def main_dropdown(self):
        """Return the dcc.Dropdown which will be used to switch between pages (i.e. for placement in sidebar etc)
        Note: It will be initialized by the BaseLayout instance (which has the list of Main pages which exist through
        a call to _make_main_callbacks
        """
        if self._main_dd is None:
            dd_id = self.id('dd-main')
            dd = dcc.Dropdown(id=dd_id, )
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
            self._main_dd.value = main_ids[0]
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
            if inp is not None and inp in outs:
                outs[inp] = False  # Set selected page to False (not hidden)
            else:
                outs[next(iter(outs))] = False  # Otherwise set first page to False (not hidden)
            if len(outs) > 1:
                return list(outs.values())
            else:
                return False  # Only one page, do not hide

        return func

    @sidebar_input_wrapper
    def input_box(self, *, name: Optional[str] = None, id_name: Optional[str] = None, val_type='number', debounce=True,
                  placeholder: str = '', persistence=True,
                  **kwargs):
        """Note: name is required for wrapper to add prefix"""
        inp = dbc.Input(id=self.id(id_name), type=val_type, placeholder=placeholder, debounce=debounce, **kwargs,
                        persistence=persistence, persistence_type='local')
        return inp

    @sidebar_input_wrapper
    def dropdown(self, *, name: Optional[str] = None, id_name: str, multi=False, placeholder='Select',
                 persistence=True):
        """Note: name is required for wrapper to add prefix"""
        if multi is False:
            dd = dbc.Select(id=self.id(id_name), placeholder=placeholder, persistence=persistence,
                            persistence_type='local')
        else:
            dd = dcc.Dropdown(id=self.id(id_name), placeholder=placeholder, style={'widht': '80%'}, multi=True,
                              persistence=persistence, persistence_type='local')
        return dd

    @sidebar_input_wrapper
    def toggle(self, *, name: Optional[str] = None, id_name: str, persistence=True):
        """Note: name is required for wrapper to add prefix"""
        tog = dbc.Checklist(id=self.id(id_name), options=[{'label': '', 'value': True}], switch=True,
                            persistence=persistence, persistence_type='local')
        return tog

    @sidebar_input_wrapper(add_addon=False)
    def slider(self, *, name: Optional[str] = None, id_name: str, updatemode='mouseup', range_type='slider',
               persistence=True):
        """
        Note: name is required for wrapper to add prefix
        Args:
            name (): Name to display in prefix box
            id_name (): Dash ID
            updatemode (): Whether updates should be processed as dragged, or on mouseup
            range_type (): If 'single', only one handle on slider, if 'range' then two handles for range
            persistence (): Whether the state of this should be held in local memory

        Returns:

        """
        if range_type == 'slider':
            slider = dcc.Slider(id=self.id(id_name), updatemode=updatemode, persistence=persistence,
                                persistence_type='local')
        elif range_type == 'range':
            slider = dcc.RangeSlider(id=self.id(id_name), updatemode=updatemode, persistence=persistence,
                                     persistence_type='local')
        else:
            raise ValueError(f'{range_type} not recognised. Should be in ["slider", "range"]')
        return slider

    @sidebar_input_wrapper
    def checklist(self, *, name: Optional[str] = None, id_name: str,
                  options: Optional[List[dict]] = None, persistence=True) -> dbc.Checklist:
        """Note: name is required for wrapper to add prefix"""
        if options is None:
            options = []
        checklist = dbc.Checklist(id=self.id(id_name), options=options, switch=False, persistence=persistence,
                                  persistence_type='local')
        return checklist

    @sidebar_input_wrapper(add_addon=False, add_label=True)
    def table(self, *, name: Optional[str] = None, id_name: str, dataframe: Optional[pd.Dataframe] = None,
              **kwargs) -> dbc.Table:
        """https://dash.plotly.com/datatable"""
        # table = dbc.Table(dataframe, id=self.id(id_name), striped=True, bordered=True, hover=True)
        if dataframe is not None:
            cols = [{'name': n, 'id': n} for n in dataframe.columns()]
            data = dataframe.to_dict('records')
        else:
            cols, data = None, None
        table = dash_table.DataTable(id=self.id(id_name), columns=cols, data=dataframe, **kwargs)
        return table

    @sidebar_input_wrapper(add_addon=False)
    def button(self, *, name: str, id_name: str, data: Optional[pd.Dataframe] = None, color='secondary') -> dbc.Button:
        button = dbc.Button(name, id=self.id(id_name), color=color)
        return button

    # Not really inputs like everything else, just using the wrapper to add to input list
    @sidebar_input_wrapper(add_addon=False)
    def div(self, *, id_name: str, **kwargs):
        """Note: This is not good for wrapping around other things, unless specifically assigning children later"""
        div = html.Div(id=self.id(id_name), **kwargs)
        return div
    # @sidebar_input_wrapper(add_addon=False)
    # def div(self, *, id_name: str, hidden: bool = False) -> html.Div:
    #     """Mostly used for hiding sections with 'hidden' attribute"""
    #     div = html.Div(id=self.id(id_name), hidden=hidden)
    #     return div


"""Generate layout of page to be used in app

Examples:
layout = BasePageLayout().layout() 
"""

if __name__ == '__main__':
    pass
