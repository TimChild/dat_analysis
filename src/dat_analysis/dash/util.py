from __future__ import annotations
from jupyter_dash import JupyterDash
import re
from dictor import dictor
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash import Input, Output, State, MATCH, ALL, ALLSMALLER
from dash_extensions.enrich import DashProxy, ServersideOutput, ServersideOutputTransform, ServerStore
from dash_extensions.snippets import get_triggered
from dash import dash_table
import slugify
import plotly.colors as pc
import uuid
import socket
import os
import sys
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Any

from dat_analysis.useful_functions import fig_to_data_json, fig_to_igor_itx, ensure_list
from dat_analysis.plotting.plotly.util import fig_waterfall, limit_max_datasize
from ..core_util import sig_fig


TEMPDIR = '_tempdir'
os.makedirs(TEMPDIR, exist_ok=True)


class MyDash(JupyterDash, DashProxy):
    """Allow use of dash-extensions while maintaining the jupyter-dash behavior

    When asking for a method on this class it will look in JupyterDash first, then DashProxy if it didn't find it in JupyterDash.
    """
    pass


def make_app() -> MyDash:
    """Make a new instance of the dash app (then add layout and callbacks)"""
    app = MyDash(__name__, transforms=[ServersideOutputTransform()], external_stylesheets=[dbc.themes.BOOTSTRAP], )
    return app


def make_layout_section(
        *,
        stores: list[dcc.Store] = None,
        inputs: list[InputComponent] = None,
        outputs: list[OutputComponent] = None,
):
    """Make a reasonable layout for a general set of inputs and outputs"""
    stores, inputs, outputs = [[] if v is None else v for v in [stores, inputs, outputs]]
    input_layout = make_input_layout(inputs)
    output_layout = make_output_layout(outputs)
    # return output_layout
    layout = dbc.Container(
        [
            dbc.Row([dbc.Col([input_layout, *stores])]),
            html.Hr(),
            dbc.Row([dbc.Col(output_layout)]),
            html.Hr(),
        ],
        fluid=True,
    )
    return layout


_NOT_SET = object()


def ensure_float(v, default=_NOT_SET):
    """Ensure value is a float (otherwise returns default if set)"""
    try:
        v = float(v)
    except ValueError as e:
        if default is not _NOT_SET:
            v = default
        else:
            raise e
    return v


def get_unused_port() -> int:
    """Find an open socket/port E.g. for running Dash app
    https://unix.stackexchange.com/questions/55913/whats-the-easiest-way-to-find-an-unused-local-port
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    addr = s.getsockname()
    s.close()
    return addr[1]


def add_label(component, label: str):
    """Combine a label with a component when placing in layout"""
    return html.Div([label, component])


class ComponentMixin:
    def __init__(self, id: str = None, *args, **kwargs):
        super().__init__(id=id, *args, **kwargs)

    def default_input_property(self):
        """Override this to provide a default input property for a specific component type"""
        raise NotImplementedError(f"No default input propery set for {type(self)}")

    def default_output_property(self):
        """Override this to provide a default output property for a specific component type"""
        raise NotImplementedError(f"No default output propery set for {type(self)}")

    def as_input(self, input_property=None, state=False):
        input_property = (
            input_property
            if input_property is not None
            else self.default_input_property()
        )
        if state:
            return State(self.id, input_property)
        else:
            return Input(self.id, input_property)

    def as_state(self, input_property=None):
        return self.as_input(input_property=input_property, state=True)

    def as_output(self, output_property=None, serverside=False):
        output_property = (
            output_property
            if output_property is not None
            else self.default_output_property()
        )
        if serverside:
            # TODO: Can I subclass ServersideOutput in a way to that allows me to store non-picklable objects?
            return ServersideOutput(self.id, output_property)
        else:
            return Output(self.id, output_property)

    def layout(self):
        """Override to change how this component is added to the layout"""
        # Returning self is equivalent to just putting the component directly in the layout
        return self


class InputComponent(ComponentMixin):
    # Note: Not using """ ... """ to avoid overriding docstrings
    # Add more functionality specific to input components
    # E.g. dbc.Input, dcc.Dropdown, dcc.Slider, etc
    def __init__(self, id, *args, display_name: str = None, **kwargs):
        super().__init__(id, *args, **kwargs)
        self.display_name = display_name if display_name is not None else id

    def layout(self):
        return add_label(self, self.display_name)


class OutputComponent(ComponentMixin):
    # Note: Not using """ ... """ to avoid overriding docstrings
    # Add more functionality specific to output components
    # E.g. dcc.Graph, dcc.Table, html.Div
    pass


class Components:
    """For now collect together my components in a class... Later can switch them to their own file if it makes sense"""

    class Input(InputComponent, dbc.Input):
        def default_output_property(self):
            return "value"

        def default_input_property(self):
            return "value"

    class Dropdown(InputComponent, dcc.Dropdown):
        def __init__(self, id: str, options=None, value=None, display_name=None, **kwargs):
            super().__init__(id=id, options=options, value=value, display_name=display_name, **kwargs)

        def default_input_property(self):
            return "value"

    class Slider(InputComponent, dcc.Slider):
        def default_input_property(self):
            return "value"

    class RangeSlider(InputComponent, dcc.RangeSlider):
        def default_input_property(self):
            return "value"

    class Button(InputComponent, dbc.Button):
        def __init__(self, id: str, *args, button_text: str = None, display_name: str = None, **kwargs):
            button_text = button_text if button_text is not None else id
            children = kwargs.pop('children', button_text)  # only replace if not already passed
            super().__init__(id, *args, children=children, display_name=display_name, **kwargs)

        def default_input_property(self):
            return 'n_clicks'

    class Interval(InputComponent, dcc.Interval):
        def default_input_property(self):
            return 'n_intervals'

        def layout(self):
            return self

    class RadioItems(InputComponent, dbc.RadioItems):
        def default_input_property(self):
            return 'value'

    class Collapse(OutputComponent, html.Div):
        """
        A collapsable div that handles the Callbacks for collapsing/expanding

        # Requires
        None - self contained (just put content in)

        # Provides
        None - self contained.

        """

        # Functions to create pattern-matching callbacks of the subcomponents
        class ids:
            @staticmethod
            def generic(aio_id, key: str):
                return {
                    'component': 'CollapseAIO',
                    'subcomponent': f'generic',
                    'key': key,
                    'aio_id': aio_id,
                }

        # Make the ids class a public class
        ids = ids

        def __init__(self, aio_id=None, content: html.Div = None, button_text: str = 'Expand', start_open=False):
            if aio_id is None:
                aio_id = str(uuid.uuid4())

            layout = [
                dbc.Button(button_text, id=self.ids.generic(aio_id, 'expand-button')),
                collapse := dbc.Collapse(children=content, id=self.ids.generic(aio_id, 'collapse'), is_open=start_open),
            ]
            self.collapse = collapse
            super().__init__(children=layout)  # html.Div contains layout

        @classmethod
        def run_callbacks(cls, app):
            """Run the callbacks if they don't already exist in the app"""
            if app is None:
                return False
            if getattr(app, '_CollapseCallbacksMade', False) is False:
                app.callback(
                    Output(cls.ids.generic(MATCH, 'collapse'), 'is_open'),
                    Input(cls.ids.generic(MATCH, 'expand-button'), 'n_clicks'),
                    State(cls.ids.generic(MATCH, 'collapse'), 'is_open'),
                )

                def toggle_collapse(clicks, is_open: bool):
                    if clicks:
                        return not is_open
                    return is_open

                app._CollapseCallbacksMade = True

        # @staticmethod
        # @callback(
        #     Output(ids.generic(MATCH, 'collapse'), 'is_open'),
        #     Input(ids.generic(MATCH, 'expand-button'), 'n_clicks'),
        #     State(ids.generic(MATCH, 'collapse'), 'is_open'),
        # )
        # def toggle_collapse(clicks, is_open: bool):
        #     if clicks:
        #         return not is_open
        #     return is_open

    class DataTable(InputComponent, dash_table.DataTable):
        @staticmethod
        def setup_df(df: pd.DataFrame) -> pd.DataFrame:
            """Setup df to show properly in dash_table"""
            df = df.copy()
            if 'id' not in df.columns:
                df['id'] = df.index
            df = df.reset_index()
            return df

        @classmethod
        def df_to_data(cls, df: pd.DataFrame) -> dict:
            """Convert df into an update DataTable understands
            Note: This is for the 'data' ONLY, not the 'columns'
            """
            df = cls.setup_df(df)
            return df.to_dict('records')

        @classmethod
        def df_to_columns(cls, df: pd.DataFrame) -> list[dict]:
            """Convert df columns into an update DataTable understands"""
            df = cls.setup_df(df)
            return cls.setup_columns(df, df.columns)

        @staticmethod
        def setup_columns(df: pd.DataFrame, columns: list[str]) -> list[dict]:
            """Setup columns to show properly in dash_table"""
            column_dicts = []
            for col in columns:
                if col == 'id':
                    continue
                dtype = df[col].dtype
                column = {
                    "name": str(col), "id": str(col),
                    "deletable": False, "selectable": False,
                    "type": table_type(df[col])
                }
                if np.issubdtype(dtype, np.number):
                    if np.issubdtype(dtype, np.integer):
                        # column["format"] = {'specifier': ''}
                        pass
                    else:
                        column["format"] = {'specifier': '.4g'}
                else:
                    pass
                column_dicts.append(column)
            return column_dicts

        def __init__(self, id: str, df: pd.DataFrame, display_name=None,
                     enable_filtering=False,
                     enable_sorting=True,
                     enable_row_selection=False,
                     **kwargs):
            super().__init__(
                id=id,
                display_name=display_name,  # Note: InputComponent parameter
                columns=self.df_to_columns(df),
                data=self.df_to_data(df),
                editable=False,
                filter_action="native" if enable_filtering else 'none',
                sort_action="native" if enable_sorting else 'none',
                sort_mode="multi",
                column_selectable=False,  # "single",
                row_selectable="multi" if enable_row_selection else False,
                row_deletable=False,
                selected_columns=[],
                selected_rows=[],
                page_action="native",
                page_current=0,
                page_size=10,
                style_table={'overflowX': 'scroll'},
            ),

        def default_input_property(self):
            """
                https://dash.plotly.com/datatable/interactivity
                Property Notes:
                    - derived_virtual_data: None on first callback (but matches original df, otherwise use pd.DataFrame(data)
                    - derived_virtual_selected_rows
                    - derived_virtual_selected_columns
            """
            return 'derived_virtual_data'

        def default_output_property(self):
            """Likely that you want to update both 'data' and 'columns'"""
            raise NotImplementedError(
                "No default ouptut for DataTable. Likely that you want to update both 'data' and 'columns' with something like\n`outputs=[table.as_output('data'), table.as_output('columns')],`")

    class Store(OutputComponent, dcc.Store):
        def __init__(self, id=None, **kwargs):
            id = id if id is not None else str(uuid.uuid4())
            super().__init__(id, **kwargs)

        def default_output_property(self):
            return "data"

        def default_input_property(self):
            return "data"

    class Div(OutputComponent, html.Div):
        def __init__(self, id=None, header=None, **kwargs):
            if header is None and id is not None:
                header = id
            self.header = header
            id = id if id is not None else str(uuid.uuid4())
            super().__init__(id, **kwargs)

        def default_output_property(self):
            return 'children'

        def layout(self):
            if self.header:
                return html.Div([html.H4(self.header), self])
            else:
                return super().layout()

    class Markdown(OutputComponent, dcc.Markdown):
        def __init__(self, heading: str = None, **kwargs):
            id = kwargs.pop("id", str(uuid.uuid4()))
            super().__init__(id=id, **kwargs)
            self.heading = heading

        def layout(self):
            if self.heading is not None:
                return html.Div([dcc.Markdown(f"{self.heading}"), self])
            else:
                return self

        def default_output_property(self):
            return "children"

        def default_input_property(self):
            return "children"

    class Graph(OutputComponent, html.Div):
        # Params for downloading figures
        _WIDTH = 1000  # Default is ~1000
        _HEIGHT = 450  # Default is ~450
        _SCALE = 2  # Default is 1 (higher scales up the whole figure when downloaded, so higher quality)

        # Defaults for other parameters
        _MAX_DISPLAYED_X = 500
        _MAX_DISPLAYED_Y = 500

        # Functions to create pattern-matching callbacks of the subcomponents
        class ids:
            @staticmethod
            def generic(aio_id, key: str):
                return {
                    "component": "Graph",
                    "subcomponent": f"generic",
                    "key": key,
                    "aio_id": aio_id,
                }

            @staticmethod
            def input(aio_id, key: str):
                return {
                    "component": "Graph",
                    "subcomponent": f"input",
                    "key": key,
                    "aio_id": aio_id,
                }

            @staticmethod
            def button(aio_id, key: str):
                return {
                    "component": "Graph",
                    "subcomponent": f"button",
                    "key": key,
                    "aio_id": aio_id,
                }

            @staticmethod
            def graph(aio_id):
                return {
                    "component": "Graph",
                    "subcomponent": f"figure",
                    "aio_id": aio_id,
                }

            @staticmethod
            def update_figure_store(aio_id):
                """Updating this store with a fig.to_dict() will be passed on to the graph figure without requiring a
                second callback
                I.e. The callback to update the actual figure is defined in the AIO, so can't be duplicated elsewhere
                """
                return {
                    "component": "Graph",
                    "subcomponent": f"update_figure",
                    "aio_id": aio_id,
                }

        # Make the ids class a public class
        ids = ids

        def __init__(self, aio_id=None, figure=None, app=None, **graph_kwargs):
            """Note: pass 'app' to setup the callbacks"""
            # Set defaults
            figure = figure if figure else go.Figure()

            if aio_id is None:
                aio_id = str(uuid.uuid4())

            # Create components
            self.update_figure_store_id = self.ids.update_figure_store(aio_id)
            update_fig_store = Components.Store(id=self.update_figure_store_id, data=figure)
            self.graph_id = self.ids.graph(aio_id)
            figure = limit_max_datasize(figure, max_x=self._MAX_DISPLAYED_X, max_y=self._MAX_DISPLAYED_Y,
                                        resample_x='decimate', resample_y='downsample')
            fig = dcc.Graph(
                id=self.graph_id,
                figure=figure,
                **graph_kwargs,
                config=dict(
                    toImageButtonOptions={
                        "format": "png",  # one of png, svg, jpeg, webp
                        # 'filename': 'custom_image',
                        "height": self._HEIGHT,
                        "width": self._WIDTH,
                        "scale": self._SCALE,  # Multiply title/legend/axis/canvas sizes by this factor
                    }
                ),
            )

            download_buttons = dbc.ButtonGroup(
                [
                    dbc.Button(children=name, id=self.ids.button(aio_id, name))
                    for name in ["HTML", "Jpeg", "SVG", "Data", "Igor"]
                ], style={'max_width': '500px'}
            )

            options_layout = dbc.Form(
                [
                    dbc.Label(
                        children="Download Name",
                        html_for=self.ids.input(aio_id, "downloadName"),
                    ),
                    Components.Input(id=self.ids.input(aio_id, "downloadName")),
                    html.Hr(),
                    download_buttons,
                    html.Hr(),
                    add_label(
                        dbc.RadioButton(id=self.ids.generic(aio_id, "tog-full-density")),
                        "Full Density",
                    ),
                    add_label(
                        html.Div([
                            Components.Input(id=self.ids.generic(aio_id, "inp-max-x"), display_name='Max X points',
                                             inputmode='numeric', value=self._MAX_DISPLAYED_X),
                            Components.Input(id=self.ids.generic(aio_id, "inp-max-y"), display_name='Max y points',
                                             inputmode='numeric', value=self._MAX_DISPLAYED_Y),
                        ],),
                        "Max display in X and Y",
                    ),
                    add_label(
                        dbc.RadioButton(id=self.ids.generic(aio_id, "tog-waterfall")),
                        "Waterfall",
                    ),
                ],
            )

            options_button = dbc.Button(
                id=self.ids.button(aio_id, "optsPopover"),
                children="Options",
                size="sm",
                color="light",
            )
            options_popover = dbc.Popover(
                children=dbc.PopoverBody(options_layout),
                target=options_button.id,
                trigger="legacy",
            )

            # Make Layout
            full_layout = html.Div(
                [
                    update_fig_store,
                    dcc.Download(id=self.ids.generic(aio_id, "download")),
                    options_popover,
                    dcc.Loading(fig, type="default"),
                    html.Div(
                        children=options_button,
                        style={"position": "absolute", "top": 0, "left": 0},
                    ),
                ],
                style={"position": "relative"},
            )

            # Run Callbacks (only if necessary)
            self.run_callbacks(app)

            super().__init__(id=aio_id, children=full_layout)  # html.Div contains layout

        def default_output_property(self):
            """Update the Store component, then the Graph component will update from there"""
            return 'data'

        def as_output(self, serverside=True):
            # # TODO: When serverside=True, make this update a local store with full fig (HD data),
            # # TODO: Then only update the displayed fig with a reduced amount of that data
            # # TODO: When download button clicked, can pull from the HD data
            if serverside:
                return ServersideOutput(self.update_figure_store_id, 'data')
            else:
                return Output(self.update_figure_store_id, 'data')
            # return Output(self.update_figure_store_id, 'data')

        @classmethod
        def run_callbacks(cls, app):
            """Run the callbacks if they don't already exist in the app"""
            if app is None:
                return False
            if getattr(app, '_GraphCallbacksMade', False) is False:
                app.callback(Output(cls.ids.graph(MATCH), "figure"),
                             Input(cls.ids.update_figure_store(MATCH), "data"),
                             Input(cls.ids.generic(MATCH, "tog-waterfall"), "value"),
                             Input(cls.ids.generic(MATCH, "tog-full-density"), "value"),
                             Input(cls.ids.generic(MATCH, "inp-max-x"), "value"),
                             Input(cls.ids.generic(MATCH, "inp-max-y"), "value"),
                             State(cls.ids.graph(MATCH), "figure"),
                             prevent_initial_call=True,
                             )(cls.update_figure)

                app.callback(
                    Output(cls.ids.generic(MATCH, "download"), "data"),
                    Input(cls.ids.button(MATCH, ALL), "n_clicks"),
                    State(cls.ids.input(MATCH, "downloadName"), "value"),
                    State(cls.ids.graph(MATCH), "figure"),
                    State(cls.ids.update_figure_store(MATCH), "data"),
                    prevent_initial_callback=True,
                )(cls.download)
                app._GraphCallbacksMade = True
            return True

        @staticmethod
        def update_figure(update_fig, waterfall, full_density, max_x, max_y, existing_fig):
            max_x, max_y = [int(m) for m in [max_x, max_y]]
            # if dash.ctx.triggered and dash.ctx.triggered_id.get("subcomponent", None) == "update_figure":
            #     fig = go.Figure(update_fig)
            #     fig = limit_max_datasize(fig, max_x=1000, max_y=1000, resample_x='decimate', resample_y='downsample')
            # else:
            #     fig = go.Figure(existing_fig)
            fig = go.Figure(update_fig)
            if not full_density:
                fig = limit_max_datasize(fig, max_x=max_x, max_y=max_y, resample_x='decimate', resample_y='downsample')

            fig = fig_waterfall(fig, waterfall)  # Set waterfall state if appropriate
            # # TODO: Probably can remove this formatting (should have good defaults set elsewhere)
            # fig.update_layout(
            #     template="plotly_white",
            #     xaxis=dict(
            #         mirror=True,
            #         ticks="outside",
            #         showline=True,
            #         linecolor="black",
            #     ),
            #     yaxis=dict(
            #         mirror=True,
            #         ticks="outside",
            #         showline=True,
            #         linecolor="black",
            #     ),
            # )
            if fig:
                return fig
            else:
                fig = go.Figure()
                fig.update_layout(title='Nothing to show')
                return fig

        @staticmethod
        def download(selected, download_name, figure, stored_fig):
            def make_filename_safe(fname):
                # fname = fname.replace(':', '--')
                fname = slugify.slugify(fname)
                return fname

            triggered = get_triggered()
            if triggered.id and triggered.id["subcomponent"] == "button" and figure:
                selected = triggered.id["key"].lower()
                fig = go.Figure(figure)
                if not download_name:
                    download_name = (
                        fig.layout.title.text if fig.layout.title.text else "fig"
                    )
                download_name = download_name.split(".")[
                    0
                ]  # To remove any extensions in name
                download_name = make_filename_safe(download_name)
                if selected == "html":
                    return dict(
                        content=fig.to_html(),
                        filename=f"{download_name}.html",
                        type="text/html",
                    )
                elif selected == "jpeg":
                    filepath = os.path.join(TEMPDIR, f"{str(uuid.uuid4())}.jpg")
                    fig.write_image(
                        filepath,
                        format="jpg",
                        width=Components.Graph._WIDTH,
                        height=Components.Graph._HEIGHT,
                        scale=Components.Graph._SCALE,
                    )
                    return dcc.send_file(
                        filepath, f"{download_name}.jpg", type="image/jpg"
                    )
                elif selected == "svg":
                    filepath = os.path.join(TEMPDIR, f"{str(uuid.uuid4())}.svg")
                    fig.write_image(filepath, format="svg")
                    return dcc.send_file(
                        filepath, f"{download_name}.svg", type="image/svg+xml"
                    )
                elif selected == "data":
                    fig = go.Figure(stored_fig)
                    filepath = os.path.join(TEMPDIR, f"{str(uuid.uuid4())}.json")
                    fig_to_data_json(fig, filepath)
                    return dcc.send_file(
                        filepath, f"{download_name}.json", type="application/json"
                    )
                elif selected == "igor":
                    fig = go.Figure(stored_fig)
                    filepath = os.path.join(TEMPDIR, f"{str(uuid.uuid4())}.itx")
                    fig_to_igor_itx(fig, filepath)
                    return dcc.send_file(
                        filepath, f"{download_name}.itx", type="application/json"
                    )
                return dash.no_update


def make_input_layout(inputs: list[InputComponent]):
    return html.Div([inp.layout() for inp in inputs])


def make_output_layout(outputs: list[OutputComponent]):
    return html.Div([out.layout() for out in outputs])


def table_type(df_column):
    # modified from https://dash.plotly.com/datatable/filtering
    # Note - this only works with Pandas >= 1.0.0
    if sys.version_info < (3, 0):  # Pandas 1.0.0 does not support Python 2
        return 'any'
    if isinstance(df_column.dtype, pd.DatetimeTZDtype):
        return 'datetime',
    elif (isinstance(df_column.dtype, pd.StringDtype) or
          isinstance(df_column.dtype, pd.BooleanDtype) or
          isinstance(df_column.dtype, pd.CategoricalDtype) or
          isinstance(df_column.dtype, pd.PeriodDtype)):
        return 'text'
    elif np.issubdtype(df_column.dtype, np.number):
        return 'numeric'
    else:
        return 'any'


def generate_conditional_format_styles(df: pd.DataFrame, n_bins: int=10, columns: [str, list[str]]='all', mode: str='per_column', colorscale: Any ='blues'):
    """Generates styles for conditionally coloring numeric columns of a DataTable
     Note: They can be applied to dash_table.DataTable via the `style_data_conditional` argument

     Inspired from: https://dash.plotly.com/datatable/conditional-formatting

     Args:
         df: DataFrame to apply conditional formatting to
         n_bins: How many color bins to split the numbers into
         columns: default 'all' -- Which columns to apply to (single column, list of columns, or 'all')
         mode: default 'per_column' -- How to apply color formatting, either across 'all' or 'per_column'
         colorscale: default 'blues' -- Colorscale to use (Anything that pc.sample_colorscale accepts)
     """
    df = df.select_dtypes('number')
    if columns == 'all':
        if 'id' in df:
            df = df.drop(['id'], axis=1)
    else:
        columns = ensure_list(columns)  # Ensures output of df[columns] is a df not a series
        df = df[columns]

    df_max = df.max().max()
    df_min = df.min().min()

    styles = []
    for column in df:
        match mode:
            case 'per_column':
                _max = df[column].max()
                _min = df[column].min()
            case 'all':
                _max = df_max
                _min = df_min
            case _:
                raise NotImplementedError(f"{mode} not implemented, should be one of ['per_column', 'all']")

        bounds = np.linspace(0, 1, n_bins+1)  # +1 because want n_bin gaps between n_bin+1 bounds
        ranges = bounds*(_max-_min)+_min
        colors = pc.sample_colorscale(colorscale, n_bins, 0, 1, colortype='rgb')

        for color, min_range, max_range in zip(colors, ranges[:-1], ranges[1:]):
            # use re to extract the numbers within the parentheses
            numbers = re.findall(r'\d+', color)
            # convert the numbers to integers
            r, g, b = map(int, numbers)
            if np.all([v > 200 for v in [r, g, b]]):
                text_color = 'black'
            else:
                text_color = 'white'
            styles.append({
                'if': {
                    'filter_query': f'{{{column}}} >= {min_range} && {{{column}}} <= {max_range}',
                    'column_id': column
                },
                'backgroundColor': color,
                'color': text_color,
            })

    return styles

def generate_conditional_format_legends(styles: list[dict]) -> dict[str, Any]:
    """Generate a legend corresponding to conditional formatted styles

    Args:
        styles: list of styles generated by `generate_conditional_format_styles`
    """
    legends = {}
    for style in styles:
        col = dictor(style, 'if.column_id')
        if col not in legends:
            legends[col] = []

        pattern = r'\{.*?\}.*?(\d+(?:\.\d+)?)\s*\&\&\s*\{.*?\}.*?(\d+(?:\.\d+)?)'  # match curly braces and floats with the && operator in between
        match = re.search(pattern, dictor(style, 'if.filter_query'))  # search for the pattern in the string

        if not match:
            raise RuntimeError(f'No min_range/max_range found in style.if.filter_query')
        min_range = float(match.group(1))
        max_range = float(match.group(2))

        legends[col].append(
            html.Div(style={'display': 'inline-block', 'width': '60px'}, children=[
                html.Div(
                    style={
                        'backgroundColor': style['backgroundColor'],
                        'borderLeft': '1px rgb(50, 50, 50) solid',
                        'height': '10px'
                    }
                ),
                html.Small(round((min_range+max_range)/2, 2), style={'paddingLeft': '2px'})
            ])
        )

    for k in legends:
        legends[k] = html.Div(legends[k], style={'padding': '5px 0 5px 0'})

    return legends


def conditional_format_legends_to_div(legends: dict, mode='per_column') -> html.Div:
    """Convert the dictionary of legends into either a list of legends for each column, or a single legend from the
    first of the columns

    Args:
        legends: the legends dict generated by `generate_conditional_format_legend`
        mode: 'per_column' makes a legend for each column separately, 'all' just makes a single legend (from first col)
    """
    match mode:
        case 'per_column':
            full_legend = dash.html.Div([
                add_label(leg, k)
                for k, leg in legends.items()
            ])
        case 'all':
            full_legend = next(iter(legends.values()))
        case _:
            raise NotImplementedError
    return full_legend
