"""
All dash things specific to Dat analysis should be implemented here. BaseClasses should be general to any Dash app.

"""
from singleton_decorator import singleton
import time
from src.Dash.BaseClasses import BasePageLayout, BaseMain, BaseSideBar
from src.Dash.DatPlotting import OneD, TwoD, ThreeD
from dash_extensions.enrich import Input, Output, State
import dash_html_components as html
from src.Dash.app import app
import dash_bootstrap_components as dbc
from typing import Optional
import abc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from src.DatObject.Make_Dat import DatHandler
get_dat = DatHandler().get_dat


# Dash layouts for Dat Specific
class DatDashPageLayout(BasePageLayout, abc.ABC):
    def top_bar_layout(self):
        layout = super().top_bar_layout()
        return layout


class DatDashMain(BaseMain, abc.ABC):
    def graph_area(self, name: str, title: Optional[str] = None, default_fig: go.Figure = None,
                   datnum_id: str = None):
        if datnum_id is None:
            raise ValueError(f'Need datnum_id to know what datnum to save to')
        graph = super().graph_area(name, title, default_fig, save_option_kwargs=dict(datnum_id=datnum_id))
        return graph

    def graph_save_options(self, graph_id, datnum_id=None):
        if datnum_id is None:
            raise ValueError(f'Need id of whatever chooses datnum to know where to save to dat')
        layout = dbc.Row([
            dbc.Col(self._save_to_dat_button(graph_id), width='auto'),
            dbc.Col(self._download_button(graph_id, 'html'), width='auto'),
            dbc.Col(self._download_button(graph_id, 'jpg'), width='auto'),
            dbc.Col(self._download_button(graph_id, 'svg'), width='auto'),
            dbc.Col(self._download_name(graph_id), width='auto'),
        ], no_gutters=True)
        self._run_graph_save_callbacks(graph_id, datnum_id)
        return layout

    def _run_graph_save_callbacks(self, graph_id, datnum_id=None):
        super()._run_graph_save_callbacks(graph_id)
        self._save_to_dat_callback(graph_id, datnum_id)

    def _save_to_dat_callback(self, graph_id, datnum_id):  # FIXME: This is dat specific... should not be in here
        def save_to_dat(clicks, fig, datnum, save_name):
            if clicks and datnum and fig:
                dat = get_dat(datnum)
                fig = go.Figure(fig)
                if not save_name:
                    save_name = fig.layout.title.text
                    if not save_name:
                        save_name = dat.Figures._generate_fig_name(fig, overwrite=False)
                dat.Figures.save_fig(fig, save_name, sub_group_name='Dash', overwrite=True)
                return True
            else:
                raise PreventUpdate
        app.callback(Output(f'{graph_id}_div-fake-output', 'hidden'), Input(f'{graph_id}_but-dat-save', 'n_clicks'), State(graph_id, 'figure'), State(datnum_id, 'value'), State(f'{graph_id}_inp-download-name', 'value'))(save_to_dat)

    def _save_to_dat_button(self, graph_id):
        button = [dbc.Button('Save to Dat', id=f'{graph_id}_but-dat-save'), html.Div(id=f'{graph_id}_div-fake-output', style={'display': 'none'})]
        return button


class DatDashSideBar(BaseSideBar, abc.ABC):
    pass


@singleton
class NameResetter:
    def __init__(self):
        self.last_t = time.time()
        self._suffix = 0

    def get_resetting_fig_name(self):
        """
        Generates a new fig name each time it is called within a time window. Otherwise resets to the beginning
        Returns:
            (str): Unique name which expires after a time

        Examples:
            Want to be able to save all recent dash figures so they can be opened elsewhere, but don't want to keep filling
            up the datHDF with them over and over again. So this will give unique names, and then reset after a time
        """
        if time.time()-self.last_t > 10:  # If last graph was made more than 10 seconds ago then reset
            self._suffix = 0
        name = f'DashFig_{self._suffix}'
        self._suffix += 1
        self.last_t = time.time()
        return name


# Plotting classes for Dash specific
class DashOneD(OneD):
    def save_to_dat(self, fig, name: Optional[str] = None, sub_group_name: Optional[str] = None, overwrite=True):
        if not sub_group_name:
            sub_group_name = 'Dash'
        if name:
            super().save_to_dat(fig, name, sub_group_name, overwrite)
        else:
            super().save_to_dat(fig, NameResetter().get_resetting_fig_name(), sub_group_name, overwrite)

    def _default_autosave(self, fig, name: Optional[str] = None):
        super()._default_autosave(fig, NameResetter().get_resetting_fig_name())


class DashTwoD(TwoD):
    def save_to_dat(self, fig, name: Optional[str] = None, sub_group_name: Optional[str] = None, overwrite=True):
        if not sub_group_name:
            sub_group_name = 'Dash'
        if name:
            super().save_to_dat(fig, name, sub_group_name, overwrite)
        else:
            super().save_to_dat(fig, NameResetter().get_resetting_fig_name(), sub_group_name, overwrite)

    def _plot_autosave(self, fig, name: Optional[str] = None):
        super()._plot_autosave(fig, NameResetter().get_resetting_fig_name())


class DashThreeD(ThreeD):
    def save_to_dat(self, fig, name: Optional[str] = None, sub_group_name: Optional[str] = None, overwrite=True):
        if not sub_group_name:
            sub_group_name = 'Dash'
        if name:
            super().save_to_dat(fig, name, sub_group_name, overwrite)
        else:
            super().save_to_dat(fig, NameResetter().get_resetting_fig_name(), sub_group_name, overwrite)

    def _plot_autosave(self, fig, name: Optional[str] = None):
        super()._plot_autosave(fig, NameResetter().get_resetting_fig_name())
