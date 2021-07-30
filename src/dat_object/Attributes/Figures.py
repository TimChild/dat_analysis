from __future__ import annotations
import os
from dictor import dictor
import plotly.graph_objects as go
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Union, List, Dict, Optional
import h5py
import logging

from src.dat_object.Attributes.DatAttribute import DatAttribute
from src.hdf_util import with_hdf_write, with_hdf_read, NotFoundInHdfError, DatDataclassTemplate
import src.hdf_util as HDU

if TYPE_CHECKING:
    from src.dat_object.dat_hdf import DatHDF

logger = logging.getLogger(__name__)


@dataclass
class FigSave(DatDataclassTemplate):
    """
    Dataclass for saving figures to DatHDF. This is mostly a wrapper around the plotly go.Figure() dict, but will
    let me change things in the future more easily
    """
    fig: dict
    name: str
    date_saved: Optional[datetime] = None  # This is only intended to be initialized when loading from HDF

    def save_to_hdf(self, parent_group: h5py.Group, name: Optional[str] = None):
        self.date_saved = datetime.now()
        group = super().save_to_hdf(parent_group, name)
        HDU.set_attr(group, '_identifier', 'FigSave')  # So I can easily look for all groups with this group attr
        # TODO: any benefit to adding a hash thing to this?

    def to_fig(self) -> go.Figure:
        return go.Figure(self.fig)


class Figures(DatAttribute):
    version = '1.0.0'
    group_name = 'Figures'
    description = 'A place to store all figures which are plotted, partly so that it is easy to see a history of what' \
                  'has been plotted, but also so that figures can be shared between kernels and with Dash apps (i.e. ' \
                  'they can save data to the HDF here, and it can be read anywhere else)'

    def __init__(self, dat: DatHDF):
        import src.plotting.plotly.dat_plotting as DP
        super().__init__(dat)
        self.OneD = DP.OneD(self.dat)
        self.TwoD = DP.TwoD(self.dat)

    @property
    def all_fig_names(self) -> List[str]:
        """Something which returns all the figure names (and group name?) of saved figures"""
        full_paths = self._full_fig_paths()
        return [p.split('/')[-1] for p in full_paths]

    @property
    def dash_fig_names(self) -> List[str]:
        full_paths = self._full_fig_paths(sub_group='Dash')
        return [p.split('/')[-1] for p in full_paths]

    @property
    def last_fig(self) -> go.Figure:
        fig_times = self.figs_by_date
        time_figs = {v: k for k, v in fig_times.items()}
        name = tuple(sorted(time_figs.items()))[-1][1]  # latest time, name of fig
        fig = self.get_fig(name)
        return fig

    @property
    @with_hdf_read
    def figs_by_date(self) -> Dict[str, datetime]:
        fig_times = {}
        for path in self._full_fig_paths():
            group = self.hdf.get(path)
            name = HDU.get_attr(group, 'name')
            time = HDU.get_attr(group, 'date_saved')
            fig_times[name] = time
        return fig_times

    @with_hdf_read
    def _full_fig_paths(self, sub_group: Optional[str] = None):
        if sub_group:
            group = self.hdf.group.get(sub_group)
        else:
            group = self.hdf.group
        full_paths = HDU.find_all_groups_names_with_attr(group, '_identifier', 'FigSave')
        return full_paths

    @with_hdf_write
    def initialize_minimum(self):
        self.hdf.group.require_group('Dash')
        self.initialized = True

    @with_hdf_write
    def save_fig(self, fig, name: Optional[str] = None, sub_group_name: Optional[str] = None, overwrite=False):
        if not name:
            name = self._generate_fig_name(fig, overwrite=overwrite)
        elif name in self.all_fig_names and overwrite is False:
            name = self._generate_unique(name)
        name = name.replace('/', '-')  # To prevent saving in subgroups
        self._save_fig(fig, name, sub_group_name)

    def _generate_fig_name(self, fig: Union[go.Figure, dict], overwrite: bool = False):
        existing = self.all_fig_names

        if isinstance(fig, go.Figure):
            fig = fig.to_dict()

        title = dictor(fig, 'layout.title', None)
        if title:
            name = title
        else:
            plot_type = dictor(fig, 'data', [{}])[0].get('type', None)
            if plot_type is None:
                raise ValueError(f'Are you sure "fig" contains info?: {fig}')
            ylabel = dictor(fig, 'layout.yaxis.title.text', 'y')
            xlabel = dictor(fig, 'layout.xaxis.title.text', 'x')
            name = f'{plot_type}_{ylabel} vs {xlabel}'

        if name not in existing or overwrite:
            return name
        else:
            return self._generate_unique(name)

    def _generate_unique(self, name: str):
        existing = self.all_fig_names
        i = 0
        while True:
            new_name = f'{name}_{i}'
            if new_name not in existing:
                return new_name
            i += 1

    @with_hdf_write
    def _save_fig(self, fig: Union[go.Figure, dict], name: str, sub_group_name: Optional[str] = None):
        if sub_group_name:
            group = self.hdf.group.require_group(sub_group_name)
        else:
            group = self.hdf.group
        if isinstance(fig, go.Figure):
            fig = fig.to_dict()
        name = name.replace('/', '-')  # To prevent saving in subgroups
        FigSave(fig, name).save_to_hdf(group, name)

    def get_fig(self, name: str, sub_group_name: Optional[str] = None):
        name = name.replace('/', '-')  # Because I make this substitution when saving
        return self._get_fig(name, sub_group_name)

    @with_hdf_read
    def _get_fig(self, name: str, sub_group_name: Optional[str] = None) -> go.Figure:
        name = name.replace('/', '-')  # Because I make this substitution when saving
        full_paths_to_name = [v for v in self._full_fig_paths() if v.split('/')[-1] == name]
        if sub_group_name and full_paths_to_name:
            paths = [p for p in full_paths_to_name if sub_group_name in p]
        else:
            paths = full_paths_to_name

        if len(paths) > 1:
            raise ValueError(f'Found multiple paths to {name}: {paths}')
        elif len(paths) == 0:
            raise NotFoundInHdfError(f'Fig {name} not found [sub_group_name={sub_group_name}]')
        else:
            path = paths[0]

        fig_save: FigSave = self.get_group_attr(name, group_name=os.path.split(path)[0], DataClass=FigSave)
        logger.info(f'Figure {fig_save.name} was saved at {fig_save.date_saved}')
        return fig_save.to_fig()


