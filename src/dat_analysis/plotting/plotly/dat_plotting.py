"""
This is where all general dat plotting functions should live... To use in other pages, import the more general plotting
function from here, and make a little wrapper plotting function which calls with the relevant arguments

Sep21 -- Basically the idea that when plotting a single dat or dats, there is quite a lot of information which is
usually useful but time consuming to add (e.g. datnum in the title). So initializing a plotter which knows of the dat
allows it to fill in blanks where it can.
Can also control some more general behaviours, like plotting template and resampling methods etc.
"""
from __future__ import annotations

import plotly.graph_objects as go
import numpy as np
import logging
import abc
from typing import Optional, Union, List, Tuple, Iterable, TYPE_CHECKING

from dat_analysis.useful_functions import ARRAY_LIKE
from dat_analysis.core_util import resample_data

if TYPE_CHECKING:
    from dat_analysis.dat_object.dat_hdf import DatHDF

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_NOT_SET = object()


class DatPlotter(abc.ABC):
    """Generally useful functions for all Dat Plotters"""

    MAX_POINTS = 1000  # Maximum number of points to plot in x or y otherwise resampled to be below this.
    RESAMPLE_METHOD = 'bin'  # Resample down to MAX_POINTS points by binning or just down sampling (i.e every nth)
    TEMPLATE = 'plotly_white'

    def __init__(self, dat: Optional[DatHDF] = _NOT_SET, dats: Optional[Iterable[DatHDF]] = None):
        """Initialize with a dat or dats to provide some ability to get defaults"""
        if dat is not _NOT_SET:
            self.dat = dat
        elif dats is not None:
            self.dat = dats[0]
        else:
            if dat == _NOT_SET:
                logger.warning(f'No Dat supplied, no values will be supplied by default. Set dat=None to suppress this '
                               f'warning')
            self.dat = None
        self.dats = dats

    def figure(self,
               xlabel: Optional[str] = None, ylabel: Optional[str] = None,
               title: Optional[str] = None,
               fig_kwargs: Optional[dict] = None) -> go.Figure:
        """
        Generates a go.Figure only using defaults from dat where possible.
        Use this as a starting point to add multiple traces. Or if only adding one trace, use 'plot' instead.
        Args:
            xlabel (): X label for figure
            ylabel (): Y label for figure
            title (): Title for figure
            fig_kwargs (): Other fig_kwargs which are accepted by go.Figure()

        Returns:
            (go.Figure): Figure without any data, only axis labels and title etc.
        """
        if fig_kwargs is None:
            fig_kwargs = {}

        xlabel = self._get_xlabel(xlabel)
        ylabel = self._get_ylabel(ylabel)

        fig = go.Figure(**fig_kwargs)
        fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel, title=title, template=self.TEMPLATE)
        return fig

    @abc.abstractmethod
    def plot(self, trace_kwargs: Optional[dict], fig_kwargs: Optional[dict]) -> go.Figure:
        """Override to make something which returns a competed plotly go.Figure

        Note: Should also include: trace_kwargs: Optional[dict] = None, fig_kwargs: Optional[dict] = None
        """
        pass

    @abc.abstractmethod
    def trace(self, trace_kwargs: Optional[dict]) -> go.Trace:
        """Override to make something which returns the completed trace only

        Note: Should also include: trace_kwargs: Optional[dict] = None
        """
        pass

    @staticmethod
    def add_textbox(fig: go.Figure, text: str, position: Union[str, Tuple[float, float]],
                    fontsize=10) -> None:
        """
        Adds <text> to figure in a text box.
        Args:
            fig (): Figure to add text box to
            text (): Text to add
            position (): Absolute position on figure to add to (e.g. (0.5,0.9) for center top, or 'CT' for center top, or 'T' for center top)

        Returns:
            None: Modifies figure passed in
        """
        if isinstance(position, str):
            position = get_position_from_string(position)
        text = text.replace('\n', '<br>')
        fig.add_annotation(text=text,
                           xref='paper', yref='paper',
                           x=position[0], y=position[1],
                           showarrow=False,
                           bordercolor='#111111',
                           borderpad=3,
                           borderwidth=1,
                           opacity=0.8,
                           bgcolor='#F5F5F5',
                           font=dict(size=fontsize)
                           )

    def add_line(self, fig: go.Figure, value: float, mode: str = 'horizontal',
                 color: Optional[str] = None, linewidth: float = 1, linetype: str = 'solid') -> go.Figure:
        """
        Convenience for adding a line to a graph
        Args:
            fig (): Figure to add line to
            value (): Where to put line
            mode (): horizontal or vertical
            color(): Color of line
            linewidth (): in px
            linetype: 'solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot', or e.g. ("5px,10px,2px,2px")

        Returns:
            (go.Figure): Returns original figure with line added
        """

        def _add_line(x0, x1, xref, y0, y1, yref):
            fig.add_shape(dict(y0=y0, y1=y1, yref=yref, x0=x0, x1=x1, xref=xref,
                               type='line',
                               line=dict(color=color, width=linewidth, dash=linetype),
                               ))

        def add_vertical(x):
            _add_line(x0=x, x1=x, xref='x', y0=0, y1=1, yref='paper')

        def add_horizontal(y):
            _add_line(x0=0, x1=1, xref='paper', y0=y, y1=y, yref='y')

        if mode == 'horizontal':
            add_horizontal(y=value)
        elif mode == 'vertical':
            add_vertical(x=value)
        else:
            raise NotImplementedError(f'{mode} not recognized')
        return fig

    def save_to_dat(self, fig, name: Optional[str] = None, sub_group_name: Optional[str] = None,
                    overwrite: bool = False):
        """Saves to the Figures attribute of the dat"""
        if self.dat is not None:
            self.dat.Figures.save_fig(fig, name=name, sub_group_name=sub_group_name, overwrite=overwrite)

    def _resample_data(self, data: np.ndarray,
                       x: Optional[np.ndarray] = None,
                       y: Optional[np.ndarray] = None,
                       z: Optional[np.ndarray] = None):
        """
        Resamples given data using self.MAX_POINTS and self.RESAMPLE_METHOD.
        Will always return data, then optionally ,x, y, z incrementally (i.e. can do only x or only x, y but cannot do
        e.g. x, z)
        Args:
            data (): Data to resample down to < self.MAX_POINTS in each dimension
            x (): Optional x array to resample the same amount as data
            y (): Optional y ...
            z (): Optional z ...

        Returns:
            (Any): Matching combination of what was passed in (e.g. data, x, y ... or data only, or data, x, y, z)
        """
        return resample_data(data=data, x=x, y=y, z=z,
                             max_num_pnts=self.MAX_POINTS, resample_method=self.RESAMPLE_METHOD)

    def _get_x(self, x):
        if x is None and self.dat:
            return self.dat.Data.x
        return x

    def _get_y(self, y):
        if y is None and self.dat:
            return self.dat.Data.y
        return y

    def _get_xlabel(self, xlabel):
        if xlabel is None and self.dat:
            return self.dat.Logs.xlabel
        return xlabel

    def _get_ylabel(self, ylabel):
        if ylabel is None and self.dat:
            return self.dat.Logs.ylabel
        return ylabel


class OneD(DatPlotter):
    """
    For 1D plotting
    """

    def trace(self, data: ARRAY_LIKE, data_err: Optional[ARRAY_LIKE] = None,
              x: Optional[ARRAY_LIKE] = None, text: Optional[ARRAY_LIKE] = None,
              mode: Optional[str] = None,
              name: Optional[str] = None,
              hover_data: Optional[ARRAY_LIKE] = None,
              hover_template: Optional[str] = None,
              trace_kwargs: Optional[dict] = None) -> go.Scatter:
        """Just generates a trace for a figure

        Args: hover_data: Shape should be (N-datas per pt, data.shape)  Note: plotly does this the other way around (
            which is wrong)

        """
        data, data_err, x = [np.asanyarray(arr) if arr is not None else None for arr in [data, data_err, x]]
        if data.ndim != 1:
            logger.warning('Raising an error')
            raise ValueError(f'data.shape: {data.shape}. Invalid shape, should be 1D for a 1D trace')

        if trace_kwargs is None:
            trace_kwargs = {}
        x = self._get_x(x)
        mode = self._get_mode(mode)

        data, x = self._resample_data(data, x)  # Makes sure not plotting more than self.MAX_POINTS in any dim
        if hover_data:  # Also needs same dimensions in x
            hover_data = np.asanyarray(hover_data)
            if (s := hover_data.shape[1:]) == data.shape:
                hover_data = np.moveaxis(hover_data, 0, -1)  # This is how plotly likes the shape
            elif (s := hover_data.shape[:-1]) == data.shape:
                pass
            else:
                raise ValueError(f"hover_data.shape ({hover_data.shape}) doesn't match data.shape ({data.shape})")
            hover_data = self._resample_data(hover_data)

        if data.shape != x.shape or x.ndim > 1 or data.ndim > 1:
            raise ValueError(f'Trying to plot data with different shapes or dimension > 1. '
                             f'(x={x.shape}, data={data.shape} for dat{self.dat.datnum if self.dat else "--"}.')
        if text is not None and 'text' not in mode:
            mode += '+text'
        trace = go.Scatter(x=x,
                           y=data,
                           error_y=dict(
                               type='data', array=data_err, visible=True
                           ),
                           text=text,
                           mode=mode,
                           name=name,
                           textposition='top center',
                           **trace_kwargs)
        if hover_data is not None and hover_template:
            trace.update(customdata=hover_data, hovertemplate=hover_template)
        return trace

    def plot(self, data: ARRAY_LIKE, data_err: Optional[ARRAY_LIKE] = None,
             x: Optional[ARRAY_LIKE] = None, text: Optional[ARRAY_LIKE] = None,
             xlabel: Optional[str] = None, ylabel: Optional[str] = None,
             trace_name: Optional[str] = None,
             title: Optional[str] = None,
             mode: Optional[str] = None,
             trace_kwargs: Optional[dict] = None, fig_kwargs: Optional[dict] = None) -> go.Figure:
        """Creates a figure and adds trace to it"""
        fig = self.figure(xlabel=xlabel, ylabel=ylabel,
                          title=title,
                          fig_kwargs=fig_kwargs)
        trace = self.trace(data=data, data_err=data_err,
                           x=x, text=text, mode=mode, name=trace_name, trace_kwargs=trace_kwargs)
        fig.add_trace(trace)
        self._default_autosave(fig, name=title)
        return fig

    def _get_mode(self, mode):
        if mode is None:
            mode = 'markers'
        return mode

    def _get_ylabel(self, ylabel):
        if ylabel is None:
            ylabel = 'Arbitrary'
        return ylabel

    def _default_autosave(self, fig: go.Figure, name: Optional[str] = None):
        self.save_to_dat(fig, name=name)


class TwoD(DatPlotter):
    """
    For 2D plotting
    """

    def plot(self, data: np.ndarray, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
             xlabel: Optional[str] = None, ylabel: Optional[str] = None,
             title: Optional[str] = None,
             plot_type: Optional[str] = None,
             trace_kwargs: Optional[dict] = None, fig_kwargs: Optional[dict] = None):
        if fig_kwargs is None:
            fig_kwargs = {}
        if plot_type is None:
            plot_type = 'heatmap'
        xlabel = self._get_xlabel(xlabel)
        ylabel = self._get_ylabel(ylabel)

        fig = go.Figure(self.trace(data=data, x=x, y=y, trace_type=plot_type, trace_kwargs=trace_kwargs), **fig_kwargs)
        fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel, title=title, template='plotly_white')
        self._plot_autosave(fig, name=title)
        return fig

    def trace(self, data: ARRAY_LIKE, x: Optional[ARRAY_LIKE] = None, y: Optional[ARRAY_LIKE] = None,
              trace_type: Optional[str] = None,
              trace_kwargs: Optional[dict] = None) -> Union[go.Heatmap, List[go.Scatter3d]]:
        if data.ndim != 2:
            raise ValueError(f'data.shape: {data.shape}. Invalid shape, should be 2D for a 2D trace')
        if trace_type is None:
            trace_type = 'heatmap'
        if trace_kwargs is None:
            trace_kwargs = {}
        x, y, data = [np.asanyarray(arr) if arr is not None else None for arr in [x, y, data]]
        x = self._get_x(x)
        y = self._get_y(y)

        if x.shape[0] != data.shape[-1] or y.shape[0] != data.shape[0]:
            raise ValueError(f'Bad array shape -- data.shape: {data.shape}, x.shape: {x.shape}, y.shape: {y.shape}')
        data, x = self._resample_data(data, x)  # Makes sure not plotting more than self.MAX_POINTS in any dim

        if trace_type == 'heatmap':
            trace = go.Heatmap(x=x, y=y, z=data, **trace_kwargs)
        elif trace_type == 'waterfall':
            trace = [go.Scatter3d(mode='lines', x=x, y=[yval] * len(x), z=row, name=f'{yval:.3g}', **trace_kwargs) for
                     row, yval in zip(data, y)]
        else:
            raise ValueError(f'{trace_type} is not a recognized trace type for TwoD.trace')
        return trace

    def _plot_autosave(self, fig: go.Figure, name: Optional[str] = None):
        self.save_to_dat(fig, name=name)


class ThreeD(DatPlotter):
    """
    For 3D plotting
    """

    def plot(self, trace_kwargs: Optional[dict] = None, fig_kwargs: Optional[dict] = None) -> go.Figure:
        raise NotImplementedError

    def trace(self, trace_kwargs: Optional[dict] = None) -> go.Trace:
        # data, x = self._resample_data(data, x)  # Makes sure not plotting more than self.MAX_POINTS in any dim
        # if data.ndim != 3:
        #     raise ValueError(f'data.shape: {data.shape}. Invalid shape, should be 3D for a 3D trace')
        raise NotImplementedError


def get_position_from_string(text_pos: str) -> Tuple[float, float]:
    """
    Get position to place things in figure using short string for convenience

    Args:
        text_pos (): one or two letter string to specify a position
            e.g. 'C' = center, 'B' = bottom, 'TR' = top right, etc.

    Returns:

    """
    assert isinstance(text_pos, str)
    ps = dict(C=0.5, B=0.1, T=0.9, L=0.1, R=0.9)

    text_pos = text_pos.upper()
    if not all([l in ps for l in text_pos]) or len(text_pos) not in [1, 2]:
        raise ValueError(f'{text_pos} is not a valid position. It must be 1 or 2 long, with only {ps.keys()}')

    if len(text_pos) == 1:
        if text_pos == 'C':
            position = (ps['C'], ps['C'])
        elif text_pos == 'B':
            position = (ps['C'], ps['B'])
        elif text_pos == 'T':
            position = (ps['C'], ps['T'])
        elif text_pos == 'L':
            position = (ps['L'], ps['C'])
        elif text_pos == 'R':
            position = (ps['R'], ps['C'])
        else:
            raise NotImplementedError
    elif len(text_pos) == 2:
        a, b = text_pos
        if a in ['T', 'B'] or b in ['L', 'R']:
            position = (ps[b], ps[a])
        else:
            position = (ps[a], ps[b])
    else:
        raise NotImplementedError
    return position
