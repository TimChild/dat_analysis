from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter
import copy
import uuid
from scipy.signal import filtfilt, iirnotch
import logging
from typing import Union, Optional, TYPE_CHECKING

from dat_analysis.plotting.plotly.util import default_fig, heatmap, error_fill
from dat_analysis.core_util import get_data_index, get_matching_x, bin_data, decimate, center_data, mean_data, resample_data, ensure_list
from dat_analysis.analysis_tools.data_aligning import subtract_data, subtract_data_1d

if TYPE_CHECKING:
    from dat_analysis.dat.dat_hdf import DatHDF


@dataclass
class PlottingInfo:
    x_label: str = None
    y_label: str = None
    title: str = None
    coloraxis_title: str = None
    datnum: int = None  # TODO: Remove

    @classmethod
    def from_dat(cls, dat: DatHDF, title: str = None):
        inst = cls(
            x_label=dat.Logs.x_label,
            y_label=dat.Logs.y_label,
            title=f"Dat{dat.datnum}: {title}",
            datnum=dat.datnum,
        )
        return inst

    def update_layout(self, fig: go.Figure):
        """Updates layout of figure (only with non-None values)"""
        updates = {
            k: v
            for k, v in {
                "xaxis_title_text": self.x_label,
                "yaxis_title_text": self.y_label,
                "title_text": self.title,
                "coloraxis_colorbar_title_text": self.coloraxis_title,

            }.items()
            if v is not None
        }
        return fig.update_layout(**updates)


@dataclass
class Data:
    data: np.ndarray
    x: np.ndarray
    y: np.ndarray = None
    xerr: np.ndarray = None
    yerr: np.ndarray = None
    plot_info: PlottingInfo = PlottingInfo()

    def __post_init__(self, *args, **kwargs):
        # If data is 2D and no y provided, just assign index values
        self.x = np.asanyarray(self.x) if self.x is not None else self.x
        self.y = np.asanyarray(self.y) if self.y is not None else self.y
        self.data = np.asanyarray(self.data) if self.data is not None else self.data
        if self.data.ndim == 2 and self.y is None:
            self.y = np.arange(self.data.shape[0])
        if self.plot_info is None:
            self.plot_info = PlottingInfo()

        if args:
            logging.warning(f'Data got unexpected __post_init__ args {args}')
        if kwargs:
            logging.warning(f'Data got unexpected __post_init__ kwargs {kwargs}')

    def plot(self, limit_datapoints=True, **trace_kwargs):
        """
            Generate a quick 1D or 2D plot of data
            Args:
                limit_datapoints: Whether to do automatic downsampling before plotting (very useful for large 2D datasets)
        """
        fig = default_fig()
        if self.data.ndim == 1:
            fig.add_traces(self.get_traces(**trace_kwargs))
        elif self.data.ndim == 2:
            if limit_datapoints:
                fig.add_trace(heatmap(x=self.x, y=self.y, data=self.data, **trace_kwargs))
            else:
                fig.add_trace(go.Heatmap(x=self.x, y=self.y, z=self.data, **trace_kwargs))
        else:
            raise RuntimeError(f"data is not 1D or 2D, got data.shape = {self.data.ndim}")
        if self.plot_info:
            fig = self.plot_info.update_layout(fig)
        return fig

    def get_traces(self, max_num_pnts=10000, error_fill_threshold=50, **first_trace_kwargs) -> list[
        Union[go.Scatter, go.Heatmap]]:
        traces = []
        group_key = first_trace_kwargs.pop('legendgroup', uuid.uuid4().hex)
        if self.data.ndim == 1:
            if max_num_pnts:
                data, x = resample_data(self.data, x=self.x, max_num_pnts=max_num_pnts, resample_method='downsample',
                                        resample_x_only=True)
            # Note: Scattergl might cause issues with lots of plots, if so just go back to go.Scatter only
            # scatter_func = go.Scatter if x.shape[0] < 1000 else go.Scattergl
            scatter_func = go.Scatter
            trace = scatter_func(x=x, y=data, legendgroup=group_key, **first_trace_kwargs)
            traces.append(trace)
            if self.yerr is not None:
                yerr = resample_data(self.yerr, max_num_pnts=max_num_pnts, resample_method='downsample',
                                     resample_x_only=True)
                if len(x) <= error_fill_threshold:
                    trace.update(error_y=dict(array=yerr))
                else:
                    # Note: error_fill also switched to scattergl after 1000 points
                    traces.append(error_fill(x, data, yerr, legendgroup=group_key))
            if self.xerr is not None:
                xerr = resample_data(self.xerr, max_num_pnts=max_num_pnts, resample_method='downsample',
                                     resample_x_only=True)
                if len(x) <= error_fill_threshold:
                    trace.update(error_x=dict(array=xerr), legendgroup=group_key)
                else:
                    pass  # Too many to plot, don't do anything
        elif self.data.ndim == 2:
            traces.append(heatmap(x=self.x, y=self.y, data=self.data, **first_trace_kwargs))
        else:
            raise RuntimeError(f"data is not 1D or 2D, got data.shape = {self.data.ndim}")
        return traces

    def copy(self) -> Data:
        """Make a copy of self s.t. changing the copy does not affect the original data"""
        return copy.deepcopy(self)

    def center(self, centers):
        centered, new_x = center_data(self.x, self.data, centers, return_x=True)
        new_data = self.copy()
        new_data.x = new_x
        new_data.data = centered
        return new_data

    def mean(self, centers=None, axis=None):
        axis = axis if axis else 0
        if centers is not None:
            if axis != 0:
                raise NotImplementedError
            averaged, new_x, averaged_err = mean_data(self.x, self.data, centers, return_x=True, return_std=True)
        else:
            averaged, averaged_err = np.mean(self.data, axis=axis), np.std(self.data, axis=axis)
            if self.data.ndim == 2:
                if axis == 0:
                    new_x = self.x
                elif axis in [1, -1]:
                    new_x = self.y
                else:
                    raise ValueError(f'axis {axis} not valid/implemented')
            elif self.data.ndim == 1 and axis == 0:
                # Only return average value (not really Data anymore)
                return averaged
            else:
                raise ValueError(f'axis {axis} not valid/implemented')
        new_data = self.copy()
        new_data.x = new_x
        new_data.y = None
        new_data.plot_info.y_label = new_data.plot_info.coloraxis_title
        new_data.plot_info.title = f"{new_data.plot_info.title} Averaged in axis {axis}{' after aligning' if centers is not None else ''}"
        new_data.data = averaged
        new_data.yerr = averaged_err
        return new_data

    def __add__(self, other: Data) -> Data:
        return self.add(other)

    def __sub__(self, other: Data) -> Data:
        return self.subtract(other)

    def subtract(self, other_data: Data) -> Data:
        new_data = self.copy()
        if self.data.ndim == 1:
            new_data.x, new_data.data = subtract_data_1d(self.x, self.data, other_data.x, other_data.data)
        elif self.data.ndim == 2:
            new_data.x, new_data.y, new_data.data = subtract_data(self.x, self.y, self.data, other_data.x, other_data.y,
                                                                  other_data.data)
        else:
            raise NotImplementedError(f"Subtracting for data with ndim == {self.data.ndim} not implemented")
        return new_data

    def add(self, other_data: Data) -> Data:
        od = other_data.copy()
        od.data = -1 * od.data
        return self.subtract(od)

    def diff(self, axis=-1) -> Data:
        """Differentiate Data long specified axis

        Note: size reduced by 1 in differentiation axis
        """
        data = self.copy()
        diff = np.diff(self.data, axis=axis)
        data.data = diff
        data.x = get_matching_x(self.x, shape_to_match=diff.shape[-1])
        if self.y is not None:
            data.y = get_matching_x(self.y, shape_to_match=diff.shape[0])
        else:
            data.y = None
        return data

    def smooth(self, axis=-1, window_length=10, polyorder=3) -> Data:
        """Smooth data using method savgol_filter"""
        data = self.copy()
        data.data = savgol_filter(self.data, window_length, polyorder)
        data.plot_info.title = f'{data.plot_info.title} Smoothed ({window_length})'
        return data

    def decimate(
            self,
            decimate_factor: int = None,
            numpnts: int = None,
            measure_freq: float = None,
            desired_freq: float = None,
    ):
        """Decimate data (i.e. lowpass then downsample)
        Note: Use either decimate_factor, numpnts, or (measure_freq and desired_freq)
        """
        data = self.copy()
        data.data = decimate(
            self.data.astype(float),
            measure_freq=measure_freq,
            desired_freq=desired_freq,
            decimate_factor=decimate_factor,
            numpnts=numpnts,
        )
        data.yerr = None  # I don't think this makes sense after decimating
        data.x = get_matching_x(self.x.astype(float), shape_to_match=data.data.shape[-1])
        data.plot_info.title = f'{data.plot_info.title} Decimated ({len(data.x)} points)'
        return data

    def notch_filter(
            self,
            notch_freq: Union[float, list[float]],
            Q: float,
            measure_freq: float,
            fill_nan_values: float = None,
    ):
        notch_freqs = ensure_list(notch_freq)
        data = self.copy()
        if np.sum(np.isnan(data.data)) > 0:
            if fill_nan_values is None:
                raise ValueError(
                    f"Data must either contain no NaNs or `fill_nan_values` must be provided"
                )
            data.data[np.isnan(data.data)] = fill_nan_values

        for notch_freq in notch_freqs:
            b, a = iirnotch(notch_freq, Q, fs=measure_freq)
            data.data = filtfilt(b, a, data.data)
        data.plot_info.title = f'{data.plot_info.title} Notch Filtered ({notch_freqs} Hz)'
        return data

    def bin(self, bin_x=1, bin_y=1) -> Data:
        """Bin data
        Args:
            bin_x: binsize for x-axis
            bin_y: binsize for y-axis
        """
        data = self.copy()
        data.data = bin_data(self.data, bin_x=bin_x, bin_y=bin_y)
        data.x = bin_data(self.x, bin_x=bin_x)
        if data.y is not None:
            data.y = bin_data(self.y, bin_x=bin_y)
        return data

    def __getitem__(self, key: tuple):
        """Allow slicing like a numpy array"""
        new_data = self.copy()

        new_data.data = self.data[key]
        if new_data.yerr is not None:
            new_data.yerr = self.yerr[key]

        if self.data.ndim == 2:
            if isinstance(key, tuple) and len(key) == 2:
                pass
            elif isinstance(key, slice):
                key = tuple([key, ...])
            elif isinstance(key, int):
                key = tuple([key, ...])
            else:
                raise NotImplementedError
            new_data.x = self.x[key[1]]
            new_data.y = self.y[key[0]]
        elif self.data.ndim == 1:
            new_data.x = self.x[key]
        else:
            raise NotImplementedError
        return new_data

    def slice_values(
            self,
            x_range: tuple[Optional[int], Optional[int]] = None,
            y_range: tuple[Optional[int], Optional[int]] = None,
    ):
        """Select data based on x and y axes"""

        def replace_nones_in_indexes(arr: np.ndarray, indexes: tuple):
            """Replace None with either 0 or last index value"""
            new_indexes = list(indexes)
            if indexes[0] is None:
                indexes[0] = 0
            if indexes[1] is None:
                indexes[1] = arr.shape[-1]
            return indexes

        x_slice = ...
        if x_range is not None:
            x_indexes = get_data_index(self.x, x_range)
            x_indexes = replace_nones_in_indexes(self.x, x_indexes)
            if None not in x_indexes:
                x_indexes = min(x_indexes), max(x_indexes)
            x_slice = slice(x_indexes[0], x_indexes[1] + 1)
        if self.data.ndim == 1:
            return self[x_slice]
        elif self.data.ndim == 2:
            y_slice = ...
            if y_range is not None:
                y_indexes = get_data_index(self.y, y_range)
                y_indexes = replace_nones_in_indexes(self.y, y_indexes)
                if None not in y_indexes:
                    y_indexes = min(y_indexes), max(y_indexes)
                y_slice = slice(y_indexes[0], y_indexes[1] + 1)
            return self[y_slice, x_slice]
        else:
            raise NotImplementedError

    def _ipython_display_(self):
        """Make this object act like a figure when calling display(data) or leaving at the end of a jupyter cell"""
        return self.plot()._ipython_display_()