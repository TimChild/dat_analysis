from __future__ import annotations

import os.path
from dataclasses import dataclass, field
from xmlrpc.client import Boolean
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from scipy.signal import periodogram
import copy
import uuid
from scipy.signal import filtfilt, iirnotch
import logging
from typing import Union, Optional, TYPE_CHECKING, TypeVar
import warnings

from dat_analysis.plotting.plotly.util import (
    default_fig,
    heatmap,
    error_fill,
    figures_to_subplots,
)
from dat_analysis.core_util import (
    get_data_index,
    get_matching_x,
    bin_data,
    decimate,
    center_data,
    mean_data,
    resample_data,
    ensure_list,
)
from dat_analysis.general_io import save_to_txt, save_to_mat, save_to_igor_itx
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

    def __post_init__(self):
        if self.datnum is not None:
            warnings.warn(
                f"Use of datnum in PlottingInfo is deprecated. PlotInfo should be more general than being tied to a specific dat in any way."
            )

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

    def copy(self):
        return copy.deepcopy(self)


D = TypeVar("D", bound="Data")


@dataclass
class Data:
    data: np.ndarray = None
    x: np.ndarray = None
    y: np.ndarray = None
    xerr: np.ndarray = None
    yerr: np.ndarray = None
    plot_info: PlottingInfo = field(default_factory=PlottingInfo)

    def __post_init__(self, *args, **kwargs):
        # If data is 2D and no y provided, just assign index values
        self.x = np.asanyarray(self.x) if self.x is not None else self.x
        self.y = np.asanyarray(self.y) if self.y is not None else self.y
        if self.data is not None:
            self.data = np.asanyarray(self.data)
            if self.x is None:
                self.x = np.arange(self.data.shape[-1])
            if self.data.ndim == 2 and self.y is None:
                self.y = np.arange(self.data.shape[0])
        if self.plot_info is None:
            self.plot_info = PlottingInfo()

        if args:
            logging.warning(f"Data got unexpected __post_init__ args {args}")
        if kwargs:
            logging.warning(
                f"Data got unexpected __post_init__ kwargs {kwargs}")

    def plot(self, resample=True, **trace_kwargs):
        """
        Generate a quick 1D or 2D plot of data
        Args:
            resample: Whether to do automatic downsampling before plotting (very useful for large 2D datasets)
        """
        fig = default_fig()
        if self.data.ndim == 1:
            fig.add_traces(self.get_traces(**trace_kwargs))
        elif self.data.ndim == 2:
            fig.add_trace(
                heatmap(
                    x=self.x,
                    y=self.y,
                    data=self.data,
                    resample=resample,
                    **trace_kwargs,
                )
            )
        else:
            raise RuntimeError(
                f"data is not 1D or 2D, got data.shape = {self.data.ndim}"
            )
        if self.plot_info:
            fig = self.plot_info.update_layout(fig)
        return fig

    def get_traces(
        self, max_num_pnts=10000, error_fill_threshold=50, **first_trace_kwargs
    ) -> list[Union[go.Scatter, go.Heatmap]]:
        traces = []
        group_key = first_trace_kwargs.pop("legendgroup", uuid.uuid4().hex)
        if self.data.ndim == 1:
            if max_num_pnts:
                data, x = resample_data(
                    self.data,
                    x=self.x,
                    max_num_pnts=max_num_pnts,
                    resample_method="downsample",
                    resample_x_only=True,
                )
            # Note: Scattergl might cause issues with lots of plots, if so just go back to go.Scatter only
            # scatter_func = go.Scatter if x.shape[0] < 1000 else go.Scattergl
            scatter_func = go.Scatter
            trace = scatter_func(
                x=x, y=data, legendgroup=group_key, **first_trace_kwargs
            )
            traces.append(trace)
            if self.yerr is not None:
                yerr = resample_data(
                    self.yerr,
                    max_num_pnts=max_num_pnts,
                    resample_method="downsample",
                    resample_x_only=True,
                )
                if len(x) <= error_fill_threshold:
                    trace.update(error_y=dict(array=yerr))
                else:
                    # Note: error_fill also switched to scattergl after 1000 points
                    traces.append(error_fill(
                        x, data, yerr, legendgroup=group_key))
            if self.xerr is not None:
                xerr = resample_data(
                    self.xerr,
                    max_num_pnts=max_num_pnts,
                    resample_method="downsample",
                    resample_x_only=True,
                )
                if len(x) <= error_fill_threshold:
                    trace.update(error_x=dict(array=xerr),
                                 legendgroup=group_key)
                else:
                    pass  # Too many to plot, don't do anything
        elif self.data.ndim == 2:
            traces.append(
                heatmap(x=self.x, y=self.y, data=self.data,
                        **first_trace_kwargs)
            )
        else:
            raise RuntimeError(
                f"data is not 1D or 2D, got data.shape = {self.data.ndim}"
            )
        return traces

    def copy(self) -> D:
        """Make a copy of self s.t. changing the copy does not affect the original data"""
        return copy.deepcopy(self)

    def center(self, centers) -> D:
        centered, new_x = center_data(
            self.x, self.data, centers, return_x=True)
        new_data = self.copy()
        new_data.x = new_x
        new_data.data = centered
        return new_data

    def mean(self, centers=None, axis=None) -> D:
        axis = axis if axis else 0
        if centers is not None:
            if axis != 0:
                raise NotImplementedError
            averaged, new_x, averaged_err = mean_data(
                self.x, self.data, centers, return_x=True, return_std=True
            )
        else:
            averaged, averaged_err = np.mean(self.data, axis=axis), np.std(
                self.data, axis=axis
            )
            if self.data.ndim == 2:
                if axis == 0:
                    new_x = self.x
                elif axis in [1, -1]:
                    new_x = self.y
                else:
                    raise ValueError(f"axis {axis} not valid/implemented")
            elif self.data.ndim == 1 and axis == 0:
                # Only return average value (not really Data anymore)
                return averaged
            else:
                raise ValueError(f"axis {axis} not valid/implemented")
        new_data = self.copy()
        new_data.x = new_x
        new_data.y = None
        new_data.plot_info.y_label = new_data.plot_info.coloraxis_title
        new_data.plot_info.title = f"{new_data.plot_info.title} Averaged in axis {axis}{' after aligning' if centers is not None else ''}"
        new_data.data = averaged
        new_data.yerr = averaged_err
        return new_data

    def __add__(self, other: D) -> D:
        return self.add(other)

    def __sub__(self, other: D) -> D:
        return self.subtract(other)

    def subtract(self, other_data: D) -> D:
        new_data = self.copy()
        if self.data.ndim == 1:
            new_data.x, new_data.data = subtract_data_1d(
                self.x, self.data, other_data.x, other_data.data
            )
        elif self.data.ndim == 2:
            new_data.x, new_data.y, new_data.data = subtract_data(
                self.x, self.y, self.data, other_data.x, other_data.y, other_data.data
            )
        else:
            raise NotImplementedError(
                f"Subtracting for data with ndim == {self.data.ndim} not implemented"
            )
        return new_data

    def add(self, other_data: D) -> D:
        od = other_data.copy()
        od.data = -1 * od.data
        return self.subtract(od)

    def diff(self, axis=-1) -> D:
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

    def smooth(self, axis=-1, window_length=10, polyorder=3) -> D:
        """Smooth data using method savgol_filter"""
        data = self.copy()
        data.data = savgol_filter(
            self.data, window_length, polyorder, axis=axis)
        data.plot_info.title = f"{data.plot_info.title} Smoothed ({window_length})"
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
        data.x = get_matching_x(
            self.x.astype(float), shape_to_match=data.data.shape[-1]
        )
        data.plot_info.title = (
            f"{data.plot_info.title} Decimated ({len(data.x)} points)"
        )
        return data

    def notch_filter(
        self,
        notch_freq: Union[float, list[float]],
        Q: Union[float, list[float]],
        measure_freq: float,
        fill_nan_values: float = None,
    ) -> D:
        notch_freqs = ensure_list(notch_freq)
        Qs = ensure_list(Q)
        data = self.copy()
        if np.sum(np.isnan(data.data)) > 0:
            if fill_nan_values is None:
                raise ValueError(
                    f"Data must either contain no NaNs or `fill_nan_values` must be provided"
                )
            data.data[np.isnan(data.data)] = fill_nan_values

        for notch_freq_index, notch_freq in enumerate(notch_freqs):
            Q_factor = Qs[0] if len(Qs) == 1 else Qs[notch_freq_index]
            b, a = iirnotch(notch_freq, Q_factor, fs=measure_freq)
            data.data = filtfilt(b, a, data.data)
        data.plot_info.title = (
            f"{data.plot_info.title} Notch Filtered ({notch_freqs} Hz)"
        )
        return data

    def bin(self, bin_x=1, bin_y=1) -> D:
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
        data.plot_info.title = (
            f"{data.plot_info.title} Binned (x={bin_x}, y={bin_y})"
        )
        return data

    def __getitem__(self, key: tuple):
        """Allow slicing like a numpy array"""
        new_data = self.copy()

        # Slice should always apply to data like normal
        new_data.data = self.data[key]

        # Different behavior for the rest based on ndim
        if self.data.ndim == 1:
            new_data.x = self.x[key]
            if new_data.yerr is not None:
                new_data.yerr = self.yerr[key]
            if new_data.xerr is not None:
                new_data.xerr = self.xerr[key]
        elif self.data.ndim == 2:
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
            if new_data.yerr is not None:
                if new_data.yerr.ndim == 1:
                    new_data.yerr = self.yerr[key[0]]
                else:
                    new_data.yerr = self.yerr[key]
            if new_data.xerr is not None:
                if new_data.xerr.ndim == 1:
                    new_data.xerr = self.xerr[key[1]]
                else:
                    new_data.xerr = self.xerr[key]
        else:
            raise NotImplementedError(
                f'Not implemented for data with ndim = {self.data.ndim}')
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

    def power_spectrum(
        self, measure_frequency, density=True, logx=True, logy=False, or_optional_kwargs=None
    ) -> PowerSpectrumData:
        """
        Calculates periodorgam based on measure frequency
        Parameters
        ----------
        density : Boolean, optional
            True returns power spectral density V^2/Hz
            False returns power spectrum V^2
        logx : Boolean, optional
            True scales x-axis of plot to log
        logy : Boolean, optional
            True scales f-axis of plot to log

        """

        def get_1d_power_spectrum(data, measure_frequency, density=density):
            """
            calculate periodogram on 1d set of data
            """
            if density:
                f, Pxx = periodogram(
                    data, measure_frequency, scaling="density")
            else:
                f, Pxx = periodogram(
                    data, measure_frequency, scaling="spectrum")

            integrated = np.cumsum(Pxx)

            return np.array(f), np.array(Pxx), np.array(integrated)

        # data = dat.Data.get_data(data_type)
        # scan_vars = dat.Logs.scan_vars
        # measure_freq = scan_vars["measureFreq"]
        new_data_x, new_data_pxx, new_data_integrated = [], [], []

        if self.data.ndim == 1:
            new_data_x, new_data_pxx, new_data_integrated = get_1d_power_spectrum(
                self.data, measure_frequency, density=density)

        else:
            for data_row in self.data:
                new_data_x, Pxx, integrated = get_1d_power_spectrum(
                    data_row, measure_frequency, density=density
                )
                new_data_pxx.append(Pxx)
                new_data_integrated.append(integrated)

        new_data = PowerSpectrumData(x=new_data_x, y=self.copy().y, data=new_data_pxx, integrated=new_data_integrated,
                                     plot_info=PlottingInfo(
                                         title=f'{self.plot_info.title} power spectrum'),
                                     density=density, logx=logx, logy=logy)
        return new_data
        # return np.array(f), np.array(Pxxs), np.array(integrateds)

        raise NotImplementedError
        # fs, power = calculate_power_spectrum(self.data, measure_freq, etc...)
        #
        # # Something like this for plot_info
        # new_plot_info = PlottingInfo(x_label='Frequency /Hz', y_label='power', title=self.plot_info.title+' power spectrum')
        #
        # # Returning a subclass of Data allows to override the plotting behavior
        # new_data = PowerSpectrumData(data=power, x=fs, y=self.y, plot_info=new_plot_info)
        # return new_data

    def save_to_txt(
        self, name_prefix: str = None, filepath: str = "data.txt", overwrite=False
    ):
        """Saves Data to .txt file via np.savetxt"""
        datas, names, filepath = self._prepare_for_saving(
            name_prefix=name_prefix, filepath=filepath, overwrite=overwrite
        )
        save_to_txt(datas, names, file_path=filepath)

    def save_to_mat(
        self, name_prefix: str = None, filepath: str = "data.mat", overwrite=False
    ):
        """Saves Data to .mat file"""
        datas, names, filepath = self._prepare_for_saving(
            name_prefix=name_prefix, filepath=filepath, overwrite=overwrite
        )
        save_to_mat(datas, names, file_path=filepath)

    def save_to_itx(
        self, name: str = "data", filepath: str = "data.itx", overwrite=False
    ):
        """Saves Data to .itx file"""
        if not overwrite and os.path.exists(filepath):
            raise FileExistsError(
                f"Already a file at {filepath}, set overwrite=True or change filepath to save"
            )

        # Note: this is more complicated than to .mat or .txt because Igor waves can store x, y, axis_labels with data
        save_to_igor_itx(
            file_path=filepath,
            xs=[self.x],
            datas=[self.data],
            names=[name],
            ys=[self.y],
            x_labels=[self.plot_info.x_label],
            y_labels=[self.plot_info.y_label],
        )

    def _prepare_for_saving(
        self, name_prefix: str = None, filepath: str = "data.txt", overwrite=False
    ) -> tuple[list[np.ndarray], list[str], str]:
        name_prefix = name_prefix if name_prefix else ""
        datas, names = [], []
        for attr in ["x", "y", "xerr", "yerr", "data"]:
            arr = getattr(self, attr)
            if arr is not None:
                datas.append(arr)
                names.append(
                    f"{name_prefix}{'_' if name_prefix else ''}{attr}")
        if not overwrite and os.path.exists(filepath):
            raise FileExistsError(
                f"Already a file at {filepath}, set overwrite=True or change filepath to save"
            )
        return datas, names, filepath

    def _ipython_display_(self):
        """Make this object act like a figure when calling display(data) or leaving at the end of a jupyter cell"""
        return self.plot()._ipython_display_()


@dataclass
class InterlacedData(Data):
    """E.g. Combining +/- conductance data or averaging the same bias CS data"""

    num_setpoints: int = None

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        if self.num_setpoints is None:
            raise ValueError(
                f"must specify `num_setpoints` for InterlacedData")

    @classmethod
    def from_Data(cls, data: D, num_setpoints: int) -> D:
        """Convert a regulare Data class to InterlacedData"""
        d = data.copy()
        if isinstance(d, cls):
            # Already InterlacedData, just update num_setpoints
            d.num_setpoints = num_setpoints
            return d
        inst = cls(**d.__dict__, num_setpoints=num_setpoints)
        return inst

    @classmethod
    def get_num_interlaced_setpoints(cls, scan_vars) -> int:
        """
        Helper function to not have to remember how to do this every time
        Returns:
            (int): number of y-interlace setpoints
        """
        if scan_vars.get("interlaced_y_flag", 0):
            num = len(scan_vars["interlaced_setpoints"].split(
                ";")[0].split(","))
        else:
            num = 1
        return num

    def separate_setpoints(self) -> list[Data]:
        new_y = np.linspace(
            self.y[0],
            self.y[-1],
            int(self.y.shape[0] / self.num_setpoints),
        )
        new_datas = []
        for i in range(self.num_setpoints):
            d_ = copy.deepcopy(self.__dict__)
            d_.pop("num_setpoints")
            new_data = Data(**d_)
            new_data.plot_info.title = f"Interlaced Data Setpoint {i}"
            new_data.y = new_y
            new_data.data = self.data[i:: self.num_setpoints]
            new_datas.append(new_data)
        return new_datas

    def combine_setpoints(
        self, setpoints: list[int], mode: str = "mean", centers=None
    ) -> D:
        """
        Combine separate parts of interlaced data by averaging or difference

        Args:
            setpoints: which interlaced setpoints to combine
            mode: `mean` or `difference`
        """
        if mode not in (modes := ["mean", "difference"]):
            raise NotImplementedError(
                f"{mode} not implemented, must be in {modes}")

        if centers is not None:
            data = self.center(centers)
        else:
            data = self
        datas = np.array(data.separate_setpoints())[list(setpoints)]
        new_data = datas[0].copy()
        if mode == "mean":
            new_data.data = np.nanmean([data.data for data in datas], axis=0)
        elif mode == "difference":
            if len(datas) != 2:
                raise ValueError(
                    f"Got {setpoints}, expected 2 exactly for mode `difference`"
                )
            new_data.data = datas[0].data - datas[1].data
        else:
            raise RuntimeError
        new_data.plot_info.title = f"Combined by {mode} of {setpoints}"
        return new_data

    def plot_separated(self, shared_data=False, resample=True) -> go.Figure:
        """
        Plot each part of interlaced data on an individual heatmap
        Args:
            shared_data (): Whether to share a colorscale between the plots
            resample (): Whether to resample down before plotting (large interactive plots slow down jupyter)

        Returns:
            figure with subplots for each interlaced setpoint
        """
        figs = []
        for i, d in enumerate(self.separate_setpoints()):
            fig = default_fig()
            fig.add_trace(heatmap(d.x, d.y, d.data, resample=resample))
            fig.update_layout(title=f"Interlace Setpoint: {i}")
            figs.append(fig)

        title = self.plot_info.title if self.plot_info.title else "Separated Data"
        fig = figures_to_subplots(
            figs,
            title=title,
            shared_data=shared_data,
        )
        if self.plot_info:
            fig = self.plot_info.update_layout(fig)
            fig.update_layout(
                title=f"{fig.layout.title.text} Interlaced Separated")
            # Note: Only updates xaxis1 by default, so update other axes
            fig.update_xaxes(title=self.plot_info.x_label)
            fig.update_yaxes(title=self.plot_info.y_label)
        return fig

    def center(self, centers) -> InterlacedData:
        """If passed a list of list of centers, flatten back to apply to whole dataset before calling super().center(
        ...)"""
        if len(centers) == self.num_setpoints:
            # Flatten back to a single center per row for the whole dataset
            centers = np.array(centers).flatten(order="F")
        return super().center(centers)

    def _ipython_display_(self):
        """Make this object act like a figure when calling display(data) or leaving at the end of a jupyter cell"""
        return self.plot_separated()._ipython_display_()


@dataclass
class PowerSpectrumData(Data):
    integrated: np.ndarray = None
    logx: Boolean = True
    logy: Boolean = False
    density: Boolean = True,

    # This overrides the .plot method of a regular Data object
    def plot(self, resample=False, **trace_kwargs):
        # You can start with the figure that the normal .plot method would give by doing this
        fig = super().plot(resample=resample, **trace_kwargs)

        y_title = "Power Spectral Density (nA^2/Hz)" if self.density else "Power Spectrum (nA^2)"
        y_title = f"log {y_title}" if self.logy else f"{y_title}"
        x_title = "log Frequency (Hz)" if self.logx else "Frequency (Hz)"

        fig.update_xaxes(type="log") if self.logx else fig.update_xaxes(
            type="linear")
        fig.update_yaxes(type="log") if self.logy else fig.update_yaxes(
            type="linear")

        if self.data.ndim == 1:
            fig.add_trace(go.Scatter(x=self.x, y=self.integrated, yaxis='y2'))
            y_title_integrated = "Cumulative Sum (nA^2)" if self.density else "Cumulative Sum (nA^2 Hz)"
            fig.update_layout(
                xaxis=dict(domain=(0, 0.9)),
                yaxis2=dict(title=f"{y_title_integrated}",
                            anchor="x",
                            overlaying="y",
                            side="right",
                            position=0.15,
                            showgrid=False,))

        fig.update_layout(
            yaxis_title=f"{y_title}",
            xaxis_title=f"{x_title}",)
        # And then edit/add to that figure here (or just make a figure from scratch if that seems better)

        return fig
