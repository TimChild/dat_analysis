from __future__ import annotations
from dataclasses import dataclass, InitVar, field
from hashlib import md5
from typing import Union, Optional, Callable, Any, TYPE_CHECKING, Tuple
from deprecation import deprecated
import plotly.graph_objects as go
import plotly.colors as pc
import abc
import re
import h5py
import lmfit as lm
import numpy as np
import pandas as pd
import logging

from .. import core_util as CU
from ..core_util import _NOT_SET
from ..hdf_util import (
    params_from_HDF,
    params_to_HDF,
    NotFoundInHdfError,
    HDFStoreableDataclass,
)
from .data import Data, PlottingInfo

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def are_params_equal(params1: lm.Parameters, params2: lm.Parameters):
    """Check if two sets of lm.Parameters are equal to each other
    Note: The default param1 == param2, behaviour only checks names and values (not vary, expr etc)

    Checks:
        - name
        - value
        - min
        - max
        - expr
    Does not check:
        - stderr
        - Anything else not listed above
    """
    if params1 != params2:
        return False
    for par in params1.keys():
        p1 = params1[par]
        p2 = params2[par]
        for attr in ["vary", "expr", "min", "max"]:
            if getattr(p1, attr) != getattr(p2, attr):
                return False
    return True


@dataclass
class FitResult(Data):
    # Note: Don't actually store the fit, it will prevent pickling data
    fit: InitVar[lm.model.ModelResult] = None
    params: lm.Parameters = field(init=False)
    best_fit: np.ndarray = field(init=False)
    init_fit: np.ndarray = field(init=False)
    fit_x: np.ndarray = field(init=False)

    def __post_init__(self, fit=None):
        super().__post_init__()
        if fit:
            self.params = fit.params  # Params are pickleable
            self.fit_report = fit.fit_report()
            self.best_fit = fit.best_fit
            self.init_fit = fit.init_fit
            self.fit_x = fit.userkws.get(
                "x", None
            )  # best_fit and init_fit are data only

    @classmethod
    def from_fit(cls, data: Data, fit: lm.model.ModelResult) -> FitResult:
        inst = cls(**data.__dict__, fit=fit)
        return inst

    def plot(self, *args, plot_init=False, **kwargs):
        fig = super().plot(*args, **kwargs)
        fig.add_trace(
            go.Scatter(x=self.fit_x, y=self.best_fit, name="fit", mode="lines")
        )
        if plot_init:
            fig.add_trace(
                go.Scatter(x=self.fit_x, y=self.init_fit, name="init", mode="lines")
            )
        amp = np.nanmax(self.data) - np.nanmin(self.data)
        fig.update_layout(
            yaxis_range=[
                np.nanmin(self.data) - 0.2 * amp,
                np.nanmax(self.data) + 0.2 * amp,
            ]
        )
        return fig


@dataclass
class SimultaneousFitResult:
    """Note: This is different enough from Data and FitResult that it makes sense to be its own class"""

    datas: list[Data]
    # Note: Don't actually store the fit, it will prevent pickling data
    fit: InitVar[lm.model.ModelResult]
    plot_info: Optional[PlottingInfo] = None
    params: lm.Parameters = field(init=False)
    chisqr: float = field(init=False)
    redchi: float = field(init=False)

    def __post_init__(self, fit):
        if not np.all(d.data.ndim == 1 for d in self.datas):
            logging.warning(
                f"Support for non-1D data not implemented, use at your own risk"
            )
        self.params = fit.params  # Params are pickleable
        self.chisqr = fit.chisqr
        self.redchi = fit.redchi
        if self.plot_info is None:
            self.plot_info = PlottingInfo(
                title="Datas that were Fit<br>(Note: only params are stored here, no way to plot fits)",
                x_label="",
                y_label="",
            )

    @property
    def individual_params(self) -> list[lm.Parameters]:
        """List of Parameters, one for each dataset fit"""
        return separate_simultaneous_params(self.params)

    def plot(
        self,
        plot_init: bool = False,
        waterfall: bool = False,
        waterfall_spacing: float = None,
    ) -> go.Figure:
        if waterfall and not waterfall_spacing:
            waterfall_spacing = 0.2 * np.nanmax([d.data for d in self.datas])

        fig = go.Figure()
        colors = pc.qualitative.D3
        for i, data in enumerate(self.datas):
            color = colors[i % len(colors)]
            traces = data.get_traces()
            data_trace = traces[0]  # Only the data, no errors
            data_trace.update(mode="markers", marker=dict(size=2, color=color))
            if waterfall:
                for t in traces:
                    t.y += waterfall_spacing
            fig.add_trace(data_trace)
        if self.plot_info:
            fig = self.plot_info.update_layout(fig)
        return fig

    def _repr_html_(self):
        return self.params._repr_html_()


def separate_simultaneous_params(params: lm.Parameters) -> list[lm.Parameters]:
    """
    Separate out the combined simultaneous fitting parameters back into the individual fitting parameters
    """
    all_keys = [k for k in params]
    param_nums = [int(re.search("_(\d+)", k).groups()[0]) for k in all_keys]
    max_num = max(param_nums)
    unique = set([re.search("(.+)_\d+", k).groups()[0] for k in all_keys])
    params_list = []
    for i in range(max_num + 1):
        pars = lm.Parameters()
        for k in unique:
            p = params[f"{k}_{i}"]
            vary = False if p.vary is False or p.expr else True
            pars.add(k, value=p.value, min=p.min, max=p.max, vary=vary)
            pars[k].stderr = p.stderr
        params_list.append(pars)
    return params_list


class GeneralFitter(abc.ABC):
    """
    A base class for fitting data to a function
    I.e. This provides a generally useful framework and useful functions for any individual fitting
    This should be subclassed for any specific type of fitting

    This is intended to remain pickleable (i.e. not actually hold onto the lmfit ModelResults or references to functions etc)
    """

    def __init__(self, data: Data):
        self.data = data

        # Caching variables (to make it very fast to ask for the same fit again)
        self._last_params: lm.Parameters = None
        # Note: ModelResult is not pickleable, FitResult is (so can be passed in dash)
        self._last_fit_result: FitResult = None

    @property
    def fit_result(self) -> FitResult:
        """Nice way to access the self._last_fit_result value (i.e. equivalent to a fit result)"""
        if self._last_fit_result:
            return self._last_fit_result
        else:
            raise RuntimeError(f"No FitResult exists yet, run a fit first")

    def __repr__(self):
        return f"FitToNRG(data={repr(self.data)})"

    def _ipython_display_(self):
        return self.plot_fit()._ipython_display_()

    def fit(
        self, params: Optional[lm.Parameters] = _NOT_SET, max_x_points=1000
    ) -> FitResult:
        """Do the fit, with optional parameters to have more control over max/mins/initial/vary etc
        Args:
            params: Optionally provide the params to use for fitting (fitter.make_params() to get initial guesses)
            max_x_points: If dimension in x is larger than this, the data will be binned before fitting (for speed)

        Returns:
            FitResult: Note, this is not the lmfit.model.ModelResult, but it has many of the same attributes
                (the lmfit.model.ModelResult is not pickleable, and that causes issues in Dash)
        """
        if params is _NOT_SET:
            params = self._last_params  # Even if None
        if not params:
            params = self.make_params()

        # Only do expensive calculation if necessary
        if self._last_fit_result is None or self._are_params_new(params):
            self._last_params = params.copy()

            # Collect data to fit
            x = self.data.x
            data = self.data.data
            model = self.model()

            # Resample if very large before fitting (faster fitting, minimal effect on fit)
            if max_x_points and len(x) > max_x_points:
                x, data = CU.resample_data(
                    data,
                    x,
                    max_num_pnts=max_x_points,
                    resample_method="bin",
                    resample_x_only=True,
                )

            # Fit to the function
            fit = model.fit(
                data.astype(np.float32),
                params,
                x=x.astype(np.float32),
                nan_policy="omit",
                method=self._default_fit_method(),
            )

            # Cache for quicker return if same params asked again
            self._last_fit_result = FitResult.from_fit(self.data, fit)
        return self._last_fit_result

    def _default_fit_method(self) -> str:
        """Override this to change the fitting method (anything lmfit method)"""
        return "leastsq"

    @classmethod
    def model(cls) -> lm.models.Model:
        """Override to return model for fitting
        Examples:
            return lm.models.Model(nrg.NRG_func_generator(which='i_sense')) + lm.models.Model(simple_quadratic)
        """
        raise NotImplementedError()

    def eval(self, x: np.ndarray, params=_NOT_SET) -> np.ndarray:
        """Evaluate the model for the x values provided, and optionally the provided params"""
        if params is _NOT_SET:
            params = self._last_params  # Even if None
        if not params:
            params = self.make_params()

        model = self.model()
        return model.eval(x=x, params=params)

    @abc.abstractmethod
    def make_params(self) -> lm.Parameters:
        """Override to return default params for fitting (using self.data: Data)"""
        raise NotImplementedError

    def plot_fit(
        self, params: Optional[lm.Parameters] = _NOT_SET, plot_init=False
    ) -> go.Figure:
        """Plot the fit result and data, and optionally the initial fit"""
        if params is not _NOT_SET and not are_params_equal(params, self._last_params):
            logging.warning(f"Plotting fit with different parameters to last fit")

        fit_data = self.fit(params=params)
        fig = fit_data.plot(plot_init=plot_init)
        return fig

    def _are_params_new(self, params: lm.Parameters) -> bool:
        """Checks if params are different to the last params used for fitting"""
        if params is None:
            params = self.make_params()
        last_is_none = self._last_params is None
        pars_not_equal = not are_params_equal(params, self._last_params)
        if last_is_none or pars_not_equal:
            return True
        return False


class GeneralSimultaneousFitter(abc.ABC):
    def __init__(
        self,
        # ftns: list[NewFitToNRG],
        datas: Union[list[Data], list[FitResult]],
    ):
        """Carry out fitting on multiple datasets simultaneously

        Args:
            datas: list of datas to fit simultaneously
                (if passing FitResult, the params there can be used for initial fits)
        """
        # Data to fit
        self.datas: Union[list[Data], list[FitResult]] = datas

        # Caching Params
        self._last_fit_result: SimultaneousFitResult = None
        self._last_params: lm.Parameters = None

    @property
    def fit_result(self) -> SimultaneousFitResult:
        """Nice way to access the self._last_fit_result value (i.e. equivalent to a fit result)"""
        if self._last_fit_result:
            return self._last_fit_result
        else:
            raise RuntimeError(f"No SimultaneousFitResult exists yet, run a fit first")

    def _func(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Override this with the function that should be evaluated to generate residuals
        Note: **kwargs will contain a dict[par_name: value]

        Examples:
            return nrg.nrg_func(x=x, **par_dict, data_name='i_sense')
        """
        raise NotImplementedError(
            "Need to implement self._func(...) for whatever you want to fit"
        )

    def fit(
        self,
        which: str = "charge",
        params: lm.Parameters = _NOT_SET,
    ) -> SimultaneousFitResult:
        """Fit the data simultaneosly"""
        params = self._figure_out_params(params)

        # Only do expensive calculation if necessary
        if self._last_fit_result is None or self._are_params_new(params):
            self._last_params = params

            # Collect Data to Fit
            xs, datas = self._collect_datas()

            # Do fit
            objective = self._get_objective_function()
            fit = lm.minimize(
                objective,
                params,
                method=self._default_fit_method(),
                args=(xs, datas),
                nan_policy="omit",
            )
            self._last_fit_result = SimultaneousFitResult(
                datas=self.datas, fit=fit, plot_info=None
            )
        return self._last_fit_result

    def _default_fit_method(self):
        return "powell"

    def eval_dataset(
        self, dataset_index: int, x: np.ndarray = None, params=None, initial=False
    ) -> Data:
        """Equivalent to `eval`, but the `datset_index` also needs to be supplied to know which params to be plotting for
        Args:
            initial: Evaluate the initial fit instead
        """
        if x is None:
            x = self._last_fit_result.datas[dataset_index].x
        if params is None:
            params = self._last_fit_result.individual_params[dataset_index]
        par_dict = (
            params.valuesdict()
            if not initial
            else {k: p.init_value for k, p in params.items()}
        )
        data = self._func(x=x, **par_dict)
        return Data(
            x=x,
            data=data,
            plot_info=PlottingInfo(
                x_label="",
                y_label="",
                title=f"Dataset {dataset_index} Eval{' Initial' if initial else ''}",
            ),
        )

    def plot_fits(
        self,
        plot_init: bool = False,
        waterfall: bool = False,
        waterfall_spacing: float = None,
    ) -> go.Figure:
        fit_data = self._last_fit_result

        y_shift = 0  # For waterfall
        fig = go.Figure()
        colors = pc.qualitative.D3
        for i, data in enumerate(self.datas):
            traces = []
            color = colors[i % len(colors)]
            data_trace = data.get_traces()[0]  # Only the data, no errors
            data_trace.update(
                mode="markers", name=f"data {i}", marker=dict(size=2, color=color)
            )
            traces.append(data_trace)

            fit_trace = self.eval_dataset(i).get_traces(max_num_pnts=500)[0]
            fit_trace.update(
                mode="lines",
                name=f"fit {i}",
                legendgroup=data_trace.legendgroup,
                line=dict(color=color, width=2),
            )
            traces.append(fit_trace)
            if plot_init:
                init_trace = self.eval_dataset(i, initial=True).get_traces(
                    max_num_pnts=500
                )[0]
                init_trace.update(
                    mode="lines",
                    name=f"init {i}",
                    legendgroup=data_trace.legendgroup,
                    opacity=0.6,
                    line=dict(color=color, width=2),
                )
                traces.append(init_trace)
            if waterfall:
                y_shift += (
                    waterfall_spacing
                    if waterfall_spacing
                    else 0.2 * (np.nanmax(traces[0].y) - np.nanmin(traces[0].y))
                )
                for t in traces:
                    t.update(y=t.y + y_shift)
            fig.add_traces(traces)
        return fig

    def make_params(self) -> lm.Parameters:
        params = self._make_per_dataset_params()
        combined_params = self.combine_params(params)
        return combined_params.copy()

    def update_all_param(
        self,
        params: lm.Parameters,
        param_name: str,
        value=_NOT_SET,
        vary=_NOT_SET,
        min=_NOT_SET,
        max=_NOT_SET,
    ) -> lm.Parameters:
        """Update all params that are <param_name>_i"""
        for i in range(len(self.datas)):
            for k, v in zip(["value", "vary", "min", "max"], [value, vary, min, max]):
                if v is not _NOT_SET:
                    setattr(params[f"{param_name}_{i}"], k, v)
        return params

    def make_param_shared(
        self, params: lm.Parameters, share_params: Union[str, list[str]]
    ):
        """
        Make `share_param` be shared between all simultaneous fits (i.e. vary together)
        Note: this can easily be achieved (as well as more complex relations) by modifying param.expr directly
        """
        share_params = CU.ensure_list(share_params)
        for i in range(1, len(self.datas)):
            for share_param in share_params:
                params[f"{share_param}_{i}"].expr = f"{share_param}_0"
        return params

    def combine_params(self, params_list: list[lm.Parameters]) -> lm.Parameters:
        """Combine list of lm.Parameters into one lm.Parameters with unique names"""
        combined_params = lm.Parameters()
        for i, pars in enumerate(params_list):
            for par_name in pars.keys():
                combined_params[f"{par_name}_{i}"] = pars[par_name]
        return combined_params

    def _make_per_dataset_params(self) -> list[lm.Parameters]:
        """Make a single Parameters object containing fitting parameters for each dataset"""
        params_list = []
        for i, data in enumerate(self.datas):
            # Get initial pars
            pars = self._get_initial_dataset_params(data)

            # Add to the list of params for each dataset
            params_list.append(pars)
        return params_list

    def _figure_out_params(self, params: lm.Parameters) -> lm.Parameters:
        """Some logic about figuring out which params to use for fitting
        I.e. Use params provided if provided, else default
        """
        if params is _NOT_SET:
            params = self._last_params  # Even if None

        # If not passed, and last_params is None
        if not params:
            params = self.make_params()
        return params

    def _are_params_new(self, params: lm.Parameters) -> bool:
        if params is None:
            params = self.make_params()
        last_is_none = self._last_params is None
        pars_not_equal = not are_params_equal(params, self._last_params)
        if last_is_none or pars_not_equal:
            return True
        return False

    def _func_dataset(self, params: lm.Parameters, i: int, x: np.ndarray):
        # Collect params for this dataset (and strip off _<index> for the names
        pars = {
            re.match("(.+)_\d+", k).groups()[0]: v
            for k, v in params.items()
            if int(re.search("_(\d+)", k).groups()[0]) == i
        }
        return self._func(x=x, **pars)

    def _get_objective_function(self):
        """Objective function to minimize for all data in one go"""

        def objective(params, xs, datas):
            resids = []
            for i, (row, x) in enumerate(zip(datas, xs)):
                resids.extend(row - self._func_dataset(params, i, x))

            # 1D resids required for lm.minimize
            return resids

        return objective

    def _collect_datas(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Collect the xs and datas from each of the FitToNRG instances"""
        xs = []
        datas = []
        for data in self.datas:
            if data.x.shape[0] > 1000:
                data = data.decimate(numpnts=1000)
            xs.append(data.x.astype(np.float32))
            datas.append(data.data.astype(np.float32))
        return xs, datas

    def _separate_combined_params(
        self, combined_params: lm.Parameters
    ) -> list[lm.Parameters]:
        """Separate out the combined simultaneous fitting parameters back into the individual fitting parameters
        Note: Potentially a better way to do this is:
            - copy the full params
            - remove unwanted params
            - change name (remove _xx) from desired params
            This would keep any other useful things associated with each param that I have not considered below
        """
        params_list = separate_simultaneous_params(combined_params)
        return params_list

    def _get_initial_dataset_params(self, data: Union[Data, FitResult]):
        # if isinstance(data, FitResult) and data.params:  # TODO: switch back to this, only doesn't work with autoreload
        if hasattr(data, "params") and data.params:
            return data.params.copy()
        else:
            raise NotImplementedError(
                f"Either pass FitResult which has params or override _get_initial_dataset_params(...) to add a way to guess params from data"
            )


################## Pre 3.2.0 #############################


class Values(object):
    """Object to store Init/Best values in and stores Keys of those values in self.keys"""

    def __init__(self):
        self.keys = []

    def __getattr__(self, item):
        if (
            item.startswith("__") or item.startswith("_") or item == "keys"
        ):  # So don't complain about things like __len__
            return super().__getattribute__(
                item
            )  # Come's here looking for Ipython variables
        else:
            if item in self.keys:
                return super().__getattribute__(item)
            else:
                msg = f"{item} does not exist. Valid keys are {self.keys}"
                print(msg)
                logger.warning(msg)
                return None

    def get(self, item, default=None):
        if item in self.keys:
            val = self.__getattr__(item)
        else:
            val = default
        if default is not None and val is None:
            return default
        return val

    def __setattr__(self, key, value):
        if (
            key.startswith("__")
            or key.startswith("_")
            or key == "keys"
            or not isinstance(value, (np.number, float, int, type(None)))
        ):  # So don't complain about
            # things like __len__ and don't keep key of random things attached to class
            super().__setattr__(key, value)
        else:  # probably is something I want the key of
            self.keys.append(key)
            super().__setattr__(key, value)

    def __repr__(self):
        string = ""
        for key in self.keys:
            v = getattr(self, key)
            if v is not None:
                string += f"{key}={self.__getattr__(key):.5g}\n"
            else:
                string += f"{key}=None\n"
        return string

    def to_df(self):
        df = pd.DataFrame(
            data=[[self.get(k) for k in self.keys]], columns=[k for k in self.keys]
        )
        return df


@dataclass
class FitInfo(HDFStoreableDataclass):
    params: Union[lm.Parameters, None] = None
    init_params: lm.Parameters = None
    func_name: Union[str, None] = None
    fit_report: Union[str, None] = None
    model: Union[lm.Model, None] = None
    best_values: Union[Values, None] = None
    init_values: Union[Values, None] = None
    success: bool = None
    hash: Optional[int] = None

    # Will only exist when set from fit, or after recalculate_fit
    fit_result: Union[lm.model.ModelResult, None] = None

    @property
    def reduced_chi_sq(self):
        return float(
            re.search(r"reduced chi-square\s*=\s(.*)", self.fit_report).groups()[0]
        )

    def init_from_fit(self, fit: lm.model.ModelResult, hash_: Optional[int] = None):
        """Init values from fit result"""
        if fit is None:
            logger.warning(f"Got None for fit to initialize from. Not doing anything.")
            return None
        assert isinstance(fit, lm.model.ModelResult)
        self.params = fit.params
        self.init_params = fit.init_params
        self.func_name = fit.model.func.__name__
        self.fit_report = fit.fit_report()
        self.success = fit.success
        self.model = fit.model
        self.best_values = Values()
        self.init_values = Values()
        for key in self.params.keys():
            par = self.params[key]
            self.best_values.__setattr__(par.name, par.value)
            self.init_values.__setattr__(par.name, par.init_value)
        self.hash = hash_
        self.fit_result = fit

    def init_from_hdf(self, group: h5py.Group):
        """Init values from HDF file"""
        self.params = params_from_HDF(group)
        self.init_params = params_from_HDF(group.get("init_params"), initial=True)
        self.func_name = group.attrs.get("func_name", None)
        self.fit_report = group.attrs.get("fit_report", None)
        # self.model = lm.models.Model(
        #     self._get_func()
        # )  # TODO: Figure out a good way to do this or remove this (cannot pickle when model or func is stored as an
        self.model = None  # I don't think it's a good idea to rely on this
        # attribute)
        self.success = group.attrs.get("success", None)
        self.best_values = Values()
        self.init_values = Values()
        for key in self.params.keys():
            par = self.params[key]
            self.best_values.__setattr__(par.name, par.value)
            self.init_values.__setattr__(par.name, par.init_value)

        temp_hash = group.attrs.get("hash")
        if temp_hash is not None:
            self.hash = int(temp_hash)
        else:
            self.hash = None
        self.fit_result = None

    def save_to_hdf(self, parent_group: h5py.Group, name: Optional[str] = None):
        if name is None:
            name = self._default_name()
        parent_group = parent_group.require_group(name)

        if self.params is None:
            logger.warning(
                f"No params to save for {self.func_name} fit. Not doing anything"
            )
            return None
        params_to_HDF(self.params, parent_group)
        params_to_HDF(self.init_params, parent_group.require_group("init_params"))
        parent_group.attrs[
            "description"
        ] = "FitInfo"  # Overwrites what params_to_HDF sets
        parent_group.attrs["func_name"] = self.func_name
        parent_group.attrs["fit_report"] = self.fit_report
        parent_group.attrs["success"] = self.success
        if self.hash is not None:
            parent_group.attrs["hash"] = int(self.hash)

    def eval_fit(self, x: np.ndarray):
        """Return best fit for x array using params"""
        return self.model.eval(self.params, x=x)

    def eval_init(self, x: np.ndarray):
        """Return init fit for x array using params"""
        init_pars = CU.edit_params(
            self.params,
            list(self.params.keys()),
            [par.init_value for par in self.params.values()],
        )
        return self.model.eval(init_pars, x=x)

    def recalculate_fit(
        self, x: np.ndarray, data: np.ndarray, auto_bin=False, min_bins=1000
    ):
        """Fit to data with x array and update self"""
        assert data.ndim == 1
        data, x = CU.remove_nans(data, x)
        if auto_bin is True and len(data) > min_bins:
            logger.info(
                f"Binning data of len {len(data)} into {min_bins} before fitting"
            )
            x, data = CU.old_bin_data([x, data], round(len(data) / min_bins))
        fit = self.model.fit(
            data.astype(np.float32), self.params, x=x, nan_policy="omit"
        )
        self.init_from_fit(fit, self.hash)

    def edit_params(
        self, param_names=None, values=None, varys=None, mins=None, maxs=None
    ):
        self.params = CU.edit_params(
            self.params, param_names, values, varys, mins, maxs
        )

    def to_df(self):
        val_df = self.best_values.to_df()
        val_df["success"] = self.success
        return val_df

    def __hash__(self):
        if self.hash is None:
            raise AttributeError(f"hash value stored as None so hashing not supported")
        return int(self.hash)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return hash(other) == hash(self)
        return False

    def __repr__(self):
        return self.fit_report

    def __getstate__(self):
        """For dumping to pickle"""
        return self.__dict__

    def __setstate__(self, state):
        """For loading from pickle"""
        self.__dict__.update(state)

    @classmethod
    def from_fit(cls, fit, hash_: Optional[int] = None):
        """Use FitIdentifier to generate hash (Should be done before binning data to be able to check if
        matches before doing expensive processing)"""
        inst = cls()
        inst.init_from_fit(fit, hash_)
        return inst

    @classmethod
    def from_hdf(cls, parent_group: h5py.Group, name: str = None):
        if name is None:
            name = cls._default_name()
        fg = parent_group.get(name)
        if fg is None:
            raise NotFoundInHdfError(f"{name} not found in {parent_group.name}")
        inst = cls()
        inst.init_from_hdf(fg)
        return inst


@deprecated(
    deprecated_in="3.2.0",
    details="This has not been used for a long time and should be thought about more carefully if implementing again",
)
@dataclass
class FitIdentifier:
    initial_params: lm.Parameters
    func: Callable  # Or should I just use func name here?  # TODO: definitely BAD to store func as attribute (not
    # pickleable)
    data: InitVar[np.ndarray]
    data_hash: str = field(init=False)

    def __post_init__(self, data: np.ndarray):
        assert isinstance(self.initial_params, lm.Parameters)
        self.data_hash = self._hash_data(data)

    @staticmethod
    def _hash_data(data: np.ndarray):
        if data.ndim == 1:
            data = data[
                ~np.isnan(data)
            ]  # Because fits omit NaN data so this will make data match the fit data.
        return md5(data.tobytes()).hexdigest()

    def __hash__(self):
        """The default hash of FitIdentifier which will allow comparison between instances
        Using hashlib hashes makes this deterministic rather than runtime specific, so can compare to saved values
        """
        pars_hash = self._hash_params()
        func_hash = self._hash_func()
        data_hash = self.data_hash
        h = md5(pars_hash.encode())
        h.update(func_hash.encode())
        h.update(data_hash.encode())
        return int(h.hexdigest(), 16)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if hash(self) == hash(other):
                return True
        return False

    def _hash_params(self) -> str:
        h = md5(str(sorted(self.initial_params.valuesdict().items())).encode())
        return h.hexdigest()

    def _hash_func(self) -> str:
        # hash(self.func)   # Works pretty well, but if function is executed later even if it does the same thing it
        # will change
        h = md5(str(self.func.__name__).encode())
        return h.hexdigest()

    def generate_name(self):
        """Will be something reproducible and easy to read. Note: not totally guaranteed to be unique."""
        return str(hash(self))[0:5]


def calculate_fit(
    x: np.ndarray,
    data: np.ndarray,
    params: lm.Parameters,
    func: Union[Callable, lm.Model],
    auto_bin=True,
    min_bins=1000,
    generate_hash=False,
    warning_id: Optional[str] = None,
    method: str = "leastsq",
) -> FitInfo:
    """
    Calculates fit on data (Note: assumes that 'x' is the independent variable in fit_func)
    Args:
        x (np.ndarray): x_array (Note: fit_func should have variable with name 'x')
        data (np.ndarray): Data to fit
        params (lm.Parameters): Initial parameters for fit
        func (Callable): Function or lm.Model to fit to
        auto_bin (bool): if True will bin data into >= min_bins
        min_bins: How many bins to use for binning (actual num will lie between min_bins >= actual > min_bins*1.5)
        generate_hash: Whether to hash the data and fit params for comparison in future
        warning_id: String to use warning messages if fits aren't completely successful

    Returns:
        (FitInfo): FitInfo instance (with FitInfo.fit_result filled)
    """

    def sanitize_params(pars: lm.Parameters) -> lm.Parameters:
        # for par in pars:
        #     pars[par].value = np.float32(pars[par].value)  # VERY infrequently causes issues for calculating
        #     # uncertainties with np.float64 dtype
        return pars

    # Create lm.model.Model if necessary
    model = lm.model.Model(func) if not isinstance(func, lm.Model) else func

    if generate_hash:
        hash_ = hash(
            FitIdentifier(params, func, data)
        )  # Needs to be done BEFORE binning data.
    else:
        hash_ = None

    if (
        auto_bin and data.shape[-1] > min_bins * 2
    ):  # between 1-2x min_bins won't actually end up binning
        bin_size = int(
            np.floor(data.shape[-1] / min_bins)
        )  # Will end up with >= self.AUTO_BIN_SIZE pts
        x, data = [CU.bin_data(arr, bin_x=bin_size) for arr in [x, data]]

    params = sanitize_params(params)
    try:
        fit = FitInfo.from_fit(
            model.fit(
                data.astype(np.float32),
                params,
                x=x.astype(np.float32),
                nan_policy="omit",
                method=method,
            ),
            hash_,
        )
        if (
            fit.fit_result.covar is None and fit.success is True
        ):  # Failed to calculate uncertainties even though fit
            # was successful
            logger.debug(f"{warning_id}: Fit successful but uncertainties failed")
        elif fit.success is False:
            logger.warning(f"{warning_id}: A fit failed")
    except TypeError as e:
        logger.error(f"{e} while fitting {warning_id}")
        fit = None
    return fit


def get_data_in_range(
    x: np.ndarray,
    data: np.ndarray,
    width: Optional[float],
    center: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if center is None:
        center = 0
    if width is not None:
        x, data = np.copy(x), np.copy(data)

        start_ind, end_ind = CU.get_data_index(
            x, [center - width, center + width], is_sorted=True
        )

        x[:start_ind] = np.nan
        x[end_ind + 1 :] = np.nan

        data[:start_ind] = np.nan
        data[end_ind + 1 :] = np.nan
    return x, data


@deprecated(
    deprecated_in="3.2.0",
    details="Should be part of a subclass of general Fitting class",
)
@dataclass(frozen=True)
class PlaneParams:
    nx: float
    ny: float
    const: float


@deprecated(
    deprecated_in="3.2.0",
    details="Should be part of a subclass of general Fitting class",
)
@dataclass(frozen=True)
class PlaneFit:
    params: PlaneParams

    def eval(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.params.nx * x + self.params.ny * y[:, None] + self.params.const


@deprecated(
    deprecated_in="3.2.0",
    details="Should be part of a subclass of general Fitting class",
)
def plane_fit(x, y, data) -> PlaneFit:
    xx, yy = np.meshgrid(x, y)

    x, y, data = xx.flatten(), yy.flatten(), data.flatten()

    # do fit
    tmp_A = []
    tmp_b = []
    for i in range(len(x)):
        tmp_A.append([x[i], y[i], 1])
        tmp_b.append(data[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)

    # Manual solution
    fit = (A.T * A).I * A.T * b
    errors = b - A * fit
    residual = np.linalg.norm(errors)

    # Or use Scipy
    # from scipy.linalg import lstsq
    # fit, residual, rnk, s = lstsq(A, b)

    # print("solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
    # print("errors: \n", errors)
    # print("residual:", residual)
    vals = np.array(fit[:, 0])
    plane = PlaneFit(PlaneParams(nx=vals[0], ny=vals[1], const=vals[2]))
    return plane
