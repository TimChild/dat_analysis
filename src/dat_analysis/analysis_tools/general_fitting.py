from __future__ import annotations

from dataclasses import dataclass, InitVar, field
from hashlib import md5
from typing import Union, Optional, Callable, Any, TYPE_CHECKING, Tuple
from deprecation import deprecated

import re
import h5py
import lmfit as lm
import numpy as np
import pandas as pd
import logging

from .. import core_util as CU, useful_functions as U
from ..hdf_util import (
    params_from_HDF,
    params_to_HDF,
    NotFoundInHdfError,
    HDFStoreableDataclass,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


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
            re.search(r"(?:reduced chi-square\s*=\s)(.*)", self.fit_report).groups()[0]
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
        self.model = lm.models.Model(
            self._get_func()
        )  #  TODO: Figure out a good way to do this or remove this (cannot pickle when model or func is stored as an attribute)
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
    func: Callable  # Or should I just use func name here?  # TODO: definitely BAD to store func as attribute (not pickleable)
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

        start_ind, end_ind = U.get_data_index(
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
