"""
Quadratic fitting and some HDF storeable dataclasses that are useful for recording the heating applied in entropy measurements.

I.e. mostly intended to be used to fit DC bias heating measurements where the heating is expected to be quadratic in
the small heating limit, where this fit result will be used to calculate the heat being applied in an entropy
measurement where the heating bias is varied rapidly. """
import numpy as np
from .general_fitting import FitInfo
import lmfit as lm
from typing import Union, Tuple, List, Iterable, Optional, Dict, Any, TYPE_CHECKING
from ..hdf_util import HDFStoreableDataclass
from dataclasses import dataclass, field
from deprecation import deprecated

if TYPE_CHECKING:
    import h5py


@deprecated(deprecated_in='3.2.0', details='Should be replaced by something that subclasses the fitting class.')
def fit_quadratic(x: np.ndarray, data: np.ndarray, force_centered: Union[bool, float] = False) -> FitInfo:
    """
    Quadratic fit to thetas vs x.

    Args:
        x ():
        data ():
        force_centered (): Whether to force quadratic fit to be centered at 0 (or provided value)

    Returns:
        (FitInfo): HDF storable version version of lmfit.Result()
    """
    # lmfit built in model
    model = lm.models.QuadraticModel()

    # Get starting params
    params = model.guess(data, x)

    if force_centered:
        if force_centered is True:
            force_centered = 0
        params['b'].value = force_centered
        params['b'].vary = False

    fit = FitInfo.from_fit(model.fit(data, params=params, x=x, nan_policy='omit'))
    return fit


@deprecated(deprecated_in='3.2.0', details='Possibly worth re-writing, or maybe this is actually OK to use (in which case remove the deprecation warning)')
def delta_from_min_of_quadratic(fit: FitInfo, x_val: Union[float, np.ndarray, Iterable[float]]) -> Union[float, np.ndarray]:
    """Calculates the average difference in value between minimum of quadratic and at x-values provided"""

    # Eval at min point
    min_ = fit.eval_fit(-fit.best_values.b / 2)

    # If passed only one value, want to return only one value
    ret_float = True if isinstance(x_val, float) else False

    # Vectorized operation
    x_val = np.asanyarray(x_val)
    deltas = fit.eval_fit(x_val) - min_

    if ret_float:
        return deltas[0]
    else:
        return deltas


@deprecated(deprecated_in='3.2.0', details='Not thought about this for a while, likely needs updating to be used again')
@dataclass
class QuadraticFitInfo(HDFStoreableDataclass):
    """
    HDF storable Quadratic Fit information (i.e. the data that was fit along with best fit values and fit result)

    Note: Should only include the data that is intended to be included in the quadratic fit

    E.g. Useful for calculating amount of heating being applied in an entropy measurement based on more careful DC bias measurements
    """
    quad_vals: List[float]
    quad_fit: FitInfo = field(repr=False)
    x: np.ndarray = field(repr=False)
    thetas: np.ndarray = field(repr=False)

    @classmethod
    def from_data(cls, biases: Union[Iterable[float], np.ndarray],
                  thetas: Union[Iterable[float], np.ndarray],
                  force_centered: Union[bool, float] = False):
        """
        Make DCbias info from data (i.e. biases and thetas)
        Args:
            biases (): Biases
            thetas (): Thetas
            force_centered (): Whether to force the minimum of the quad fit to be at zero bias or not

        Returns:
            Info about DCBias fit which is storeable in DatHDF
        """
        fit = fit_quadratic(biases, thetas, force_centered=force_centered)
        quad_vals = [fit.best_values.get(x) for x in ['a', 'b', 'c']]
        inst = cls(quad_vals=quad_vals, quad_fit=fit, x=biases, thetas=thetas)
        return inst

    # Below here is just for saving and loading to HDF
    @staticmethod
    def ignore_keys_for_hdf() -> Optional[Union[str, List[str]]]:
        return ['quad_fit']

    def additional_save_to_hdf(self, dc_group: h5py.Group):
        if self.quad_fit is not None:
            self.quad_fit.save_to_hdf(dc_group, 'quad_fit')

    @staticmethod
    def additional_load_from_hdf(dc_group: h5py.Group) -> Dict[str, Any]:
        ret = {}
        if 'quad_fit' in dc_group.keys():
            fit = FitInfo.from_hdf(dc_group, name='quad_fit')
            ret['quad_fit'] = fit
        return ret


@deprecated(deprecated_in='3.2.0', details='Not thought about this for a while, likely needs updating to be used again')
@dataclass
class HeatingInfo(HDFStoreableDataclass):
    """DatHDF savable information about Heating (including the QuadraticFitInfo used to calculate this)"""
    quadratic_fit_info: QuadraticFitInfo
    biases: List[float]
    dTs: List[float]
    avg_bias: Union[float]
    avg_dT: Union[float]

    @classmethod
    def from_data(cls, quadratic_fit_info: QuadraticFitInfo, bias: Union[float, Iterable[float]]):
        bias = list(np.asanyarray(bias))
        dTs = delta_from_min_of_quadratic(quadratic_fit_info.quad_fit, bias)
        inst = cls(quadratic_fit_info=quadratic_fit_info,
                   biases=bias, avg_bias=float(np.nanmean(bias)), dTs=list(dTs), avg_dT=float(np.nanmean(dTs)))
        return inst

    # Below here is just for saving and loading to HDF
    @staticmethod
    def ignore_keys_for_hdf() -> Optional[Union[str, List[str]]]:
        return ['dc_bias_info']

    def additional_save_to_hdf(self, dc_group: h5py.Group):
        self.quadratic_fit_info.save_to_hdf(dc_group, 'dc_bias_info')

    @staticmethod
    def additional_load_from_hdf(dc_group: h5py.Group) -> Dict[str, Any]:
        return {'dc_bias_info': QuadraticFitInfo.from_hdf(dc_group, 'dc_bias_info')}
