"""Functions which takes non dat specific data (i.e. from 1 or multiple dats) in order to return DC bias info
The info returned from here can be stored in dats if necessary, but this should not generally use dats as input
as the way data is recorded varies for this type of analysis.

DCbias is for calculating how much heating is being applied in entropy sensing measurements.
"""
from __future__ import annotations
import numpy as np
from src.analysis_tools.general_fitting import FitInfo
import lmfit as lm
from typing import Union, Tuple, List, Iterable, Optional, Dict, Any, TYPE_CHECKING
from src.hdf_util import DatDataclassTemplate
from dataclasses import dataclass, field

if TYPE_CHECKING:
    import h5py
    from src.dat_object.dat_hdf import DatHDF


def fit_quad(x: np.ndarray, thetas: np.ndarray, force_centered: Union[bool, float] = False) -> FitInfo:
    """
    Quadratic fit to thetas vs x.
    Args:
        x ():
        thetas ():
        force_centered (): Whether or not to force quadratic fit to be centered at 0 (or provided float)

    Returns:
        (FitInfo): HDF storable version version of lmfit.Result()
    """
    # lmfit built in model
    model = lm.models.QuadraticModel()

    # Get starting params
    params = model.guess(thetas, x)

    if force_centered:
        if force_centered is True:
            force_centered = 0
        params['b'].value = force_centered
        params['b'].vary = False

    fit = FitInfo.from_fit(model.fit(thetas, params=params, x=x, nan_policy='omit'))
    return fit


def dTs_from_fit(fit: FitInfo, bias: Union[float, np.ndarray, Iterable[float]]) -> Union[float, np.ndarray]:
    """Calculates the average difference in theta between minimum of quadratic and theta at bias(es)"""

    # Eval at min point
    min_ = fit.eval_fit(-fit.best_values.b / 2)

    # If passed only one value, want to return only one value
    ret_float = True if isinstance(bias, float) else False

    # Vectorized operation
    bias = np.asanyarray(bias)
    dTs = fit.eval_fit(bias) - min_

    if ret_float:
        return dTs[0]
    else:
        return dTs


@dataclass
class DCbiasInfo(DatDataclassTemplate):
    """Information about DCBias fit which is storable in DatHDF"""
    quad_vals: List[float]
    quad_fit: FitInfo = field(repr=False)
    x: np.ndarray = field(repr=False)
    thetas: np.ndarray = field(repr=False)

    @classmethod
    def from_data(cls, biases: Union[Iterable[float], np.ndarray],
                  thetas: Union[Iterable[float], np.ndarray],
                  force_centered: Union[bool, float] = False) -> DCbiasInfo:
        """
        Make DCbias info from data (i.e. biases and thetas)
        Args:
            biases (): Biases
            thetas (): Thetas
            force_centered (): Whether to force the minimum of the quad fit to be at zero bias or not

        Returns:
            Info about DCBias fit which is storeable in DatHDF
        """
        fit = fit_quad(biases, thetas, force_centered=force_centered)
        quad_vals = [fit.best_values.get(x) for x in ['a', 'b', 'c']]
        inst = cls(quad_vals=quad_vals, quad_fit=fit, x=biases, thetas=thetas)
        return inst

    @classmethod
    def from_dats(cls, dats: Iterable[DatHDF], bias_key: str,
                  force_centered: Union[bool, float] = False,
                  fit_name: Optional[str] = None) -> DCbiasInfo:
        """
        Helper for making DCbiasInfo from list of dats which make up a DCbias measurement. (i.e. if each dat is a repeat
        for a fixed bias).
        Args:
            dats (): List of DatHDFs which make up measurement
            bias_key (): FastDac Key of channel that was setting bias (e.g. 'R2T(10M)')
            force_centered (): Whether to force quad fit to be centered at 0 or given float value
            fit_name (): Optionally choose to use thetas from a named Transition fit

        Returns:
            (DCbiasInfo): Filled DCbiasInfo

        """
        x = [dat.Logs.dacs[bias_key] for dat in dats]
        if fit_name:
            z = [dat.Transition.get_fit(name=fit_name).best_values.theta for dat in dats]
        else:
            z = [dat.Transition.avg_fit.best_values.theta for dat in dats]
        return cls.from_data(x, z, force_centered=force_centered)

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


@dataclass
class HeatingInfo(DatDataclassTemplate):
    """DatHDF savable information about Heating (including DCBias info)"""
    dc_bias_info: DCbiasInfo
    biases: List[float]
    dTs: List[float]
    avg_bias: Union[float]
    avg_dT: Union[float]

    @classmethod
    def from_data(cls, dc_info: DCbiasInfo, bias: Union[float, Iterable[float]]) -> HeatingInfo:
        bias = list(np.asanyarray(bias))
        dTs = dTs_from_fit(dc_info.quad_fit, bias)

        inst = cls(dc_bias_info=dc_info,
                   biases=bias, avg_bias=float(np.nanmean(bias)), dTs=list(dTs), avg_dT=float(np.nanmean(dTs)))
        return inst

    @classmethod
    def from_dat(cls, dat: DatHDF, dcbias_info: DCbiasInfo) -> HeatingInfo:
        """
        Helper for making HeatingInfo from dat. Will assume that dat.SquareEntropy exists
        Args:
            dat (): Dat to get HeatingInfo for
            dcbias_info (): The relevant DCbiasInfo for this dat (i.e. DCbias taken at same fridge temp and conditions)

        Returns:
            (HeatingInfo): HeatingInfo for dat

        """
        biases = [dat.SquareEntropy.square_awg.AWs[0][0][i] for i in [1, 3]]
        return cls.from_data(dc_info=dcbias_info, bias=biases)

    # Below here is just for saving and loading to HDF
    @staticmethod
    def ignore_keys_for_hdf() -> Optional[Union[str, List[str]]]:
        return ['dc_bias_info']

    def additional_save_to_hdf(self, dc_group: h5py.Group):
        self.dc_bias_info.save_to_hdf(dc_group, 'dc_bias_info')

    @staticmethod
    def additional_load_from_hdf(dc_group: h5py.Group) -> Dict[str, Any]:
        return {'dc_bias_info': DCbiasInfo.from_hdf(dc_group, 'dc_bias_info')}


if __name__ == '__main__':
    from src.dat_object.make_dat import get_dats, get_dat

    dc_bias_dats = {
        100: tuple(range(4284, 4295)),
        50: tuple(range(8593, 8599))
    }

    dc_bias_keys = {
        100: 'R2T(10M)',
        50: 'R2T/0.001'
    }

    # Get dats used for DC bias
    all_dats = get_dats(dc_bias_dats[50])

    # Get the bias and theta for each dat (or could be from a single measurement)
    x = [dat.Logs.fds[dc_bias_keys[50]] for dat in all_dats]
    z = [dat.Transition.avg_fit.best_values.theta for dat in all_dats]

    # Initialize DCbiasInfo (which will fit the quadratic either with center forced at bias = 0 or not.
    dc_info = DCbiasInfo.from_data(x, z, force_centered=False)
    dc_info_centered = DCbiasInfo.from_data(x, z, force_centered=True)
    # print(dc_info.quad_fit)

    # Can use the DCBiasInfo to get HeatingInfo for a dat (DCbias should be for conditions of dat obviously)
    dat = get_dat(2197)  # Not the right dat, just for example purposes
    heating_info = HeatingInfo.from_dat(dat, dc_info)
    # print(heating_info.avg_dT)
