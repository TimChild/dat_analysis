"""
For calculating, fitting, and integrated Entropy measurements
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, TYPE_CHECKING, List, Dict, Any
import re
import logging

import lmfit as lm
import numpy as np
import pandas as pd

from .new_procedures import Process, DataPlotter, PlottableData
from .square_wave import get_transition_parts
from .general_fitting import calculate_fit, FitInfo
from .. import core_util as CU
from ..characters import DELTA

if TYPE_CHECKING:
    import h5py

logger = logging.getLogger(__name__)


FIT_NUM_BINS = 1000  # Much faster to bin data down to this size before fitting, and negligible impact on fit result


def entropy_nik_shape(x, mid, theta, const, dS, dT):
    """Weakly coupled single dot entropy shape"""
    arg = ((x - mid) / (2 * theta))
    return -dT * ((x - mid) / (2 * theta) - 0.5 * dS) * (np.cosh(arg)) ** (-2) + const


def get_param_estimates(x_array, data, mids=None, thetas=None) -> List[lm.Parameters]:
    if data.ndim == 1:
        return [_get_param_estimates_1d(x_array, data, mids, thetas)]
    elif data.ndim == 2:
        mids = mids if mids is not None else [None] * data.shape[0]
        thetas = thetas if thetas is not None else [None] * data.shape[0]
        return [_get_param_estimates_1d(x_array, z, mid, theta) for z, mid, theta in zip(data, mids, thetas)]


def _get_param_estimates_1d(x, z, mid=None, theta=None) -> lm.Parameters:
    """Returns estimate of params and some reasonable limits. Const forced to zero!!"""
    params = lm.Parameters()
    dT = np.nanmax(z) - np.nanmin(z)
    if mid is None:
        mid = (x[np.nanargmax(z)] + x[np.nanargmin(z)]) / 2  #
    if theta is None:
        theta = abs((x[np.nanargmax(z)] - x[np.nanargmin(z)]) / 2.5)

    params.add_many(('mid', mid, True, None, None, None, None),
                    ('theta', theta, True, 0, 500, None, None),
                    ('const', 0, False, None, None, None, None),
                    ('dS', 0, True, -5, 5, None, None),
                    ('dT', dT, True, -10, 50, None, None))
    return params


def fit_entropy_1d(x, z, params: lm.Parameters = None, auto_bin=False):
    entropy_model = lm.Model(entropy_nik_shape)
    z = pd.Series(z, dtype=np.float32)
    if np.count_nonzero(~np.isnan(z)) > 10:  # Don't try fit with not enough data
        z, x = CU.remove_nans(z, x)
        if auto_bin is True and len(z) > FIT_NUM_BINS:
            logger.debug(f'Binning data of len {len(z)} before fitting')
            bin_size = int(np.ceil(len(z) / FIT_NUM_BINS))
            x, z = CU.bin_data([x, z], bin_size)
        if params is None:
            params = get_param_estimates(x, z)[0]

        result = entropy_model.fit(z, x=x, params=params, nan_policy='omit')
        return result
    else:
        return None


def fit_entropy_data(x, z, params: Optional[Union[List[lm.Parameters], lm.Parameters]] = None, auto_bin=False):
    if params is None:
        params = [None] * z.shape[0]
    else:
        params = CU.ensure_params_list(params, z)
    if z.ndim == 1:  # 1D data
        return [fit_entropy_1d(x, z, params[0], auto_bin=auto_bin)]
    elif z.ndim == 2:  # 2D data
        fit_result_list = []
        for i in range(z.shape[0]):
            fit_result_list.append(fit_entropy_1d(x, z[i, :], params[i], auto_bin=auto_bin))
        return fit_result_list


def integrate_entropy(data, scaling):
    """Integrates entropy data with scaling factor along last axis

    Args:
        data (np.ndarray): Entropy data
        scaling (float): scaling factor from dT, amplitude, dx

    Returns:
        np.ndarray: Integrated entropy units of Kb with same shape as original array
    """

    return np.nancumsum(data, axis=-1) * scaling


def scaling(dt, amplitude, dx):
    """Calculate scaling factor for integrated entropy from dt, amplitude, dx

    Args:
        dt (float): The difference in theta of hot and cold (in units of plunger gate).
            Note: Using lock-in dT is 1/2 the peak to peak, for Square wave it is the full dT
        amplitude (float): The amplitude of charge transition from the CS
        dx (float): How big the DAC steps are in units of plunger gate
            Note: Relative to the data passed in, not necessarily the original x_array

    Returns:
        float: Scaling factor to multiply cumulative sum of data by to convert to entropy
    """
    return dx / amplitude / dt


@dataclass
class EntropySignalProcess(Process):
    """
    Taking data which has been separated into the 4 parts of square heating wave and making entropy signal (2D)
    by averaging together cold and hot parts, then subtracting to get entropy signal
    """

    def set_inputs(self, x: np.ndarray, separated_data: np.ndarray,
                   ):
        self.inputs = dict(
            x=x,
            separated_data=separated_data,
        )

    def process(self):
        x = self.inputs['x']
        data = self.inputs['separated_data']
        data = np.atleast_3d(data)  # rows, steps, 4 heating setpoints
        cold = np.nanmean(np.take(data, get_transition_parts('cold'), axis=2), axis=2)
        hot = np.nanmean(np.take(data, get_transition_parts('hot'), axis=2), axis=2)
        entropy = cold-hot
        self.outputs = {
            'x': x,  # Worth keeping x-axis even if not modified
            'entropy': entropy
        }
        return self.outputs

    def get_input_plotter(self) -> DataPlotter:
        return DataPlotter(data=None, xlabel='Sweepgate /mV', ylabel='Repeats', data_label='Current /nA')

    def get_output_plotter(self,
                           y: Optional[np.ndarray] = None,
                           xlabel: str = 'Sweepgate /mV', data_label: str = f'{DELTA} Current /nA',
                           title: str = 'Entropy Signal',
                           ) -> DataPlotter:
        x = self.outputs['x']
        data = self.outputs['entropy']

        data = PlottableData(
            data=data,
            x=x,
        )

        plotter = DataPlotter(
            data=data,
            xlabel=xlabel,
            data_label=data_label,
            title=title,
        )
        return plotter


@dataclass
class EntropyFitProcess(Process):
    def set_inputs(self, x, entropy_data):
        self.inputs['x'] = x
        self.inputs['data'] = entropy_data
        # TODO: Add option to input initial param guesses

    def process(self):
        x = self.inputs['x']
        data = self.inputs['data']
        ndim = data.ndim  # To know whether to return a single or list of fits in the end

        data = np.atleast_2d(data)  # Might as well always assume 2D data to fit
        fits = fit_entropy_data(x, data, params=None)  # TODO: Add option to input initial param guesses
        fits = [FitInfo.from_fit(fit) for fit in fits]

        self.outputs['fits'] = fits
        if ndim == 1:
            return self.outputs['fits'][0]
        else:
            return self.outputs['fits']

    @staticmethod
    def ignore_keys_for_hdf() -> Optional[Union[str, List[str]]]:
        return ['outputs']

    def additional_save_to_hdf(self, dc_group: h5py.Group):
        if self.outputs:
            outputs_group = dc_group.require_group('outputs')
            fits_group = outputs_group.require_group('fits')
            for i, fit in enumerate(self.outputs['fits']):
                fit: FitInfo
                fit.save_to_hdf(fits_group, f'row{i}')

    @classmethod
    def additional_load_from_hdf(cls, dc_group: h5py.Group) -> Dict[str, Any]:
        additional_load = {}
        output = cls.load_output_only(dc_group)
        if output:
            additional_load = {'outputs': output}
        return additional_load

    @classmethod
    def load_output_only(cls, group: h5py.Group) -> dict:
        outputs = {}
        if 'outputs' in group.keys() and 'fits' in group['outputs'].keys():
            fit_group = group.get('outputs/fits')
            fits = []
            for k in sorted(fit_group.keys()):
                fits.append(FitInfo.from_hdf(fit_group, k))
            outputs['fits'] = fits
        return outputs


@dataclass
class EntropyIntegrationProcess(Process):
    def set_inputs(self, x, data, dT, amp):
        self.inputs['x'] = x
        self.inputs['data'] = data
        self.inputs['dT'] = dT
        self.inputs['amp'] = amp
        # TODO: Add more options for where to define zero on integration

    def process(self):
        dt, amp, x, data = [self.inputs[k] for k in ['dT', 'amp', 'x', 'data']]
        sf = scaling(dt=dt, amplitude=amp, dx=np.mean(np.diff(x)))

        self.outputs['scaling'] = sf
        self.outputs['integrated'] = integrate_entropy(data, sf)
        return self.outputs['integrated']



