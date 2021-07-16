from __future__ import annotations
import numpy as np
from typing import List, Union, Tuple, Optional, Callable, Any, Iterable, Dict, TYPE_CHECKING
import lmfit as lm
import pandas as pd
from dataclasses import dataclass
import logging

import src.hdf_util
from src.hdf_util import NotFoundInHdfError, with_hdf_read, with_hdf_write
import src.core_util as CU
from src.dat_object.Attributes import DatAttribute as DA

if TYPE_CHECKING:
    from src.dat_object.dat_hdf import DatHDF

logger = logging.getLogger(__name__)
FIT_NUM_BINS = 1000

_pars = lm.Parameters()
_pars.add_many(('mid', 0, True, None, None, None, None),
               ('theta', 20, True, 0, 500, None, None),
               ('const', 0, False, None, None, None, None),
               ('dS', 0, True, -5, 5, None, None),
               ('dT', 5, True, -10, 50, None, None))
DEFAULT_PARAMS = _pars


def entropy_nik_shape(x, mid, theta, const, dS, dT):
    """fit to entropy curve"""
    arg = ((x - mid) / (2 * theta))
    return -dT * ((x - mid) / (2 * theta) - 0.5 * dS) * (np.cosh(arg)) ** (-2) + const


class Entropy(DA.FittingAttribute):
    version = '2.0.0'
    group_name = 'Entropy'
    description = 'Fitting to entropy shape (either measured by lock-in or from square heating)'
    DEFAULT_DATA_NAME = 'entropy_signal'

    def __init__(self, dat: DatHDF):
        super().__init__(dat)
        self._integrated_entropy = None
        self._integration_infos = {}

    @property
    def integrated_entropy(self):
        """Returns DEFAULT integrated entropy if previously calculated
        Note: Needs to be calculated with a passed in dT first using dat.Entropy.get_integrated_entropy()
        """
        if self._integrated_entropy is None:
            self._integrated_entropy = self.get_integrated_entropy()
        return self._integrated_entropy

    @property
    def integration_info(self):
        return self.get_integration_info('default')

    def set_integration_info(self,
                             dT: float,
                             amp: Optional[float] = None,
                             dx: Optional[float] = None,
                             sf: Optional[float] = None,
                             name: Optional[str] = None,
                             overwrite=False) -> IntegrationInfo:
        """
        Sets information required to calculate integrated entropy in HDF.
        Note: Mostly dT is required, others can be calculated from dat
        Args:
            dT (): Heating amount (will default to calculating from dc_info and biases)
            amp (): Charge sensor sensitivity (will default to dat.Transition.avg_fit.best_values.amp)
            dx (): Step size between measurements in gate potential (will default to step size of self.x)
            sf (): Scaling factor for integration (will default to calculating based on dT, amp, dx)
            name (): Name to save integration info under (will default to 'default')
            overwrite (): Whether to overwrite an existing IntegrationInfo

        Returns:
            (bool): True if successfully saved

        """
        if name is None:
            name = 'default'
        if self._integration_info_exists(name) and overwrite is False:
            raise FileExistsError(f'{name} IntegrationInfo already exists, to overwrite set overwrite=True')

        if amp is None:
            amp = self.dat.Transition.avg_fit.best_values.amp
        if dx is None:
            dx = abs((self.x[-1] - self.x[0]) / self.x.shape[-1])  # Should be same for avg_x or x
        if sf is None:
            sf = scaling(dT, amp, dx)

        int_info = IntegrationInfo(dT=dT, amp=amp, dx=dx, sf=sf)
        self._save_integration_info(name, int_info)
        self._integration_infos[name] = int_info
        return int_info

    def get_integration_info(self, name: Optional[str] = None) -> IntegrationInfo:
        """
        Returns named integration info (i.e. all things relevant to calculating integrated entropy).
        This also acts as a caching function to make things faster
        Args:
            name (): Name of integration info to look for (will default to 'default')

        Returns:
            (IntegrationInfo): Info relevant to calculating integrated entropy
        """
        if name is None:
            name = 'default'

        if name not in self._integration_infos:
            if self._integration_info_exists(name):
                self._integration_infos[name] = self._get_integration_info_from_hdf(name)
            else:
                raise NotFoundInHdfError(f'No IntegrationInfo found for dat{self.dat.datnum} with name {name}.\n'
                                         f'Use dat.Entropy.set_integration_info(..., name={name}) first')
        return self._integration_infos[name]

    @with_hdf_read
    def get_integration_info_names(self) -> List[str]:
        group = self.hdf.group.get('IntegrationInfo')
        return list(group.keys())

    # @lru_cache
    def get_integrated_entropy(self,
                               row: Optional[int] = None,
                               name: Optional[str] = None,
                               data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculates integrated entropy given optional info. Will look for saved scaling factor info if available in HDF

        Args:
            row (): Optionally specify a row of data to integrate, None will default to using avg_data
            name (): Optional name to look for or save scaling factor info under
            data (): nD Data to integrate (last axis is integrated)
                (Only use to override data being integrated, will by default use row or avg)

        Returns:
            (np.ndarray): Integrated entropy data
        """
        if name is None:
            name = 'default'

        # Get data to integrate
        if data is None:  # Which should usually be the case
            if row is None:
                use_avg = True
            else:
                assert type(row) == int
                use_avg = False
            if use_avg:
                data = self.avg_data
            else:
                data = self.data[row]

        int_info = self.get_integration_info(name)

        integrated = integrate_entropy(data, int_info.sf)
        return integrated

    @with_hdf_read
    def _get_integration_info_from_hdf(self, name: str) -> Optional[IntegrationInfo]:
        group = self.hdf.group.get('IntegrationInfo')
        return self.get_group_attr(name, check_exists=True, group_name=group.name, DataClass=IntegrationInfo)

    @with_hdf_read
    def _integration_info_exists(self, name: str) -> bool:
        group = self.hdf.group.get('IntegrationInfo')
        if name in group:
            return True
        return False

    @with_hdf_write
    def _save_integration_info(self, name: str, info: IntegrationInfo):
        group = self.hdf.group.get('IntegrationInfo')
        info.save_to_hdf(group, name)

    def default_data_names(self) -> List[str]:
        # return ['x', 'entropy_signal']
        raise RuntimeError(f'I am overriding set_default_data_descriptors, this should not be called')

    def clear_caches(self):
        super().clear_caches()
        # self.get_integrated_entropy.cache_clear()
        self._integrated_entropy = None
        self._integration_infos = {}

    def get_centers(self):
        if 'centers' in self.specific_data_descriptors_keys:
            return self.get_data('centers')
        else:
            return self.dat.Transition.get_centers()

    def get_default_params(self, x: Optional[np.ndarray] = None,
                           data: Optional[np.ndarray] = None) -> Union[List[lm.Parameters], lm.Parameters]:
        if x is not None and data is not None:
            params = get_param_estimates(x, data)
            if len(params) == 1:
                params = params[0]
            return params
        else:
            return DEFAULT_PARAMS

    def get_default_func(self) -> Callable[[Any], float]:
        return entropy_nik_shape

    @with_hdf_write
    def initialize_additional_FittingAttribute_minimum(self):
        group = self.hdf.group
        ii_group = group.require_group('IntegrationInfo')
        ii_group.attrs['Description'] = 'Stored information required to integrate entropy signal (i.e. dT, amp, scale ' \
                                        'factor).\nIf dT and amp are used to calculate scale factor, then all three are' \
                                        'stored, otherwise only scale factor is stored.\n' \
                                        'Multiplying entropy by scale factor gives integrated entropy'

    def set_default_data_descriptors(self):
        """
            Overriding to either get Square Entropy signal, or Lock-in Entropy signal rather than just looking for
            normal saved data

            Set the data descriptors required for fitting (e.g. x, and i_sense)
            Returns:

        """
        try:
            descriptor = self.get_descriptor('entropy_signal')
            x = self.get_descriptor('x')
            self.set_data_descriptor(descriptor, 'entropy_signal')  # Only copy descriptor if already exists
            self.set_data_descriptor(x, 'x')
        except NotFoundInHdfError:
            x, data, centers = get_entropy_signal_from_dat(self.dat)  # Get x as well, because Square Entropy makes it's own x
            self.set_data('entropy_signal', data)  # Save dataset because being calculated
            self.set_data('x', x)
            if centers is not None:
                centers = centers - np.average(centers)  # So that when making average_x it doesn't shift things further
                self.set_data('centers', centers)


@dataclass
class IntegrationInfo(src.hdf_util.DatDataclassTemplate):
    dT: Optional[float]
    amp: Optional[float]
    dx: Optional[float]
    sf: Optional[float]

    def to_df(self) -> pd.DataFrame:
        df = pd.DataFrame(data=[[getattr(self, k) for k in self.__annotations__]],
                          columns=[k for k in self.__annotations__])
        return df

    def integrate(self, data: np.ndarray) -> np.ndarray:
        return integrate_entropy(data, self.sf)


def get_entropy_signal_from_dat(dat: DatHDF) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    x = dat.Data.get_data('x')
    centers = None  # So that I can store centers if using Square Entropy which generates centers
    if dat.Logs.awg is not None:  # Assuming square wave heating, getting entropy signal from i_sense
        entropy_signal = dat.SquareEntropy.entropy_signal
        x = dat.SquareEntropy.x
        centers = np.array(dat.SquareEntropy.default_Output.centers_used)
    elif all([k in dat.Data.keys for k in ['entropy_x', 'entropy_y']]):  # Both x and y present, generate R and use that as signal
        entx, enty = [dat.Data.get_data(k) for k in ['entropy_x', 'entropy_y']]
        try:
            centers = dat.Transition.get_centers()
            logger.info(f'Using centers from dat.Transition to average entropyx/y data to best determine phase from avg')
        except NotFoundInHdfError:
            centers = None
        entropy_signal, entropy_angle = calc_r(entx, enty, x, centers=centers)
    elif 'entropy_x' in dat.Data.keys or 'entropy' in dat.Data.keys:  # Only entropy_x recorded so use that as entropy signal
        if 'entropy_x' in dat.Data.keys:
            entropy_signal = dat.Data.get_data('entropy_x')
        elif 'entropy' in dat.Data.keys:
            entropy_signal = dat.Data.get_data('entropy')
        else:
            raise ValueError
    else:
        raise NotFoundInHdfError(f'Did not find AWG in Logs and did not find entropy_x, entropy_y or entropy in data keys')
    return x, entropy_signal, centers


# class NewEntropy(DA.FittingAttribute):
#     version = '1.1'
#     group_name = 'Entropy'
#
#     """
#     Versions:
#         1.1 -- 20-7-20: Changed average_data to use centers not center_ids. Better way to average data
#     """
#
#     def __init__(self, dat):
#         self.angle = None  # type: Union[float, None]
#         super().__init__(dat)
#
#     def get_from_HDF(self):
#         super().get_from_HDF()  # Gets self.x/y/avg_fit/all_fits
#         dg = self.group.get('Data', None)
#         if dg is not None:
#             self.data = dg.get('entropy_r', None)
#             self.avg_data = dg.get('avg_entropy_r', None)
#             self.avg_data_err = dg.get('avg_entropy_r_err', None)
#         self.angle = self.group.attrs.get('angle', None)
#
#     def update_HDF(self):
#         super().update_HDF()
#         self.group.attrs['angle'] = self.angle
#
#     def recalculate_entr(self, centers, x_array=None):
#         """
#         Recalculate entropy r from 'entropy x' and 'entropy y' in HDF using center positions provided on x_array if
#         provided otherwise on original x_array.
#
#         Args:
#             centers (np.ndarray):  Center positions in units of x_array (either original or passed)
#             x_array (np.ndarray):  Option to pass an x_array that centers were defined on
#
#         Returns:
#             None: Sets self.data, self.angle, self.avg_data
#         """
#         x = x_array if x_array is not None else self.x
#         dg = self.group['Data']
#         entx = dg.get('entropy_x', None)
#         enty = dg.get('entropy_y', None)
#         assert entx not in [None, np.nan]
#         if enty is None or enty.size == 1:
#             entr = CU.center_data(x, entx, centers)  # To match entr which gets centered by calc_r
#             angle = 0.0
#         else:
#             entr, angle = calc_r(entx, enty, x=x, centers=centers)
#         self.data = entr
#         self.angle = angle
#
#         self.set_avg_data(centers='None')  # Because entr is already centered now
#         self.update_HDF()
#
#     def _set_data_hdf(self, **kwargs):
#         super()._set_data_hdf(data_name='entropy_r')
#
#     def run_row_fits(self, params=None, **kwargs):
#         super().run_row_fits(entropy_fits, params=params)
#
#     def _set_row_fits_hdf(self):
#         super()._set_row_fits_hdf()
#
#     def set_avg_data(self, centers=None, x_array=None):
#         if centers is not None:
#             logger.warning(f'Using centers to average entropy data, but data is likely already centered!')
#         super().set_avg_data(centers=centers, x_array=x_array)  # sets self.avg_data/avg_data_err and saves to HDF
#
#     def _set_avg_data_hdf(self):
#         dg = self.group['Data']
#         HDU.set_data(dg, 'avg_entropy_r', self.avg_data)
#         HDU.set_data(dg, 'avg_entropy_r_err', self.avg_data_err)
#
#     def run_avg_fit(self, params=None, **kwargs):
#         super().run_avg_fit(entropy_fits, params=params)  # sets self.avg_fit and saves to HDF
#
#     def _set_avg_fit_hdf(self):
#         super()._set_avg_fit_hdf()
#
#     def _check_default_group_attrs(self):
#         super()._check_default_group_attrs()
#
#     def _get_centers_from_transition(self):
#         assert 'Transition' in self.hdf.keys()
#         tg = self.hdf['Transition']  # type: h5py.Group
#         rg = tg.get('Row fits', None)
#         if rg is None:
#             raise AttributeError("No Rows Group in self.hdf['Transition'], this must be initialized first")
#         fit_infos = DA.rows_group_to_all_FitInfos(rg)
#         x = self.x
#         return CU.get_data_index(x, [fi.best_values.mid for fi in fit_infos])


def calc_r(entx, enty, x=None, centers=None):
    """
    Calculate R using constant phase determined at largest signal value of averaged data

    Args:
        entx (np.ndarray):  Entropy x signal (1D or 2D)
        enty (np.ndarray):  Entropy y signal (1D or 2D)
        x (np.ndarray): x_array for centering data with center values
        centers (np.ndarray): Center of transition to center data on

    Returns:
        (np.ndarray, float): 1D or 2D entropy r, phase angle
    """

    entx = np.atleast_2d(entx)
    enty = np.atleast_2d(enty)

    if x is None or centers is None:
        logger.warning('Not using centers to center data because x or centers missing')
        entxav = np.nanmean(entx, axis=0)
        entyav = np.nanmean(enty, axis=0)
    else:
        entxav = CU.mean_data(x, entx, centers, return_std=False)
        entyav = CU.mean_data(x, enty, centers, return_std=False)

    x_max, y_max, which = _get_max_and_sign_of_max(entxav, entyav)  # Gets max of x and y at same location
    # and which was bigger
    angle = np.arctan(y_max / x_max)

    entr = np.array([x * np.cos(angle) + y * np.sin(angle) for x, y in zip(entx, enty)])
    entangle = angle

    if entr.shape[0] == 1:  # Return to 1D if only one row of data
        entr = np.squeeze(entr, axis=0)
    return entr, entangle


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


def entropy_1d(x, z, params: lm.Parameters = None, auto_bin=False):
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


def entropy_fits(x, z, params: Optional[Union[List[lm.Parameters], lm.Parameters]] = None, auto_bin=False):
    if params is None:
        params = [None] * z.shape[0]
    else:
        params = CU.ensure_params_list(params, z)
    if z.ndim == 1:  # 1D data
        return [entropy_1d(x, z, params[0], auto_bin=auto_bin)]
    elif z.ndim == 2:  # 2D data
        fit_result_list = []
        for i in range(z.shape[0]):
            fit_result_list.append(entropy_1d(x, z[i, :], params[i], auto_bin=auto_bin))
        return fit_result_list


def _get_max_and_sign_of_max(x, y) -> Tuple[float, float, np.array]:
    """Returns value of x, y at the max position of the larger of the two and which was larger...
     i.e. x and y value at index=10 if max([x,y]) is x at x[10] and 'x' because x was larger

    Args:
        x (np.ndarray): x data (can be nD but probably better to average first to use 1D)
        y (np.ndarray): y data (can be nD but probably better to average first to use 1D)

    Returns:
        (float, float, str): x_max, y_max, which was larger of 'x' and 'y'
    """

    if np.nanmax(np.abs(x)) > np.nanmax(np.abs(y)):
        which = 'x'
        x_max, y_max = _get_values_at_max(x, y)
    else:
        which = 'y'
        y_max, x_max = _get_values_at_max(y, x)
    return x_max, y_max, which


def _get_values_at_max(larger, smaller) -> Tuple[float, float]:
    """Returns values of larger and smaller at position of max in larger

    Useful for calculating phase difference between x and y entropy data. Best to do at
    place where there is a large signal and then hold constant over the rest of the data

    Args:
        larger (np.ndarray): Data with the largest abs value
        smaller (np.ndarray): Data with the smaller abs value to be evaluated at the same index as the larger data

    Returns:
        (float, float): max(abs) of larger, smaller at same index
    """
    assert larger.shape == smaller.shape
    if np.abs(np.nanmax(larger)) > np.abs(np.nanmin(larger)):
        large_max = np.nanmax(larger)
        index = np.nanargmax(larger)
    else:
        large_max = float(np.nanmin(larger))
        index = np.nanargmin(larger)
    small_max = smaller[index]
    return large_max, small_max


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
