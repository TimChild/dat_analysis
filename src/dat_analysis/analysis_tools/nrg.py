from __future__ import annotations
from dataclasses import dataclass
from deprecation import deprecated
from functools import lru_cache
from typing import Optional, Union, Callable, Tuple, List
import logging
import copy
import os
import lmfit as lm
import numpy as np
import scipy.io
from scipy.interpolate import RectBivariateSpline, interp1d
from importlib.resources import files

from dat_analysis.analysis_tools.general_fitting import FitInfo, calculate_fit

from dat_analysis.core_util import get_data_index, Data1D

logger = logging.getLogger(__name__)


@dataclass
class NRGData:
    ens: np.ndarray
    ts: np.ndarray
    conductance: np.ndarray
    dndt: np.ndarray
    entropy: np.ndarray
    occupation: np.ndarray
    int_dndt: np.ndarray
    gs: np.ndarray

    @classmethod
    def from_mat(cls, _use_wide=True):
        """
        Loads NRG data from .mat files

        Note: _use_wide should ALWAYS be True, it is here for testing purposes only
        Args:
            _use_wide ():  For testing purposes only. If set to False, then gamma broadened data will not be wide enough

        Returns:

        """
        if _use_wide:
            return cls._from_new_mat()
        else:
            return cls._from_new_mat_narrow_only()

    @classmethod
    @deprecated(details='Use "from_new_mat" instead')
    @lru_cache
    def from_old_mat(cls) -> NRGData:
        """Loads NRG data from .mat
        Note: this is the older NRG data which covers a much wider range of G/T, but isn't wide enough for G/T < 0.1 and
        isn't dense enough for G/T > ~10 ish. This is left here only for testing purposes
        """
        path = files('dat_analysis.resources').joinpath('NRGResults.mat')
        data = scipy.io.loadmat(path)
        return cls(
            ens=np.tile(data['Ens'].flatten(), (len(data['Ts'].flatten()), 1)),  # New data has ens for each row
            ts=data['Ts'].flatten(),
            conductance=data['Conductance_mat'].T,
            dndt=data['DNDT_mat'].T,
            entropy=data['Entropy_mat'].T,
            occupation=data['Occupation_mat'].T,
            int_dndt=data['intDNDT_mat'].T,
            gs=np.array([0.001] * len(data['Ts'].flatten()))  # New data has gamma for each row
        )

    @classmethod
    @lru_cache
    def _from_new_mat(cls):
        """Combines two new NRG datasets (the first has good thermally broadened data, but isn't wide enough for
        gamma broadened data. The second is more gamma broadened data only over a wider range but with the same density
        of points (i.e. a differently shaped array)

        This combines both and adds NoNs to the narrower data so that they can still be treated as arrays.
        """
        def pad_to_shape(arr: np.ndarray, desired_x_shape: int):
            """Pads array with NaNs so that it has a given x dimension"""
            if arr.shape[-1] > desired_x_shape:
                raise RuntimeError(f'{arr.shape[-1]} > {desired_x_shape}')
            diff = desired_x_shape - arr.shape[-1]
            pads = [(0, 0)] * (arr.ndim - 1)
            # pads.extend([(np.floor(diff / 2).astype(int), np.ceil(diff / 2).astype(int))])  # Equal before and after
            pads.extend([(0, diff)])  # Pad all NaNs, at end of data
            return np.pad(arr, pad_width=pads, mode='constant', constant_values=np.nan)

        NRG_DATAS = ['Mu_mat',
                     'Conductance_mat',
                     'DNDT_mat',
                     'Entropy_mat',
                     'Occupation_mat',
                     'intDNDT_mat']

        # Thermally broadened data (includes gamma broadened which isn't wide enough)
        path = files('dat_analysis.resources').joinpath('NRGResultsNew.mat')
        data = scipy.io.loadmat(path)
        rows_from_narrow = np.s_[0:10]  # 0 -> 9 are the thermal rows from first set of data
        dx_shape, dy_shape = data['Mu_mat'][:, rows_from_narrow].shape

        # Gamma broadened data (same as in above but much wider)
        path = files('dat_analysis.resources').joinpath('NRGResultsNewWide.mat')
        wide_data = scipy.io.loadmat(path)
        wx_shape, wy_shape = wide_data['Mu_mat'].shape

        common_x_shape = wx_shape  # This has the larger shape

        new_data = {}
        for k in NRG_DATAS:
            if k in data and wide_data:
                d = data[k].T[rows_from_narrow]
                padded = pad_to_shape(d, common_x_shape)
                new_data[k] = np.concatenate([padded, wide_data[k].T], axis=0)
            else:
                # Just getting shape using an array I know will exist
                full_shape = (dy_shape + wy_shape, common_x_shape)
                new_data[k] = np.zeros(full_shape)
        new_data['Ts'] = np.array([data['T'][0, 0]] * dy_shape + [wide_data['T'][0, 0]] * wy_shape)
        new_data['Gammas'] = np.concatenate([data.get('Gammas').flatten()[rows_from_narrow],
                                             wide_data.get('Gammas').flatten()])

        return cls(
            ens=new_data['Mu_mat'],
            ts=new_data['Ts'],
            conductance=new_data['Conductance_mat'],
            dndt=new_data['DNDT_mat'],
            entropy=new_data['Entropy_mat'],
            occupation=new_data['Occupation_mat'],
            int_dndt=new_data['intDNDT_mat'],
            gs=new_data['Gammas'],
        )

    @classmethod
    @deprecated(details=f"Gamma data here isn't wide enough, use the newer combined data instead")
    @lru_cache
    def _from_new_mat_narrow_only(cls) -> NRGData:
        """Loads new NRG data which has a higher density of points over the region we can access experimentally.
        Unfortunately the Gamma broadened data here is not wide enough here, but I'm leaving it for future testing
        purposes"""
        # Thermally broadened data (includes gamma broadened which isn't wide enough)
        path = files('dat_analysis.resources').joinpath('NRGResultsNew.mat')
        data = scipy.io.loadmat(path)

        return cls(
            ens=data['Mu_mat'].T,
            ts=np.array([data['T'][0, 0]] * len(data['Gammas'].flatten())),
            conductance=data['Conductance_mat'].T,
            dndt=data['DNDT_mat'].T,
            entropy=np.zeros(data['DNDT_mat'].shape).T,  # No entropy data for new NRG
            occupation=data['Occupation_mat'].T,
            int_dndt=np.zeros(data['DNDT_mat'].shape).T,  # No entropy data for new NRG
            gs=data['Gammas'].flatten(),
        )

    @classmethod
    @deprecated(details=f"This doesn't include any thermally broadened data, use the combined data instead")
    @lru_cache
    def _from_wide_mat_only(cls) -> NRGData:
        """Loads the new wider NRG data only. This doesn't include any thermally broadened data, so this is only here
         for testing purposes. """
        # Gamma broadened data (same as in above but much wider)
        path = files('dat_analysis.resources').joinpath('NRGResultsNewWide.mat')
        data = scipy.io.loadmat(path)
        return cls(
            ens=data['Mu_mat'].T,
            ts=np.array([data['T'][0, 0]] * len(data['Gammas'].flatten())),
            conductance=data['Conductance_mat'].T,
            dndt=data['DNDT_mat'].T,
            entropy=np.zeros(data['DNDT_mat'].shape).T,  # No entropy data for new NRG
            occupation=data['Occupation_mat'].T,
            int_dndt=np.zeros(data['DNDT_mat'].shape).T,  # No entropy data for new NRG
            gs=data['Gammas'].flatten(),
        )


class NrgUtil:

    """For working with 1D NRG Data. I.e. generating and fitting"""

    nrg = NRGData.from_mat()

    def __init__(self, initial_params: Optional[NRGParams] = None):
        """
        Args:
            inital_params (): For running later fits
        """
        self.initial_params = initial_params if initial_params else NRGParams(gamma=1, theta=1)

    @staticmethod
    def make_params(mid=None, amp=None, const=None, lin=None, theta=None, g=None, occ_lin=None) -> NRGParams:
        """Helper to make NRGParams given"""
        return NRGParams(gamma=g, theta=theta, center=mid, amp=amp, lin=lin, const=const, lin_occ=occ_lin)

    def init_params(self, mid=None, amp=None, const=None, lin=None, theta=None, g=None, occ_lin=None) -> NRGParams:
        """Make NRG params and set as the initial params for running get_fit method"""
        self.initial_params = self.make_params(mid=mid, amp=amp, const=const, lin=lin, theta=theta, g=g, occ_lin=occ_lin)
        return self.initial_params

    def get_occupation_x(self, orig_x: np.ndarray, params: Optional[NRGParams] = None) -> np.ndarray:
        """
        Convert from sweepgate x values to occupation x values
        Args:
            orig_x (): Sweepgate x values
            params (): NRGParams to generate Occupation data (only requires mid, theta, gamma)

        Returns:
            Occupation x values
        """
        if params is None:
            params = self.initial_params
        occupation = self.data_from_params(params=params, x=orig_x,
                                           which_data='occupation', which_x='sweepgate').data
        return occupation

    def data_from_params(self, params: Optional[NRGParams] = None,
                         x: Optional[np.ndarray] = None,
                         which_data: str = 'dndt',
                         which_x: str = 'sweepgate',
                         center = None,
                         gamma = None,
                         theta = None,
                         amp = None,
                         lin = None,
                         const = None,
                         lin_occ = None,
                         ) -> Data1D:
        """
        Return 1D NRG data using parameters only

        Args:
            params ():
            x ():
            which_data ():
            which_x (): Whether to use sweepgate, ens, or occupation as x-axis
                (sweepgate is just ens after shifting s.t. N=0.5 ~ x=0 and scaling to account for varying theta)

        Returns:

        """
        if x is None:
            x = np.linspace(-1000, 1000, 1001)
        if params is None:
            params = self.initial_params

        center = params.center if center is None else center
        gamma = params.gamma if gamma is None else gamma
        theta = params.theta if theta is None else theta
        amp = params.amp if amp is None else amp
        lin = params.lin if lin is None else lin
        const = params.const if const is None else const
        lin_occ = params.lin_occ if lin_occ is None else lin_occ

        nrg_data = nrg_func(x=x, mid=center, g=gamma, theta=theta,
                            amp=amp, lin=lin, const=const, occ_lin=lin_occ,
                            data_name=which_data)
        if which_x == 'sweepgate':  # This is the default behaviour (that the x is already scaled to be more like
            # sweepgate)
            pass
        elif which_x == 'ens':
            # Do the same x-scaling as in nrg_func to convert sweepgate to ens
            x = scale_x(x, params.center, params.gamma, params.theta)
        elif which_x == 'occupation':
            x = self.get_occupation_x(x, params)
        else:
            raise ValueError(f'{which_x} not recognized. Must be one of ("sweepgate", "ens", "occupation")')
        return Data1D(x=x, data=nrg_data)

    def data_from_fit(self, x: np.ndarray, data: np.ndarray,
                      initial_params: Optional[Union[NRGParams, lm.Parameters]] = None,
                      which_data: str = 'dndt',
                      which_x: str = 'sweepgate',
                      which_fit_data: str = 'i_sense',
                      ) -> Data1D:
        """Fits data and then returns calculate NRG based on those fit parameters"""
        fit = self.get_fit(x=x, data=data, initial_params=initial_params, which_data=which_fit_data)
        params = NRGParams.from_lm_params(fit.params)
        return self.data_from_params(params, x=x, which_data=which_data, which_x=which_x)

    def get_fit(self, x: np.ndarray, data: np.ndarray,
                initial_params: Optional[Union[NRGParams, lm.Parameters]] = None,
                which_data: str = 'i_sense',
                vary_theta: Optional[bool] = None,
                vary_gamma: Optional[bool] = None,
                ) -> FitInfo:
        """
        Fit to NRG data
        Args:
            x (): sweepgate x of data to fit
            data (): data to fit
            initial_params (): Initial fit parameters as either NRGParams or lm.Parameters for more control)
            which_data (): Which NRG data is being fit (i.e. 'i_sense' or 'dndt')  # TODO: test for anything other than 'i_sense'
            vary_theta (): Override whether theta should vary or not (None will not do anything)
            vary_gamma (): Override whether gamma should vary or not (None will not do anything)

        Returns:

        """
        if initial_params is None:
            initial_params = copy.copy(self.initial_params)

        if isinstance(initial_params, lm.Parameters):
            lm_pars = copy.copy(initial_params)
            if which_data != 'i_sense':  # remove unnecessary params
                for k in ['amp', 'lin', 'const', 'occ_lin']:
                    if k in lm_pars:
                        lm_pars.pop(k)
        else:
            lm_pars = initial_params.to_lm_params(which_data=which_data, x=x, data=data)

        if vary_theta is not None:
            lm_pars['theta'].vary = vary_theta
        if vary_gamma is not None:
            lm_pars['g'].vary = vary_gamma

        fit = calculate_fit(x=x, data=data, params=lm_pars, func=NRG_func_generator(which=which_data),
                            method='powell')
        return fit


def get_x_of_half_occ(params: lm.Parameters = None, theta=None, g=None) -> float:
    """
    Get x value where occupation = 0.5
    (because NRG data has its own energy scale and 0 in x is not quite 0.5 occupation)
    NOTE: Only works between -1000 to 1000

    Args:
        params (): lm.Parameters for which the center of occupation should be found (no theta or g if params passed)
        theta (): if not passing params, pass theta of data
        g (): if not passing params, pass gamma of data

    Returns:

    """
    if params is None and theta is not None and g is not None:
        params = NRGParams(gamma=g, theta=theta).to_lm_params()
    assert params is not None

    nrg = NrgUtil(initial_params=NRGParams.from_lm_params(params))
    occ = nrg.data_from_params(x=np.linspace(params['mid'].value - 1000, params['mid'].value + 1000, 10000),
                               which_data='occupation')
    idx = get_data_index(occ.data, 0.5)
    return occ.x[idx]


def nrg_func(x, mid, g, theta, amp: float = 1, lin: float = 0, const: float = 0, occ_lin: float = 0,
             data_name='i_sense') -> Union[float, np.ndarray]:
    """
    Returns data interpolated from NRG results. I.e. acts like an analytical function for fitting etc.

    Note:
    Note: Does not require amp, lin, const, occ_lin for anything other than 'i_sense' fitting (which just adds terms to
    occupation)
    Args:
        x ():
        mid ():
        g ():
        theta ():
        amp ():
        lin ():
        const ():
        occ_lin ():
        data_name (): Which NRG data to return (i.e. occupation, dndt, i_sense)

    Returns:

    """
    interper = _get_interpolator(t_over_gamma=theta / g, data_name=data_name)
    return interper(x, mid, g, theta, amp=amp, lin=lin, const=const, occ_lin=occ_lin)


def NRG_func_generator(which='i_sense') -> Callable[..., Union[float, np.ndarray]]:
    """
    Wraps the nrg_func in a way that can be used by lmfit. If not using lmfit, then just call nrg_func directly

    Args:
        which (): Which data to make a function for (i.e. 'i_sense', 'occupation', 'dndt', etc)

    Returns:
        nrg_func for named data
    """
    from functools import wraps

    @wraps(nrg_func)
    def wrapper(*args, **kwargs):
        return nrg_func(*args, **kwargs, data_name=which)

    return wrapper


@dataclass
class NRGParams:
    """The parameters that go into NRG fits. Easier to make this, and then this can be turned into lm.Parameters or
    can be made from lm.Parameters"""
    gamma: float
    theta: float
    center: Optional[float] = None
    amp: Optional[float] = None
    lin: Optional[float] = None
    const: Optional[float] = None
    lin_occ: Optional[float] = None

    def __post_init__(self):
        self.center = 0 if self.center is None else self.center
        self.amp = 1 if self.amp is None else self.amp
        self.lin = 0 if self.lin is None else self.lin
        self.const = 0 if self.const is None else self.const
        self.lin_occ = 0 if self.lin_occ is None else self.lin_occ

    def to_lm_params(self, which_data: str = 'i_sense', x: Optional[np.ndarray] = None,
                     data: Optional[np.ndarray] = None) -> lm.Parameters:
        if x is None:
            x = [-1000, 1000]
        if data is None:
            data = [-10, 10]

        lm_pars = lm.Parameters()
        # Make lm.Parameters with some reasonable limits etc (these are common to all)
        gamma = self.gamma if self.gamma else 0.001*self.theta  # Setting Gamma == 0 breaks fitting because of divide by gamma
        lm_pars.add_many(
            ('mid', self.center, True, np.nanmin(x), np.nanmax(x), None, None),
            ('theta', self.theta, False, 0.5, 200, None, None),
            ('g', gamma, True, self.theta/1000, self.theta*50, None, None),  # Limit to Range of NRG G/Ts
        )

        if which_data == 'i_sense':  # then add other necessary fitting parameters
            lm_pars.add_many(
                ('amp', self.amp, True, 0.01, 3, None, None),
                ('lin', self.lin, True, 0, 0.005, None, None),
                ('occ_lin', self.lin_occ, True, -0.0003, 0.0003, None, None),
                ('const', self.const, True, np.nanmin(data), np.nanmax(data), None, None),
            )
        elif which_data == 'dndt':
            lm_pars.add_many(
                ('amp', self.amp, True, 0, None, None, None),  # Rescale the arbitrary NRG dndt
                ('lin', 0, False, None, None, None, None),
                ('occ_lin', 0, False, None, None, None, None),
                ('const', 0, False, None, None, None, None),
            )
        elif which_data == 'conductance':
            lm_pars.add_many(
                ('amp', self.amp, True, 0, None, None, None),  # Amplitude of Data unlikely to match NRG
                ('lin', 0, False, None, None, None, None),
                ('occ_lin', 0, False, None, None, None, None),
                ('const', self.const, False, None, None, None, None),  # May want to allow for offset, but default to
                # not vary
            )
        else:  # Just necessary because of general nrg_func. All these paramters do nothing
            lm_pars.add_many(
                ('amp', 0, False, None, None, None, None),
                ('lin', 0, False, None, None, None, None),
                ('occ_lin', 0, False, None, None, None, None),
                ('const', 0, False, None, None, None, None),
            )
        return lm_pars

    @classmethod
    def from_lm_params(cls, params: lm.Parameters) -> NRGParams:
        d = {}
        for k1, k2 in zip(['gamma', 'theta', 'center', 'amp', 'lin', 'const', 'lin_occ'],
                          ['g', 'theta', 'mid', 'amp', 'lin', 'const', 'occ_lin']):
            par = params.get(k2, None)
            if par is not None:
                v = par.value
            elif k1 == 'gamma':
                v = 0  # This will cause issues if not set somewhere else, but no better choice here.
            else:
                v = 0 if k1 != 'amp' else 1  # Most things should default to zero except for amp
            d[k1] = v
        return cls(**d)
        
    @classmethod
    def guess_params(cls, x: np.ndarray, data: np.ndarray, theta=None, gamma=None) -> NRGParams:
        assert data.ndim == 1
        gamma = gamma if gamma is not None else 0.001
        from .transition import get_param_estimates
        lm_pars = get_param_estimates(x, data)
        theta = theta if theta is not None else lm_pars['theta'].value
        center = lm_pars['mid'].value
        amp = lm_pars['amp'].value
        const = lm_pars['const'].value
        lin = lm_pars['lin'].value        
        lin_occ = 0
        return cls(gamma=gamma, theta=theta, const=const, lin=lin, amp=amp, center=center, lin_occ=lin_occ)
    


def get_nrg_data(data_name: str):
    """Returns just the named data array from NRG data"""
    nrg = NRGData.from_mat()
    if data_name == 'i_sense':
        z = 1 - nrg.occupation
    elif data_name == 'ts':
        z = nrg.ts
    elif data_name == 'gs':
        z = nrg.gs
    elif data_name == 'occupation':
        z = nrg.occupation
    elif data_name == 'dndt':
        z = nrg.dndt
    elif data_name == 'entropy':
        z = nrg.entropy
    elif data_name == 'int_dndt':
        z = nrg.int_dndt
    elif data_name == 'conductance':
        z = nrg.conductance
    elif data_name == 'ens':
        z = nrg.ens
    else:
        raise NotImplementedError(f'{data_name} not implemented')
    return z


def scale_x(x, mid, g, theta, inverse=False):
    """
    New NRG ONLY (Jun 2021+)
    To rescale sweepgate data to match the ens of NRG (with varying theta).

    Note: The -g*(...) - theta*(...) is just to make the center roughly near OCC = 0.5 (which is helpful for fitting
    only around the transition)

    x_scaled ~ (x - mid) * nrg_theta / theta

    Args:
        x ():
        mid ():
        g ():
        theta ():
        inverse (): set True to reverse the scaling

    Returns:

    """
    if not inverse:
        x_shifted = x - mid - g * (-2.2) - theta * (-1.5)  # Just choosing values which make 0.5 occ be near 0
        x_scaled = x_shifted * 0.0001 / theta  # 0.0001 == nrg_T
        return x_scaled
    else:
        x_scaled = x / 0.0001 * theta   # 0.0001 == nrg_T
        x_shifted = x_scaled + mid + g * (-2.2) + theta * (-1.5)
        return x_shifted


def _get_interpolator(t_over_gamma: float, data_name: str = 'i_sense') -> Callable:
    """
    Generates a function which acts like a 2D interpolator between the closest t_over_gamma values of NRG data.
    Args:
        t_over_gamma ():
        data_name ():

    Returns:
        Effective interpolator function which takes same args as nrg_func
        i.e. (x, mid, g, theta, amp=1, lin=0, const=0, occ_lin=0)  where the optionals are only used for i_sense
    """
    ts, gs = [get_nrg_data(name) for name in ['ts', 'gs']]
    tgs = ts / gs
    index = get_data_index(tgs, t_over_gamma)  # get nearest value
    index = index if tgs[index] > t_over_gamma else index - 1  # want the true value to be between interpolated rows
    if index < 0:  # Asking for data outside of calculation range
        logger.debug(f'Theta/Gamma ratio {t_over_gamma:.4f} is higher than NRG range, will use {tgs[0]:.2f} instead')
        index = 0
    elif index > len(tgs) - 2:  # -2 because cached interpolator is going to look at next row as well
        logger.debug(f'Theta/Gamma ratio {t_over_gamma:.4f} is lower than NRG range, will use {tgs[-1]:.2f} instead')
        index = len(tgs) - 2
    return _cached_interpolator(lower_index=index, data_name=data_name)


@lru_cache(maxsize=100)  # Shouldn't ever be more than XX rows of NRG data (XX == size of data in .mat files)
def _cached_interpolator(lower_index: int, data_name: str) -> Callable:
    """
    Actually generates the scipy 2D interpolator for NRG data.
    This can be used for any future requests of this interpolator
    so this should be cached.

    Args:
        lower_index (): The lower index of NRG data to use for interpolation (will always interpolate between this and
            lower_index + 1)
        data_name (): Which NRG data to make an interpolator for

    Returns:
        2D interpolator function which takes x as an energy and y as a gamma/theta ratio.
    """
    def strip_x_nans(x: np.array, z: np.array) -> Tuple[np.ndarray, np.ndarray]:
        """Strip off NaNs that are in x array (and corresponding data)"""
        return x[np.where(~np.isnan(x))], z[np.where(~np.isnan(x))]

    ts, gs, ens, data = [get_nrg_data(name)[lower_index:lower_index + 2] for name in ['ts', 'gs', 'ens', data_name]]
    tgs = ts / gs

    narrower_ens, narrower_data = ens[0], data[0]  # Just the
    wider_ens, wider_data = ens[1], data[1]

    narrower_ens, narrower_data = strip_x_nans(narrower_ens, narrower_data)
    wider_ens, wider_data = strip_x_nans(wider_ens, wider_data)

    single_interper = interp1d(x=wider_ens, y=wider_data, bounds_error=False,
                               fill_value='extrapolate')  # values are saturated near edge of NRG data,
    # so effectively constants for extrapolation

    interpolated_wider_data = single_interper(x=narrower_ens)  # i.e. mapping wider data to narrower ens

    # Note: Just returns edge value if outside interp range
    # flips are because x and y must be strictly increasing
    interper = RectBivariateSpline(x=np.flip(narrower_ens),
                                   y=np.flip(np.log10(tgs)),
                                   z=np.flip(np.array([narrower_data, interpolated_wider_data]).T, axis=(0, 1)),
                                   kx=1, ky=1)
    # Note: the interpolator does not use the parts of the wider data that extend beyond the narrower data

    interp_func = _interper_to_nrg_func(interper, data_name)
    return interp_func


def _interper_to_nrg_func(interper, data_name: str):
    """Makes a function which takes normal fitting arguments and returns that function"""

    def func(x, mid, g, theta, amp=1, lin=0, const=0, occ_lin=0):
        x_scaled = scale_x(x, mid, g, theta)

        interped = interper(x_scaled, np.log10(theta / g)).flatten()
        if data_name == 'i_sense':
            interped = amp * (1 + occ_lin * (x - mid)) * interped + lin * (x - mid) + const - amp / 2
            # Note: (occ_lin*x)*Occupation is a linear term which changes with occupation,
            # not a linear term which changes with x
        elif data_name == 'dndt':
            interped *= amp
        elif data_name == 'conductance':
            interped = amp*interped + const
        return interped

    return func


if __name__ == '__main__':
    nrg = NrgUtil()
