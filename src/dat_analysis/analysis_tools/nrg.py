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
from .general_fitting import FitInfo, calculate_fit

from ..core_util import get_project_root, get_data_index, Data1D

logger = logging.getLogger(__name__)


def get_x_of_half_occ(params: lm.Parameters) -> float:
    """
    Get x value where occupation = 0.5
    (because NRG data has it's own energy scale and 0 in x is not quite 0.5 occupation)

    Args:
        params ():

    Returns:

    """

    nrg = NrgUtil(inital_params=NRGParams.from_lm_params(params))
    occ = nrg.data_from_params(x=np.linspace(params['mid'].value - 100, params['mid'].value + 100, 1000),
                               which_data='occupation')
    idx = get_data_index(occ.data, 0.5)
    return occ.x[idx]


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
    def from_old_mat(cls, path=r'D:\GitHub\dat_analysis\dat_analysis\resources\NRGResults.mat') -> NRGData:
        """Loads NRG data from .mat

        Note: this is the older NRG data which covers a much wider range of G/T, but isn't wide enough for G/T < 0.1 and
        isn't dense enough for G/T > ~10 ish. This is left here only for testing purposes
        """
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
        path = os.path.join(get_project_root(), r'resources\NRGResultsNew.mat')
        data = scipy.io.loadmat(path)
        rows_from_narrow = np.s_[0:10]  # 0 -> 9 are the thermal rows from first set of data
        dx_shape, dy_shape = data['Mu_mat'][:, rows_from_narrow].shape

        # Gamma broadened data (same as in above but much wider)
        path = os.path.join(get_project_root(), r'resources\NRGResultsNewWide.mat')
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
        path = os.path.join(get_project_root(), r'resources\NRGResultsNew.mat')
        data = scipy.io.loadmat(path)

        # Gamma broadened data (same as in above but much wider)
        path = os.path.join(get_project_root(), r'resources\NRGResultsNewWide.mat')
        wide_data = scipy.io.loadmat(path)

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
        path = os.path.join(get_project_root(), r'resources\NRGResultsNewWide.mat')
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

    def __init__(self, inital_params: Optional[NRGParams] = None):
        """
        Args:
            inital_params (): For running later fits
        """
        self.inital_params = inital_params if inital_params else NRGParams(gamma=1, theta=1)

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
            params = self.inital_params
        occupation = self.data_from_params(params=params, x=orig_x,
                                           which_data='occupation', which_x='sweepgate').data
        return occupation.data

    def data_from_params(self, params: Optional[NRGParams] = None,
                         x: Optional[np.ndarray] = None,
                         which_data: str = 'dndt',
                         which_x: str = 'sweepgate') -> Data1D:
        """Return 1D NRG data using parameters only"""
        if x is None:
            x = np.linspace(-1000, 1000, 1001)
        if params is None:
            params = self.inital_params

        nrg_data = nrg_func(x=x, mid=params.center, g=params.gamma, theta=params.theta,
                            amp=params.amp, lin=params.lin, const=params.const, occ_lin=params.lin_occ,
                            data_name=which_data)
        if which_x == 'occupation':
            x = self.get_occupation_x(x, params)
        return Data1D(x=x, data=nrg_data)

    def data_from_fit(self, x: np.ndarray, data: np.ndarray,
                      initial_params: Optional[Union[NRGParams, lm.Parameters]] = None,
                      which_data: str = 'dndt',
                      which_x: str = 'sweepgate',
                      which_fit_data: str = 'i_sense',
                      ) -> Data1D:
        fit = self.get_fit(x=x, data=data, initial_params=initial_params, which_data=which_fit_data)
        params = NRGParams.from_lm_params(fit.params)
        return self.data_from_params(params, x=x, which_data=which_data, which_x=which_x)
    def get_fit(self, x: np.ndarray, data: np.ndarray,
                initial_params: Optional[Union[NRGParams, lm.Parameters]] = None,
                which_data: str = 'i_sense'
                ) -> FitInfo:
        """
        Fit to NRG data
        Args:
            x (): sweepgate x of data to fit
            data (): data to fit
            initial_params (): Initial fit parameters as either NRGParams or lm.Parameters for more control)
            which_data (): Which NRG data is being fit (i.e. 'i_sense' or 'dndt')  # TODO: test for anything other than 'i_sense'

        Returns:

        """
        if initial_params is None:
            initial_params = self.inital_params

        if isinstance(initial_params, lm.Parameters):
            lm_pars = copy.copy(initial_params)
            if which_data != 'i_sense':  # remove unnecessary params
                for k in ['amp', 'lin', 'const', 'occ_lin']:
                    if k in lm_pars:
                        lm_pars.pop(k)
        else:
            lm_pars = initial_params.to_lm_params(which_data=which_data, x=x, data=data)

        fit = calculate_fit(x=x, data=data, params=lm_pars, func=NRG_func_generator(which=which_data),
                            method='powell')
        return fit




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
    center: Optional[float] = 0
    amp: Optional[float] = 1

    lin: Optional[float] = 0

    const: Optional[float] = 0

    lin_occ: Optional[float] = 0

    def to_lm_params(self, which_data: str = 'i_sense', x: Optional[np.ndarray] = None,
                     data: Optional[np.ndarray] = None) -> lm.Parameters:
        if x is None:
            x = [-1000, 1000]
        if data is None:
            data = [-10, 10]

        lm_pars = lm.Parameters()
        # Make lm.Parameters with some reasonable limits etc (these are common to all)
        lm_pars.add_many(
            ('mid', self.center, True, np.nanmin(x), np.nanmax(x), None, None),
            ('theta', self.theta, False, 0.5, 200, None, None),
            ('g', self.gamma, True, self.theta/1000, self.theta*50, None, None),
        )

        if which_data == 'i_sense':  # then add other necessary fitting parameters
            lm_pars.add_many(
                ('amp', self.amp, True, 0.1, 3, None, None),
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
        x_scaled = x / 0.0001  # 0.0001 == nrg_T
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
    index = get_data_index(tgs, t_over_gamma)
    index = index if tgs[index] > t_over_gamma else index - 1
    if index < 0:
        logger.warning(f'Theta/Gamma ratio {t_over_gamma:.4f} is higher than NRG range, will use {tgs[0]:.2f} instead')
        index = 0
    elif index > len(tgs) - 2:  # -2 because cached interpolator is going to look at next row as well
        logger.warning(f'Theta/Gamma ratio {t_over_gamma:.4f} is lower than NRG range, will use {tgs[-1]:.2f} instead')
        index = len(tgs) - 2
    return _cached_interpolator(lower_index=index, data_name=data_name)


@lru_cache(maxsize=100)  # Shouldn't ever be more than N rows of NRG data
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

    wide_ens, wide_data = ens[0], data[0]
    narrow_ens, narrow_data = ens[1], data[1]

    wide_ens, wide_data = strip_x_nans(wide_ens, wide_data)
    narrow_ens, narrow_data = strip_x_nans(narrow_ens, narrow_data)

    single_interper = interp1d(x=narrow_ens, y=narrow_data, bounds_error=False,
                               fill_value='extrapolate')  # TODO: extrapolate vs (0, 1)

    extrapolated_narrow_data = single_interper(x=wide_ens)

    # Note: Just returns edge value if outside of interp range
    # flips are because x and y must be strictly increasing
    interper = RectBivariateSpline(x=np.flip(wide_ens),
                                   y=np.flip(np.log10(tgs)),
                                   z=np.flip(np.array([wide_data, extrapolated_narrow_data]).T, axis=(0, 1)),
                                   kx=1, ky=1)

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
        return interped

    return func


@deprecated(deprecated_in='2021-06', details='Use "NRG_func_generator" instead')
def NRG_func_generator_old(which='i_sense') -> Callable[..., Union[float, np.ndarray]]:
    """
    Use this to generate the fitting function (i.e. to generate the equivalent of i_sense().
    It just makes sense in this case to do the setting up of the NRG data within a generator function.

    Note: RectBivariateSpline is a more efficient interp2d function for input data where x and y form a grid
    Args:

    Returns:

    """
    nrg = NRGData.from_old_mat()
    nrg_gamma = 0.001
    x_ratio = -1000
    if which == 'i_sense':
        z = 1 - nrg.occupation
    elif which == 'occupation':
        z = nrg.occupation
    elif which == 'dndt':
        z = nrg.dndt
    elif which == 'entropy':
        z = nrg.entropy
    elif which == 'int_dndt':
        z = nrg.int_dndt
    elif which == 'conductance':
        z = nrg.conductance
    else:
        raise NotImplementedError(f'{which} not implemented')
    interper = RectBivariateSpline(x=nrg.ens[0] * x_ratio, y=np.log10(nrg.ts / nrg_gamma),
                                   z=z.T, kx=1, ky=1)

    # 1-occupation to be comparable to CS data which decreases for increasing occupation
    # Log10 to help make y data more uniform for interper. Should not make a difference to fit values
    def nrg_func(x, mid, g, theta, amp=1, lin=0, const=0, occ_lin=0):
        """

        Args:
            x (): sweep gate
            mid (): center
            g (): gamma broadening
            theta (): thermal broadening
            amp (): charge sensor amplitude
            lin (): sweep gate charge sensor cross capacitance
            const (): charge sensor current average
            occ_lin (): screening of sweep gate/charge sensor cross capacitance due to increased occupation

        Returns:

        """
        x_scaled = (x - mid - g * (-1.76567) - theta * (
            -1)) / g  # To rescale varying temperature data with G instead (
        # double G is really half T). Note: The -g*(...) - theta*(...) is just to make the center roughly near OCC =
        # 0.5 (which is helpful for fitting only around the transition) x_scaled = (x - mid) / g

        # Note: the fact that NRG_gamma = 0.001 is taken into account with x_ratio above
        interped = interper(x_scaled, np.log10(theta / g)).flatten()
        if which == 'i_sense':
            interped = amp * (1 + occ_lin * (x - mid)) * interped + lin * (x - mid) + const - amp / 2
        # Note: (occ_lin*x)*Occupation is a linear term which changes with occupation,
        # not a linear term which changes with x
        return interped

    return nrg_func


if __name__ == '__main__':
    from ..dat_object.make_dat import get_dats
    from ..characters import PM
    from itertools import product
    import logging

    logging.basicConfig(level=logging.ERROR)


    def temp_nrg_fit(dat, theta=None, fit_name='forced_theta_linear_non_csq'):
        digamma_fit = dat.SquareEntropy.get_fit(fit_name=fit_name)  # Gamma = 0, but correct theta
        init_params = NRGParams.from_lm_params(digamma_fit.params)
        if theta is not None:
            init_params.theta = theta
        init_params.gamma = init_params.theta * 10
        nrg_fitter = NrgUtil(inital_params=init_params)

        out = dat.SquareEntropy.get_Outputs(name=fit_name)
        nrg_fit = nrg_fitter.get_fit(x=out.x, data=out.transition_part(which='cold'))
        return nrg_fit


    def nrg_fit_to_gamma_over_t(nrg_fit, verbose=True, datnum=None):
        g = nrg_fit.best_values.g
        gerr = nrg_fit.params['g'].stderr
        t = nrg_fit.best_values.theta
        terr = nrg_fit.params['theta'].stderr
        zerr = np.sqrt((terr / t) ** 2 + (gerr / g) ** 2)
        if verbose:
            datnum = datnum if datnum else '---'
            print(f'Dat{datnum}: G/T = {g / t:.3f}{PM}{g / t * zerr:.3f}')
        return g / t


    def fit_dats():
        dats = get_dats([2167, 2170, 2213])
        all_thetas, all_gts = [], []
        for dat in dats:
            fit_name = 'forced_theta_linear_non_csq'
            linear_theta = lm.models.LinearModel()
            slope = 3.4648e-5
            slope_err = 2.3495e-6
            intercept = 0.0509
            intercept_err = 7.765e-4

            params = linear_theta.make_params()
            possible_thetas = []
            slopes, intercepts = [(v - err, v + err) for v, err in zip([slope, intercept], [slope_err, intercept_err])]
            for slope, intercept in product(slopes, intercepts):
                params['slope'].value = slope
                params['intercept'].value = intercept
                possible_thetas.append(linear_theta.eval(x=dat.Logs.dacs['ESC'], params=params) * 100)

            thetas, gts = [], []  # For storing used ones
            for theta, text in zip([min(possible_thetas), max(possible_thetas)], ['min', 'max']):
                print(f'Dat{dat.datnum}: Using {text} Theta = {theta:.4f}')
                fit = temp_nrg_fit(dat=dat, theta=theta, fit_name=fit_name)
                # print(f'Dat{dat.datnum}:\n{fit}')
                gt = nrg_fit_to_gamma_over_t(fit, verbose=True, datnum=dat.datnum)
                thetas.append(theta)
                gts.append(gt)

            all_thetas.append(thetas)
            all_gts.append(gts)

        for dat, thetas, gts in zip(dats, all_thetas, all_gts):
            print(f'\nDat{dat.datnum}:')
            for i, text in enumerate(['Min', 'Max']):
                print(f'{text} Theta:\n'
                      f'Theta = {thetas[i]:.3f}mV\n'
                      f'Gamma = {gts[i] * thetas[i]:.3f}mV\n'
                      f'G/T = {gts[i]:.3f}\n')
            print(f'G/T={np.mean(gts):.2f}{PM}{(gts[0] - gts[1]) / 2:.2f}\n')


    import plotly.graph_objects as go
    import plotly.io as pio
    from ..plotting.plotly import TwoD, OneD

    p2d = TwoD(dat=None)
    p1d = OneD(dat=None)

    pio.renderers.default = 'browser'

    def plot_nrg_range() -> go.Figure:

        nrg = NRGData.from_mat()
        # nrg = NRGData.temp_from_wide_mat()
        nrgs = (NRGData.from_old_mat(), NRGData.from_mat())
        fig = p1d.figure(xlabel='Mu', ylabel='Occupation (arb.)', title='New NRG Occupation')
        for nrg, ttype in zip(nrgs, ['dash', 'solid']):
            every = 1
            for x, r, g, t in zip(nrg.ens[::every], nrg.occupation[::every], nrg.gs[::every], nrg.ts[::every]):
                fig.add_trace(p1d.trace(data=r, x=x, mode='lines', name=f'{g/t:.3f}', trace_kwargs=dict(line=dict(dash=ttype))))
        return fig


    def plot_nrg_comparison(which='occupation') -> go.Figure:
        new_func = NRG_func_generator(which)

        old_func = NRG_func_generator_old(which)

        x = np.linspace(-500, 500, 1001)
        T = 5

        fig = p1d.figure(xlabel='Sweepgate /mV', ylabel=which, title='Comparing New and Old NRG data')
        for func, name, ttype in zip([new_func, old_func], ['New', 'Old'], ['solid', 'dash']):
            for gt in np.logspace(np.log10(0.1), np.log10(10), 5):

                fig.add_trace(p1d.trace(x=x, data=func(x=x, mid=0, g=T*gt, theta=T), mode='lines',
                                        trace_kwargs=dict(line=dict(dash=ttype)), name=f'{name}_{gt:.2f}'))
        fig.update_layout(legend_title='G/T')
        return fig


    def plot_comparison_of_narrow_wide() -> go.Figure:
        """Comparing the row of data from new NRG narrow data with G/T = 0.1 and the new NRG wide data with the same
        G/T... Should hopefully lie on top of each other"""

        narrow_nrg = NRGData._from_new_mat_narrow_only()
        wide_nrg = NRGData._from_wide_mat_only()

        narrow_index = 10
        wide_index = 0

        fig = p1d.figure(xlabel='NRG Mu', ylabel='Occupation', title=f'Comparing common row of New Narrow and Wide NRG')
        for nrg, index, name in zip([narrow_nrg, wide_nrg], [narrow_index, wide_index], ['Narrow', 'Wide']):
            fig.add_trace(p1d.trace(data=nrg.occupation[index], x=nrg.ens[index],
                                    name=f'{name}: {nrg.gs[index]/nrg.ts[index]:.2f}', mode='lines+markers'))
        return fig


    def plot_x_axis_comparison() -> List[go.Figure()]:
        """Compare the x-axes of just the Narrow NRG data to the combined Narrow and Wide NRG data"""

        narrow_nrg = NRGData.from_mat(_use_wide=False)
        combined_nrg = NRGData.from_mat(_use_wide=True)

        step = 1
        figs = []
        for nrg, name in zip([narrow_nrg, combined_nrg], ['Narrow', 'Combined']):
            fig = p2d.figure(xlabel='None', ylabel='G/T',
                             title=f'X-axes of {name}')
            fig.add_trace(p2d.trace(data=nrg.ens[::step], x=np.linspace(0, 1, nrg.ens.shape[-1]), y=nrg.gs[::step]/nrg.ts[::step]))
            # for index in range(0, 41, 5):
            #     fig.add_trace(p1d.trace(data=nrg.ens[index], x=np.linspace(0, 1, len(nrg.ens[index])),
            #                         name=f'{name}: {nrg.gs[index]/nrg.ts[index]:.2f}', mode='lines+markers'))
            figs.append(fig)
        return figs


    # plot_comparison_of_narrow_wide().show()
    # figs = plot_x_axis_comparison()
    # [fig.show() for fig in figs]

    fig = plot_nrg_comparison(which='dndt')
    fig.show()

    # fit_dats()
    # fig = plot_nrg_range()
    # fig.show()