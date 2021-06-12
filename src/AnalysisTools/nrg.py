from __future__ import annotations
from dataclasses import dataclass
from deprecation import deprecated
from functools import lru_cache
from typing import Optional, Union, Callable
import logging
import copy
import os
import lmfit as lm

import numpy as np
import scipy.io
from scipy.interpolate import RectBivariateSpline, interp1d
from src.AnalysisTools.general_fitting import FitInfo, calculate_fit

from src.Dash.DatPlotting import Data1D
from src.CoreUtil import get_project_root, get_data_index
logger = logging.getLogger(__name__)


def nrg_func(x, mid, g, theta, amp: float = 1, lin: float = 0, const: float = 0, occ_lin: float = 0,
             data_name='i_sense') -> Union[float, np.ndarray]:
    """
    Returns data interpolated from NRG results. I.e. acts like an analytical function for fitting etc.

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
    @deprecated(details='Use "from_new_mat" instead')
    @lru_cache
    def from_mat(cls, path=r'D:\GitHub\dat_analysis\dat_analysis\resources\NRGResults.mat') -> NRGData:
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
    def from_new_mat(cls, path=os.path.join(get_project_root(), r'resources\NRGResultsNew.mat')) -> NRGData:
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
            ('g', self.gamma, True, 0.2, 4000, None, None),
        )

        if which_data == 'i_sense':  # then add other necessary fitting parameters
            lm_pars.add_many(
                ('amp', self.amp, True, 0.1, 3, None, None),
                ('lin', self.lin, True, 0, 0.005, None, None),
                ('occ_lin', self.lin_occ, True, -0.0003, 0.0003, None, None),
                ('const', self.const, True, np.nanmin(data), np.nanmax(data), None, None),
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


NEW = True


if NEW:
    @deprecated(details='Use "nrg_func" instead')
    def NRG_func_generator(which='i_sense') -> Callable[..., Union[float, np.ndarray]]:
        from functools import wraps

        @wraps(nrg_func)
        def wrapper(*args, **kwargs):
            return nrg_func(*args, **kwargs, data_name=which)
        return wrapper
else:
    @deprecated(details='Use "nrg_func" instead')
    def NRG_func_generator(which='i_sense') -> Callable[..., Union[float, np.ndarray]]:
        """
        Use this to generate the fitting function (i.e. to generate the equivalent of i_sense().
        It just makes sense in this case to do the setting up of the NRG data within a generator function.

        Note: RectBivariateSpline is a more efficient interp2d function for input data where x and y form a grid
        Args:

        Returns:

        """
        nrg = NRGData.from_mat()
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
            x_scaled = (x - mid - g * (-1.76567) - theta * (-1)) / g  # To rescale varying temperature data with G instead (
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
class NrgUtil:

    """For working with 1D NRG Data. I.e. generating and fitting"""

    nrg = NRGData.from_new_mat()

    def __init__(self, inital_params: Optional[NRGParams] = None):
        """
        Args:
            inital_params (): For running later fits
        """
        self.inital_params = inital_params if inital_params else NRGParams(gamma=1, theta=1)

    def get_occupation_x(self, orig_x: np.ndarray, params: NRGParams) -> np.ndarray:
        """
        Convert from sweepgate x values to occupation x values
        Args:
            orig_x (): Sweepgate x values
            params (): NRGParams to generate Occupation data (only requires mid, theta, gamma)

        Returns:
            Occupation x values
        """
        occupation = self.data_from_params(params=params, x=orig_x,
                                           which_data='occupation', which_x='sweepgate').data
        # TODO: Might need to think about what happens when occupation is 0 or 1 in the tails
        # interp_range = np.where(np.logical_and(occupation < 0.999, occupation > 0.001))
        #
        # interp_data = occupation[interp_range]
        # interp_x = orig_x[interp_range]
        #
        # interper = interp1d(x=interp_x, y=interp_data, assume_sorted=True, bounds_error=False)
        #
        # occ_x = interper(orig_x)
        return occupation

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
                            amp=params.amp, lin=params.lin, const=params.const, occ_lin=params.lin_occ, data_name=which_data)
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




def get_nrg_data(data_name: str):
    """Returns just the named data array from NRG data"""
    nrg = NRGData.from_new_mat()
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
    ts, gs, ens, data = [get_nrg_data(name)[lower_index:lower_index + 2] for name in ['ts', 'gs', 'ens', data_name]]
    tgs = ts / gs

    wide_ens, wide_data = ens[0], data[0]
    narrow_ens, narrow_data = ens[1], data[1]
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
        return interped

    return func


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