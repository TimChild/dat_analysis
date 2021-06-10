from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Union, Callable
import copy
import os

import lmfit as lm
import numpy as np
import scipy.io
from scipy.interpolate import RectBivariateSpline, interp2d

from src.AnalysisTools.general_fitting import FitInfo, calculate_fit
from src.Dash.DatPlotting import Data1D
from src.CoreUtil import get_project_root


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
            entropy=np.zeros(data['DNDT_mat'].shape),  # No entropy data for new NRG
            occupation=data['Occupation_mat'].T,
            int_dndt=np.zeros(data['DNDT_mat'].shape),  # No entropy data for new NRG
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
    interper = RectBivariateSpline(x=nrg.ens * x_ratio, y=np.log10(nrg.ts / nrg_gamma),
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


class NrgGenerator:
    """For generating 1D NRG Data"""

    nrg = NRGData.from_mat()

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

        nrg_func = NRG_func_generator(which_data)
        nrg_data = nrg_func(x=x, mid=params.center, g=params.gamma, theta=params.theta,
                            amp=params.amp, lin=params.lin, const=params.const, occ_lin=params.lin_occ)
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
