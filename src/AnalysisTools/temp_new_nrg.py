from src.AnalysisTools.nrg import NRGData

from typing import Callable, Union
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.io as sio
from scipy.interpolate import interp2d
import time
import plotly.io as pio
from functools import lru_cache

from src.Dash.DatPlotting import OneD, TwoD
from src.CoreUtil import get_data_index

pio.renderers.default = 'browser'

p1d = OneD(dat=None)
p2d = TwoD(dat=None)


def NRG_func_generator_new(which='i_sense') -> Callable[..., Union[float, np.ndarray]]:
    """
    Use this to generate the fitting function (i.e. to generate the equivalent of i_sense().
    It just makes sense in this case to do the setting up of the NRG data within a generator function.

    Note: RectBivariateSpline is a more efficient interp2d function for input data where x and y form a grid
    Args:

    Returns:

    """
    nrg = NRGData.from_new_mat()
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
    interper = interp2d(x=nrg.ens * x_ratio, y=np.tile(np.log10(nrg.ts / nrg.gs), (nrg.ens.shape[-1], 1)).T,
                        z=z.T, bounds_error=False, fill_value=np.nan)

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


def get_nrg_data(data_name: str):
    nrg = NRGData.from_new_mat()
    if data_name == 'i_sense':
        z = 1 - nrg.occupation
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


def get_nearest_nrg_data(gamma_over_t, data_name) -> np.ndarray:
    nrg = NRGData.from_new_mat()
    gs = nrg.gs
    ts = nrg.ts
    nrg_g_over_t = gs/ts
    nearest = get_data_index(nrg_g_over_t, gamma_over_t)

    if nrg_g_over_t[nearest] >= gamma_over_t:
        a, b = nearest-1, nearest
    else:
        a, b = nearest, nearest+1

    if data_name == 'gts':
        return nrg_g_over_t[a:b+1]
    else:
        data = get_nrg_data(data_name)
        return data[a:b+1]


@lru_cache()
def get_interpolator(gamma_over_t: float, data_name: str = 'i_sense') -> Callable:
    """
    Generates a function which acts like a 2D interpolator between the closest gamma_over_t values of NRG data.
    Args:
        gamma_over_t ():
        data_name ():

    Returns:
        Effective interpolator function which takes same args as nrg_func
        i.e. (x, mid, g, theta, amp=1, lin=0, const=0, occ_lin=0)  where the optionals are only used for i_sense
    """
    # Get the rows of data above and below in g/t
    gts = get_nearest_nrg_data(gamma_over_t, 'gts')  # Nearest Gamma/T ratios in NRG
    ens = get_nearest_nrg_data(gamma_over_t, 'ens')
    data = get_nearest_nrg_data(gamma_over_t, data_name)

    # Only use data as wide as the narrower of the two in ens
    # TODO: Could possibly interpolate the wider one to match the same ens as the narrower one and then use RectBivariateSpline instead of interp2d
    # TODO: Which is the wider array? 0 or 1?
    ids = get_data_index(ens[1], [ens[0][0], ens[0][-1]], is_sorted=False)

    # Flatten accepted ens and data into 1D array (for interp2d)
    all_ens, all_data = [np.concatenate([arr[0], arr[1][ids[0]:ids[1]]]) for arr in (ens, data)]
    all_gts = np.concatenate([np.repeat(gts[0], ens.shape[-1]), np.repeat(gts[1], (ids[1]-ids[0]))])

    interper = interp2d(x=all_ens * -1000, y=np.log10(all_gts), z=all_data, bounds_error=False, fill_value=np.nan)  # TODO: data.T?
    interp_func = interper_to_nrg_func(interper, data_name)
    return interp_func


def interper_to_nrg_func(interper, data_name: str):
    """Makes a function which takes normal fitting arguments and returns that function"""
    def func(x, mid, g, theta, amp=1, lin=0, const=0, occ_lin=0):
        x_scaled = scale_x(x, mid, g, theta)

        # TODO: Is this still True? # Note: the fact that NRG_gamma = 0.001 is taken into account with x_ratio above
        interped = interper(x_scaled, np.log10(theta/g)).flatten()
        if data_name == 'i_sense':
            interped = amp * (1 + occ_lin * (x - mid)) * interped + lin * (x - mid) + const - amp / 2
            # Note: (occ_lin*x)*Occupation is a linear term which changes with occupation,
            # not a linear term which changes with x
        return interped
    return func


def scale_x(x, mid, g, theta, inverse=False):
    """
    To rescale varying temperature data with G instead (double G is really half T).
    Note: The -g*(...) - theta*(...) is just to make the center roughly near OCC = 0.5 (which is helpful for fitting only around the transition)
    x_scaled = (x - mid) / g

    Args:
        x ():
        mid ():
        g ():
        theta ():
        inverse (): set True to reverse the scaling

    Returns:

    """
    if not inverse:
        x_shifted = x - mid - g * (-1.76567) - theta * (-1)
        x_scaled = x_shifted/g
        return x_scaled
    else:
        x_scaled = x*g
        x_shifted = x_scaled + mid + g * (-1.76567) + theta * (-1)
        return x_shifted


def nrg_func(x, mid, g, theta, amp=1, lin=0, const=0, occ_lin=0, data_name='i_sense') -> Union[float, np.ndarray]:
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
    interper = get_interpolator(gamma_over_t=g/theta, data_name=data_name)
    return interper(x, mid, g, theta, amp=amp, lin=lin, const=const, occ_lin=occ_lin)


if __name__ == '__main__':
    data = sio.loadmat('../../resources/NRGResultsNew.mat')

    for k in ['Conductance_mat', 'DNDT_mat', 'Mu_mat', 'Occupation_mat', 'T', 'Gammas']:
        print(k, data[k].shape)

    d = NRGData.from_new_mat()
    # t1 = time.time()
    # # f = NRG_func_generator_new('i_sense')
    # print(time.time()-t1)

    x = np.linspace(-100, 100, 201)
    gs = d.gs/0.0001


    arr = nrg_func(x, 0, 10, 1, data_name='i_sense')



