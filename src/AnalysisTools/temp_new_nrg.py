from src.AnalysisTools.nrg import NRGData

from typing import Callable, Union
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.io as sio
from scipy.interpolate import interp2d, interp1d, RectBivariateSpline
import time
import plotly.io as pio
from functools import lru_cache

from src.Dash.DatPlotting import OneD, TwoD
from src.CoreUtil import get_data_index

pio.renderers.default = 'browser'

p1d = OneD(dat=None)
p2d = TwoD(dat=None)


# def NRG_func_generator_new(which='i_sense') -> Callable[..., Union[float, np.ndarray]]:
#     """
#     Use this to generate the fitting function (i.e. to generate the equivalent of i_sense().
#     It just makes sense in this case to do the setting up of the NRG data within a generator function.
#
#     Note: RectBivariateSpline is a more efficient interp2d function for input data where x and y form a grid
#     Args:
#
#     Returns:
#
#     """
#     nrg = NRGData.from_new_mat()
#     x_ratio = -1000
#     if which == 'i_sense':
#         z = 1 - nrg.occupation
#     elif which == 'occupation':
#         z = nrg.occupation
#     elif which == 'dndt':
#         z = nrg.dndt
#     elif which == 'entropy':
#         z = nrg.entropy
#     elif which == 'int_dndt':
#         z = nrg.int_dndt
#     elif which == 'conductance':
#         z = nrg.conductance
#     else:
#         raise NotImplementedError(f'{which} not implemented')
#     interper = interp2d(x=nrg.ens * x_ratio, y=np.tile(np.log10(nrg.ts / nrg.gs), (nrg.ens.shape[-1], 1)).T,
#                         z=z.T, bounds_error=False, fill_value=np.nan)
#
#     # 1-occupation to be comparable to CS data which decreases for increasing occupation
#     # Log10 to help make y data more uniform for interper. Should not make a difference to fit values
#
#     def nrg_func(x, mid, g, theta, amp=1, lin=0, const=0, occ_lin=0):
#         """
#
#         Args:
#             x (): sweep gate
#             mid (): center
#             g (): gamma broadening
#             theta (): thermal broadening
#             amp (): charge sensor amplitude
#             lin (): sweep gate charge sensor cross capacitance
#             const (): charge sensor current average
#             occ_lin (): screening of sweep gate/charge sensor cross capacitance due to increased occupation
#
#         Returns:
#
#         """
#         x_scaled = (x - mid - g * (-1.76567) - theta * (-1)) / g  # To rescale varying temperature data with G instead (
#         # double G is really half T). Note: The -g*(...) - theta*(...) is just to make the center roughly near OCC =
#         # 0.5 (which is helpful for fitting only around the transition) x_scaled = (x - mid) / g
#
#         # Note: the fact that NRG_gamma = 0.001 is taken into account with x_ratio above
#         interped = interper(x_scaled, np.log10(theta / g)).flatten()
#         if which == 'i_sense':
#             interped = amp * (1 + occ_lin * (x - mid)) * interped + lin * (x - mid) + const - amp / 2
#         # Note: (occ_lin*x)*Occupation is a linear term which changes with occupation,
#         # not a linear term which changes with x
#         return interped
#
#     return nrg_func


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


def get_nearest_nrg_data(t_over_gamma, data_name) -> np.ndarray:
    """
    Get nearest two rows of NRG data to the ratio of t_over_gamma. Can be any of the data_names that exist in the
    NRG data.
    Args:
        t_over_gamma (): Theta over gamma ratio to get nearest rows for
        data_name (): Name of data to return (i.e. dndt, occupation etc)

    Returns:
        2 rows of NRG data closest to ratio of t_over_gamma
    """
    nrg = NRGData.from_new_mat()
    gs = nrg.gs
    ts = nrg.ts
    nrg_t_over_gs = ts / gs
    nearest = get_data_index(nrg_t_over_gs, t_over_gamma)

    if nrg_t_over_gs[nearest] < t_over_gamma:
        a, b = nearest - 1, nearest
    else:
        a, b = nearest, nearest + 1

    if data_name == 'tgs':
        return nrg_t_over_gs[a:b + 1]
    else:
        data = get_nrg_data(data_name)
        return data[a:b + 1]


def get_interpolator(t_over_gamma: float, data_name: str = 'i_sense') -> Callable:
    """
    Generates a function which acts like a 2D interpolator between the closest t_over_gamma values of NRG data.
    Args:
        t_over_gamma ():
        data_name ():

    Returns:
        Effective interpolator function which takes same args as nrg_func
        i.e. (x, mid, g, theta, amp=1, lin=0, const=0, occ_lin=0)  where the optionals are only used for i_sense
    """
    # Get the rows of data above and below in g/t
    # TODO: Need to cache results after I know what the nearest Gamma/T ratios are
    tgs = get_nearest_nrg_data(t_over_gamma, 'tgs')  # Nearest Gamma/T ratios in NRG
    ens = get_nearest_nrg_data(t_over_gamma, 'ens')
    data = get_nearest_nrg_data(t_over_gamma, data_name)

    # TODO: Could possibly interpolate the wider one to match the same ens as the narrower one and then use RectBivariateSpline instead of interp2d

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


    # # Only use data as wide as the narrower of the two in ens
    # ids = get_data_index(ens[1], [ens[0][0], ens[0][-1]], is_sorted=False)
    # # Flatten accepted ens and data into 1D array (for interp2d)
    # all_ens, all_data = [np.concatenate([arr[0], arr[1][ids[0]:ids[1]]]) for arr in (ens, data)]
    # all_tgs = np.concatenate([np.repeat(tgs[0], ens.shape[-1]), np.repeat(tgs[1], (ids[1]-ids[0]))])
    # interper = interp2d(x=all_ens, y=np.log10(all_tgs), z=all_data, bounds_error=False, fill_value=np.nan)  # TODO: data.T?

    interp_func = interper_to_nrg_func(interper, data_name)
    return interp_func


def interper_to_nrg_func(interper, data_name: str):
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
        # x_shifted = x - mid - g * (-1.76567) - theta * (-1)
        x_shifted = x
        x_scaled = x_shifted * 0.0001/theta  # *nrg_T
        # x_scaled = x_shifted
        return x_scaled
    else:
        # x_scaled = x*g
        x_scaled = x / 0.0001
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
    interper = get_interpolator(t_over_gamma=theta / g, data_name=data_name)
    return interper(x, mid, g, theta, amp=amp, lin=lin, const=const, occ_lin=occ_lin)


if __name__ == '__main__':
    data = sio.loadmat('../../resources/NRGResultsNew.mat')

    for k in ['Conductance_mat', 'DNDT_mat', 'Mu_mat', 'Occupation_mat', 'T', 'Gammas']:
        print(k, data[k].shape)

    d = NRGData.from_new_mat()
    # t1 = time.time()
    # # f = NRG_func_generator_new('i_sense')
    # print(time.time()-t1)

    x = np.linspace(-1000, 1000, 1001)

    arr = nrg_func(x, 0, 1, 1, data_name='dndt')
    fig = px.line(x=x, y=arr)
    arr = nrg_func(x, 0, 50, 1, data_name='dndt')
    fig.add_trace(go.Scatter(x=x, y=arr, mode='lines')).show()

    # from src.DatObject.Attributes.Transition import i_sense_digamma
    # fig = go.Figure()
    # nrg = NRGData.from_new_mat()
    # for i, (x, d, t, g) in enumerate(zip(nrg.ens, nrg.dndt, nrg.ts, nrg.gs)):
    #     # if i in [0, 10, 11, 15, 20, 30, 39]:
    #     if i in [11, 15, 20, 30, 39]:
    #     # if i == 20:
    #         digamma_x = np.linspace(x[0]*5, x[-1]*5, num=200)
    #         digamma = i_sense_digamma(digamma_x, 0, g, t, 1, 0, 0)*-1 + 0.5
    #         digamma_x = scale_x(digamma_x*10000, 0, g*10000, 0, inverse=False)
    #         fig.add_trace(p1d.trace(x=digamma_x, data=digamma, name=f'digamma {t/g:.2f}', mode='lines'))
    #         fig.add_trace(p1d.trace(x=x, data=d, name=f'nrg {t/g:.2f}'))
    #
    # fig.show()
