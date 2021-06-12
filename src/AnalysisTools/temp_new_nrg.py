from src.AnalysisTools.nrg import NRGData

from typing import Callable, Union
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.io as sio
from scipy.interpolate import interp2d, interp1d, RectBivariateSpline
import logging
import time
import plotly.io as pio
from functools import lru_cache

from src.Dash.DatPlotting import OneD, TwoD
from src.CoreUtil import get_data_index

logger = logging.getLogger(__name__)

pio.renderers.default = 'browser'

p1d = OneD(dat=None)
p2d = TwoD(dat=None)


def get_nrg_data(data_name: str):
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
    return cached_interpolator(lower_index=index, data_name=data_name)


@lru_cache(maxsize=100)  # Shouldn't ever be more than N rows of NRG data
def cached_interpolator(lower_index: int, data_name: str) -> Callable:
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
        x_shifted = x - mid - g * (-2.2) - theta * (-1.5)  # Just choosing values which make 0.5 occ be near 0
        x_scaled = x_shifted * 0.0001 / theta  # 0.0001 == nrg_T
        return x_scaled
    else:
        x_scaled = x / 0.0001  # 0.0001 == nrg_T
        x_shifted = x_scaled + mid + g * (-2.2) + theta * (-1.5)
        return x_shifted


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
    interper = get_interpolator(t_over_gamma=theta / g, data_name=data_name)
    return interper(x, mid, g, theta, amp=amp, lin=lin, const=const, occ_lin=occ_lin)


if __name__ == '__main__':
    from src.AnalysisTools.nrg import NRG_func_generator

    data = sio.loadmat('../../resources/NRGResultsNew.mat')

    d = NRGData.from_new_mat()
    # t1 = time.time()
    # # f = NRG_func_generator_new('i_sense')
    # print(time.time()-t1)

    x = np.linspace(-200, 200, 1001)

    old_func = NRG_func_generator('i_sense')
    fig = go.Figure()
    for c, g, theta in zip([1, 2, 3], [10, 10, 10], [10, 10, 10]):
        l=0
        o=0
        a=1
        c=1
        m=0
        arr = old_func(x, m, g, theta, amp=a, lin=l, occ_lin=o, const=c)
        fig.add_trace(go.Scatter(x=x, y=arr, mode='lines', name='Old', line=dict(dash='dash')))
        arr = nrg_func(x, m, g, theta, amp=a, lin=l, occ_lin=o, const=c, data_name='i_sense')
        fig.add_trace(go.Scatter(x=x, y=arr, mode='lines', name='New'))
    fig.show()
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
