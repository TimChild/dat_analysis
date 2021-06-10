from src.AnalysisTools.nrg import NRGData

from typing import Callable, Union
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.io as sio
from scipy.interpolate import interp2d
import time
import plotly.io as pio

from src.Dash.DatPlotting import OneD, TwoD

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
if __name__ == '__main__':
    data = sio.loadmat('../../resources/NRGResultsNew.mat')

    for k in ['Conductance_mat', 'DNDT_mat', 'Mu_mat', 'Occupation_mat', 'T', 'Gammas']:
        print(k, data[k].shape)

    d = NRGData.from_new_mat()
    t1 = time.time()
    f = NRG_func_generator_new('i_sense')
    print(time.time()-t1)

    x = np.linspace(-100, 100, 201)
    gs = d.gs/0.0001

    data = np.array([f(x, 0, g, 5) for g in gs])

    p2d.plot(data, x=x, y=gs).show()


