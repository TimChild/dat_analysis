from __future__ import annotations
from typing import Callable
from scipy.interpolate import RectBivariateSpline, interp1d
import scipy.io
from dataclasses import dataclass
import numpy as np
import plotly.io as pio
from functools import lru_cache
import lmfit as lm
from itertools import product


from src.DatObject.Make_Dat import get_dat
from src.Dash.DatPlotting import OneD, TwoD
from src.AnalysisTools.fitting import calculate_fit

pio.renderers.default = "browser"


# ### Just here as a hint -- taken from fitting page
# def NRG_fitter() -> Callable:
#     NRG = scipy.io.loadmat('NRGResults.mat')
#     occ = NRG["Occupation_mat"]
#     ens = np.reshape(NRG["Ens"], 401)
#     ts = np.reshape(NRG["Ts"], 70)
#     ens = np.flip(ens)
#     occ = np.flip(occ, 0)
#     interp = RectBivariateSpline(ens, np.log10(ts), occ, kx=1, ky=1)
#
#     def interpNRG(x, logt, dx=1, amp=1, center=0, lin=0, const=0):
#         ens = np.multiply(np.add(x, center), dx)
#         curr = [interp(en, logt)[0][0] for en in ens]
#         scaled_current = np.multiply(curr, amp)
#         scaled_current += const + np.multiply(lin, x)
#         return scaled_current
#
#     return interpNRG


def NRG_func_generator(which='occupation') -> Callable:
    """
    Use this to generate the fitting function (i.e. to generate the equivalent of i_sense().
    It just makes sense in this case to do the setting up of the NRG data within a generator function.

    Note: RectBivariateSpline is a more efficient interp2d function for input data where x and y form a grid
    Args:

    Returns:

    """
    nrg = NRGData.from_mat()
    nrg_gamma = 0.001
    x_ratio = 1000  # Some arbitrary ratio to make NRG equivalent to i_sense/digamma x scaling (e.g. to get same theta)
    if which == 'i_sense':
        z = nrg.occupation
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
    interper = RectBivariateSpline(x=nrg.ens*x_ratio, y=np.log10(nrg.ts/nrg_gamma),
                                   z=z.T)
    # 1-occupation to be comparable to CS data which decreases for increasing occupation
    # Log10 to help make y data more uniform for interper. Should not make a difference to fit values

    ens_scaling_interper = interp1d(x=nrg.ts, y=nrg.ts, assume_sorted=True)
    # Need to scale the energy axis with temperature

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
        x_scaled = ens_scaling_interper(theta/g)*(x-mid)/theta
        interped = interper(x_scaled, np.log10(theta/g)).flatten()
        if which == 'i_sense':
            interped = amp*(1+occ_lin*(x-mid))*interped + lin*(x-mid)+const - amp*(1+occ_lin)/2
        # Note: (occ_lin*x)*Occupation is a linear term which changes with occupation,
        # not a linear term which changes with x
        return interped
    return nrg_func


@dataclass
class NRGData:
    ens: np.ndarray
    ts: np.ndarray
    conductance: np.ndarray
    dndt: np.ndarray
    entropy: np.ndarray
    occupation: np.ndarray
    int_dndt: np.ndarray

    @classmethod
    @lru_cache
    def from_mat(cls, path=r'D:\OneDrive\GitHub\dat_analysis\dat_analysis\resources\NRGResults.mat') -> NRGData:
        import os
        print(os.path.abspath('.'))
        data = scipy.io.loadmat(path)
        return cls(
            ens=np.flip(data['Ens'].flatten(), axis=-1),
            ts=data['Ts'].flatten(),
            conductance=np.flip(data['Conductance_mat'].T, axis=-1),
            dndt=np.flip(data['DNDT_mat'].T, axis=-1),
            entropy=np.flip(data['Entropy_mat'].T, axis=-1),
            occupation=np.flip(data['Occupation_mat'].T, axis=-1),
            int_dndt=np.flip(data['intDNDT_mat'].T, axis=-1),
        )


if __name__ == '__main__':
    # Weakly coupled entropy dat
    dat = get_dat(2164)
    out = dat.SquareEntropy.get_Outputs(name='default')
    x = out.x
    data = np.nanmean(out.averaged[(0, 2,), :], axis=0)

    plotter = OneD(dat=dat)

    fig = plotter.figure(ylabel='Current /nA', title=f'Dat{dat.datnum}: Fitting Weakly coupled to NRG')

    fig.add_trace(plotter.trace(x=x, data=data, name='Data', mode='lines'))

    print(dat.SquareEntropy.get_fit(fit_name='default').best_values)
    params = lm.Parameters()
    params.add_many(
        ('mid', 0, True, None, None, None, None),
        ('theta', 3.8, True, 0.01, None, None, None),
        ('amp', 0.93, True, 0, None, None, None),
        ('lin', 0.0015, True, 0, None, None, None),
        ('occ_lin', 0, True, None, None, None, None),
        ('const', 7.2, True, None, None, None, None),
        ('g', 0, False, 0.5, 200, None, None),
    )

    fit = calculate_fit(x, data, params=params, func=NRG_func_generator())

    # model = lm.model.Model(NRG_fitter())
    # init_fit = model.eval(params=params, x=x)

    fig.add_trace((plotter.trace(x=x, data=fit.eval_init(x=x), name='Initial Fit', mode='lines')))
    fig.add_trace((plotter.trace(x=x, data=fit.eval_fit(x=x), name='Final Fit', mode='lines')))

    print(fit.best_values)

    fig.show()

